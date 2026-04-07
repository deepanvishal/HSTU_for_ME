# ============================================================
# NB_06 — Model_HSTU_provider.py
# Purpose : Train HSTU for provider-level recommendation
#           Reuses HSTUBlock + RelativeBucketedTimeAndPositionBias
#           from hstu_pytorch.py — key differentiator vs SASRec/BERT4Rec
# Key differences from SASRec/BERT4Rec:
#   delta_t_matrix used as timestamps — encodes time gaps between visits
#   Relative bucketed time + position bias inside attention
#   SiLU activation (not softmax) for attention weights
#   Causal mask (like SASRec) — next-item prediction
#   Trigger token appended as position MAX_SEQ_LEN
# Same as NB_04/05:
#   seq_matrix (N, 20, 2) — provider + specialty per slot
#   trigger_token (N, 1) — trigger_dx_id
#   InfoNCE loss with batched ops
#   from_provider_to_cands for hard negatives in collate_fn
# Sources : ./cache_provider_{SAMPLE}/ (from NB_01/02)
# Output  : ./models/hstu_provider_{SAMPLE}_ep{N}.pt
#            ./output/hstu_train_log_{SAMPLE}.csv
# ============================================================

import gc
import math
import os
import pickle
import time
from functools import partial
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from IPython.display import display, Markdown

print("Imports done")
print(f"PyTorch: {torch.__version__}")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
SAMPLE        = "5pct"
MAX_SEQ_LEN   = 20
PAD_IDX       = 0
UNK_IDX       = 1
# HSTU-specific dims — separate from embedding_dim
D_MODEL       = 128        # provider + specialty embedding dim
DELTA_T_DIM   = 32         # delta_t embedding dim — concatenated into x (matches working model)
NUM_RATINGS   = 16         # number of delta_t buckets (matches HSTU_model.py)
HSTU_DIM      = D_MODEL + DELTA_T_DIM   # = 160 — actual dim inside HSTUBlocks
LINEAR_DIM    = 64         # HSTU linear (value) projection dim per head
ATTENTION_DIM = 32         # HSTU attention (query/key) dim per head
N_HEADS       = 4
N_LAYERS      = 2
DROPOUT       = 0.2
ATTN_DROPOUT  = 0.1
N_BUCKETS     = 64         # time-gap buckets for relative bias
LR            = 1e-3
WEIGHT_DECAY  = 1e-2
BATCH_SIZE    = 1024
EPOCHS        = 1
WARMUP_STEPS  = 200
NEG_K         = 128
HARD_NEG_K    = 32
K_VALUES      = [1, 3, 5]
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

CACHE_DIR  = f"./cache_provider_{SAMPLE}"
MODEL_DIR  = "./models"
OUTPUT_DIR = "./output"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Device: {DEVICE} | Sample: {SAMPLE} | Epochs: {EPOCHS}")
display(Markdown(f"""
## Config
| Parameter | Value |
|---|---|
| Sample | {SAMPLE} |
| d_model | {D_MODEL} |
| linear_dim | {LINEAR_DIM} |
| attention_dim | {ATTENTION_DIM} |
| Heads | {N_HEADS} |
| Layers | {N_LAYERS} |
| Batch size | {BATCH_SIZE} |
| Epochs | {EPOCHS} |
| Device | {DEVICE} |
"""))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD CACHE
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 1 — Load Cache"))

with open(f"{CACHE_DIR}/provider_vocab.pkl",  "rb") as f: provider_vocab  = pickle.load(f)
with open(f"{CACHE_DIR}/specialty_vocab.pkl", "rb") as f: specialty_vocab = pickle.load(f)
with open(f"{CACHE_DIR}/dx_vocab.pkl",        "rb") as f: dx_vocab        = pickle.load(f)
with open(f"{CACHE_DIR}/from_provider_to_cands.pkl", "rb") as f:
    from_provider_to_cands = pickle.load(f)

PROVIDER_VOCAB_SIZE = len(provider_vocab)
SPEC_VOCAB_SIZE     = len(specialty_vocab)
DX_VOCAB_SIZE       = len(dx_vocab)

keys = ["seq_matrix", "delta_t_matrix", "trigger_token", "seq_lengths",
        "lab_t30", "lab_t60", "lab_t180",
        "is_t30", "is_t60", "is_t180",
        "member_ids", "trigger_dates", "trigger_dxs", "segments",
        "from_provider_ids"]

train_data = {k: np.load(f"{CACHE_DIR}/train_{k}.npy", allow_pickle=True) for k in keys}
val_data   = {k: np.load(f"{CACHE_DIR}/val_{k}.npy",   allow_pickle=True) for k in keys}

N_train = train_data["seq_matrix"].shape[0]
N_val   = val_data["seq_matrix"].shape[0]

print(f"Train: {N_train:,} | Val: {N_val:,}")
print(f"Provider vocab: {PROVIDER_VOCAB_SIZE:,} | Spec: {SPEC_VOCAB_SIZE:,} | DX: {DX_VOCAB_SIZE:,}")
print(f"Section 1 done — {time.time()-t0:.1f}s")
display(Markdown(f"**Train:** {N_train:,} | **Val:** {N_val:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATASET + COLLATE
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 2 — Dataset"))


def sparse_to_multihot(label_list, vocab_size):
    v = torch.zeros(vocab_size, dtype=torch.float32)
    if len(label_list) > 0:
        idx = torch.tensor(label_list, dtype=torch.long)
        idx = idx[idx >= 2]
        if len(idx) > 0:
            v.scatter_(0, idx, 1.0)
    return v


class HSTUProviderDataset(Dataset):
    def __init__(self, data):
        self.seq_matrix        = data["seq_matrix"]        # (N, 20, 2)
        self.delta_t_matrix    = data["delta_t_matrix"]    # (N, 20) — time gaps
        self.trigger_token     = data["trigger_token"]     # (N, 1)
        self.seq_lengths       = data["seq_lengths"]
        self.lab_t30           = data["lab_t30"]
        self.lab_t60           = data["lab_t60"]
        self.lab_t180          = data["lab_t180"]
        self.is_t30            = data["is_t30"]
        self.is_t60            = data["is_t60"]
        self.is_t180           = data["is_t180"]
        self.from_provider_ids = data["from_provider_ids"]

    def __len__(self):
        return len(self.seq_lengths)

    def __getitem__(self, idx):
        return {
            "seq":              torch.from_numpy(self.seq_matrix[idx].copy()),
            "delta_t":          torch.from_numpy(self.delta_t_matrix[idx].copy()),
            "trigger":          torch.from_numpy(self.trigger_token[idx].copy()),
            "seq_len":          torch.tensor(int(self.seq_lengths[idx]), dtype=torch.long),
            "lab_t30":          self.lab_t30[idx],
            "lab_t60":          self.lab_t60[idx],
            "lab_t180":         self.lab_t180[idx],
            "is_t30":           torch.tensor(bool(self.is_t30[idx]),  dtype=torch.bool),
            "is_t60":           torch.tensor(bool(self.is_t60[idx]),  dtype=torch.bool),
            "is_t180":          torch.tensor(bool(self.is_t180[idx]), dtype=torch.bool),
            "from_provider_id": int(self.from_provider_ids[idx]),
        }


def collate_fn(batch, vocab_size, from_provider_to_cands):
    seq      = torch.stack([b["seq"]     for b in batch])
    delta_t  = torch.stack([b["delta_t"] for b in batch])
    trigger  = torch.stack([b["trigger"] for b in batch])
    seq_len  = torch.stack([b["seq_len"] for b in batch])
    is_t30   = torch.stack([b["is_t30"]  for b in batch])
    is_t60   = torch.stack([b["is_t60"]  for b in batch])
    is_t180  = torch.stack([b["is_t180"] for b in batch])
    lab_t30  = torch.stack([sparse_to_multihot(b["lab_t30"],  vocab_size) for b in batch])
    lab_t60  = torch.stack([sparse_to_multihot(b["lab_t60"],  vocab_size) for b in batch])
    lab_t180 = torch.stack([sparse_to_multihot(b["lab_t180"], vocab_size) for b in batch])

    _empty = np.array([], dtype=np.int32)
    hard_negs = []
    for b in batch:
        fp_int = b["from_provider_id"]
        cands  = from_provider_to_cands.get(fp_int, _empty)
        if len(cands) == 0:
            hard_negs.append(_empty); continue
        all_pos = set(b["lab_t30"]) | set(b["lab_t60"]) | set(b["lab_t180"])
        hard_negs.append(cands[~np.isin(cands, list(all_pos))] if all_pos else cands)

    return {
        "seq": seq, "delta_t": delta_t, "trigger": trigger, "seq_len": seq_len,
        "lab_t30": lab_t30, "lab_t60": lab_t60, "lab_t180": lab_t180,
        "is_t30": is_t30, "is_t60": is_t60, "is_t180": is_t180,
        "hard_negs": hard_negs,
    }


_loader_kwargs = dict(num_workers=4, pin_memory=(DEVICE == "cuda"), persistent_workers=True)

train_loader = DataLoader(
    HSTUProviderDataset(train_data),
    batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=partial(collate_fn, vocab_size=PROVIDER_VOCAB_SIZE,
                       from_provider_to_cands=from_provider_to_cands),
    **_loader_kwargs,
)
val_loader = DataLoader(
    HSTUProviderDataset(val_data),
    batch_size=BATCH_SIZE * 2, shuffle=False,
    collate_fn=partial(collate_fn, vocab_size=PROVIDER_VOCAB_SIZE,
                       from_provider_to_cands={}),
    **_loader_kwargs,
)

print(f"Train loader: {len(train_loader)} batches | Val loader: {len(val_loader)} batches")
print(f"Section 2 done — {time.time()-t0:.1f}s")
display(Markdown(f"**Train batches:** {len(train_loader):,} | **Val batches:** {len(val_loader):,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — HSTU MODEL
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 3 — Model"))


class RelativeBucketedTimeAndPositionBias(nn.Module):
    """
    Learnable relative position + time-gap bias added to attention logits.
    Encodes both WHERE in sequence (position) and WHEN (time gap) each visit was.
    This is HSTU's key advantage over SASRec — explicitly models temporal dynamics.
    """
    def __init__(self, max_seq_len: int, num_buckets: int = 64) -> None:
        super().__init__()
        self._max_seq_len     = max_seq_len
        self._num_buckets     = num_buckets
        self._ts_w  = nn.Parameter(torch.empty(num_buckets + 1).normal_(0, 0.02))
        self._pos_w = nn.Parameter(torch.empty(2 * max_seq_len - 1).normal_(0, 0.02))
        self._cached_pos_bias = None

    def _compute_pos_bias(self, device):
        N = self._max_seq_len
        t = F.pad(self._pos_w[:2 * N - 1], [0, N]).repeat(N)
        t = t[:-N].reshape(N, 3 * N - 2)
        r = (2 * N - 1) // 2
        return t[:, r:-r].unsqueeze(0)    # (1, N, N)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        B, N   = timestamps.shape
        device = timestamps.device
        if self.training or self._cached_pos_bias is None:
            self._cached_pos_bias = self._compute_pos_bias(device)
        pos_bias = self._cached_pos_bias.to(device)

        ext     = torch.cat([timestamps, timestamps[:, N-1:N]], dim=1)
        diff    = ext[:, 1:].unsqueeze(2) - ext[:, :-1].unsqueeze(1)
        buckets = torch.clamp(
            (torch.log(torch.abs(diff).clamp(min=1)) / 0.301).long(),
            min=0, max=self._num_buckets
        ).detach()
        ts_bias = self._ts_w[buckets.view(-1)].view(B, N, N)
        return pos_bias + ts_bias          # (B, N, N)


class HSTUBlock(nn.Module):
    """
    HSTU attention block.
    Key differences vs standard transformer:
      - SiLU activation on attention weights instead of softmax
        → sparser, more selective attention
      - Relative time + position bias on attention logits
      - u-v gating: output = O(u * attn(v))
        → multiplicative interaction between content and attention
    """
    def __init__(self, embedding_dim, linear_dim, attention_dim,
                 num_heads, dropout_rate, attn_dropout_rate,
                 max_seq_len, num_buckets=64):
        super().__init__()
        self._num_heads  = num_heads
        self._linear_dim = linear_dim
        self._attn_dim   = attention_dim

        self._uvqk = nn.Parameter(
            torch.empty(
                embedding_dim,
                linear_dim * 2 * num_heads + attention_dim * num_heads * 2
            ).normal_(0, 0.02)
        )
        self._o         = nn.Linear(linear_dim * num_heads, embedding_dim)
        self._norm_x    = nn.LayerNorm(embedding_dim)
        self._norm_attn = nn.LayerNorm(linear_dim * num_heads)
        self._rel_bias  = RelativeBucketedTimeAndPositionBias(max_seq_len, num_buckets)
        self._dropout   = nn.Dropout(dropout_rate)
        self._attn_drop = attn_dropout_rate
        nn.init.xavier_uniform_(self._o.weight)

    def forward(self, x, timestamps, causal_mask, pad_mask):
        B, N, D = x.shape
        H, Dv, Dq = self._num_heads, self._linear_dim, self._attn_dim

        normed = self._norm_x(x)
        out    = F.silu(normed.reshape(B * N, D) @ self._uvqk)
        u, v, q, k = torch.split(out, [H*Dv, H*Dv, H*Dq, H*Dq], dim=-1)

        q = q.view(B, N, H, Dq).permute(0, 2, 1, 3).reshape(B*H, N, Dq)
        k = k.view(B, N, H, Dq).permute(0, 2, 1, 3).reshape(B*H, N, Dq)
        v = v.view(B, N, H, Dv).permute(0, 2, 1, 3).reshape(B*H, N, Dv)

        qk       = torch.bmm(q, k.transpose(1, 2)).view(B, H, N, N)
        rel_bias = self._rel_bias(timestamps).unsqueeze(1)
        qk       = qk + rel_bias
        qk       = F.silu(qk) / N
        qk       = qk.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
        qk       = qk.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2),    0.0)
        qk       = F.dropout(qk, p=self._attn_drop, training=self.training)

        attn_out = torch.bmm(qk.reshape(B*H, N, N), v).view(B, H, N, Dv)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, H*Dv)
        attn_out = self._norm_attn(attn_out)
        u_flat   = u.view(B, N, H*Dv)
        out      = self._o(self._dropout(u_flat * attn_out)) + x
        return out.masked_fill(pad_mask.unsqueeze(-1), 0.0)


class HSTUProvider(nn.Module):
    """
    HSTU for provider-level recommendation.

    Input per position: emb(provider_id) + emb(specialty_id) — summed, same as SASRec
    Timestamps:         delta_t_matrix — time gaps between visits (fed to rel_bias)
    Trigger token:      emb(trigger_dx_id) — appended as position MAX_SEQ_LEN
    Trigger timestamp:  0 (trigger is at time 0 relative to itself)
    Output:             dot-product against provider_emb table
    Loss:               batched InfoNCE
    """
    def __init__(self, provider_vocab_size, spec_vocab_size, dx_vocab_size,
                 d_model, delta_t_dim, num_ratings, hstu_dim,
                 linear_dim, attention_dim, max_seq_len,
                 num_heads, num_blocks, dropout, attn_dropout, num_buckets):
        super().__init__()
        self.d_model     = d_model
        self.hstu_dim    = hstu_dim
        self.max_seq_len = max_seq_len

        self.provider_emb  = nn.Embedding(provider_vocab_size, d_model, padding_idx=PAD_IDX)
        self.spec_emb      = nn.Embedding(spec_vocab_size,     d_model, padding_idx=PAD_IDX)
        self.dx_emb        = nn.Embedding(dx_vocab_size,       d_model, padding_idx=PAD_IDX)
        # delta_t embedding — concatenated into x (matches working HSTU_model.py)
        # rating_emb in hstu_pytorch → delta_t_emb here
        self.delta_t_emb   = nn.Embedding(num_ratings, delta_t_dim)
        # positional embedding over full sequence (MAX_SEQ_LEN + 1 trigger position)
        self.pos_emb       = nn.Embedding(max_seq_len + 1, hstu_dim)
        self.input_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            HSTUBlock(
                embedding_dim     = hstu_dim,  # 160 — after delta_t concat
                linear_dim        = linear_dim,
                attention_dim     = attention_dim,
                num_heads         = num_heads,
                dropout_rate      = dropout,
                attn_dropout_rate = attn_dropout,
                max_seq_len       = max_seq_len + 1,  # +1 for trigger token
                num_buckets       = num_buckets,
            )
            for _ in range(num_blocks)
        ])
        self.out_norm = nn.LayerNorm(hstu_dim)  # hstu_dim=160 after delta_t concat

        self.register_buffer(
            '_causal_mask',
            torch.triu(torch.ones(max_seq_len + 1, max_seq_len + 1, dtype=torch.bool),
                       diagonal=1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def encode(self, seq, delta_t, trigger, seq_len):
        """
        seq:     (B, 20, 2)  — [provider_id, specialty_id]
        delta_t: (B, 20)     — time gaps (fed to delta_t_emb AND RelBias)
        trigger: (B, 1)      — trigger_dx_id
        seq_len: (B,)

        Matches working HSTU_model.py:
          x = cat([item_emb, delta_t_emb(delta_t)], dim=-1)  → hstu_dim=160
          x = x * sqrt(hstu_dim) + pos_emb(pos)
        Returns: user_repr (B, hstu_dim) from trigger position
        """
        B, L, _ = seq.shape

        # ── Item embeddings: provider + specialty summed → (B, L, d_model) ───
        item_emb = self.provider_emb(seq[:, :, 0]) + self.spec_emb(seq[:, :, 1])

        # ── delta_t embedding → concatenate into item_emb ────────────────────
        # Matches: x = cat([embeddings, rating_emb(delta_t)], dim=-1) in HSTU_model.py
        # Cap delta_t at NUM_RATINGS-1 (same as LEAST(delta_t_bucket, NUM_RATINGS-1))
        dt_capped  = delta_t.clamp(0, self.delta_t_emb.num_embeddings - 1)
        dt_emb_seq = self.delta_t_emb(dt_capped)          # (B, L, delta_t_dim)
        x_seq      = torch.cat([item_emb, dt_emb_seq], dim=-1)  # (B, L, hstu_dim=160)

        # ── Trigger token: dx_emb padded to hstu_dim ─────────────────────────
        # dx_emb is d_model=128; pad to hstu_dim=160 to match sequence dim
        dx_emb_raw = self.dx_emb(trigger)                  # (B, 1, d_model)
        pad_size   = self.hstu_dim - self.d_model          # 32
        dx_emb_pad = F.pad(dx_emb_raw, (0, pad_size))      # (B, 1, hstu_dim)
        x          = torch.cat([x_seq, dx_emb_pad], dim=1) # (B, L+1, hstu_dim)
        TL         = L + 1

        # ── Input scaling + positional embedding ──────────────────────────────
        # Matches: x = x * sqrt(hstu_dim) + pos_emb(pos) in HSTU_model.py
        positions  = torch.arange(TL, device=seq.device).unsqueeze(0)  # (1, L+1)
        x          = x * math.sqrt(self.hstu_dim) + self.pos_emb(positions)
        x          = self.input_dropout(x)

        # ── Timestamps for RelBias: delta_t for sequence, 0 for trigger ──────
        trigger_ts = torch.zeros(B, 1, dtype=torch.long, device=seq.device)
        timestamps = torch.cat([dt_capped, trigger_ts], dim=1).long()  # (B, L+1)

        # ── Pad mask ──────────────────────────────────────────────────────────
        pad_flags  = (seq[:, :, 0] == PAD_IDX)
        trigger_ok = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
        pad_mask   = torch.cat([pad_flags, trigger_ok], dim=1)          # (B, L+1)
        x          = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        # ── HSTU blocks ───────────────────────────────────────────────────────
        for block in self.blocks:
            x = block(x, timestamps, self._causal_mask, pad_mask)
        x = self.out_norm(x)

        # User repr from trigger position (last) — conditioned on trigger_dx
        return x[:, -1, :]    # (B, hstu_dim=160)

    def score(self, user_repr, provider_ids):
        # user_repr is (B, hstu_dim=160), provider_emb is (V, d_model=128)
        # project user_repr down to d_model for dot-product
        emb = self.provider_emb(provider_ids)
        u   = user_repr[:, :self.d_model]   # use first d_model dims for scoring
        if emb.dim() == 2: return u @ emb.T
        return (u.unsqueeze(1) * emb).sum(-1)

    def forward(self, seq, delta_t, trigger, seq_len):
        user_repr = self.encode(seq, delta_t, trigger, seq_len)   # (B, hstu_dim)
        all_ids   = torch.arange(2, self.provider_emb.num_embeddings, device=seq.device)
        scores    = self.score(user_repr, all_ids)                 # (B, vocab-2)
        return scores, user_repr


def get_raw(model):
    return model.module if isinstance(model, nn.DataParallel) else model


model = HSTUProvider(
    provider_vocab_size = PROVIDER_VOCAB_SIZE,
    spec_vocab_size     = SPEC_VOCAB_SIZE,
    dx_vocab_size       = DX_VOCAB_SIZE,
    d_model             = D_MODEL,
    delta_t_dim         = DELTA_T_DIM,
    num_ratings         = NUM_RATINGS,
    hstu_dim            = HSTU_DIM,
    linear_dim          = LINEAR_DIM,
    attention_dim       = ATTENTION_DIM,
    max_seq_len         = MAX_SEQ_LEN,
    num_heads           = N_HEADS,
    num_blocks          = N_LAYERS,
    dropout             = DROPOUT,
    attn_dropout        = ATTN_DROPOUT,
    num_buckets         = N_BUCKETS,
).to(DEVICE)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {n_params:,}")
print(f"Section 3 done — {time.time()-t0:.1f}s")
display(Markdown(f"**Parameters:** {n_params:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LOSS (same batched InfoNCE as NB_04/05)
# ══════════════════════════════════════════════════════════════════════════════

def build_neg_matrix(batch_size, hard_negs_list, provider_vocab_size,
                     neg_k, hard_neg_k, device):
    rand_negs = torch.randint(2, provider_vocab_size, (batch_size, neg_k), device=device)
    if hard_neg_k == 0:
        return rand_negs
    hard_mat = torch.randint(2, provider_vocab_size, (batch_size, hard_neg_k), device=device)
    for i, hn in enumerate(hard_negs_list):
        if len(hn) == 0: continue
        if len(hn) > hard_neg_k:
            idx = torch.randperm(len(hn), device='cpu')[:hard_neg_k]
            hn_t = torch.tensor(hn[idx.numpy()], dtype=torch.long, device=device)
        else:
            hn_t = torch.tensor(hn[:len(hn)], dtype=torch.long, device=device)
        hard_mat[i, :len(hn_t)] = hn_t
    return torch.cat([rand_negs, hard_mat], dim=1)


def batched_infonce_loss(user_repr, labels_multihot, neg_matrix,
                         model_raw, temperature=0.07):
    B, d = user_repr.shape
    neg_emb     = model_raw.provider_emb(neg_matrix)
    # project to d_model for dot-product
    u_proj      = user_repr[:, :model_raw.d_model]          # (B, d_model)
    neg_scores  = torch.bmm(u_proj.unsqueeze(1),
                            neg_emb.transpose(1, 2)).squeeze(1) / temperature
    ib_scores   = (u_proj @ u_proj.T) / temperature
    ib_scores.fill_diagonal_(float('-inf'))
    all_neg_scores = torch.cat([neg_scores, ib_scores], dim=1)
    pos_rows, pos_cols = labels_multihot.nonzero(as_tuple=True)
    if len(pos_rows) == 0:
        return torch.tensor(0.0, device=user_repr.device)
    pos_emb     = model_raw.provider_emb(pos_cols)
    # user_repr is hstu_dim=160, provider_emb is d_model=128
    # use first d_model dims for dot-product (matches score())
    u_for_pos   = user_repr[pos_rows][:, :model_raw.d_model]
    pos_scores  = (u_for_pos * pos_emb).sum(dim=1) / temperature
    log_sum_neg = torch.logsumexp(all_neg_scores, dim=1)
    return (-pos_scores + log_sum_neg[pos_rows]).mean()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATION (same as NB_04/05)
# ══════════════════════════════════════════════════════════════════════════════

def metrics_at_k(scores, labels_multihot, k):
    topk_idx  = scores.topk(k, dim=1).indices
    topk_hits = labels_multihot.gather(1, topk_idx)
    hit_at_k  = (topk_hits.sum(1) > 0).float().mean().item()
    prec_at_k = topk_hits.sum(1).float().div(k).mean().item()
    n_pos     = labels_multihot.sum(1).clamp(min=1)
    rec_at_k  = topk_hits.sum(1).float().div(n_pos).mean().item()
    positions = torch.arange(1, k+1, dtype=torch.float32, device=scores.device)
    discounts = 1.0 / torch.log2(positions + 1)
    dcg       = (topk_hits.float() * discounts).sum(1)
    ideal     = torch.zeros(scores.shape[0], k, device=scores.device)
    for i in range(scores.shape[0]):
        ideal[i, :min(int(n_pos[i].item()), k)] = 1.0
    idcg    = (ideal * discounts).sum(1).clamp(min=1e-8)
    ndcg_at_k = (dcg / idcg).mean().item()
    return {"hit": hit_at_k, "prec": prec_at_k, "rec": rec_at_k, "ndcg": ndcg_at_k}


def evaluate(loader, mdl):
    mdl.eval()
    model_raw  = get_raw(mdl)
    buckets    = {"T0_30":  ("is_t30",  "lab_t30"),
                  "T30_60": ("is_t60",  "lab_t60"),
                  "T60_180":("is_t180", "lab_t180")}
    all_metrics = {b: {k: {"hit":[],"prec":[],"rec":[],"ndcg":[]}
                        for k in K_VALUES} for b in buckets}
    val_loss_sum, val_loss_n = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            seq     = batch["seq"].to(DEVICE,     non_blocking=True)
            delta_t = batch["delta_t"].to(DEVICE, non_blocking=True)
            trigger = batch["trigger"].to(DEVICE, non_blocking=True)
            seq_len = batch["seq_len"].to(DEVICE, non_blocking=True)
            hard_negs_list = batch["hard_negs"]
            B = seq.shape[0]

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                scores, user_repr = mdl(seq, delta_t, trigger, seq_len)

            neg_matrix = build_neg_matrix(B, hard_negs_list,
                                          PROVIDER_VOCAB_SIZE,
                                          NEG_K, HARD_NEG_K, DEVICE)
            for flag_key, lab_key in [("is_t30","lab_t30"),("is_t60","lab_t60"),("is_t180","lab_t180")]:
                flag = batch[flag_key].to(DEVICE)
                if flag.sum() == 0: continue
                labs = batch[lab_key].to(DEVICE)
                wl   = batched_infonce_loss(user_repr[flag], labs[flag],
                                            neg_matrix[flag], model_raw)
                val_loss_sum += wl.item(); val_loss_n += 1

            pad_scores  = torch.zeros(B, 2, device=scores.device)
            full_scores = torch.cat([pad_scores, scores], dim=1)
            for bucket, (flag_key, lab_key) in buckets.items():
                mask = batch[flag_key].to(DEVICE)
                if mask.sum() == 0: continue
                labs = batch[lab_key].to(DEVICE)
                s_m  = full_scores[mask]; l_m = labs[mask]
                for k in K_VALUES:
                    m = metrics_at_k(s_m, l_m, k)
                    for metric, val in m.items():
                        all_metrics[bucket][k][metric].append(val)

    results = {"val_loss": round(val_loss_sum / max(val_loss_n, 1), 4)}
    for bucket in buckets:
        for k in K_VALUES:
            for metric in ["hit", "prec", "rec", "ndcg"]:
                vals = all_metrics[bucket][k][metric]
                if vals:
                    results[f"{bucket}_{metric}@{k}"] = round(np.mean(vals), 4)
    return results


def print_metrics(metrics, split="Val"):
    print(f"\n  {split} Metrics:")
    for bucket in ["T0_30", "T30_60", "T60_180"]:
        row = " | ".join(
            f"Hit@{k}={metrics.get(f'{bucket}_hit@{k}',0):.4f} "
            f"NDCG@{k}={metrics.get(f'{bucket}_ndcg@{k}',0):.4f}"
            for k in K_VALUES
        )
        print(f"    {bucket}: {row}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 6 — Training"))

optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scaler       = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

total_steps  = EPOCHS * len(train_loader)
warmup_steps = min(WARMUP_STEPS, total_steps // 10)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step + 1) / float(max(warmup_steps, 1))
    progress = float(current_step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
model_raw  = get_raw(model)
best_ndcg  = 0.0
train_log  = []

print(f"Training — {EPOCHS} epochs | {len(train_loader)} batches/epoch")
print(f"Warmup: {warmup_steps} steps → cosine decay over {total_steps} total")

for epoch in range(EPOCHS):
    t_ep = time.time()
    model.train()
    total_loss, n_batches = 0.0, 0

    for batch_idx, batch in enumerate(train_loader):
        seq     = batch["seq"].to(DEVICE,     non_blocking=True)
        delta_t = batch["delta_t"].to(DEVICE, non_blocking=True)
        trigger = batch["trigger"].to(DEVICE, non_blocking=True)
        seq_len = batch["seq_len"].to(DEVICE, non_blocking=True)
        is_t30  = batch["is_t30"].to(DEVICE,  non_blocking=True)
        is_t60  = batch["is_t60"].to(DEVICE,  non_blocking=True)
        is_t180 = batch["is_t180"].to(DEVICE, non_blocking=True)
        lab_t30 = batch["lab_t30"].to(DEVICE, non_blocking=True)
        lab_t60 = batch["lab_t60"].to(DEVICE, non_blocking=True)
        lab_t180= batch["lab_t180"].to(DEVICE,non_blocking=True)
        hard_negs_list = batch["hard_negs"]
        B = seq.shape[0]

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            user_repr = model_raw.encode(seq, delta_t, trigger, seq_len)

            neg_matrix = build_neg_matrix(B, hard_negs_list,
                                          PROVIDER_VOCAB_SIZE,
                                          NEG_K, HARD_NEG_K, DEVICE)

            loss      = torch.tensor(0.0, device=DEVICE)
            n_windows = 0
            for flag, labs in [(is_t30, lab_t30), (is_t60, lab_t60), (is_t180, lab_t180)]:
                if flag.sum() == 0: continue
                u_m   = user_repr[flag]
                l_m   = labs[flag]
                neg_m = neg_matrix[flag]
                window_loss = batched_infonce_loss(u_m, l_m, neg_m, model_raw)
                loss        = loss + window_loss
                n_windows  += 1
            if n_windows > 0:
                loss = loss / n_windows

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        n_batches  += 1

        if (batch_idx + 1) % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Ep {epoch+1} | B {batch_idx+1}/{len(train_loader)} | "
                  f"Train Loss (avg): {total_loss/n_batches:.4f} | LR: {current_lr:.2e}")

    avg_loss = total_loss / max(n_batches, 1)
    ep_time  = time.time() - t_ep
    print(f"  Epoch {epoch+1} done — loss={avg_loss:.4f} | time={ep_time:.0f}s")

    val_metrics = evaluate(val_loader, model)
    val_loss    = val_metrics.get("val_loss", 0.0)
    val_ndcg    = np.mean([val_metrics.get(f"T0_30_ndcg@{k}", 0) for k in K_VALUES])
    print_metrics(val_metrics, f"Val Epoch {epoch+1}")

    train_log.append({"epoch": epoch+1,
                       "train_loss": avg_loss, "val_loss": val_loss,
                       "val_ndcg_t30": val_ndcg, **val_metrics})

    display(Markdown(
        f"**Epoch {epoch+1}/{EPOCHS}** | "
        f"Train Loss: `{avg_loss:.4f}` | Val Loss: `{val_loss:.4f}` | "
        f"Val NDCG@T30: `{val_ndcg:.4f}` | Time: {ep_time:.0f}s"
    ))

    if val_ndcg >= best_ndcg:
        best_ndcg = val_ndcg
        ckpt_path = f"{MODEL_DIR}/hstu_provider_{SAMPLE}_ep{epoch+1}.pt"
        torch.save({
            "epoch":         epoch,
            "model_state":   model_raw.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_ndcg": best_ndcg,
            "config": {
                "provider_vocab_size": PROVIDER_VOCAB_SIZE,
                "spec_vocab_size":     SPEC_VOCAB_SIZE,
                "dx_vocab_size":       DX_VOCAB_SIZE,
                "d_model":             D_MODEL,
                "delta_t_dim":         DELTA_T_DIM,
                "num_ratings":         NUM_RATINGS,
                "hstu_dim":            HSTU_DIM,
                "linear_dim":          LINEAR_DIM,
                "attention_dim":       ATTENTION_DIM,
                "max_seq_len":         MAX_SEQ_LEN,
                "num_heads":           N_HEADS,
                "num_blocks":          N_LAYERS,
                "dropout":             DROPOUT,
                "attn_dropout":        ATTN_DROPOUT,
                "num_buckets":         N_BUCKETS,
            },
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")

print(f"\nTraining done — {time.time()-t0:.1f}s")

pd.DataFrame(train_log).to_csv(f"{OUTPUT_DIR}/hstu_train_log_{SAMPLE}.csv", index=False)
print(f"Train log saved")

display(Markdown(f"""
---
## Training Complete
| Metric | Value |
|---|---|
| Best val NDCG@T30 | {best_ndcg:.4f} |
| Epochs | {EPOCHS} |
| Sample | {SAMPLE} |
| Checkpoint | hstu_provider_{SAMPLE}_ep*.pt |

Next: run NB_07 (scoring on test set).
"""))
print("NB_06 complete")


# ══════════════════════════════════════════════════════════════════════════════
# CLEANUP
# ══════════════════════════════════════════════════════════════════════════════
import gc
import torch

for var in ["seq_df", "label_df", "train_data", "val_data", "from_provider_to_cands"]:
    if var in dir():
        del globals()[var]

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU memory freed — allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")

print("Memory cleanup done")
