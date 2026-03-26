# ============================================================
# NB_04 — Model_SASRec_provider.py
# Purpose : Train SASRec for provider-level recommendation
#           Input  : emb(provider) + emb(specialty) per visit
#                    + emb(trigger_dx) appended as last token
#           Output : dot-product against provider embedding table
#           Loss   : InfoNCE — in-batch + random + hard negatives
# Sources : ./cache_provider_{SAMPLE}/ (from NB_01/02)
# Output  : ./models/sasrec_provider_{SAMPLE}_ep{N}.pt
#            ./output/sasrec_train_log_{SAMPLE}.csv
# Notes:
#   First pass: 1 epoch — validate metrics before tuning
#   Dense multi-hot labels infeasible at 31K vocab
#   collate_fn builds sparse → multi-hot on-the-fly per batch
#   Three output heads (T30/T60/T180) share same user representation
# ============================================================

import gc
import os
import pickle
import time

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
SAMPLE       = "5pct"
MAX_SEQ_LEN  = 20
PAD_IDX      = 0
UNK_IDX      = 1
D_MODEL      = 128
N_HEADS      = 4
N_LAYERS     = 2
DROPOUT      = 0.2
LR           = 1e-3
WEIGHT_DECAY = 1e-2
BATCH_SIZE   = 1024
EPOCHS        = 1              # first pass — validate before tuning
WARMUP_STEPS  = 200            # ~100-200 steps for 1-epoch dry run; scale up for full training
NEG_K        = 128            # random negatives per sample
HARD_NEG_K   = 32             # hard negatives per sample (capped)
K_VALUES     = [1, 3, 5]
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

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
| Heads | {N_HEADS} |
| Layers | {N_LAYERS} |
| Batch size | {BATCH_SIZE} |
| Epochs | {EPOCHS} |
| Device | {DEVICE} |
| NEG_K | {NEG_K} |
| HARD_NEG_K | {HARD_NEG_K} |
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
    """Convert sparse list of int_ids to dense multi-hot float tensor."""
    v = torch.zeros(vocab_size, dtype=torch.float32)
    if len(label_list) > 0:
        idx = torch.tensor(label_list, dtype=torch.long)
        # Exclude PAD=0 and UNK=1 from positive signal
        idx = idx[idx >= 2]
        if len(idx) > 0:
            v.scatter_(0, idx, 1.0)
    return v


class ProviderDataset(Dataset):
    def __init__(self, data):
        self.seq_matrix       = data["seq_matrix"]       # (N, 20, 2)
        self.trigger_token    = data["trigger_token"]     # (N, 1)
        self.seq_lengths      = data["seq_lengths"]
        self.lab_t30          = data["lab_t30"]           # ragged
        self.lab_t60          = data["lab_t60"]
        self.lab_t180         = data["lab_t180"]
        self.is_t30           = data["is_t30"]
        self.is_t60           = data["is_t60"]
        self.is_t180          = data["is_t180"]
        self.from_provider_ids = data["from_provider_ids"] # (N,) int32

    def __len__(self):
        return len(self.seq_lengths)

    def __getitem__(self, idx):
        return {
            "seq":             torch.from_numpy(self.seq_matrix[idx].copy()),
            "trigger":         torch.from_numpy(self.trigger_token[idx].copy()),
            "seq_len":         torch.tensor(int(self.seq_lengths[idx]), dtype=torch.long),
            "lab_t30":         self.lab_t30[idx],
            "lab_t60":         self.lab_t60[idx],
            "lab_t180":        self.lab_t180[idx],
            "is_t30":          torch.tensor(bool(self.is_t30[idx]),  dtype=torch.bool),
            "is_t60":          torch.tensor(bool(self.is_t60[idx]),  dtype=torch.bool),
            "is_t180":         torch.tensor(bool(self.is_t180[idx]), dtype=torch.bool),
            "from_provider_id": int(self.from_provider_ids[idx]),  # int — for collate lookup
        }


def collate_fn(batch, vocab_size, from_provider_to_cands):
    """
    Collate batch:
    - builds multi-hot labels on-the-fly from sparse lists
    - computes hard negatives per sample (exclusion of positives)
    """
    seq          = torch.stack([b["seq"]     for b in batch])          # (B, 20, 2)
    trigger      = torch.stack([b["trigger"] for b in batch])          # (B, 1)
    seq_len      = torch.stack([b["seq_len"] for b in batch])
    is_t30       = torch.stack([b["is_t30"]  for b in batch])
    is_t60       = torch.stack([b["is_t60"]  for b in batch])
    is_t180      = torch.stack([b["is_t180"] for b in batch])

    # Build multi-hot per batch — only for qualified windows
    lab_t30  = torch.stack([sparse_to_multihot(b["lab_t30"],  vocab_size) for b in batch])
    lab_t60  = torch.stack([sparse_to_multihot(b["lab_t60"],  vocab_size) for b in batch])
    lab_t180 = torch.stack([sparse_to_multihot(b["lab_t180"], vocab_size) for b in batch])

    # Hard neg exclusion here — batch of 512, not 5M triggers
    # For each sample: get candidates for its from_provider, exclude positives
    _empty = np.array([], dtype=np.int32)
    hard_negs = []
    for b in batch:
        fp_int = b["from_provider_id"]
        cands  = from_provider_to_cands.get(fp_int, _empty)
        if len(cands) == 0:
            hard_negs.append(_empty)
            continue
        # Exclude positives — combine all windows
        all_pos = set(b["lab_t30"]) | set(b["lab_t60"]) | set(b["lab_t180"])
        if all_pos:
            mask  = ~np.isin(cands, list(all_pos))
            hard_negs.append(cands[mask])
        else:
            hard_negs.append(cands)

    return {
        "seq": seq, "trigger": trigger, "seq_len": seq_len,
        "lab_t30": lab_t30, "lab_t60": lab_t60, "lab_t180": lab_t180,
        "is_t30": is_t30, "is_t60": is_t60, "is_t180": is_t180,
        "hard_negs": hard_negs,
    }


from functools import partial

_loader_kwargs = dict(
    num_workers=4,
    pin_memory=(DEVICE == "cuda"),
    persistent_workers=True,
)

train_dataset = ProviderDataset(train_data)
val_dataset   = ProviderDataset(val_data)

_empty_cands = {}   # val has no hard negs — pass empty dict

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=partial(collate_fn, vocab_size=PROVIDER_VOCAB_SIZE,
                       from_provider_to_cands=from_provider_to_cands),
    **_loader_kwargs,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2, shuffle=False,
    collate_fn=partial(collate_fn, vocab_size=PROVIDER_VOCAB_SIZE,
                       from_provider_to_cands=_empty_cands),
    **_loader_kwargs,
)

print(f"Train loader: {len(train_loader)} batches | Val loader: {len(val_loader)} batches")
print(f"Section 2 done — {time.time()-t0:.1f}s")
display(Markdown(f"**Train batches:** {len(train_loader):,} | **Val batches:** {len(val_loader):,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 3 — Model"))


class PointWiseFeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class SASRecBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.attn    = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff      = PointWiseFeedForward(dim, dropout)
        self.norm1   = nn.LayerNorm(dim)
        self.norm2   = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask, pad_mask):
        # Self-attention with causal mask (SASRec: no future tokens)
        attn_out, _ = self.attn(x, x, x,
                                attn_mask=causal_mask,
                                key_padding_mask=pad_mask,
                                need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ff(x))
        return x


class SASRecProvider(nn.Module):
    """
    SASRec for provider-level recommendation.

    Input per position: emb(provider_id) + emb(specialty_id)  [summed]
    Trigger token:      emb(trigger_dx_id)                     [appended as pos 21]
    Output head:        dot-product against provider_emb table
    Loss:               InfoNCE over positives + random + hard negatives

    seq_matrix shape:    (B, 20, 2)   — [provider_id, specialty_id] per slot
    trigger shape:       (B, 1)       — [trigger_dx_id]
    """
    def __init__(self, provider_vocab_size, spec_vocab_size, dx_vocab_size,
                 d_model, max_seq_len, num_heads, num_blocks, dropout):
        super().__init__()

        self.d_model     = d_model
        self.max_seq_len = max_seq_len

        # ── Embedding tables ──────────────────────────────────────────────────
        self.provider_emb = nn.Embedding(provider_vocab_size, d_model, padding_idx=PAD_IDX)
        self.spec_emb     = nn.Embedding(spec_vocab_size,     d_model, padding_idx=PAD_IDX)
        self.dx_emb       = nn.Embedding(dx_vocab_size,       d_model, padding_idx=PAD_IDX)
        self.pos_emb      = nn.Embedding(max_seq_len + 1,     d_model) # +1 for trigger token position

        self.emb_drop = nn.Dropout(dropout)

        # ── Transformer blocks ────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            SASRecBlock(d_model, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def encode(self, seq, trigger, seq_len):
        """
        Encode sequence + trigger into user representation.

        seq:     (B, 20, 2) — provider_id and specialty_id per slot
        trigger: (B, 1)     — trigger_dx_id
        seq_len: (B,)

        Returns: user_repr (B, d_model) — representation from last sequence position
        """
        B, L, _ = seq.shape

        # ── Composite embedding: emb(provider) + emb(specialty) ──────────────
        prov_emb = self.provider_emb(seq[:, :, 0])   # (B, L, d)
        spec_emb = self.spec_emb(seq[:, :, 1])        # (B, L, d)
        x = prov_emb + spec_emb                        # (B, L, d) — summation

        # ── Trigger token: emb(trigger_dx) ───────────────────────────────────
        # Appended as position L (after the sequence)
        dx_emb     = self.dx_emb(trigger)              # (B, 1, d)
        x          = torch.cat([x, dx_emb], dim=1)     # (B, L+1, d)
        total_len  = L + 1

        # ── Positional embeddings ─────────────────────────────────────────────
        positions  = torch.arange(total_len, device=seq.device).unsqueeze(0)  # (1, L+1)
        x          = self.emb_drop(x + self.pos_emb(positions))

        # ── Padding mask: True where position is padded ───────────────────────
        # Padded positions: seq slots where provider_id == PAD_IDX (0)
        # Trigger token (last position) is never padded
        pad_flags  = (seq[:, :, 0] == PAD_IDX)                                 # (B, L) bool
        trigger_ok = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)    # (B, 1) — not padded
        pad_mask   = torch.cat([pad_flags, trigger_ok], dim=1)                  # (B, L+1)

        # ── Causal mask: upper triangular — no future tokens ─────────────────
        causal_mask = torch.triu(
            torch.ones(total_len, total_len, device=seq.device, dtype=torch.bool),
            diagonal=1
        )  # (L+1, L+1)

        # ── Transformer blocks ────────────────────────────────────────────────
        for block in self.blocks:
            x = block(x, causal_mask, pad_mask)
        x = self.norm(x)

        # ── Extract user representation from trigger position (last) ──────────
        user_repr = x[:, -1, :]    # (B, d) — representation at trigger token position
        return user_repr

    def score(self, user_repr, provider_ids):
        """
        Dot-product scores: user_repr @ provider_emb[provider_ids].T

        user_repr:    (B, d)
        provider_ids: (K,) or (B, K)
        Returns:      (B, K) scores
        """
        emb = self.provider_emb(provider_ids)              # (K, d) or (B, K, d)
        if emb.dim() == 2:
            return user_repr @ emb.T                        # (B, K)
        else:
            return (user_repr.unsqueeze(1) * emb).sum(-1)  # (B, K)

    def forward(self, seq, trigger, seq_len):
        """Full scoring over all top80 providers — for val/test."""
        user_repr = self.encode(seq, trigger, seq_len)      # (B, d)
        # Score against all top80 providers (skip PAD=0, UNK=1)
        all_ids   = torch.arange(2, self.provider_emb.num_embeddings,
                                 device=seq.device)          # (vocab-2,)
        scores    = self.score(user_repr, all_ids)           # (B, vocab-2)
        return scores, user_repr


def get_raw(model):
    return model.module if isinstance(model, nn.DataParallel) else model


model = SASRecProvider(
    provider_vocab_size = PROVIDER_VOCAB_SIZE,
    spec_vocab_size     = SPEC_VOCAB_SIZE,
    dx_vocab_size       = DX_VOCAB_SIZE,
    d_model             = D_MODEL,
    max_seq_len         = MAX_SEQ_LEN,
    num_heads           = N_HEADS,
    num_blocks          = N_LAYERS,
    dropout             = DROPOUT,
).to(DEVICE)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {n_params:,}")
print(f"Section 3 done — {time.time()-t0:.1f}s")
display(Markdown(f"**Parameters:** {n_params:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LOSS (InfoNCE)
# ══════════════════════════════════════════════════════════════════════════════

def build_neg_matrix(batch_size, hard_negs_list, provider_vocab_size,
                     neg_k, hard_neg_k, device):
    """
    Build (B, K_total) negative id matrix in one shot.
    K_total = neg_k + hard_neg_k
    Random negatives: sampled uniformly — one call for the whole batch
    Hard negatives:   padded to hard_neg_k with random fallback
    Returns: (B, neg_k + hard_neg_k) int64 tensor
    """
    # Random negatives — one call for entire batch (B, neg_k)
    rand_negs = torch.randint(2, provider_vocab_size,
                              (batch_size, neg_k), device=device)

    if hard_neg_k == 0:
        return rand_negs

    # Hard negatives — pad each sample to hard_neg_k
    # Use random fallback for samples with fewer than hard_neg_k hard negs
    hard_mat = torch.randint(2, provider_vocab_size,
                             (batch_size, hard_neg_k), device=device)
    for i, hn in enumerate(hard_negs_list):
        if len(hn) == 0:
            continue
        n = min(len(hn), hard_neg_k)
        # Sample without replacement if enough candidates
        if len(hn) > hard_neg_k:
            idx = torch.randperm(len(hn), device='cpu')[:hard_neg_k]
            hn_t = torch.tensor(hn[idx.numpy()], dtype=torch.long, device=device)
        else:
            hn_t = torch.tensor(hn[:n], dtype=torch.long, device=device)
        hard_mat[i, :len(hn_t)] = hn_t

    return torch.cat([rand_negs, hard_mat], dim=1)   # (B, neg_k + hard_neg_k)


def batched_infonce_loss(user_repr, labels_multihot, neg_matrix,
                         model_raw, temperature=0.07):
    """
    Fully batched InfoNCE — zero Python loops over batch.

    user_repr:       (B, d)
    labels_multihot: (B, V) — 1 where positive provider
    neg_matrix:      (B, K) — negative provider int_ids

    Steps:
    1. One embedding lookup for all negatives: (B, K, d)
    2. In-batch negatives: other samples' user_repr as negatives — free
    3. Batched matmul for all scores
    4. Per-positive logsumexp loss — vectorized
    """
    B, d = user_repr.shape
    K    = neg_matrix.shape[1]
    device = user_repr.device

    # ── Negative scores — one batched embedding lookup ────────────────────────
    # neg_matrix: (B, K) → neg_emb: (B, K, d)
    neg_emb    = model_raw.provider_emb(neg_matrix)            # (B, K, d)
    neg_scores = torch.bmm(user_repr.unsqueeze(1),
                           neg_emb.transpose(1, 2)             # (B, d, K)
                           ).squeeze(1) / temperature           # (B, K)

    # ── In-batch negatives — free, just use other rows' user_repr ─────────────
    # user_repr: (B, d) — each row is a negative for all other rows
    # ib_scores[i, j] = user_repr[i] · user_repr[j] / τ  (j ≠ i diagonal masked)
    ib_scores = (user_repr @ user_repr.T) / temperature         # (B, B)
    # Mask self-similarity (diagonal) with -inf
    ib_scores.fill_diagonal_(float('-inf'))

    # ── All negative logits: explicit negs + in-batch ────────────────────────
    all_neg_scores = torch.cat([neg_scores, ib_scores], dim=1)  # (B, K+B)

    # ── Positive scores — lookup only positive provider embeddings ────────────
    # labels_multihot: (B, V) — sparse, few 1s per row
    # Get all positive indices across batch in one lookup
    pos_rows, pos_cols = labels_multihot.nonzero(as_tuple=True)  # both (n_pos,)

    if len(pos_rows) == 0:
        return torch.tensor(0.0, device=device)

    pos_emb    = model_raw.provider_emb(pos_cols)                # (n_pos, d)
    # Score each positive against its corresponding user repr
    pos_scores = (user_repr[pos_rows] * pos_emb).sum(dim=1) / temperature  # (n_pos,)

    # ── InfoNCE per positive ──────────────────────────────────────────────────
    # For each positive (i, p): loss = -score(i,p) + log(sum_exp(neg_scores[i]))
    # log_sum_neg is the same for all positives of the same sample i
    log_sum_neg = torch.logsumexp(all_neg_scores, dim=1)         # (B,)
    loss_per_pos = -pos_scores + log_sum_neg[pos_rows]           # (n_pos,)

    return loss_per_pos.mean()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def metrics_at_k(scores, labels_multihot, k):
    """
    scores:         (B, V) float — higher = more likely
    labels_multihot:(B, V) float — 1 for positive providers
    Returns dict of hit@k, precision@k, recall@k, ndcg@k
    """
    # Top-K indices
    topk_idx   = scores.topk(k, dim=1).indices            # (B, K)
    topk_hits  = labels_multihot.gather(1, topk_idx)      # (B, K) — 1 if hit

    # Hit@K
    hit_at_k   = (topk_hits.sum(1) > 0).float().mean().item()

    # Precision@K
    prec_at_k  = topk_hits.sum(1).float().div(k).mean().item()

    # Recall@K
    n_pos      = labels_multihot.sum(1).clamp(min=1)
    recall_at_k= topk_hits.sum(1).float().div(n_pos).mean().item()

    # NDCG@K — DCG / IDCG
    positions  = torch.arange(1, k + 1, dtype=torch.float32, device=scores.device)
    discounts  = 1.0 / torch.log2(positions + 1)          # (K,)
    dcg        = (topk_hits.float() * discounts).sum(1)    # (B,)
    # IDCG: best possible DCG given n_pos actual positives
    ideal_hits = torch.zeros(scores.shape[0], k, device=scores.device)
    for i in range(scores.shape[0]):
        n = min(int(n_pos[i].item()), k)
        ideal_hits[i, :n] = 1.0
    idcg       = (ideal_hits * discounts).sum(1).clamp(min=1e-8)
    ndcg_at_k  = (dcg / idcg).mean().item()

    return {"hit": hit_at_k, "prec": prec_at_k, "rec": recall_at_k, "ndcg": ndcg_at_k}


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
            trigger = batch["trigger"].to(DEVICE, non_blocking=True)
            seq_len = batch["seq_len"].to(DEVICE, non_blocking=True)
            hard_negs_list = batch["hard_negs"]
            B = seq.shape[0]

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                scores, user_repr = mdl(seq, trigger, seq_len)  # (B, V-2), (B, d)

            # Val loss — same InfoNCE as training
            neg_matrix = build_neg_matrix(B, hard_negs_list,
                                          PROVIDER_VOCAB_SIZE,
                                          NEG_K, HARD_NEG_K, DEVICE)
            for flag_key, lab_key in [("is_t30","lab_t30"),("is_t60","lab_t60"),("is_t180","lab_t180")]:
                flag = batch[flag_key].to(DEVICE)
                if flag.sum() == 0: continue
                labs = batch[lab_key].to(DEVICE)
                wl   = batched_infonce_loss(user_repr[flag], labs[flag],
                                            neg_matrix[flag], model_raw)
                val_loss_sum += wl.item()
                val_loss_n   += 1

            # Ranking metrics
            pad_scores  = torch.zeros(B, 2, device=scores.device)
            full_scores = torch.cat([pad_scores, scores], dim=1)  # (B, V)

            for bucket, (flag_key, lab_key) in buckets.items():
                mask = batch[flag_key].to(DEVICE)
                if mask.sum() == 0: continue
                labs = batch[lab_key].to(DEVICE)
                s_m  = full_scores[mask]
                l_m  = labs[mask]
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

# Warmup + cosine decay
# Linear warmup for WARMUP_STEPS, then cosine decay over remaining steps
total_steps  = EPOCHS * len(train_loader)
warmup_steps = min(WARMUP_STEPS, total_steps // 10)   # cap at 10% of total

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step + 1) / float(max(warmup_steps, 1))
    progress = float(current_step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
print(f"Scheduler: {warmup_steps} warmup steps → cosine decay over {total_steps} total steps")

model_raw  = get_raw(model)
best_ndcg  = 0.0
train_log  = []

print(f"Training — {EPOCHS} epochs | {len(train_loader)} batches/epoch")

for epoch in range(EPOCHS):
    t_ep = time.time()
    model.train()
    total_loss, n_batches = 0.0, 0

    for batch_idx, batch in enumerate(train_loader):
        seq     = batch["seq"].to(DEVICE,     non_blocking=True)     # (B, 20, 2)
        trigger = batch["trigger"].to(DEVICE, non_blocking=True)     # (B, 1)
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
            user_repr = model_raw.encode(seq, trigger, seq_len)     # (B, d)

            # Build neg matrix — one call for whole batch (B, K)
            neg_matrix = build_neg_matrix(B, hard_negs_list,
                                          PROVIDER_VOCAB_SIZE,
                                          NEG_K, HARD_NEG_K, DEVICE)

            # Batched InfoNCE per active window — zero Python loops
            loss      = torch.tensor(0.0, device=DEVICE)
            n_windows = 0

            for flag, labs in [(is_t30, lab_t30), (is_t60, lab_t60), (is_t180, lab_t180)]:
                if flag.sum() == 0:
                    continue
                u_m   = user_repr[flag]           # (B_m, d)
                l_m   = labs[flag]                # (B_m, V) — multi-hot
                neg_m = neg_matrix[flag]          # (B_m, K)

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

        scheduler.step()   # per-batch step for warmup + cosine
        total_loss += loss.item()
        n_batches  += 1

        if (batch_idx + 1) % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Ep {epoch+1} | B {batch_idx+1}/{len(train_loader)} | "
                  f"Train Loss (avg): {total_loss/n_batches:.4f} | LR: {current_lr:.2e}")

    avg_loss = total_loss / max(n_batches, 1)
    ep_time  = time.time() - t_ep
    print(f"  Epoch {epoch+1} done — loss={avg_loss:.4f} | time={ep_time:.0f}s")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_metrics = evaluate(val_loader, model)
    val_ndcg    = np.mean([val_metrics.get(f"T0_30_ndcg@{k}", 0) for k in K_VALUES])
    print_metrics(val_metrics, f"Val Epoch {epoch+1}")

    val_loss = val_metrics.get("val_loss", 0.0)
    train_log.append({"epoch": epoch+1,
                       "train_loss": avg_loss, "val_loss": val_loss,
                       "val_ndcg_t30": val_ndcg, **val_metrics})

    display(Markdown(
        f"**Epoch {epoch+1}/{EPOCHS}** | "
        f"Train Loss: `{avg_loss:.4f}` | Val Loss: `{val_loss:.4f}` | "
        f"Val NDCG@T30: `{val_ndcg:.4f}` | Time: {ep_time:.0f}s"
    ))

    # ── Save checkpoint ───────────────────────────────────────────────────────
    if val_ndcg >= best_ndcg:
        best_ndcg = val_ndcg
        ckpt_path = f"{MODEL_DIR}/sasrec_provider_{SAMPLE}_ep{epoch+1}.pt"
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
                "max_seq_len":         MAX_SEQ_LEN,
                "num_heads":           N_HEADS,
                "num_blocks":          N_LAYERS,
                "dropout":             DROPOUT,
            },
        }, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path} (val_ndcg={best_ndcg:.4f})")

print(f"\nTraining done — {time.time()-t0:.1f}s")


# ── Save train log ─────────────────────────────────────────────────────────────
log_path = f"{OUTPUT_DIR}/sasrec_train_log_{SAMPLE}.csv"
pd.DataFrame(train_log).to_csv(log_path, index=False)
print(f"Train log saved: {log_path}")

display(Markdown(f"""
---
## Training Complete
| Metric | Value |
|---|---|
| Best val NDCG@T30 | {best_ndcg:.4f} |
| Epochs | {EPOCHS} |
| Sample | {SAMPLE} |
| Checkpoint | sasrec_provider_{SAMPLE}_ep*.pt |

Next: run NB_07 to score on test set.
"""))
print("NB_04 complete")

# ══════════════════════════════════════════════════════════════════════════════
# CLEANUP — Free memory after notebook completes
# ══════════════════════════════════════════════════════════════════════════════
import gc
import torch

# Delete all large dataframes and arrays still in scope
for var in ["seq_df", "label_df", "hn_df", "prov_df", "spec_df", "dx_df",
            "prov_spec_df", "seq_df2", "trans_df", "grouped_trans",
            "seq_matrix", "delta_t_mat", "lab_t30", "lab_t60", "lab_t180",
            "train_data", "val_data", "test_data",
            "from_to_by_specialty", "from_provider_to_cands",
            "hard_neg_candidates", "all_arrays"]:
    if var in dir():
        del globals()[var]

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU memory freed — allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")

print("Memory cleanup done")
