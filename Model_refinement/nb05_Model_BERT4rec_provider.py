# ============================================================
# NB_05 — Model_BERT4Rec_provider.py
# Purpose : Train BERT4Rec for provider-level recommendation
#           Same input/output/loss as NB_04 SASRec
#           Key difference: bidirectional attention + masked training
#           Training: randomly mask 15% of sequence positions
#           Inference: mask last real position, predict from there
# Sources : ./cache_provider_{SAMPLE}/ (from NB_01/02)
# Output  : ./models/bert4rec_provider_{SAMPLE}_ep{N}.pt
#            ./output/bert4rec_train_log_{SAMPLE}.csv
# Architecture differences vs NB_04 SASRec:
#   No causal mask during training — full bidirectional attention
#   MASK token added to provider vocab (MASK_IDX = PROVIDER_VOCAB_SIZE)
#   At inference: last real position replaced with MASK token
#   Prediction from MASK position (not trigger position)
# ============================================================

import gc
import os
import pickle
import time
from functools import partial

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
BATCH_SIZE   = 512
EPOCHS       = 1
MASK_PROB    = 0.15           # fraction of real positions masked during training
NEG_K        = 128
HARD_NEG_K   = 32
K_VALUES     = [1, 3, 5]
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CACHE_DIR  = f"./cache_provider_{SAMPLE}"
MODEL_DIR  = "./models"
OUTPUT_DIR = "./output"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Device: {DEVICE} | Sample: {SAMPLE} | Mask prob: {MASK_PROB}")
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
| Mask prob | {MASK_PROB} |
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
# MASK token is one beyond the top80 vocab
MASK_IDX = PROVIDER_VOCAB_SIZE   # special token — not a real provider

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
print(f"MASK_IDX: {MASK_IDX} (beyond top80 vocab)")
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


class BERT4RecProviderDataset(Dataset):
    def __init__(self, data, mask_for_inference=False):
        self.seq_matrix        = data["seq_matrix"]
        self.trigger_token     = data["trigger_token"]
        self.seq_lengths       = data["seq_lengths"]
        self.lab_t30           = data["lab_t30"]
        self.lab_t60           = data["lab_t60"]
        self.lab_t180          = data["lab_t180"]
        self.is_t30            = data["is_t30"]
        self.is_t60            = data["is_t60"]
        self.is_t180           = data["is_t180"]
        self.mask_for_inference = mask_for_inference
        self.from_provider_ids = data["from_provider_ids"]

    def __len__(self):
        return len(self.seq_lengths)

    def __getitem__(self, idx):
        seq     = self.seq_matrix[idx].copy()         # (20, 2)
        seq_len = int(self.seq_lengths[idx])
        L       = seq.shape[0]
        pad_start = L - seq_len

        if self.mask_for_inference:
            # Mask the most recent real position (pos L-1 = recency_rank 1)
            mask_pos         = L - 1
            seq[mask_pos, 0] = MASK_IDX               # mask provider_id only
            target_mask      = np.zeros(L + 1, dtype=np.float32)
            target_mask[mask_pos] = 1.0
        else:
            # Random masking: 15% of real positions
            target_mask = np.zeros(L + 1, dtype=np.float32)
            if seq_len > 0:
                rand_vals    = np.random.random(seq_len)
                mask_flags   = rand_vals < MASK_PROB
                if not mask_flags.any():
                    mask_flags[np.random.randint(seq_len)] = True
                real_pos     = np.arange(pad_start, L)
                masked_pos   = real_pos[mask_flags]
                seq[masked_pos, 0] = MASK_IDX          # mask provider_id only
                target_mask[masked_pos] = 1.0
            # Trigger position (index L) is never masked

        return {
            "seq":              torch.from_numpy(seq),
            "trigger":          torch.from_numpy(self.trigger_token[idx].copy()),
            "seq_len":          torch.tensor(seq_len, dtype=torch.long),
            "target_mask":      torch.from_numpy(target_mask),
            "lab_t30":          self.lab_t30[idx],
            "lab_t60":          self.lab_t60[idx],
            "lab_t180":         self.lab_t180[idx],
            "is_t30":           torch.tensor(bool(self.is_t30[idx]),  dtype=torch.bool),
            "is_t60":           torch.tensor(bool(self.is_t60[idx]),  dtype=torch.bool),
            "is_t180":          torch.tensor(bool(self.is_t180[idx]), dtype=torch.bool),
            "from_provider_id": int(self.from_provider_ids[idx]),
        }


def collate_fn(batch, vocab_size, from_provider_to_cands):
    seq         = torch.stack([b["seq"]         for b in batch])
    trigger     = torch.stack([b["trigger"]     for b in batch])
    seq_len     = torch.stack([b["seq_len"]     for b in batch])
    target_mask = torch.stack([b["target_mask"] for b in batch])
    is_t30      = torch.stack([b["is_t30"]      for b in batch])
    is_t60      = torch.stack([b["is_t60"]      for b in batch])
    is_t180     = torch.stack([b["is_t180"]     for b in batch])
    lab_t30     = torch.stack([sparse_to_multihot(b["lab_t30"],  vocab_size) for b in batch])
    lab_t60     = torch.stack([sparse_to_multihot(b["lab_t60"],  vocab_size) for b in batch])
    lab_t180    = torch.stack([sparse_to_multihot(b["lab_t180"], vocab_size) for b in batch])
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
        "seq": seq, "trigger": trigger, "seq_len": seq_len,
        "target_mask": target_mask,
        "lab_t30": lab_t30, "lab_t60": lab_t60, "lab_t180": lab_t180,
        "is_t30": is_t30, "is_t60": is_t60, "is_t180": is_t180,
        "hard_negs": hard_negs,
    }


_loader_kwargs = dict(num_workers=4, pin_memory=(DEVICE == "cuda"), persistent_workers=True)

train_dataset = BERT4RecProviderDataset(train_data, mask_for_inference=False)
val_dataset   = BERT4RecProviderDataset(val_data,   mask_for_inference=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=partial(collate_fn, vocab_size=PROVIDER_VOCAB_SIZE,
                                             from_provider_to_cands=from_provider_to_cands),
                          **_loader_kwargs)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE*2, shuffle=False,
                          collate_fn=partial(collate_fn, vocab_size=PROVIDER_VOCAB_SIZE,
                                             from_provider_to_cands={}),
                          **_loader_kwargs)

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
    def forward(self, x): return self.net(x)


class BERT4RecBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        # No causal mask — bidirectional attention
        self.attn    = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff      = PointWiseFeedForward(dim, dropout)
        self.norm1   = nn.LayerNorm(dim)
        self.norm2   = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        # No causal_mask — BERT4Rec attends in both directions
        attn_out, _ = self.attn(x, x, x,
                                key_padding_mask=pad_mask,
                                need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ff(x))
        return x


class BERT4RecProvider(nn.Module):
    """
    BERT4Rec for provider-level recommendation.

    Key difference from SASRec:
      - Bidirectional attention (no causal mask)
      - MASK token replaces masked provider_ids during training
      - provider_emb table extended by 1 for MASK_IDX
      - Prediction from masked positions (not last position)
      - trigger_dx still appended as last token (not masked)

    At inference: last real position is masked, prediction from there.
    """
    def __init__(self, provider_vocab_size, spec_vocab_size, dx_vocab_size,
                 d_model, max_seq_len, num_heads, num_blocks, dropout):
        super().__init__()
        self.d_model     = d_model
        self.max_seq_len = max_seq_len

        # provider_emb extended by 1 for MASK token (MASK_IDX = provider_vocab_size)
        self.provider_emb = nn.Embedding(provider_vocab_size + 1, d_model, padding_idx=PAD_IDX)
        self.spec_emb     = nn.Embedding(spec_vocab_size, d_model, padding_idx=PAD_IDX)
        self.dx_emb       = nn.Embedding(dx_vocab_size,   d_model, padding_idx=PAD_IDX)
        self.pos_emb      = nn.Embedding(max_seq_len + 1, d_model)  # +1 for trigger
        self.emb_drop     = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            BERT4RecBlock(d_model, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)

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

    def encode(self, seq, trigger, target_mask):
        """
        seq:         (B, 20, 2)  — provider_id (may be MASK_IDX), specialty_id
        trigger:     (B, 1)      — trigger_dx_id
        target_mask: (B, 21)     — 1 at masked positions (incl trigger slot)

        Returns: hidden (B, 21, d) — full sequence hidden states
        """
        B, L, _ = seq.shape

        # Composite embedding — MASK token has its own embedding
        prov_emb = self.provider_emb(seq[:, :, 0])   # (B, L, d)
        spec_emb = self.spec_emb(seq[:, :, 1])        # (B, L, d)
        x        = prov_emb + spec_emb                 # (B, L, d)

        # Append trigger token
        dx_emb = self.dx_emb(trigger)                  # (B, 1, d)
        x      = torch.cat([x, dx_emb], dim=1)          # (B, L+1, d)
        TL     = L + 1

        # Positional embeddings
        positions = torch.arange(TL, device=seq.device).unsqueeze(0)
        x = self.emb_drop(x + self.pos_emb(positions))

        # Padding mask: True where position is padded (provider_id == PAD and not masked)
        pad_flags  = (seq[:, :, 0] == PAD_IDX) & (target_mask[:, :L] == 0)
        trigger_ok = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
        pad_mask   = torch.cat([pad_flags, trigger_ok], dim=1)   # (B, L+1)

        # Bidirectional attention — no causal mask
        for block in self.blocks:
            x = block(x, pad_mask)
        return self.norm(x)    # (B, L+1, d)

    def forward(self, seq, trigger, target_mask):
        """
        Full scoring from masked positions.
        Returns scores (B, V-2) — scored against top80 providers.
        At inference, prediction from the last masked position.
        """
        hidden    = self.encode(seq, trigger, target_mask)   # (B, L+1, d)

        # Find masked positions — predict from first masked position per sample
        # target_mask: 1 at masked positions
        mask_pos  = target_mask[:, :self.max_seq_len].float().argmax(dim=1)  # (B,)
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        user_repr = hidden[batch_idx, mask_pos, :]                           # (B, d)

        # Dot-product against top80 providers (skip PAD=0, UNK=1, MASK=vocab_size)
        all_ids   = torch.arange(2, len(provider_vocab),
                                 device=seq.device)           # (vocab-2,)
        emb       = self.provider_emb(all_ids)                # (vocab-2, d)
        scores    = user_repr @ emb.T                         # (B, vocab-2)
        return scores, user_repr

    def score(self, user_repr, provider_ids):
        emb = self.provider_emb(provider_ids)
        if emb.dim() == 2: return user_repr @ emb.T
        return (user_repr.unsqueeze(1) * emb).sum(-1)


def get_raw(model):
    return model.module if isinstance(model, nn.DataParallel) else model


model = BERT4RecProvider(
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
# SECTION 4 — LOSS + NEGATIVES (same as NB_04)
# ══════════════════════════════════════════════════════════════════════════════

def build_neg_matrix(batch_size, hard_negs_list, provider_vocab_size,
                     neg_k, hard_neg_k, device):
    rand_negs = torch.randint(2, provider_vocab_size,
                              (batch_size, neg_k), device=device)
    if hard_neg_k == 0:
        return rand_negs
    hard_mat = torch.randint(2, provider_vocab_size,
                             (batch_size, hard_neg_k), device=device)
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
    neg_scores  = torch.bmm(user_repr.unsqueeze(1),
                            neg_emb.transpose(1, 2)).squeeze(1) / temperature
    ib_scores   = (user_repr @ user_repr.T) / temperature
    ib_scores.fill_diagonal_(float('-inf'))
    all_neg_scores = torch.cat([neg_scores, ib_scores], dim=1)
    pos_rows, pos_cols = labels_multihot.nonzero(as_tuple=True)
    if len(pos_rows) == 0:
        return torch.tensor(0.0, device=user_repr.device)
    pos_emb     = model_raw.provider_emb(pos_cols)
    pos_scores  = (user_repr[pos_rows] * pos_emb).sum(dim=1) / temperature
    log_sum_neg = torch.logsumexp(all_neg_scores, dim=1)
    return (-pos_scores + log_sum_neg[pos_rows]).mean()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATION (same metrics as NB_04)
# ══════════════════════════════════════════════════════════════════════════════

def metrics_at_k(scores, labels_multihot, k):
    topk_idx  = scores.topk(k, dim=1).indices
    topk_hits = labels_multihot.gather(1, topk_idx)
    hit_at_k  = (topk_hits.sum(1) > 0).float().mean().item()
    prec_at_k = topk_hits.sum(1).float().div(k).mean().item()
    n_pos     = labels_multihot.sum(1).clamp(min=1)
    rec_at_k  = topk_hits.sum(1).float().div(n_pos).mean().item()
    positions = torch.arange(1, k + 1, dtype=torch.float32, device=scores.device)
    discounts = 1.0 / torch.log2(positions + 1)
    dcg       = (topk_hits.float() * discounts).sum(1)
    ideal     = torch.zeros(scores.shape[0], k, device=scores.device)
    for i in range(scores.shape[0]):
        n = min(int(n_pos[i].item()), k)
        ideal[i, :n] = 1.0
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
            seq     = batch["seq"].to(DEVICE,         non_blocking=True)
            trigger = batch["trigger"].to(DEVICE,     non_blocking=True)
            tm      = batch["target_mask"].to(DEVICE, non_blocking=True)
            hard_negs_list = batch["hard_negs"]
            B = seq.shape[0]

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                scores, user_repr = mdl(seq, trigger, tm)

            # Val loss
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
            full_scores = torch.cat([pad_scores, scores], dim=1)
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

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
model_raw = get_raw(model)
best_ndcg = 0.0
train_log = []

print(f"Training — {EPOCHS} epochs | {len(train_loader)} batches/epoch")

for epoch in range(EPOCHS):
    t_ep = time.time()
    model.train()
    total_loss, n_batches = 0.0, 0

    for batch_idx, batch in enumerate(train_loader):
        seq     = batch["seq"].to(DEVICE,         non_blocking=True)
        trigger = batch["trigger"].to(DEVICE,     non_blocking=True)
        tm      = batch["target_mask"].to(DEVICE, non_blocking=True)
        is_t30  = batch["is_t30"].to(DEVICE,  non_blocking=True)
        is_t60  = batch["is_t60"].to(DEVICE,  non_blocking=True)
        is_t180 = batch["is_t180"].to(DEVICE, non_blocking=True)
        lab_t30 = batch["lab_t30"].to(DEVICE, non_blocking=True)
        lab_t60 = batch["lab_t60"].to(DEVICE, non_blocking=True)
        lab_t180= batch["lab_t180"].to(DEVICE,non_blocking=True)
        hard_negs_list = batch["hard_negs"]
        B = seq.shape[0]

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            hidden    = model_raw.encode(seq, trigger, tm)     # (B, 21, d)

            # Get user repr from masked positions
            mask_pos    = tm[:, :MAX_SEQ_LEN].float().argmax(dim=1)
            batch_idx_t = torch.arange(B, device=DEVICE)
            user_repr   = hidden[batch_idx_t, mask_pos, :]     # (B, d)

            # Build neg matrix — one call for whole batch
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

    if val_ndcg >= best_ndcg:
        best_ndcg = val_ndcg
        ckpt_path = f"{MODEL_DIR}/bert4rec_provider_{SAMPLE}_ep{epoch+1}.pt"
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
                "mask_prob":           MASK_PROB,
                "mask_idx":            MASK_IDX,
            },
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")

print(f"\nTraining done — {time.time()-t0:.1f}s")

log_path = f"{OUTPUT_DIR}/bert4rec_train_log_{SAMPLE}.csv"
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
| Checkpoint | bert4rec_provider_{SAMPLE}_ep*.pt |

Next: run NB_06 (HSTU) or NB_07 (scoring).
"""))
print("NB_05 complete")

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
