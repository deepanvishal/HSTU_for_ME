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
BATCH_SIZE   = 512
EPOCHS       = 1              # first pass — validate before tuning
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
with open(f"{CACHE_DIR}/train_hard_neg_candidates.pkl", "rb") as f:
    hard_neg_candidates = pickle.load(f)

PROVIDER_VOCAB_SIZE = len(provider_vocab)
SPEC_VOCAB_SIZE     = len(specialty_vocab)
DX_VOCAB_SIZE       = len(dx_vocab)

keys = ["seq_matrix", "delta_t_matrix", "trigger_token", "seq_lengths",
        "lab_t30", "lab_t60", "lab_t180",
        "is_t30", "is_t60", "is_t180",
        "member_ids", "trigger_dates", "trigger_dxs", "segments"]

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
    def __init__(self, data, hard_negs=None):
        self.seq_matrix    = data["seq_matrix"]          # (N, 20, 2)
        self.trigger_token = data["trigger_token"]        # (N, 1)
        self.seq_lengths   = data["seq_lengths"]
        self.lab_t30       = data["lab_t30"]              # ragged
        self.lab_t60       = data["lab_t60"]
        self.lab_t180      = data["lab_t180"]
        self.is_t30        = data["is_t30"]
        self.is_t60        = data["is_t60"]
        self.is_t180       = data["is_t180"]
        self.hard_negs     = hard_negs                    # dict or None

    def __len__(self):
        return len(self.seq_lengths)

    def __getitem__(self, idx):
        hard_neg = self.hard_negs[idx] if (self.hard_negs and idx in self.hard_negs) else np.array([], dtype=np.int32)
        return {
            "seq":          torch.from_numpy(self.seq_matrix[idx].copy()),    # (20, 2)
            "trigger":      torch.from_numpy(self.trigger_token[idx].copy()), # (1,)
            "seq_len":      torch.tensor(int(self.seq_lengths[idx]), dtype=torch.long),
            "lab_t30":      self.lab_t30[idx],   # kept as list for collate_fn
            "lab_t60":      self.lab_t60[idx],
            "lab_t180":     self.lab_t180[idx],
            "is_t30":       torch.tensor(bool(self.is_t30[idx]),  dtype=torch.bool),
            "is_t60":       torch.tensor(bool(self.is_t60[idx]),  dtype=torch.bool),
            "is_t180":      torch.tensor(bool(self.is_t180[idx]), dtype=torch.bool),
            "hard_neg":     hard_neg,
        }


def collate_fn(batch, vocab_size):
    """
    Collate batch — builds multi-hot labels on-the-fly from sparse lists.
    Dense multi-hot at 31K × N would be too large to store; we build per batch.
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

    # Hard neg: list of arrays (variable length) — kept as list
    hard_negs = [b["hard_neg"] for b in batch]

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

train_dataset = ProviderDataset(train_data, hard_negs=hard_neg_candidates)
val_dataset   = ProviderDataset(val_data,   hard_negs=None)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=partial(collate_fn, vocab_size=PROVIDER_VOCAB_SIZE),
    **_loader_kwargs,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2, shuffle=False,
    collate_fn=partial(collate_fn, vocab_size=PROVIDER_VOCAB_SIZE),
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

def infonce_loss(user_repr, pos_ids, neg_ids, model_raw, temperature=0.07):
    """
    InfoNCE loss over positives and negatives.

    user_repr: (B, d)
    pos_ids:   list of tensors, one per sample — positive provider int_ids
    neg_ids:   (B, K) — random + hard negative int_ids

    For each sample: loss = -log(exp(u·p/τ) / sum(exp(u·n/τ) for all n))
    """
    loss_total = torch.tensor(0.0, device=user_repr.device)
    n_valid    = 0

    for i in range(len(user_repr)):
        u   = user_repr[i]                      # (d,)
        pos = pos_ids[i]                        # variable length
        if len(pos) == 0:
            continue

        pos_t = torch.tensor(pos, dtype=torch.long, device=u.device)
        neg_t = neg_ids[i]                      # (K,)

        # Scores for positives and negatives
        pos_emb = model_raw.provider_emb(pos_t)  # (|pos|, d)
        neg_emb = model_raw.provider_emb(neg_t)  # (K, d)

        pos_scores = (u @ pos_emb.T) / temperature   # (|pos|,)
        neg_scores = (u @ neg_emb.T) / temperature   # (K,)

        # Per-positive loss: -pos_score + log(sum_exp(all scores))
        all_scores = torch.cat([pos_scores, neg_scores])
        log_sum    = torch.logsumexp(all_scores, dim=0)
        loss_i     = (-pos_scores + log_sum).mean()

        loss_total = loss_total + loss_i
        n_valid   += 1

    return loss_total / max(n_valid, 1)


def sample_negatives(batch_size, hard_negs_list, provider_vocab_size,
                     neg_k, hard_neg_k, device):
    """
    For each sample: neg_k random + up to hard_neg_k hard negatives.
    Returns list of tensors, one per sample.
    """
    neg_ids = []
    for i in range(batch_size):
        # Random negatives — uniform over [2, vocab_size)
        rand_neg = torch.randint(2, provider_vocab_size,
                                 (neg_k,), device=device)

        # Hard negatives
        hn = hard_negs_list[i]
        if len(hn) > 0:
            # Cap at hard_neg_k, sample without replacement if enough
            if len(hn) > hard_neg_k:
                idx = torch.randperm(len(hn))[:hard_neg_k]
                hn  = torch.tensor(hn[idx], dtype=torch.long, device=device)
            else:
                hn  = torch.tensor(hn, dtype=torch.long, device=device)
            neg = torch.cat([rand_neg, hn])
        else:
            neg = rand_neg

        neg_ids.append(neg)
    return neg_ids


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
    model_raw = get_raw(mdl)
    results   = {}

    buckets = {"T0_30": ("is_t30", "lab_t30"),
               "T30_60": ("is_t60", "lab_t60"),
               "T60_180": ("is_t180", "lab_t180")}

    accum = {b: {k: [] for k in ["hit", "prec", "rec", "ndcg"]}
             for b in buckets for k in K_VALUES}

    with torch.no_grad():
        for batch in loader:
            seq     = batch["seq"].to(DEVICE, non_blocking=True)
            trigger = batch["trigger"].to(DEVICE, non_blocking=True)
            seq_len = batch["seq_len"].to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                scores, _ = mdl(seq, trigger, seq_len)    # (B, V-2)

            # Pad scores back to full vocab size for label alignment
            # scores correspond to provider ids [2, PROVIDER_VOCAB_SIZE)
            # Prepend two zeros for PAD=0 and UNK=1
            B = scores.shape[0]
            pad_scores = torch.zeros(B, 2, device=scores.device)
            full_scores = torch.cat([pad_scores, scores], dim=1)  # (B, V)

            for bucket, (flag_key, lab_key) in buckets.items():
                mask = batch[flag_key].to(DEVICE)
                if mask.sum() == 0:
                    continue
                labs  = batch[lab_key].to(DEVICE)       # (B, V)
                s_m   = full_scores[mask]
                l_m   = labs[mask]

                for k in K_VALUES:
                    m = metrics_at_k(s_m, l_m, k)
                    for metric, val in m.items():
                        accum[bucket][k] = accum[bucket].get(k, {})
                        if metric not in accum[bucket]:
                            accum[bucket][metric] = []
                        # Fix: store per bucket+k+metric
    # Recompute properly
    bucket_k_metrics = {}
    for bucket, (flag_key, lab_key) in buckets.items():
        bucket_k_metrics[bucket] = {}

    # Re-evaluate cleanly with proper accumulation
    all_metrics = {b: {k: {"hit":[],"prec":[],"rec":[],"ndcg":[]}
                        for k in K_VALUES} for b in buckets}

    with torch.no_grad():
        for batch in loader:
            seq     = batch["seq"].to(DEVICE, non_blocking=True)
            trigger = batch["trigger"].to(DEVICE, non_blocking=True)
            seq_len = batch["seq_len"].to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                scores, _ = mdl(seq, trigger, seq_len)
            B = scores.shape[0]
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

    results = {}
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

            # Sample negatives
            neg_ids = sample_negatives(B, hard_negs_list,
                                       PROVIDER_VOCAB_SIZE,
                                       NEG_K, HARD_NEG_K, DEVICE)

            # Compute InfoNCE per active window
            loss = torch.tensor(0.0, device=DEVICE)
            n_windows = 0

            for flag, labs in [(is_t30, lab_t30), (is_t60, lab_t60), (is_t180, lab_t180)]:
                if flag.sum() == 0:
                    continue
                # Get positive provider ids per sample from multi-hot labels
                # labs: (B, V) — 1 where positive
                u_m   = user_repr[flag]
                l_m   = labs[flag]                              # (B_m, V)
                n_m   = flag.sum().item()

                # Extract positive ids per sample
                pos_ids_batch = []
                for j in range(n_m):
                    pos = l_m[j].nonzero(as_tuple=True)[0].cpu().numpy()
                    pos_ids_batch.append(pos)

                neg_ids_m = [neg_ids[k] for k, v in enumerate(flag.tolist()) if v]

                window_loss = infonce_loss(u_m, pos_ids_batch, neg_ids_m, model_raw)
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

        total_loss += loss.item()
        n_batches  += 1

        if (batch_idx + 1) % 100 == 0:
            print(f"  Ep {epoch+1} | B {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {total_loss/n_batches:.4f}")

    scheduler.step()
    avg_loss = total_loss / max(n_batches, 1)
    ep_time  = time.time() - t_ep
    print(f"  Epoch {epoch+1} done — loss={avg_loss:.4f} | time={ep_time:.0f}s")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_metrics = evaluate(val_loader, model)
    val_ndcg    = np.mean([val_metrics.get(f"T0_30_ndcg@{k}", 0) for k in K_VALUES])
    print_metrics(val_metrics, f"Val Epoch {epoch+1}")

    train_log.append({"epoch": epoch+1, "loss": avg_loss,
                       "val_ndcg_t30": val_ndcg, **val_metrics})

    display(Markdown(
        f"**Epoch {epoch+1}/{EPOCHS}** — "
        f"Loss: {avg_loss:.4f} | Val NDCG@T30: {val_ndcg:.4f} | "
        f"Time: {ep_time:.0f}s"
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
