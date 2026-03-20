# ============================================================
# Model_01_sasrec_train.py
# Purpose : Train SASRec on pre-built train dataset
# Input   : ./cache_model_data_{SAMPLE}/train_*.npy + vocab.pkl
# Output  : /home/jupyter/models/sasrec_{SAMPLE}_{TS}.pt
#           /home/jupyter/models/sasrec_{SAMPLE}_{TS}_vocab.pkl
# ============================================================
import gc
import os
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from datetime import datetime, timezone
from IPython.display import display, Markdown

print("Imports done")

# ── HARDWARE ──────────────────────────────────────────────────────────────────
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count()

display(Markdown(f"""
## System Hardware
| Resource | Value |
|---|---|
| Device | {DEVICE.upper()} |
| GPUs | {NUM_GPUS} |
| GPU names | {", ".join([torch.cuda.get_device_name(i) for i in range(NUM_GPUS)]) if NUM_GPUS else "N/A"} |
| GPU memory | {", ".join([f"{torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB" for i in range(NUM_GPUS)]) if NUM_GPUS else "N/A"} |
| CPU cores | {os.cpu_count()} |
| PyTorch | {torch.__version__} |
"""))

# ── CONFIG ────────────────────────────────────────────────────────────────────
SAMPLE        = "1pct"
MAX_SEQ_LEN   = 20
EMBEDDING_DIM = 128
NUM_HEADS     = 4
NUM_BLOCKS    = 2
DROPOUT       = 0.2
BATCH_SIZE    = 512
EPOCHS        = 10
LR            = 4e-4
WEIGHT_DECAY  = 1e-5
PATIENCE      = 3
K_VALUES      = [1, 3, 5]
WINDOWS       = ["T0_30", "T30_60", "T60_180"]
PAD_IDX       = 0
NUM_WORKERS   = 2

RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

CACHE_DIR  = f"./cache_model_data_{SAMPLE}"
MODEL_DIR  = "/home/jupyter/models"
CHECKPOINT = f"{MODEL_DIR}/sasrec_{SAMPLE}_{RUN_TIMESTAMP}.pt"
VOCAB_PATH = f"{MODEL_DIR}/sasrec_{SAMPLE}_{RUN_TIMESTAMP}_vocab.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)

_loader_kwargs = dict(pin_memory=(DEVICE == "cuda"), num_workers=NUM_WORKERS)
if NUM_WORKERS > 0:
    _loader_kwargs.update(prefetch_factor=2, persistent_workers=True)

print(f"Config — sample={SAMPLE}, device={DEVICE}")
print(f"Run timestamp : {RUN_TIMESTAMP}")
print(f"Checkpoint    : {CHECKPOINT}")
print(f"Cache dir     : {CACHE_DIR}")

display(Markdown(f"""
## Config
| Parameter | Value |
|---|---|
| Sample | {SAMPLE} |
| Max sequence length | {MAX_SEQ_LEN} |
| Embedding dim | {EMBEDDING_DIM} |
| Heads / Blocks | {NUM_HEADS} / {NUM_BLOCKS} |
| Dropout | {DROPOUT} |
| Batch size | {BATCH_SIZE} |
| Epochs | {EPOCHS} |
| LR | {LR} |
| Patience | {PATIENCE} |
"""))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD TRAIN + VAL DATA FROM CACHE
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 1 — Load Train/Val Data"))
print("Section 1 — loading from cache...")

KEYS = ["seq_matrix", "lab_t30", "lab_t60", "lab_t180",
        "seq_lengths", "is_t30", "is_t60", "is_t180",
        "trigger_dates", "member_ids", "trigger_dxs", "segments"]

train_data = {k: np.load(f"{CACHE_DIR}/train_{k}.npy", allow_pickle=True) for k in KEYS}
val_data   = {k: np.load(f"{CACHE_DIR}/val_{k}.npy",   allow_pickle=True) for k in KEYS}

with open(f"{CACHE_DIR}/vocab.pkl", "rb") as f:
    specialty_vocab = pickle.load(f)

NUM_SPECIALTIES = len(specialty_vocab)
N_train = train_data["seq_matrix"].shape[0]
N_val   = val_data["seq_matrix"].shape[0]

print(f"Train: {N_train:,} | Val: {N_val:,} | Vocab: {NUM_SPECIALTIES:,}")
print(f"Train date range: {train_data['trigger_dates'][0]} → {train_data['trigger_dates'][-1]}")
print(f"Val   date range: {val_data['trigger_dates'][0]} → {val_data['trigger_dates'][-1]}")
print(f"Section 1 done — time={time.time()-t0:.1f}s")
display(Markdown(f"**Train:** {N_train:,} | **Val:** {N_val:,} | **Vocab:** {NUM_SPECIALTIES:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATASET AND DATALOADER
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 2 — Dataset"))
print("Section 2 — building dataset...")


class SpecialtyDataset(Dataset):
    def __init__(self, data):
        self.seq      = data["seq_matrix"]
        self.lengths  = data["seq_lengths"]
        self.lab_t30  = data["lab_t30"]
        self.lab_t60  = data["lab_t60"]
        self.lab_t180 = data["lab_t180"]
        self.is_t30   = data["is_t30"]
        self.is_t60   = data["is_t60"]
        self.is_t180  = data["is_t180"]

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        return {
            "sequence":  torch.from_numpy(self.seq[idx].copy()),
            "seq_len":   torch.tensor(int(self.lengths[idx]),  dtype=torch.long),
            "label_t30": torch.from_numpy(self.lab_t30[idx]),
            "label_t60": torch.from_numpy(self.lab_t60[idx]),
            "label_t180":torch.from_numpy(self.lab_t180[idx]),
            "is_t30":    torch.tensor(bool(self.is_t30[idx]),  dtype=torch.bool),
            "is_t60":    torch.tensor(bool(self.is_t60[idx]),  dtype=torch.bool),
            "is_t180":   torch.tensor(bool(self.is_t180[idx]), dtype=torch.bool),
        }


train_loader = DataLoader(SpecialtyDataset(train_data),
                          batch_size=BATCH_SIZE, shuffle=True, **_loader_kwargs)
val_loader   = DataLoader(SpecialtyDataset(val_data),
                          batch_size=BATCH_SIZE * 2, shuffle=False, **_loader_kwargs)

print(f"Train loader: {len(train_loader)} batches | Val loader: {len(val_loader)} batches")
print(f"Section 2 done — time={time.time()-t0:.1f}s")
display(Markdown(f"**Batches — Train:** {len(train_loader)} | **Val:** {len(val_loader)} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SASREC MODEL
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 3 — SASRec Model"))
print("Section 3 — building model...")


class PointWiseFeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SASRecBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.attn    = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn     = PointWiseFeedForward(dim, dropout)
        self.norm1   = nn.LayerNorm(dim)
        self.norm2   = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask, pad_mask):
        a, _ = self.attn(x, x, x, attn_mask=causal_mask,
                         key_padding_mask=pad_mask, need_weights=False)
        x = self.norm1(x + self.dropout(a))
        x = self.norm2(x + self.ffn(x))
        return x


class SASRec(nn.Module):
    def __init__(self, num_specialties, embedding_dim, max_seq_len,
                 num_heads, num_blocks, dropout, num_classes):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.item_emb  = nn.Embedding(num_specialties + 1, embedding_dim, padding_idx=PAD_IDX)
        self.pos_emb   = nn.Embedding(max_seq_len, embedding_dim)
        self.emb_drop  = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            SASRecBlock(embedding_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.norm      = nn.LayerNorm(embedding_dim)
        self.head_t30  = nn.Linear(embedding_dim, num_classes)
        self.head_t60  = nn.Linear(embedding_dim, num_classes)
        self.head_t180 = nn.Linear(embedding_dim, num_classes)
        causal = torch.triu(
            torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1
        )
        self.register_buffer("causal_mask", causal)
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

    def forward(self, seq, lengths):
        B, L     = seq.shape
        pos      = torch.arange(L, device=seq.device).unsqueeze(0)
        x        = self.emb_drop(self.item_emb(seq) + self.pos_emb(pos))
        pad_mask = (seq == PAD_IDX)
        for block in self.blocks:
            x = block(x, self.causal_mask[:L, :L], pad_mask)
        x = self.norm(x)
        idx      = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, self.embedding_dim)
        seq_repr = x.gather(1, idx).squeeze(1)
        return (
            self.head_t30(seq_repr),
            self.head_t60(seq_repr),
            self.head_t180(seq_repr)
        )


def get_raw(m):
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    if isinstance(m, nn.DataParallel):
        m = m.module
    return m


model = SASRec(
    num_specialties=NUM_SPECIALTIES,
    embedding_dim=EMBEDDING_DIM,
    max_seq_len=MAX_SEQ_LEN,
    num_heads=NUM_HEADS,
    num_blocks=NUM_BLOCKS,
    dropout=DROPOUT,
    num_classes=NUM_SPECIALTIES
).to(DEVICE)

if NUM_GPUS > 1:
    print(f"Wrapping in DataParallel — {NUM_GPUS} GPUs")
    model = nn.DataParallel(model)

raw_check = get_raw(model)
print(f"Attribute check: emb_drop={hasattr(raw_check,'emb_drop')} | "
      f"item_emb={hasattr(raw_check,'item_emb')} | "
      f"blocks={hasattr(raw_check,'blocks')}")

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Section 3 done — {n_params:,} parameters, time={time.time()-t0:.1f}s")
display(Markdown(f"**Parameters:** {n_params:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EVALUATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
print("Section 4 — defining evaluation functions...")

_disc_cache = {
    k: 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float32))
    for k in K_VALUES
}
_bce_eval = nn.BCEWithLogitsLoss()


def metrics_at_k(pred, label, k):
    disc       = _disc_cache[k].to(pred.device)
    topk       = torch.topk(pred, k, dim=1).indices
    hits       = label.gather(1, topk)
    hit        = (hits.sum(1) > 0).float()
    prec       = hits.sum(1) / k
    rec        = hits.sum(1) / label.sum(1).clamp(min=1)
    dcg        = (hits.float() * disc).sum(1)
    n_ideal    = label.sum(1).clamp(max=k).long()
    ranks      = torch.arange(1, k + 1, device=pred.device)
    ideal_mask = ranks.unsqueeze(0) <= n_ideal.unsqueeze(1)
    idcg       = (disc.unsqueeze(0) * ideal_mask.float()).sum(1)
    ndcg       = dcg / idcg.clamp(min=1e-8)
    return hit.mean(), prec.mean(), rec.mean(), ndcg.mean()


def evaluate(loader, mdl):
    raw = get_raw(mdl)
    raw.eval()
    sums, counts       = defaultdict(float), defaultdict(int)
    val_loss_sum       = 0.0
    val_loss_count     = 0

    with torch.no_grad():
        for batch in loader:
            seq   = batch["sequence"].to(DEVICE, non_blocking=True)
            lens  = batch["seq_len"].to(DEVICE, non_blocking=True)
            l30   = batch["label_t30"].to(DEVICE, non_blocking=True)
            l60   = batch["label_t60"].to(DEVICE, non_blocking=True)
            l180  = batch["label_t180"].to(DEVICE, non_blocking=True)
            m30   = batch["is_t30"].to(DEVICE, non_blocking=True)
            m60   = batch["is_t60"].to(DEVICE, non_blocking=True)
            m180  = batch["is_t180"].to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                p30, p60, p180 = raw(seq, lens)

            batch_loss = torch.tensor(0.0, device=DEVICE)
            n_windows  = 0
            if m30.sum()  > 0:
                batch_loss = batch_loss + _bce_eval(p30[m30].float(),  l30[m30].float())
                n_windows += 1
            if m60.sum()  > 0:
                batch_loss = batch_loss + _bce_eval(p60[m60].float(),  l60[m60].float())
                n_windows += 1
            if m180.sum() > 0:
                batch_loss = batch_loss + _bce_eval(p180[m180].float(), l180[m180].float())
                n_windows += 1
            if n_windows > 0:
                val_loss_sum   += (batch_loss / n_windows).item()
                val_loss_count += 1

            for k in K_VALUES:
                for tag, pred, lbl, mask in [
                    ("T0_30",   p30,  l30,  m30),
                    ("T30_60",  p60,  l60,  m60),
                    ("T60_180", p180, l180, m180),
                ]:
                    n = mask.sum().item()
                    if n == 0:
                        continue
                    hit, prec, rec, ndcg = metrics_at_k(
                        pred[mask].float(), lbl[mask].float(), k
                    )
                    sums[f"{tag}_hit@{k}"]  += hit.item()  * n
                    sums[f"{tag}_prec@{k}"] += prec.item() * n
                    sums[f"{tag}_rec@{k}"]  += rec.item()  * n
                    sums[f"{tag}_ndcg@{k}"] += ndcg.item() * n
                    counts[f"{tag}@{k}"]    += n

    result = {}
    for tag in WINDOWS:
        for k in K_VALUES:
            n = counts[f"{tag}@{k}"]
            for metric in ["hit", "prec", "rec", "ndcg"]:
                key = f"{tag}_{metric}@{k}"
                result[key] = sums[key] / n if n > 0 else 0.0
    result["val_loss"] = val_loss_sum / max(val_loss_count, 1)
    return result


def print_metrics(metrics, split="Val"):
    rows = ["| Window | K | Hit | Precision | Recall | NDCG |", "|---|---|---|---|---|---|"]
    for w in WINDOWS:
        for k in K_VALUES:
            rows.append(
                f"| {w} | {k} "
                f"| {metrics.get(f'{w}_hit@{k}', 0):.4f} "
                f"| {metrics.get(f'{w}_prec@{k}', 0):.4f} "
                f"| {metrics.get(f'{w}_rec@{k}', 0):.4f} "
                f"| {metrics.get(f'{w}_ndcg@{k}', 0):.4f} |"
            )
    display(Markdown(f"**{split}**\n" + "\n".join(rows)))


print("Evaluation functions defined")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 5 — Training"))
print(f"Section 5 — {EPOCHS} epochs, {len(train_loader)} batches/epoch")

optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler     = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
bce        = nn.BCEWithLogitsLoss()
best_ndcg  = 0.0
no_improve = 0

for epoch in range(EPOCHS):
    t_ep = time.time()
    model.train()
    total_loss, n_batches = 0.0, 0

    for batch_idx, batch in enumerate(train_loader):
        seq   = batch["sequence"].to(DEVICE, non_blocking=True)
        lens  = batch["seq_len"].to(DEVICE, non_blocking=True)
        l30   = batch["label_t30"].to(DEVICE, non_blocking=True)
        l60   = batch["label_t60"].to(DEVICE, non_blocking=True)
        l180  = batch["label_t180"].to(DEVICE, non_blocking=True)
        m30   = batch["is_t30"].to(DEVICE, non_blocking=True)
        m60   = batch["is_t60"].to(DEVICE, non_blocking=True)
        m180  = batch["is_t180"].to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            p30, p60, p180 = model(seq, lens)
            loss = torch.tensor(0.0, device=DEVICE)
            if m30.sum()  > 0: loss = loss + bce(p30[m30].float(),   l30[m30].float())
            if m60.sum()  > 0: loss = loss + bce(p60[m60].float(),   l60[m60].float())
            if m180.sum() > 0: loss = loss + bce(p180[m180].float(), l180[m180].float())

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        n_batches  += 1

        if (batch_idx + 1) % 50 == 0:
            print(f"  Ep {epoch+1} | B {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {total_loss/n_batches:.4f}")

    scheduler.step()
    avg_loss  = total_loss / max(n_batches, 1)
    print(f"  Epoch {epoch+1} train done — loss: {avg_loss:.4f} | Running eval...")

    val_m    = evaluate(val_loader, model)
    val_loss = val_m.get("val_loss", 0)
    val_ndcg = np.mean([val_m.get(f"T0_30_ndcg@{k}", 0) for k in K_VALUES])
    ep_time  = time.time() - t_ep

    display(Markdown(
        f"**Epoch {epoch+1}/{EPOCHS}** — "
        f"Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | "
        f"Val NDCG@T0_30: {val_ndcg:.4f} | "
        f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {ep_time:.1f}s"
    ))
    print(f"  Epoch {epoch+1} — Train: {avg_loss:.4f} | Val: {val_loss:.4f} | NDCG: {val_ndcg:.4f}")
    print_metrics(val_m, f"Val Epoch {epoch+1}")

    if val_ndcg > best_ndcg:
        best_ndcg, no_improve = val_ndcg, 0

        torch.save({
            "epoch":            epoch,
            "model_state":      get_raw(model).state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
            "best_val_ndcg":    best_ndcg,
            "specialty_vocab":  specialty_vocab,
            "idx_to_specialty": {v - 1: k for k, v in specialty_vocab.items()
                                  if isinstance(v, int) and v > 0},
            "config": {
                "model":           "SASRec",
                "sample":          SAMPLE,
                "run_timestamp":   RUN_TIMESTAMP,
                "num_specialties": NUM_SPECIALTIES,
                "embedding_dim":   EMBEDDING_DIM,
                "max_seq_len":     MAX_SEQ_LEN,
                "num_heads":       NUM_HEADS,
                "num_blocks":      NUM_BLOCKS,
                "dropout":         DROPOUT,
                "pad_idx":         PAD_IDX,
            },
            "preprocessing": {
                "lookback_days": 365,
                "seq_ordering":  "visit_date DESC",
                "padding":       "left",
                "date_format":   "YYYY-MM-DD",
                "train_cutoff":  "2024-01-01",
            },
            "output_heads": {
                "head_t30":  "T0_30  — days 1-30 after trigger",
                "head_t60":  "T30_60 — days 31-60 after trigger",
                "head_t180": "T60_180 — days 61-180 after trigger",
            },
        }, CHECKPOINT)

        with open(VOCAB_PATH, "wb") as f:
            pickle.dump({
                "specialty_vocab":  specialty_vocab,
                "idx_to_specialty": {v - 1: k for k, v in specialty_vocab.items()
                                     if isinstance(v, int) and v > 0},
                "pad_idx":         PAD_IDX,
                "num_specialties": NUM_SPECIALTIES,
            }, f)

        print(f"  Checkpoint : {CHECKPOINT}")
        print(f"  Vocab      : {VOCAB_PATH}")
        print(f"  Best NDCG  : {best_ndcg:.4f}")
        display(Markdown(f"Checkpoint saved — NDCG: {best_ndcg:.4f}"))
    else:
        no_improve += 1
        print(f"  No improvement ({no_improve}/{PATIENCE})")
        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            display(Markdown(f"Early stopping at epoch {epoch+1}"))
            break

print(f"Section 5 done — time={time.time()-t0:.1f}s")
display(Markdown(f"**Training complete** | Best NDCG: {best_ndcg:.4f} | Checkpoint: `{CHECKPOINT}`"))
print("Model_01_sasrec_train complete")
