# ============================================================
# Model_05_bert4rec_score.py
# Purpose : Load test dataset + BERT4Rec checkpoint
#           Evaluate on test set
#           Write metrics + trigger scores to BQ
#           Generate visualizations
# Input   : ./cache_model_data_{SAMPLE}/test_*.npy
#           /home/jupyter/models/bert4rec_{SAMPLE}_{TS}.pt
# Output  : A870800_gen_rec_model_metrics  (APPEND)
#           A870800_gen_rec_trigger_scores (APPEND)
#           bert4rec_metrics_{SAMPLE}.png
#           bert4rec_ndcg_{SAMPLE}.png
# ============================================================
import os
import pickle
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from datetime import datetime, timezone
from google.cloud import bigquery
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import seaborn as sns

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
"""))

# ── CONFIG ────────────────────────────────────────────────────────────────────
SAMPLE      = "5pct"
K_VALUES    = [1, 3, 5]
WINDOWS     = ["T0_30", "T30_60", "T60_180"]
PAD_IDX     = 0
NUM_WORKERS = 2

# ── SET CHECKPOINT PATH ───────────────────────────────────────────────────────
# Update this to the .pt file saved by Model_02_bert4rec_train
CHECKPOINT = "/home/jupyter/models/bert4rec_1pct_YYYY-MM-DD_HH-MM-SS.pt"
# To find latest: ls -lt /home/jupyter/models/bert4rec_*.pt | head -1

DS        = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
CACHE_DIR       = f"./cache_test_data_{SAMPLE}"
TRAIN_CACHE_DIR = f"./cache_train_data_{SAMPLE}"

RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

_loader_kwargs = dict(pin_memory=(DEVICE == "cuda"), num_workers=NUM_WORKERS)
if NUM_WORKERS > 0:
    _loader_kwargs.update(prefetch_factor=2, persistent_workers=True)

client = bigquery.Client(project="anbc-hcb-dev")

print(f"Config — sample={SAMPLE}, device={DEVICE}")
print(f"Checkpoint     : {CHECKPOINT}")
print(f"Score timestamp: {RUN_TIMESTAMP}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 1 — Load Checkpoint"))
print("Section 1 — loading checkpoint...")

ckpt             = torch.load(CHECKPOINT, weights_only=False)
cfg              = ckpt["config"]
specialty_vocab  = ckpt["specialty_vocab"]
idx_to_specialty = ckpt["idx_to_specialty"]
NUM_SPECIALTIES  = cfg["num_specialties"]
MASK_IDX         = cfg["mask_idx"]

print(f"Checkpoint loaded — epoch {ckpt['epoch']+1}, best val NDCG: {ckpt['best_val_ndcg']:.4f}")
print(f"MASK_IDX: {MASK_IDX}")
print(f"idx_to_specialty entries: {len(idx_to_specialty):,}")
display(Markdown(f"**Best Val NDCG:** {ckpt['best_val_ndcg']:.4f} | **Epoch:** {ckpt['epoch']+1}"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BERT4Rec MODEL
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 2 — BERT4Rec Model"))
print("Section 2 — rebuilding model from checkpoint...")


class PointWiseFeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class BERT4RecBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.attn    = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn     = PointWiseFeedForward(dim, dropout)
        self.norm1   = nn.LayerNorm(dim)
        self.norm2   = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        a, _ = self.attn(x, x, x, key_padding_mask=pad_mask, need_weights=False)
        x = self.norm1(x + self.dropout(a))
        x = self.norm2(x + self.ffn(x))
        return x


class BERT4Rec(nn.Module):
    def __init__(self, num_specialties, mask_idx, embedding_dim, max_seq_len,
                 num_heads, num_blocks, dropout, num_classes):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mask_idx      = mask_idx
        vocab_size         = mask_idx + 1
        self.item_emb  = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.pos_emb   = nn.Embedding(max_seq_len, embedding_dim)
        self.emb_drop  = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            BERT4RecBlock(embedding_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.norm      = nn.LayerNorm(embedding_dim)
        self.head_t30  = nn.Linear(embedding_dim, num_classes)
        self.head_t60  = nn.Linear(embedding_dim, num_classes)
        self.head_t180 = nn.Linear(embedding_dim, num_classes)

    def forward(self, seq, target_mask):
        B, L     = seq.shape
        pos      = torch.arange(L, device=seq.device).unsqueeze(0)
        x        = self.emb_drop(self.item_emb(seq) + self.pos_emb(pos))
        pad_mask = (seq == PAD_IDX)
        for block in self.blocks:
            x = block(x, pad_mask)
        x        = self.norm(x)
        tm_exp   = target_mask.unsqueeze(-1)
        denom    = target_mask.sum(dim=1, keepdim=True)
        seq_repr = (x * tm_exp).sum(dim=1) / denom.clamp(min=1)
        return (
            self.head_t30(seq_repr),
            self.head_t60(seq_repr),
            self.head_t180(seq_repr)
        )


model = BERT4Rec(
    num_specialties=cfg["num_specialties"],
    mask_idx=cfg["mask_idx"],
    embedding_dim=cfg["embedding_dim"],
    max_seq_len=cfg["max_seq_len"],
    num_heads=cfg["num_heads"],
    num_blocks=cfg["num_blocks"],
    dropout=cfg["dropout"],
    num_classes=cfg["num_specialties"],
).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded — {n_params:,} parameters, time={time.time()-t0:.1f}s")
display(Markdown(f"**Parameters:** {n_params:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOAD TEST DATA
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 3 — Load Test Data"))
print("Section 3 — loading test data from cache...")

KEYS = ["seq_matrix", "lab_t30", "lab_t60", "lab_t180",
        "seq_lengths", "is_t30", "is_t60", "is_t180",
        "trigger_dates", "member_ids", "trigger_dxs", "segments"]

test_data = {k: np.load(f"{CACHE_DIR}/test_{k}.npy", allow_pickle=True) for k in KEYS}
N_test    = test_data["seq_matrix"].shape[0]
print(f"Test data loaded — {N_test:,} records")
print(f"Section 3 done — time={time.time()-t0:.1f}s")
display(Markdown(f"**Test records:** {N_test:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EVALUATE ON TEST SET
# BERT4Rec inference: mask last real position
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 4 — Test Evaluation"))
print("Section 4 — evaluating on test set...")


class BERT4RecDataset(Dataset):
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
        seq     = self.seq[idx].copy()
        seq_len = int(self.lengths[idx])
        L       = len(seq)

        # Inference: mask last real position
        mask_pos              = L - 1
        seq[mask_pos]         = MASK_IDX
        target_mask           = np.zeros(L, dtype=np.float32)
        target_mask[mask_pos] = 1.0

        return {
            "sequence":    torch.from_numpy(seq),
            "seq_len":     torch.tensor(seq_len, dtype=torch.long),
            "target_mask": torch.from_numpy(target_mask),
            "label_t30":   torch.from_numpy(self.lab_t30[idx]),
            "label_t60":   torch.from_numpy(self.lab_t60[idx]),
            "label_t180":  torch.from_numpy(self.lab_t180[idx]),
            "is_t30":      torch.tensor(bool(self.is_t30[idx]),  dtype=torch.bool),
            "is_t60":      torch.tensor(bool(self.is_t60[idx]),  dtype=torch.bool),
            "is_t180":     torch.tensor(bool(self.is_t180[idx]), dtype=torch.bool),
        }


_disc_cache = {
    k: 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float32))
    for k in K_VALUES
}


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


BATCH_SIZE  = 1024
test_loader = DataLoader(
    BERT4RecDataset(test_data),
    batch_size=BATCH_SIZE, shuffle=False, **_loader_kwargs
)

sums, counts = defaultdict(float), defaultdict(int)

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        seq   = batch["sequence"].to(DEVICE, non_blocking=True)
        tm    = batch["target_mask"].to(DEVICE, non_blocking=True)
        l30   = batch["label_t30"].to(DEVICE, non_blocking=True)
        l60   = batch["label_t60"].to(DEVICE, non_blocking=True)
        l180  = batch["label_t180"].to(DEVICE, non_blocking=True)
        m30   = batch["is_t30"].to(DEVICE, non_blocking=True)
        m60   = batch["is_t60"].to(DEVICE, non_blocking=True)
        m180  = batch["is_t180"].to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            p30, p60, p180 = model(seq, tm)

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

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(test_loader)}")

test_metrics = {}
for tag in WINDOWS:
    for k in K_VALUES:
        n = counts[f"{tag}@{k}"]
        for metric in ["hit", "prec", "rec", "ndcg"]:
            key = f"{tag}_{metric}@{k}"
            test_metrics[key] = sums[key] / n if n > 0 else 0.0

rows = ["| Window | K | Hit | Precision | Recall | NDCG |", "|---|---|---|---|---|---|"]
for w in WINDOWS:
    for k in K_VALUES:
        rows.append(
            f"| {w} | {k} "
            f"| {test_metrics.get(f'{w}_hit@{k}', 0):.4f} "
            f"| {test_metrics.get(f'{w}_prec@{k}', 0):.4f} "
            f"| {test_metrics.get(f'{w}_rec@{k}', 0):.4f} "
            f"| {test_metrics.get(f'{w}_ndcg@{k}', 0):.4f} |"
        )
display(Markdown("**TEST RESULTS**\n" + "\n".join(rows)))
print(f"Section 4 done — time={time.time()-t0:.1f}s")
display(Markdown(f"**Section 4:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SCORE TRIGGERS AND WRITE TO BIGQUERY
# Vectorized: topk + hit@k + ndcg@k computed on GPU per batch
# Python loop only assembles strings — unavoidable for BQ row format
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 5 — Score Triggers"))
print("Section 5 — scoring triggers (vectorized)...")

score_loader = DataLoader(
    BERT4RecDataset(test_data),
    batch_size=BATCH_SIZE, shuffle=False, **_loader_kwargs
)

# Precompute discount vectors on GPU once
SCORE_K    = 5
disc_score = (1.0 / torch.log2(
    torch.arange(2, SCORE_K + 2, dtype=torch.float32)
)).to(DEVICE)                                            # [5]

# Precompute idx_to_specialty as numpy array for fast indexing
max_idx     = max(idx_to_specialty.keys()) + 1
spec_lookup = np.array(["UNK"] * max_idx, dtype=object)
for idx, sp in idx_to_specialty.items():
    spec_lookup[idx] = sp

BUCKET_KEY = {"T0_30": "lab_t30", "T30_60": "lab_t60", "T60_180": "lab_t180"}

all_rows   = []
record_idx = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(score_loader):
        seq   = batch["sequence"].to(DEVICE, non_blocking=True)
        tm    = batch["target_mask"].to(DEVICE, non_blocking=True)
        m30   = batch["is_t30"].to(DEVICE, non_blocking=True)
        m60   = batch["is_t60"].to(DEVICE, non_blocking=True)
        m180  = batch["is_t180"].to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            p30, p60, p180 = model(seq, tm)

        p30  = torch.sigmoid(p30.float())
        p60  = torch.sigmoid(p60.float())
        p180 = torch.sigmoid(p180.float())
        bs   = seq.size(0)

        for window, pred, mask in [
            ("T0_30",   p30,  m30),
            ("T30_60",  p60,  m60),
            ("T60_180", p180, m180),
        ]:
            n_qual = mask.sum().item()
            if n_qual == 0:
                continue

            pred_m  = pred[mask].float()                 # [n, C]
            qual_ri = record_idx + mask.nonzero(as_tuple=True)[0].cpu().numpy()
            lbl_np  = test_data[BUCKET_KEY[window]][qual_ri]  # [n, C]
            lbl_m   = torch.from_numpy(lbl_np).to(DEVICE)

            # Vectorized top-5 — one call for all n records
            top5_vals, top5_idx = torch.topk(pred_m, SCORE_K, dim=1)  # [n, 5]

            # Vectorized hits matrix [n, 5]
            hits = lbl_m.gather(1, top5_idx)                  # [n, 5]

            # hit@k — [n] for k=1,3,5
            hit1 = (hits[:, :1].sum(1) > 0).float()
            hit3 = (hits[:, :3].sum(1) > 0).float()
            hit5 = (hits[:, :5].sum(1) > 0).float()

            # ndcg@k vectorized
            n_true = lbl_m.sum(1)

            def vec_ndcg(k):
                d    = disc_score[:k]
                dcg  = (hits[:, :k].float() * d).sum(1)
                ni   = n_true.clamp(max=k).long()
                idcg = torch.stack([d[:ni_i].sum() for ni_i in ni])
                return (dcg / idcg.clamp(min=1e-8)).cpu().numpy()

            ndcg1 = vec_ndcg(1)
            ndcg3 = vec_ndcg(3)
            ndcg5 = vec_ndcg(5)

            # Pull GPU tensors to CPU once per window per batch
            top5_idx_cpu  = top5_idx.cpu().numpy()
            top5_vals_cpu = top5_vals.cpu().numpy()
            hit1_cpu      = hit1.cpu().numpy()
            hit3_cpu      = hit3.cpu().numpy()
            hit5_cpu      = hit5.cpu().numpy()

            # Python loop — only string assembly
            for j, ri in enumerate(qual_ri):
                top5_specs  = [str(spec_lookup[top5_idx_cpu[j, r]])
                               for r in range(SCORE_K)]
                top5_scores = [round(float(top5_vals_cpu[j, r]), 4)
                               for r in range(SCORE_K)]

                true_positions = np.where(lbl_np[j] > 0)[0]
                true_specs     = [str(spec_lookup[p])
                                  for p in true_positions
                                  if p < max_idx and spec_lookup[p] != "UNK"]

                all_rows.append({
                    "member_id":        str(test_data["member_ids"][ri]),
                    "trigger_date":     str(test_data["trigger_dates"][ri]),
                    "trigger_dx":       str(test_data["trigger_dxs"][ri]),
                    "member_segment":   str(test_data["segments"][ri]),
                    "time_bucket":      window,
                    "true_labels":      "|".join(sorted(true_specs)),
                    "top5_predictions": "|".join(top5_specs),
                    "top5_scores":      "|".join(str(s) for s in top5_scores),
                    "hit_at_1":         float(hit1_cpu[j]),
                    "hit_at_3":         float(hit3_cpu[j]),
                    "hit_at_5":         float(hit5_cpu[j]),
                    "ndcg_at_1":        round(float(ndcg1[j]), 4),
                    "ndcg_at_3":        round(float(ndcg3[j]), 4),
                    "ndcg_at_5":        round(float(ndcg5[j]), 4),
                    "model":            "BERT4Rec",
                    "sample":           SAMPLE,
                    "run_timestamp":    RUN_TIMESTAMP,
                })

        record_idx += bs

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(score_loader)} | "
                  f"Rows so far: {len(all_rows):,}")

print(f"Scored {len(all_rows):,} trigger-window pairs")

BATCH_BQ = 100_000
schema = [
    bigquery.SchemaField("member_id",        "STRING"),
    bigquery.SchemaField("trigger_date",     "STRING"),
    bigquery.SchemaField("trigger_dx",       "STRING"),
    bigquery.SchemaField("member_segment",   "STRING"),
    bigquery.SchemaField("time_bucket",      "STRING"),
    bigquery.SchemaField("true_labels",      "STRING"),
    bigquery.SchemaField("top5_predictions", "STRING"),
    bigquery.SchemaField("top5_scores",      "STRING"),
    bigquery.SchemaField("hit_at_1",         "FLOAT64"),
    bigquery.SchemaField("hit_at_3",         "FLOAT64"),
    bigquery.SchemaField("hit_at_5",         "FLOAT64"),
    bigquery.SchemaField("ndcg_at_1",        "FLOAT64"),
    bigquery.SchemaField("ndcg_at_3",        "FLOAT64"),
    bigquery.SchemaField("ndcg_at_5",        "FLOAT64"),
    bigquery.SchemaField("model",            "STRING"),
    bigquery.SchemaField("sample",           "STRING"),
    bigquery.SchemaField("run_timestamp",    "STRING"),
]
job_cfg = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND", schema=schema)

for start in range(0, len(all_rows), BATCH_BQ):
    chunk = pd.DataFrame(all_rows[start:start + BATCH_BQ])
    client.load_table_from_dataframe(
        chunk, f"{DS}.A870800_gen_rec_trigger_scores",
        job_config=job_cfg
    ).result()
    print(f"  Written {start:,} — {min(start+BATCH_BQ, len(all_rows)):,}")

print(f"Section 5 done — {len(all_rows):,} rows, time={time.time()-t0:.1f}s")
display(Markdown(f"**5:** {len(all_rows):,} trigger scores written | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — WRITE METRICS TO BIGQUERY
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 6 — Write Metrics"))
print("Section 6 — writing metrics to BQ...")

metric_rows = [
    {
        "model":          "BERT4Rec",
        "sample":         SAMPLE,
        "run_timestamp":  RUN_TIMESTAMP,
        "time_bucket":    w,
        "k":              k,
        "member_segment": "ALL",
        "hit_at_k":       round(test_metrics.get(f"{w}_hit@{k}",  0), 4),
        "precision_at_k": round(test_metrics.get(f"{w}_prec@{k}", 0), 4),
        "recall_at_k":    round(test_metrics.get(f"{w}_rec@{k}",  0), 4),
        "ndcg_at_k":      round(test_metrics.get(f"{w}_ndcg@{k}", 0), 4),
    }
    for w in WINDOWS for k in K_VALUES
]
metrics_df = pd.DataFrame(metric_rows)
print(metrics_df.to_string(index=False))

client.load_table_from_dataframe(
    metrics_df,
    f"{DS}.A870800_gen_rec_model_metrics",
    job_config=bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema=[
            bigquery.SchemaField("model",          "STRING"),
            bigquery.SchemaField("sample",         "STRING"),
            bigquery.SchemaField("run_timestamp",  "STRING"),
            bigquery.SchemaField("time_bucket",    "STRING"),
            bigquery.SchemaField("k",              "INT64"),
            bigquery.SchemaField("member_segment", "STRING"),
            bigquery.SchemaField("hit_at_k",       "FLOAT64"),
            bigquery.SchemaField("precision_at_k", "FLOAT64"),
            bigquery.SchemaField("recall_at_k",    "FLOAT64"),
            bigquery.SchemaField("ndcg_at_k",      "FLOAT64"),
        ]
    )
).result()
print(f"Section 6 done — time={time.time()-t0:.1f}s")
display(Markdown(f"Written to `A870800_gen_rec_model_metrics` | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 7 — Visualization"))
print("Section 7 — loading metrics and plotting...")

plot_df = client.query(f"""
    SELECT time_bucket, k, hit_at_k, precision_at_k, recall_at_k, ndcg_at_k
    FROM `{DS}.A870800_gen_rec_model_metrics`
    WHERE model = 'BERT4Rec'
      AND sample = '{SAMPLE}'
      AND run_timestamp = '{RUN_TIMESTAMP}'
      AND member_segment = 'ALL'
    ORDER BY time_bucket, k
""").to_dataframe()
print(f"Loaded {len(plot_df)} metric rows")

if plot_df.empty:
    print("WARNING: No metrics found — check Section 6")
else:
    METRICS  = ["hit_at_k", "precision_at_k", "recall_at_k", "ndcg_at_k"]
    MLABELS  = {"hit_at_k": "Hit@K", "precision_at_k": "Precision@K",
                "recall_at_k": "Recall@K", "ndcg_at_k": "NDCG@K"}
    WCOLORS  = {"T0_30": "#5DBE7E", "T30_60": "#F7C948", "T60_180": "#F4845F"}
    WMARKERS = {"T0_30": "o", "T30_60": "s", "T60_180": "^"}

    # Chart 1 — all metrics by K and window
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()
    for i, metric in enumerate(METRICS):
        ax = axes[i]
        for window in WINDOWS:
            sub = plot_df[plot_df["time_bucket"] == window].sort_values("k")
            if sub.empty:
                continue
            ax.plot(sub["k"], sub[metric], color=WCOLORS[window],
                    marker=WMARKERS[window], linewidth=2, markersize=8, label=window)
            for _, row in sub.iterrows():
                ax.annotate(f"{row[metric]:.3f}", (row["k"], row[metric]),
                            textcoords="offset points", xytext=(5, 4),
                            fontsize=8, color=WCOLORS[window])
        ax.set_title(MLABELS[metric], fontsize=11, fontweight="bold")
        ax.set_xlabel("K"); ax.set_ylabel(MLABELS[metric])
        ax.set_xticks(K_VALUES); ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4); ax.set_ylim(0, 1.05)
    fig.suptitle(f"BERT4Rec — {SAMPLE} | Run: {RUN_TIMESTAMP}",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"bert4rec_metrics_{SAMPLE}.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Chart 2 — NDCG heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = plot_df.pivot_table(
        index="time_bucket", columns="k", values="ndcg_at_k"
    ).reindex(WINDOWS)
    sns.heatmap(pivot, ax=ax, cmap="YlGn", annot=True, fmt=".3f",
                annot_kws={"size": 12}, linewidths=0.5,
                cbar_kws={"label": "NDCG@K"})
    ax.set_title(f"BERT4Rec NDCG — {SAMPLE} | Run: {RUN_TIMESTAMP}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("K"); ax.set_ylabel("Time Window")
    plt.tight_layout()
    plt.savefig(f"bert4rec_ndcg_{SAMPLE}.png", dpi=150, bbox_inches="tight")
    plt.show()

    # K=3 summary table
    display(Markdown("### K=3 Summary"))
    k3 = plot_df[plot_df["k"] == 3][[
        "time_bucket", "hit_at_k", "precision_at_k", "recall_at_k", "ndcg_at_k"
    ]].rename(columns={
        "time_bucket": "Window", "hit_at_k": "Hit@3",
        "precision_at_k": "Prec@3", "recall_at_k": "Recall@3", "ndcg_at_k": "NDCG@3"
    }).reset_index(drop=True)
    display(k3)
    print(k3.to_string(index=False))

print(f"Section 7 done — time={time.time()-t0:.1f}s")
display(Markdown(f"**Section 7:** {time.time()-t0:.1f}s"))
print("Model_05_bert4rec_score complete")
