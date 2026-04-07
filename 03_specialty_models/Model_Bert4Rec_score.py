# ============================================================
# Model_05_bert4rec_score.py
# Purpose : Load test dataset + BERT4Rec checkpoint
#           Single forward pass — aggregate metrics + per-row scores
#           Write trigger scores + aggregate metrics to BQ
#           Generate visualizations
# Input   : ./cache_test_data_{SAMPLE}/test_*.npy
#           /home/jupyter/models/bert4rec_{SAMPLE}_{TS}.pt
# Output  : A870800_gen_rec_trigger_scores (APPEND)
#           A870800_gen_rec_model_metrics  (APPEND)
#           bert4rec_metrics_{SAMPLE}.png
#           bert4rec_ndcg_{SAMPLE}.png
# ============================================================
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
SAMPLE      = "1pct"
K_VALUES    = [1, 3, 5]
WINDOWS     = ["T0_30", "T30_60", "T60_180"]
PAD_IDX     = 0
BATCH_SIZE  = 1024
NUM_WORKERS = 2

# ── SET CHECKPOINT PATH ───────────────────────────────────────────────────────
# Update to the .pt file saved by Model_02_bert4rec_train
CHECKPOINT = "/home/jupyter/models/bert4rec_1pct_YYYY-MM-DD_HH-MM-SS.pt"
# To find latest: ls -lt /home/jupyter/models/bert4rec_*.pt | head -1

DS        = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
CACHE_DIR = f"./cache_test_data_{SAMPLE}"

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

ckpt             = torch.load(CHECKPOINT, weights_only=False)
cfg              = ckpt["config"]
specialty_vocab  = ckpt["specialty_vocab"]
idx_to_specialty = ckpt["idx_to_specialty"]
NUM_SPECIALTIES  = cfg["num_specialties"]

print(f"Checkpoint loaded — epoch {ckpt['epoch']+1}, best val NDCG: {ckpt['best_val_ndcg']:.4f}")
print(f"idx_to_specialty: {len(idx_to_specialty):,}")
display(Markdown(f"**Best Val NDCG:** {ckpt['best_val_ndcg']:.4f} | **Epoch:** {ckpt['epoch']+1} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BERT4REC MODEL
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 2 — BERT4Rec Model"))


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
        return self.head_t30(seq_repr), self.head_t60(seq_repr), self.head_t180(seq_repr)


MASK_IDX = cfg["mask_idx"]

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
print(f"Model loaded — {n_params:,} parameters")
display(Markdown(f"**Parameters:** {n_params:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOAD TEST DATA
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 3 — Load Test Data"))

KEYS = ["seq_matrix", "lab_t30", "lab_t60", "lab_t180",
        "seq_lengths", "is_t30", "is_t60", "is_t180",
        "trigger_dates", "member_ids", "trigger_dxs", "segments"]

test_data = {k: np.load(f"{CACHE_DIR}/test_{k}.npy", allow_pickle=True) for k in KEYS}
N_test    = test_data["seq_matrix"].shape[0]
print(f"Test data loaded — {N_test:,} records")
display(Markdown(f"**Test records:** {N_test:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SINGLE FORWARD PASS
# Per-row trigger scores + aggregate metrics in one pass.
# Labels fetched from numpy BUCKET_ARRAYS — not loaded via DataLoader.
# Aggregate accumulation reuses per-row GPU tensors — no redundant ops.
# Constant columns (model/sample/run_timestamp) assigned after DF construction.
# seq_len excluded from ScoreDataset — not used by BERT4Rec forward pass.
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 4 — Score + Evaluate (Single Pass)"))
print("Section 4 — single forward pass: per-row scores + aggregate metrics...")


class ScoreDataset(Dataset):
    def __init__(self, data):
        self.seq     = data["seq_matrix"]
        self.lengths = data["seq_lengths"]
        self.is_t30  = data["is_t30"]
        self.is_t60  = data["is_t60"]
        self.is_t180 = data["is_t180"]

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        seq     = self.seq[idx].copy()
        L       = len(seq)
        # Inference: mask last real position
        mask_pos              = L - 1
        seq[mask_pos]         = MASK_IDX
        target_mask           = np.zeros(L, dtype=np.float32)
        target_mask[mask_pos] = 1.0

        return {
            "sequence":    torch.from_numpy(seq),
            "target_mask": torch.from_numpy(target_mask),
            "is_t30":      torch.tensor(bool(self.is_t30[idx]),  dtype=torch.bool),
            "is_t60":      torch.tensor(bool(self.is_t60[idx]),  dtype=torch.bool),
            "is_t180":     torch.tensor(bool(self.is_t180[idx]), dtype=torch.bool),
        }


# ── Precompute once outside all loops ─────────────────────────────────────────
SCORE_K = 5

disc_score   = (1.0 / torch.log2(
    torch.arange(2, SCORE_K + 2, dtype=torch.float32)
)).to(DEVICE)
disc_cumsums = {k: disc_score[:k].cumsum(0) for k in [1, 3, 5]}

max_idx     = max(idx_to_specialty.keys()) + 1
spec_lookup = np.array(["UNK"] * max_idx, dtype=object)
for i, sp in idx_to_specialty.items():
    spec_lookup[i] = sp

member_ids_arr    = test_data["member_ids"]
trigger_dates_arr = test_data["trigger_dates"]
trigger_dxs_arr   = test_data["trigger_dxs"]
segments_arr      = test_data["segments"]
BUCKET_ARRAYS     = {
    "T0_30":   test_data["lab_t30"],
    "T30_60":  test_data["lab_t60"],
    "T60_180": test_data["lab_t180"],
}

cols = {
    "member_id": [], "trigger_date": [], "trigger_dx": [], "member_segment": [],
    "time_bucket": [], "true_labels": [], "top5_predictions": [], "top5_scores": [],
    "hit_at_1": [], "hit_at_3": [], "hit_at_5": [],
    "ndcg_at_1": [], "ndcg_at_3": [], "ndcg_at_5": [],
}

metric_sums   = defaultdict(float)
metric_counts = defaultdict(int)


def vec_ndcg(k, hits, n_true):
    d    = disc_score[:k]
    dcg  = (hits[:, :k].float() * d).sum(1)
    ni   = n_true.clamp(min=1, max=k).long() - 1
    idcg = disc_cumsums[k][ni]
    return dcg / idcg.clamp(min=1e-8)                   # [n] on GPU


score_loader = DataLoader(
    ScoreDataset(test_data),
    batch_size=BATCH_SIZE, shuffle=False, **_loader_kwargs
)

record_idx = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(score_loader):
        seq  = batch["sequence"].to(DEVICE, non_blocking=True)
        tm   = batch["target_mask"].to(DEVICE, non_blocking=True)
        m30  = batch["is_t30"].to(DEVICE, non_blocking=True)
        m60  = batch["is_t60"].to(DEVICE, non_blocking=True)
        m180 = batch["is_t180"].to(DEVICE, non_blocking=True)

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

            qual_pos = mask.nonzero(as_tuple=True)[0].cpu().numpy()
            qual_ri  = record_idx + qual_pos

            pred_m  = pred[mask].float()
            lbl_np  = BUCKET_ARRAYS[window][qual_ri]
            lbl_m   = torch.from_numpy(lbl_np).to(DEVICE)
            n_true  = lbl_m.sum(1)

            top5_vals, top5_idx = torch.topk(pred_m, SCORE_K, dim=1)
            hits = lbl_m.gather(1, top5_idx)

            # Per-row metrics — computed once, reused for BQ rows + aggregate
            hit1_t  = (hits[:, :1].sum(1) > 0).float()
            hit3_t  = (hits[:, :3].sum(1) > 0).float()
            hit5_t  = (hits[:, :5].sum(1) > 0).float()
            ndcg1_t = vec_ndcg(1, hits, n_true)
            ndcg3_t = vec_ndcg(3, hits, n_true)
            ndcg5_t = vec_ndcg(5, hits, n_true)

            # Aggregate accumulation — reuses above tensors, no redundant computation
            n1c = n_true.clamp(min=1)
            hs1 = hits[:, :1].float().sum(1)
            hs3 = hits[:, :3].float().sum(1)
            hs5 = hits.float().sum(1)

            metric_sums[f"{window}_hit@1"]  += hit1_t.sum().item()
            metric_sums[f"{window}_prec@1"] += hs1.sum().item()
            metric_sums[f"{window}_rec@1"]  += (hs1 / n1c).sum().item()
            metric_sums[f"{window}_ndcg@1"] += ndcg1_t.sum().item()
            metric_counts[f"{window}@1"]    += n_qual

            metric_sums[f"{window}_hit@3"]  += hit3_t.sum().item()
            metric_sums[f"{window}_prec@3"] += (hs3 / 3).sum().item()
            metric_sums[f"{window}_rec@3"]  += (hs3 / n1c).sum().item()
            metric_sums[f"{window}_ndcg@3"] += ndcg3_t.sum().item()
            metric_counts[f"{window}@3"]    += n_qual

            metric_sums[f"{window}_hit@5"]  += hit5_t.sum().item()
            metric_sums[f"{window}_prec@5"] += (hs5 / 5).sum().item()
            metric_sums[f"{window}_rec@5"]  += (hs5 / n1c).sum().item()
            metric_sums[f"{window}_ndcg@5"] += ndcg5_t.sum().item()
            metric_counts[f"{window}@5"]    += n_qual

            # Single GPU→CPU transfer per tensor group
            hits_cpu      = torch.stack([hit1_t, hit3_t, hit5_t], dim=1).cpu().numpy()
            ndcg_cpu      = torch.stack([ndcg1_t, ndcg3_t, ndcg5_t], dim=1).cpu().numpy()
            top5_idx_cpu  = top5_idx.cpu().numpy()
            top5_vals_cpu = top5_vals.cpu().numpy()

            # Python loop — string assembly only
            for j, ri in enumerate(qual_ri):
                top5_specs  = list(spec_lookup[top5_idx_cpu[j]])
                top5_scores = [round(float(v), 4) for v in top5_vals_cpu[j]]

                true_pos   = np.where(lbl_np[j] > 0)[0]
                true_specs = list(spec_lookup[true_pos[true_pos < max_idx]])

                cols["member_id"].append(str(member_ids_arr[ri]))
                cols["trigger_date"].append(str(trigger_dates_arr[ri]))
                cols["trigger_dx"].append(str(trigger_dxs_arr[ri]))
                cols["member_segment"].append(str(segments_arr[ri]))
                cols["time_bucket"].append(window)
                cols["true_labels"].append("|".join(sorted(true_specs)))
                cols["top5_predictions"].append("|".join(top5_specs))
                cols["top5_scores"].append("|".join(str(s) for s in top5_scores))
                cols["hit_at_1"].append(float(hits_cpu[j, 0]))
                cols["hit_at_3"].append(float(hits_cpu[j, 1]))
                cols["hit_at_5"].append(float(hits_cpu[j, 2]))
                cols["ndcg_at_1"].append(round(float(ndcg_cpu[j, 0]), 4))
                cols["ndcg_at_3"].append(round(float(ndcg_cpu[j, 1]), 4))
                cols["ndcg_at_5"].append(round(float(ndcg_cpu[j, 2]), 4))

        record_idx += bs

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(score_loader)} | "
                  f"Rows so far: {len(cols['member_id']):,}")

# Aggregate test_metrics
test_metrics = {}
for tag in WINDOWS:
    for k in K_VALUES:
        n = metric_counts[f"{tag}@{k}"]
        for metric in ["hit", "prec", "rec", "ndcg"]:
            key = f"{tag}_{metric}@{k}"
            test_metrics[key] = round(metric_sums[key] / n, 4) if n > 0 else 0.0

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

# Build DataFrame — constant columns assigned once outside hot loop
n_scored  = len(cols["member_id"])
scores_df = pd.DataFrame(cols)
scores_df["model"]         = "BERT4Rec"
scores_df["sample"]        = SAMPLE
scores_df["run_timestamp"] = RUN_TIMESTAMP
del cols
print(f"Single pass done — {n_scored:,} trigger-window pairs scored")
print(f"Section 4 done — time={time.time()-t0:.1f}s")
display(Markdown(f"**Scored:** {n_scored:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — WRITE TRIGGER SCORES TO BIGQUERY
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 5 — Write Trigger Scores"))
print("Section 5 — writing trigger scores to BQ...")

BATCH_BQ = 100_000
schema_scores = [
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
job_cfg = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND", schema=schema_scores)

for start in range(0, n_scored, BATCH_BQ):
    chunk = scores_df.iloc[start:start + BATCH_BQ]
    client.load_table_from_dataframe(
        chunk, f"{DS}.A870800_gen_rec_trigger_scores",
        job_config=job_cfg
    ).result()
    print(f"  Written {start:,} — {min(start+BATCH_BQ, n_scored):,}")

del scores_df
print(f"Section 5 done — {n_scored:,} rows, time={time.time()-t0:.1f}s")
display(Markdown(f"**5:** {n_scored:,} rows written | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — WRITE AGGREGATE METRICS TO BIGQUERY
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 6 — Write Aggregate Metrics"))
print("Section 6 — writing aggregate metrics to BQ...")

metric_rows = [
    {
        "model":          "BERT4Rec",
        "sample":         SAMPLE,
        "run_timestamp":  RUN_TIMESTAMP,
        "time_bucket":    w,
        "k":              k,
        "member_segment": "ALL",
        "hit_at_k":       test_metrics.get(f"{w}_hit@{k}",  0),
        "precision_at_k": test_metrics.get(f"{w}_prec@{k}", 0),
        "recall_at_k":    test_metrics.get(f"{w}_rec@{k}",  0),
        "ndcg_at_k":      test_metrics.get(f"{w}_ndcg@{k}", 0),
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
# Uses metrics_df already in memory — no BQ roundtrip
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 7 — Visualization"))
print("Section 7 — plotting...")

plot_df = metrics_df.copy()

METRICS  = ["hit_at_k", "precision_at_k", "recall_at_k", "ndcg_at_k"]
MLABELS  = {"hit_at_k": "Hit@K", "precision_at_k": "Precision@K",
            "recall_at_k": "Recall@K", "ndcg_at_k": "NDCG@K"}
WCOLORS  = {"T0_30": "#5DBE7E", "T30_60": "#F7C948", "T60_180": "#F4845F"}
WMARKERS = {"T0_30": "o", "T30_60": "s", "T60_180": "^"}

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
