# ============================================================
# NB_08 — SASRec Training and Evaluation
# Purpose : Train SASRec on stratified 1% member sample
#           Evaluate on same 1% test set as Markov baseline
# Sources : A870800_gen_rec_sample_members_{SAMPLE} (pre-built)
#           A870800_gen_rec_model_train_{SAMPLE}     (pre-built)
#           A870800_gen_rec_model_test_{SAMPLE}      (pre-built)
#           A870800_gen_rec_visits                   (sequences)
# Metrics : Hit@K, Precision@K, Recall@K, NDCG@K
#           K = 1, 3, 5 per T0_30, T30_60, T60_180
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
from google.cloud import bigquery
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import seaborn as sns

print("Imports done")

# ══════════════════════════════════════════════════════════════════════════════
# QA HELPER — call after every DataFrame creation or modification
# ══════════════════════════════════════════════════════════════════════════════
def qa_df(df, label, sample_n=3, check_cols=None):
    """Print shape, nulls, dtypes, value counts and sample rows for QA."""
    print(f"\n{'='*60}")
    print(f"QA: {label}")
    print(f"  Shape       : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  Columns     : {list(df.columns)}")

    # Null check
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    if len(null_cols) > 0:
        print(f"  NULLS FOUND :")
        for col, n in null_cols.items():
            print(f"    {col}: {n:,} nulls ({n/len(df)*100:.1f}%)")
    else:
        print(f"  Nulls       : none")

    # Key column distributions if specified
    if check_cols:
        for col in check_cols:
            if col in df.columns:
                vc = df[col].value_counts(dropna=False).head(5)
                print(f"  {col} top values:")
                for val, cnt in vc.items():
                    print(f"    {val}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    # Date range for trigger_date if present
    if "trigger_date" in df.columns:
        print(f"  trigger_date: {df['trigger_date'].min()} → {df['trigger_date'].max()}")

    # Unique counts for key identifier columns
    for col in ["member_id", "trigger_dx", "specialty", "specialty_id",
                "label_specialty", "time_bucket"]:
        if col in df.columns:
            print(f"  unique {col}: {df[col].nunique():,}")

    # Sample rows
    print(f"  Sample rows (first {sample_n}):")
    print(df.head(sample_n).to_string(index=False))
    print(f"{'='*60}\n")


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
SAMPLE        = "1pct"     # change to "5pct" or "10pct" to switch sample
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
NUM_WORKERS   = min(4 * max(NUM_GPUS, 1), 16)

DS            = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
CACHE_DIR     = f"./cache_sasrec_{SAMPLE}"
CHECKPOINT    = f"./checkpoints/sasrec_{SAMPLE}_best.pt"
LOAD_CACHE    = False

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("./checkpoints", exist_ok=True)

# DataLoader kwargs — prefetch_factor only valid when num_workers > 0
_loader_kwargs = dict(pin_memory=(DEVICE == "cuda"), num_workers=NUM_WORKERS)
if NUM_WORKERS > 0:
    _loader_kwargs.update(prefetch_factor=2, persistent_workers=True)

client = bigquery.Client(project="anbc-hcb-dev")
print(f"Config done — sample={SAMPLE}, device={DEVICE}, num_workers={NUM_WORKERS}")

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
# SECTION 1 — PULL TRAIN SEQUENCES
# Source: triggers_qualified + visits — flat rows per visit
# Uses pre-built sample_members — no on-the-fly sampling
# QUALIFY uses recency_rank alias — no duplicate window function
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 1 — Pull Train Sequences"))
print("Section 1 starting — pulling train sequences from BQ...")

SEQ_CACHE   = f"{CACHE_DIR}/train_seq.parquet"
VOCAB_CACHE = f"{CACHE_DIR}/vocab.pkl"

if LOAD_CACHE and os.path.exists(SEQ_CACHE) and os.path.exists(VOCAB_CACHE):
    print("Loading sequences from cache...")
    seq_df = pd.read_parquet(SEQ_CACHE)
    with open(VOCAB_CACHE, "rb") as f:
        specialty_vocab = pickle.load(f)
    print(f"Cache loaded — {len(seq_df):,} rows")
else:
    print("Querying BigQuery for train sequences...")
    seq_df = client.query(f"""
        WITH triggers AS (
            SELECT DISTINCT
                t.member_id
                ,t.trigger_date
                ,t.trigger_dx
                ,t.member_segment
                ,t.is_t30_qualified
                ,t.is_t60_qualified
                ,t.is_t180_qualified
            FROM `{DS}.A870800_gen_rec_triggers_qualified` t
            INNER JOIN `{DS}.A870800_gen_rec_sample_members_{SAMPLE}` s
                ON t.member_id = s.member_id
            WHERE t.is_left_qualified = TRUE
              AND t.trigger_date < DATE '2024-01-01'
              AND t.has_claims_12m_before = TRUE
        ),
        ranked AS (
            SELECT
                t.member_id
                ,CAST(t.trigger_date AS STRING)              AS trigger_date
                ,t.trigger_dx
                ,t.member_segment
                ,t.is_t30_qualified
                ,t.is_t60_qualified
                ,t.is_t180_qualified
                ,v.specialty_ctg_cd                          AS specialty
                ,ROW_NUMBER() OVER (
                    PARTITION BY t.member_id, t.trigger_date, t.trigger_dx
                    ORDER BY v.visit_date DESC
                )                                            AS recency_rank
            FROM triggers t
            JOIN `{DS}.A870800_gen_rec_visits` v
                ON t.member_id = v.member_id
                AND v.visit_date < t.trigger_date
                AND v.visit_date >= DATE_SUB(t.trigger_date, INTERVAL 365 DAY)
            WHERE v.specialty_ctg_cd IS NOT NULL
              AND v.specialty_ctg_cd != ''
        )
        SELECT *
        FROM ranked
        WHERE recency_rank <= {MAX_SEQ_LEN}
        ORDER BY member_id, trigger_date, trigger_dx, recency_rank DESC
    """).to_dataframe()
    print(f"BQ returned {len(seq_df):,} sequence rows")
    qa_df(seq_df, "seq_df — raw from BQ",
          check_cols=["member_segment", "is_t30_qualified", "is_t60_qualified", "is_t180_qualified"])

    # Build vocabulary from training data only
    all_specs = sorted(seq_df["specialty"].dropna().unique().tolist())
    specialty_vocab = {s: i + 1 for i, s in enumerate(all_specs)}
    specialty_vocab["PAD"] = PAD_IDX
    seq_df["specialty_id"] = seq_df["specialty"].map(specialty_vocab).fillna(PAD_IDX).astype(int)
    print(f"Vocabulary built — {len(specialty_vocab):,} specialties")
    qa_df(seq_df, "seq_df — after specialty_id mapped",
          check_cols=["specialty_id", "recency_rank"])

    seq_df.to_parquet(SEQ_CACHE, index=False)
    with open(VOCAB_CACHE, "wb") as f:
        pickle.dump(specialty_vocab, f)
    print("Sequences cached to disk")

# Ensure trigger_date is string in consistent format YYYY-MM-DD
# BQ CAST gives "2022-01-15", pandas Timestamp.astype(str) may give "2022-01-15 00:00:00"
# Fix: always take first 10 chars
seq_df["trigger_date"] = seq_df["trigger_date"].astype(str).str[:10]
qa_df(seq_df, "seq_df — after trigger_date normalized to YYYY-MM-DD",
      check_cols=["recency_rank"])

NUM_SPECIALTIES = len(specialty_vocab)
print(f"Section 1 done — vocab={NUM_SPECIALTIES}, rows={len(seq_df):,}, time={time.time()-t0:.1f}s")
display(Markdown(f"**Vocab:** {NUM_SPECIALTIES:,} | **Rows:** {len(seq_df):,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PULL TRAIN LABELS
# Read directly from model_train_{SAMPLE} — already filtered
# trigger_date cast to STRING in BQ for consistent key matching
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 2 — Pull Train Labels"))
print("Section 2 starting — pulling train labels from BQ...")

LABEL_CACHE = f"{CACHE_DIR}/train_labels.parquet"

if LOAD_CACHE and os.path.exists(LABEL_CACHE):
    print("Loading labels from cache...")
    label_df = pd.read_parquet(LABEL_CACHE)
else:
    print("Querying BigQuery for train labels...")
    label_df = client.query(f"""
        SELECT
            member_id
            ,CAST(trigger_date AS STRING)                AS trigger_date
            ,trigger_dx
            ,time_bucket
            ,ARRAY_AGG(DISTINCT label_specialty
                ORDER BY label_specialty)                AS true_label_set
        FROM `{DS}.A870800_gen_rec_model_train_{SAMPLE}`
        WHERE label_specialty IS NOT NULL
        GROUP BY member_id, trigger_date, trigger_dx, time_bucket
    """).to_dataframe()
    label_df.to_parquet(LABEL_CACHE, index=False)
    print(f"Labels cached — {len(label_df):,} rows")
    qa_df(label_df, "label_df — raw from BQ",
          check_cols=["time_bucket"])

# Ensure consistent string format
label_df["trigger_date"] = label_df["trigger_date"].astype(str).str[:10]
qa_df(label_df, "label_df — after trigger_date normalized",
      check_cols=["time_bucket"])

# Critical check — do trigger_date formats match between seq_df and label_df?
seq_dates  = set(seq_df["trigger_date"].unique())
lbl_dates  = set(label_df["trigger_date"].unique())
overlap    = seq_dates & lbl_dates
print(f"QA — trigger_date overlap: seq={len(seq_dates):,} | label={len(lbl_dates):,} | overlap={len(overlap):,}")
if len(overlap) == 0:
    print("  CRITICAL: No trigger_date overlap — label lookup will fail entirely")
elif len(overlap) < len(seq_dates) * 0.5:
    print(f"  WARNING: Less than 50% overlap — many triggers will have empty labels")

print(f"Section 2 done — {len(label_df):,} label rows, time={time.time()-t0:.1f}s")
display(Markdown(f"**Label rows:** {len(label_df):,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BUILD DATASET
# FIXED: trigger_date string format consistent between seq and labels
# FIXED: zero-length sequences filtered out
# FIXED: multihot uses consistent key format
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 3 — Build Dataset"))
print("Section 3 starting — building records...")

def build_records(seq_df, label_df, specialty_vocab, max_seq_len, num_specialties):
    print(f"  Building label lookup from {len(label_df):,} rows...")
    # Both trigger_date are already normalized to YYYY-MM-DD strings
    label_wide = {}
    for row in label_df.itertuples(index=False):
        label_wide[(row.member_id, row.trigger_date, row.trigger_dx, row.time_bucket)] = row.true_label_set

    print(f"  Label lookup built — {len(label_wide):,} keys")

    print(f"  Grouping {len(seq_df):,} sequence rows by trigger...")
    grouped = (
        seq_df
        .sort_values(["member_id", "trigger_date", "trigger_dx", "recency_rank"])
        .groupby(
            ["member_id", "trigger_date", "trigger_dx",
             "member_segment", "is_t30_qualified", "is_t60_qualified", "is_t180_qualified"],
            sort=False
        )["specialty_id"]
        .apply(list)
    )
    print(f"  Grouped into {len(grouped):,} triggers")

    # QA grouped sequences
    seq_lengths = grouped.apply(len)
    print(f"  Sequence length stats: min={seq_lengths.min()} | "
          f"median={seq_lengths.median():.0f} | "
          f"max={seq_lengths.max()} | "
          f"zero_len={( seq_lengths == 0).sum():,}")

    records = []
    skipped_empty = 0

    for key, ids in grouped.items():
        member_id, trigger_date, trigger_dx, seg, t30, t60, t180 = key

        # Filter zero-length sequences — no pre-trigger signal
        if len(ids) == 0:
            skipped_empty += 1
            continue

        ids     = ids[-max_seq_len:]
        padded  = [PAD_IDX] * (max_seq_len - len(ids)) + ids
        seq_len = len(ids)

        # Build multi-hot label vectors
        def multihot(bucket):
            vec = np.zeros(num_specialties, dtype=np.float32)
            for sp in label_wide.get((member_id, trigger_date, trigger_dx, bucket), []):
                idx = specialty_vocab.get(sp)
                if idx and idx > 0:
                    vec[idx - 1] = 1.0
            return vec

        records.append({
            "trigger_date": trigger_date,  # kept for temporal train/val sort
            "is_t30":    bool(t30),
            "is_t60":    bool(t60),
            "is_t180":   bool(t180),
            "sequence":  np.array(padded, dtype=np.int64),
            "seq_len":   seq_len,
            "label_t30": multihot("T0_30"),
            "label_t60": multihot("T30_60"),
            "label_t180":multihot("T60_180"),
        })

    # Sanity check — how many records have non-empty labels
    has_any_label = sum(1 for r in records if r["label_t30"].sum() + r["label_t60"].sum() + r["label_t180"].sum() > 0)
    has_t30  = sum(1 for r in records if r["label_t30"].sum()  > 0)
    has_t60  = sum(1 for r in records if r["label_t60"].sum()  > 0)
    has_t180 = sum(1 for r in records if r["label_t180"].sum() > 0)
    t30_qual  = sum(1 for r in records if r["is_t30"])
    t60_qual  = sum(1 for r in records if r["is_t60"])
    t180_qual = sum(1 for r in records if r["is_t180"])

    print(f"\n  {'='*50}")
    print(f"  Records QA Summary")
    print(f"  {'='*50}")
    print(f"  Total records built   : {len(records):,}")
    print(f"  Skipped (empty seq)   : {skipped_empty:,}")
    print(f"  Records with any label: {has_any_label:,} ({has_any_label/max(len(records),1)*100:.1f}%)")
    print(f"  T0_30  qualified      : {t30_qual:,}  | have labels: {has_t30:,}")
    print(f"  T30_60 qualified      : {t60_qual:,}  | have labels: {has_t60:,}")
    print(f"  T60_180 qualified     : {t180_qual:,} | have labels: {has_t180:,}")
    if has_any_label == 0:
        print("  CRITICAL: Zero records have any labels — model will train on noise")
        print("  Check trigger_date key format match between seq_df and label_df")
    elif has_any_label < len(records) * 0.3:
        print(f"  WARNING: Only {has_any_label/len(records)*100:.1f}% records have labels — check join keys")
    print(f"  {'='*50}\n")

    return records


class SpecialtyDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "sequence":  torch.from_numpy(r["sequence"]),
            "seq_len":   torch.tensor(r["seq_len"], dtype=torch.long),
            "label_t30": torch.from_numpy(r["label_t30"]),
            "label_t60": torch.from_numpy(r["label_t60"]),
            "label_t180":torch.from_numpy(r["label_t180"]),
            "is_t30":    torch.tensor(r["is_t30"], dtype=torch.bool),
            "is_t60":    torch.tensor(r["is_t60"], dtype=torch.bool),
            "is_t180":   torch.tensor(r["is_t180"], dtype=torch.bool),
        }


all_records = build_records(seq_df, label_df, specialty_vocab, MAX_SEQ_LEN, NUM_SPECIALTIES)

# Train/val split — last 10% by trigger date order
all_records.sort(key=lambda r: r.get("trigger_date", ""))
n_val         = max(1, int(len(all_records) * 0.1))
train_records = all_records[:-n_val]
val_records   = all_records[-n_val:]
print(f"  Train: {len(train_records):,} | Val: {len(val_records):,}")

# QA train/val split
train_dates = [r["trigger_date"] for r in train_records]
val_dates   = [r["trigger_date"] for r in val_records]
print(f"  Train date range: {min(train_dates)} → {max(train_dates)}")
print(f"  Val   date range: {min(val_dates)} → {max(val_dates)}")
# Check no val date is earlier than train max — confirms temporal split
if max(val_dates) < max(train_dates):
    print("  WARNING: Val set has dates earlier than train max — check sort")
else:
    print("  Train/val split looks correct — val dates are after train dates")

train_loader = DataLoader(SpecialtyDataset(train_records),
                          batch_size=BATCH_SIZE, shuffle=True, **_loader_kwargs)
val_loader   = DataLoader(SpecialtyDataset(val_records),
                          batch_size=BATCH_SIZE * 2, shuffle=False, **_loader_kwargs)

print(f"Section 3 done — time={time.time()-t0:.1f}s")
display(Markdown(f"**Train:** {len(train_records):,} | **Val:** {len(val_records):,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SASREC MODEL
# FIXED: causal mask precomputed as register_buffer
# FIXED: DataParallel before compile
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 4 — SASRec Model"))
print("Section 4 starting — building model...")


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

        # Precompute causal mask once — reused every forward pass
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

        # Gather last real (non-padding) position
        idx      = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, self.embedding_dim)
        seq_repr = x.gather(1, idx).squeeze(1)

        return (
            torch.sigmoid(self.head_t30(seq_repr)),
            torch.sigmoid(self.head_t60(seq_repr)),
            torch.sigmoid(self.head_t180(seq_repr))
        )


def get_raw(m):
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    if isinstance(m, nn.DataParallel):
        m = m.module
    return m


# Build model — compile inner module first, then wrap DataParallel
base_model = SASRec(
    num_specialties=NUM_SPECIALTIES,
    embedding_dim=EMBEDDING_DIM,
    max_seq_len=MAX_SEQ_LEN,
    num_heads=NUM_HEADS,
    num_blocks=NUM_BLOCKS,
    dropout=DROPOUT,
    num_classes=NUM_SPECIALTIES
).to(DEVICE)

if hasattr(torch, "compile"):
    print("Compiling inner model with torch.compile...")
    base_model = torch.compile(base_model)

if NUM_GPUS > 1:
    print(f"Wrapping in DataParallel — {NUM_GPUS} GPUs")
    model = nn.DataParallel(base_model)
else:
    model = base_model

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Section 4 done — {n_params:,} parameters, time={time.time()-t0:.1f}s")
display(Markdown(f"**Parameters:** {n_params:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATION FUNCTIONS
# FIXED: idcg vectorized — no CPU loop over batch
# FIXED: evaluate counts use same key format as sums — no fragile parsing
# ══════════════════════════════════════════════════════════════════════════════
print("Section 5 — defining evaluation functions...")


def metrics_at_k(pred, label, k):
    """
    pred:  [B, C] float — model scores per class
    label: [B, C] float — multi-hot ground truth
    """
    topk     = torch.topk(pred, k, dim=1).indices      # [B, K]
    hits     = label.gather(1, topk)                    # [B, K]
    hit      = (hits.sum(1) > 0).float()
    prec     = hits.sum(1) / k
    rec      = hits.sum(1) / label.sum(1).clamp(min=1)

    disc     = 1.0 / torch.log2(
        torch.arange(2, k + 2, dtype=torch.float32, device=pred.device)
    )
    dcg      = (hits.float() * disc).sum(1)

    # Vectorized IDCG — no Python loop over batch
    n_ideal  = label.sum(1).clamp(max=k).long()         # [B]
    ranks    = torch.arange(1, k + 1, device=pred.device)
    ideal_mask = ranks.unsqueeze(0) <= n_ideal.unsqueeze(1)  # [B, K]
    idcg     = (disc.unsqueeze(0) * ideal_mask.float()).sum(1)

    ndcg     = dcg / idcg.clamp(min=1e-8)
    return hit.mean(), prec.mean(), rec.mean(), ndcg.mean()


def evaluate(loader, mdl):
    raw = get_raw(mdl)
    raw.eval()
    # Use same key for sums and counts — no fragile string parsing
    sums   = defaultdict(float)
    counts = defaultdict(int)

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

            for k in K_VALUES:
                for tag, pred, lbl, mask in [
                    ("T0_30",   p30,  l30,  m30),
                    ("T30_60",  p60,  l60,  m60),
                    ("T60_180", p180, l180, m180),
                ]:
                    n = mask.sum().item()
                    if n == 0:
                        continue
                    hit, prec, rec, ndcg = metrics_at_k(pred[mask].float(), lbl[mask].float(), k)
                    sums[f"{tag}_hit@{k}"]  += hit.item()  * n
                    sums[f"{tag}_prec@{k}"] += prec.item() * n
                    sums[f"{tag}_rec@{k}"]  += rec.item()  * n
                    sums[f"{tag}_ndcg@{k}"] += ndcg.item() * n
                    counts[f"{tag}@{k}"]    += n

    # Average — clean loop, no None tuple trick
    result = {}
    for tag in WINDOWS:
        for k in K_VALUES:
            n = counts[f"{tag}@{k}"]
            for metric in ["hit", "prec", "rec", "ndcg"]:
                key = f"{tag}_{metric}@{k}"
                result[key] = sums[key] / n if n > 0 else 0.0
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
# SECTION 6 — TRAINING
# FIXED: loss initialized as tensor — no crash when all masks empty
# Added batch progress print every 50 batches
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 6 — Training"))
print(f"Section 6 starting — training for {EPOCHS} epochs, {len(train_loader)} batches/epoch")

optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler     = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
bce        = nn.BCELoss()
best_ndcg  = 0.0
no_improve = 0

for epoch in range(EPOCHS):
    t_ep = time.time()
    model.train()
    total_loss = 0.0
    n_batches  = 0

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

            # FIXED: loss starts as a tensor — safe when all window masks empty
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
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {total_loss/n_batches:.4f}")

    scheduler.step()
    avg_loss  = total_loss / max(n_batches, 1)
    print(f"  Epoch {epoch+1} training done — avg loss: {avg_loss:.4f} | Running eval...")

    val_m     = evaluate(val_loader, model)
    val_ndcg  = np.mean([val_m.get(f"T0_30_ndcg@{k}", 0) for k in K_VALUES])
    ep_time   = time.time() - t_ep

    display(Markdown(
        f"**Epoch {epoch+1}/{EPOCHS}** — Loss: {avg_loss:.4f} | "
        f"Val NDCG@T0_30 avg: {val_ndcg:.4f} | "
        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
        f"Time: {ep_time:.1f}s"
    ))
    print_metrics(val_m, f"Val Epoch {epoch+1}")

    if val_ndcg > best_ndcg:
        best_ndcg, no_improve = val_ndcg, 0
        torch.save({
            "epoch":           epoch,
            "model_state":     get_raw(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "specialty_vocab": specialty_vocab,
            "best_val_ndcg":   best_ndcg,
            "config": {
                "num_specialties": NUM_SPECIALTIES,
                "embedding_dim":   EMBEDDING_DIM,
                "max_seq_len":     MAX_SEQ_LEN,
                "num_heads":       NUM_HEADS,
                "num_blocks":      NUM_BLOCKS,
                "dropout":         DROPOUT,
            }
        }, CHECKPOINT)
        print(f"  Checkpoint saved — best NDCG: {best_ndcg:.4f}")
        display(Markdown(f"Checkpoint saved — NDCG: {best_ndcg:.4f}"))
    else:
        no_improve += 1
        print(f"  No improvement ({no_improve}/{PATIENCE})")
        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            display(Markdown(f"Early stopping at epoch {epoch+1}"))
            break

print(f"Section 6 done — total time={time.time()-t0:.1f}s")
display(Markdown(f"**Section 6 total:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TEST EVALUATION
# Load best checkpoint — evaluate on model_test_{SAMPLE}
# Same vocab as train — unseen specialties mapped to PAD
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 7 — Test Evaluation"))
print("Section 7 starting — loading best checkpoint...")

ckpt    = torch.load(CHECKPOINT, weights_only=False)
cfg     = ckpt["config"]
t_model = SASRec(**cfg, num_classes=cfg["num_specialties"]).to(DEVICE)
t_model.load_state_dict(ckpt["model_state"])
print(f"Checkpoint loaded — epoch {ckpt['epoch']+1}, best NDCG: {ckpt['best_val_ndcg']:.4f}")

print("Pulling test sequences from BQ...")
test_seq_df = client.query(f"""
    WITH triggers AS (
        SELECT DISTINCT
            member_id
            ,trigger_date
            ,trigger_dx
            ,member_segment
            ,is_t30_qualified
            ,is_t60_qualified
            ,is_t180_qualified
        FROM `{DS}.A870800_gen_rec_model_test_{SAMPLE}`
        WHERE label_specialty IS NOT NULL
    ),
    ranked AS (
        SELECT
            t.member_id
            ,CAST(t.trigger_date AS STRING)              AS trigger_date
            ,t.trigger_dx
            ,t.member_segment
            ,t.is_t30_qualified
            ,t.is_t60_qualified
            ,t.is_t180_qualified
            ,v.specialty_ctg_cd                          AS specialty
            ,ROW_NUMBER() OVER (
                PARTITION BY t.member_id, t.trigger_date, t.trigger_dx
                ORDER BY v.visit_date DESC
            )                                            AS recency_rank
        FROM triggers t
        JOIN `{DS}.A870800_gen_rec_visits` v
            ON t.member_id = v.member_id
            AND v.visit_date < t.trigger_date
            AND v.visit_date >= DATE_SUB(t.trigger_date, INTERVAL 365 DAY)
        WHERE v.specialty_ctg_cd IS NOT NULL
          AND v.specialty_ctg_cd != ''
    )
    SELECT *
    FROM ranked
    WHERE recency_rank <= {MAX_SEQ_LEN}
    ORDER BY member_id, trigger_date, trigger_dx, recency_rank DESC
""").to_dataframe()
test_seq_df["trigger_date"]  = test_seq_df["trigger_date"].astype(str).str[:10]
test_seq_df["specialty_id"]  = test_seq_df["specialty"].map(specialty_vocab).fillna(PAD_IDX).astype(int)
print(f"Test sequences pulled — {len(test_seq_df):,} rows")
qa_df(test_seq_df, "test_seq_df — after mapping",
      check_cols=["specialty_id", "recency_rank"])

# QA — how many test specialties are OOV (not in train vocab)
oov = test_seq_df["specialty"].map(specialty_vocab).isna().sum()
print(f"  OOV specialties in test sequences: {oov:,} ({oov/len(test_seq_df)*100:.1f}%) — mapped to PAD")

print("Pulling test labels from BQ...")
test_label_df = client.query(f"""
    SELECT
        member_id
        ,CAST(trigger_date AS STRING)                    AS trigger_date
        ,trigger_dx
        ,time_bucket
        ,ARRAY_AGG(DISTINCT label_specialty
            ORDER BY label_specialty)                    AS true_label_set
    FROM `{DS}.A870800_gen_rec_model_test_{SAMPLE}`
    WHERE label_specialty IS NOT NULL
    GROUP BY member_id, trigger_date, trigger_dx, time_bucket
""").to_dataframe()
test_label_df["trigger_date"] = test_label_df["trigger_date"].astype(str).str[:10]
print(f"Test labels pulled — {len(test_label_df):,} rows")
qa_df(test_label_df, "test_label_df — after trigger_date normalized",
      check_cols=["time_bucket"])

# Critical check — overlap between test seq and test label trigger_dates
t_seq_dates = set(test_seq_df["trigger_date"].unique())
t_lbl_dates = set(test_label_df["trigger_date"].unique())
t_overlap   = t_seq_dates & t_lbl_dates
print(f"QA — test trigger_date overlap: seq={len(t_seq_dates):,} | label={len(t_lbl_dates):,} | overlap={len(t_overlap):,}")
if len(t_overlap) == 0:
    print("  CRITICAL: No test trigger_date overlap — test evaluation will return all zeros")

print("Building test records...")
test_records = build_records(test_seq_df, test_label_df, specialty_vocab, MAX_SEQ_LEN, NUM_SPECIALTIES)
test_loader  = DataLoader(
    SpecialtyDataset(test_records),
    batch_size=BATCH_SIZE * 2, shuffle=False, **_loader_kwargs
)
print(f"Running test evaluation on {len(test_records):,} records...")
test_metrics = evaluate(test_loader, t_model)
print_metrics(test_metrics, "TEST RESULTS")

print(f"Section 7 done — time={time.time()-t0:.1f}s")
display(Markdown(f"**Section 7:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — WRITE METRICS TO BIGQUERY
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 8 — Write Metrics to BigQuery"))
print("Section 8 — writing metrics to BQ...")

rows = [
    {
        "model":          "SASRec",
        "sample":         SAMPLE,
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
metrics_df = pd.DataFrame(rows)
print(metrics_df.to_string(index=False))

client.load_table_from_dataframe(
    metrics_df,
    f"{DS}.A870800_gen_rec_model_metrics",
    job_config=bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema=[
            bigquery.SchemaField("model",          "STRING"),
            bigquery.SchemaField("sample",         "STRING"),
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

print(f"Metrics written to A870800_gen_rec_model_metrics")
print(f"Section 8 done — time={time.time()-t0:.1f}s")
display(Markdown(f"Written to `A870800_gen_rec_model_metrics` | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — VISUALIZATION vs MARKOV BASELINE
# Markov comparison reads from markov_metrics_{SAMPLE} — apples to apples
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 9 — SASRec vs Markov Baseline"))
print("Section 9 — loading Markov metrics and plotting...")

markov_df = client.query(f"""
    SELECT time_bucket, k, hit_at_k, precision_at_k, recall_at_k, ndcg_at_k
    FROM `{DS}.A870800_gen_rec_markov_metrics_{SAMPLE}`
    WHERE member_segment = 'ALL'
""").to_dataframe()
print(f"Markov metrics loaded — {len(markov_df)} rows")

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
        m_sub = markov_df[markov_df["time_bucket"] == window].sort_values("k")
        s_sub = metrics_df[metrics_df["time_bucket"] == window].sort_values("k")
        if not m_sub.empty:
            ax.plot(m_sub["k"], m_sub[metric], color=WCOLORS[window],
                    marker=WMARKERS[window], linewidth=2, linestyle="--",
                    alpha=0.7, label=f"Markov {window}")
        if not s_sub.empty:
            ax.plot(s_sub["k"], s_sub[metric], color=WCOLORS[window],
                    marker=WMARKERS[window], linewidth=2.5, linestyle="-",
                    label=f"SASRec {window}")
    ax.set_title(MLABELS[metric], fontsize=11, fontweight="bold")
    ax.set_xlabel("K"); ax.set_ylabel(MLABELS[metric])
    ax.set_xticks(K_VALUES); ax.legend(fontsize=7, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4); ax.set_ylim(0, 1.05)

fig.suptitle(f"SASRec vs Markov — {SAMPLE} (dashed=Markov, solid=SASRec)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"sasrec_vs_markov_{SAMPLE}.png", dpi=150, bbox_inches="tight")
plt.show()

# K=3 comparison table
print("\nK=3 Summary:")
summary = []
for w in WINDOWS:
    for model_name, src, k_col in [("Markov", markov_df, "k"), ("SASRec", metrics_df, "k")]:
        sub = src[(src["time_bucket"] == w) & (src[k_col] == 3)]
        if sub.empty:
            continue
        row = {
            "Model": model_name, "Window": w,
            "Hit@3":    f"{sub['hit_at_k'].values[0]:.4f}",
            "Prec@3":   f"{sub['precision_at_k'].values[0]:.4f}",
            "Recall@3": f"{sub['recall_at_k'].values[0]:.4f}",
            "NDCG@3":   f"{sub['ndcg_at_k'].values[0]:.4f}",
        }
        summary.append(row)
        print(f"  {model_name} {w}: Hit={row['Hit@3']} Prec={row['Prec@3']} Rec={row['Recall@3']} NDCG={row['NDCG@3']}")

display(pd.DataFrame(summary).reset_index(drop=True))
print(f"Section 9 done — time={time.time()-t0:.1f}s")
display(Markdown(f"**Section 9:** {time.time()-t0:.1f}s"))
print("NB_08 complete")
