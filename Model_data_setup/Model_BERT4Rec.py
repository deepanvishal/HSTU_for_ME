# ============================================================
# NB_09 — BERT4Rec Training and Evaluation
# Sources : A870800_gen_rec_train_sequences_{SAMPLE} (pre-built)
#           A870800_gen_rec_test_sequences_{SAMPLE}  (pre-built)
#           A870800_gen_rec_model_train_{SAMPLE}     (pre-built)
#           A870800_gen_rec_model_test_{SAMPLE}      (pre-built)
# Metrics : Hit@K, Precision@K, Recall@K, NDCG@K
#           K = 1, 3, 5 per T0_30, T30_60, T60_180
# Key diff from SASRec:
#   - Bidirectional attention — no causal mask
#   - Random masked item prediction during training
#   - Inference: mask last real position only
# ============================================================
import gc
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


# ══════════════════════════════════════════════════════════════════════════════
# QA HELPER
# ══════════════════════════════════════════════════════════════════════════════
def qa_df(df, label, sample_n=3, check_cols=None):
    print(f"\n{'='*60}")
    print(f"QA: {label}")
    print(f"  Shape    : {df.shape[0]:,} rows x {df.shape[1]} cols")
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    if len(null_cols) > 0:
        for col, n in null_cols.items():
            print(f"  NULL {col}: {n:,} ({n/len(df)*100:.1f}%)")
    else:
        print(f"  Nulls    : none")
    if check_cols:
        for col in check_cols:
            if col in df.columns:
                vc = df[col].value_counts(dropna=False).head(5)
                print(f"  {col} top: {dict(vc)}")
    if "trigger_date" in df.columns:
        print(f"  trigger_date: {df['trigger_date'].min()} → {df['trigger_date'].max()}")
    for col in ["member_id", "trigger_dx", "specialty", "specialty_id", "time_bucket"]:
        if col in df.columns:
            print(f"  unique {col}: {df[col].nunique():,}")
    print(f"  Sample:\n{df.head(sample_n).to_string(index=False)}")
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
SAMPLE        = "1pct"
MAX_SEQ_LEN   = 20
MASK_PROB     = 0.2
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
# MASK_IDX set after vocab load: NUM_SPECIALTIES + 1
NUM_WORKERS   = 2

RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

DS            = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
CACHE_DIR     = f"./cache_bert4rec_{SAMPLE}"
MODEL_DIR     = "/home/jupyter/models"
CHECKPOINT    = f"{MODEL_DIR}/bert4rec_{SAMPLE}_{RUN_TIMESTAMP}.pt"
VOCAB_PATH    = f"{MODEL_DIR}/bert4rec_{SAMPLE}_{RUN_TIMESTAMP}_vocab.pkl"
LOAD_CACHE    = False

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

_loader_kwargs = dict(pin_memory=(DEVICE == "cuda"), num_workers=NUM_WORKERS)
if NUM_WORKERS > 0:
    _loader_kwargs.update(prefetch_factor=2, persistent_workers=True)

client = bigquery.Client(project="anbc-hcb-dev")

print(f"Config done — sample={SAMPLE}, device={DEVICE}, num_workers={NUM_WORKERS}")
print(f"Run timestamp : {RUN_TIMESTAMP}")
print(f"Checkpoint    : {CHECKPOINT}")
print(f"Vocab path    : {VOCAB_PATH}")

display(Markdown(f"""
## Config
| Parameter | Value |
|---|---|
| Sample | {SAMPLE} |
| Max sequence length | {MAX_SEQ_LEN} |
| Mask probability | {MASK_PROB} |
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
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 1 — Pull Train Sequences"))
print("Section 1 starting...")

SEQ_CACHE   = f"{CACHE_DIR}/train_seq.parquet"
VOCAB_CACHE = f"{CACHE_DIR}/vocab.pkl"

if LOAD_CACHE and os.path.exists(SEQ_CACHE) and os.path.exists(VOCAB_CACHE):
    print("Loading from cache...")
    seq_df = pd.read_parquet(SEQ_CACHE)
    with open(VOCAB_CACHE, "rb") as f:
        specialty_vocab = pickle.load(f)
    print(f"Loaded {len(seq_df):,} rows")
else:
    print("Reading from BQ...")
    seq_df = client.query(f"""
        SELECT
            member_id
            ,CAST(trigger_date AS STRING)                AS trigger_date
            ,trigger_dx
            ,member_segment
            ,is_t30_qualified
            ,is_t60_qualified
            ,is_t180_qualified
            ,specialty_ctg_cd                            AS specialty
            ,recency_rank
        FROM `{DS}.A870800_gen_rec_train_sequences_{SAMPLE}`
        ORDER BY member_id, trigger_date, trigger_dx, recency_rank
    """).to_dataframe()
    print(f"BQ returned {len(seq_df):,} rows")
    qa_df(seq_df, "seq_df raw from BQ",
          check_cols=["member_segment", "is_t30_qualified"])

    all_specs       = sorted(seq_df["specialty"].dropna().unique().tolist())
    specialty_vocab = {s: i + 1 for i, s in enumerate(all_specs)}
    specialty_vocab["PAD"] = PAD_IDX
    seq_df["specialty_id"] = (
        seq_df["specialty"].map(specialty_vocab).fillna(PAD_IDX).astype(int)
    )
    print(f"Vocab built — {len(specialty_vocab):,} specialties")

    seq_df.to_parquet(SEQ_CACHE, index=False)
    with open(VOCAB_CACHE, "wb") as f:
        pickle.dump(specialty_vocab, f)
    print("Cached to disk")

seq_df["trigger_date"] = seq_df["trigger_date"].astype(str).str[:10]
qa_df(seq_df, "seq_df after trigger_date normalized", check_cols=["recency_rank"])

NUM_SPECIALTIES = len(specialty_vocab)
MASK_IDX        = NUM_SPECIALTIES + 1   # must be set before Dataset is defined
print(f"Section 1 done — vocab={NUM_SPECIALTIES}, MASK_IDX={MASK_IDX}, time={time.time()-t0:.1f}s")
display(Markdown(f"**Vocab:** {NUM_SPECIALTIES:,} | **MASK_IDX:** {MASK_IDX} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PULL TRAIN LABELS
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 2 — Pull Train Labels"))
print("Section 2 starting...")

LABEL_CACHE = f"{CACHE_DIR}/train_labels.parquet"

if LOAD_CACHE and os.path.exists(LABEL_CACHE):
    print("Loading from cache...")
    label_df = pd.read_parquet(LABEL_CACHE)
else:
    print("Reading from BQ...")
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
    print(f"Cached — {len(label_df):,} rows")

label_df["trigger_date"] = label_df["trigger_date"].astype(str).str[:10]
qa_df(label_df, "label_df after trigger_date normalized", check_cols=["time_bucket"])

seq_dates = set(seq_df["trigger_date"].unique())
lbl_dates = set(label_df["trigger_date"].unique())
overlap   = seq_dates & lbl_dates
print(f"trigger_date overlap: seq={len(seq_dates):,} label={len(lbl_dates):,} overlap={len(overlap):,}")
if len(overlap) == 0:
    print("CRITICAL: No trigger_date overlap — all labels will be empty")

print(f"Section 2 done — {len(label_df):,} rows, time={time.time()-t0:.1f}s")
display(Markdown(f"**Label rows:** {len(label_df):,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BUILD DATASET
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 3 — Build Dataset"))
print("Section 3 starting...")


def build_records_fast(seq_df, label_df, specialty_vocab, max_seq_len, num_specialties):
    print(f"  Building label lookup from {len(label_df):,} rows...")
    label_wide = {
        (row.member_id, row.trigger_date, row.trigger_dx, row.time_bucket): row.true_label_set
        for row in label_df.itertuples(index=False)
    }
    print(f"  Label lookup: {len(label_wide):,} keys")

    print(f"  Grouping {len(seq_df):,} rows...")
    grouped = (
        seq_df
        .sort_values(["member_id", "trigger_date", "trigger_dx", "recency_rank"])
        .groupby(
            ["member_id", "trigger_date", "trigger_dx",
             "member_segment", "is_t30_qualified",
             "is_t60_qualified", "is_t180_qualified"],
            sort=False
        )["specialty_id"]
        .apply(list)
    )
    N = len(grouped)
    print(f"  Grouped into {N:,} triggers")

    seq_lens_all = grouped.apply(len)
    print(f"  Seq length — min={seq_lens_all.min()} "
          f"median={seq_lens_all.median():.0f} "
          f"max={seq_lens_all.max()} "
          f"zero={(seq_lens_all == 0).sum():,}")

    # Pre-allocate for all N entries
    seq_matrix  = np.zeros((N, max_seq_len), dtype=np.int64)
    seq_lengths = np.zeros(N, dtype=np.int64)
    lab_t30     = np.zeros((N, num_specialties), dtype=np.float32)
    lab_t60     = np.zeros((N, num_specialties), dtype=np.float32)
    lab_t180    = np.zeros((N, num_specialties), dtype=np.float32)
    is_t30_arr  = np.zeros(N, dtype=bool)
    is_t60_arr  = np.zeros(N, dtype=bool)
    is_t180_arr = np.zeros(N, dtype=bool)

    # Lists for ALL N entries so valid mask applies uniformly
    trigger_dates_all = []
    member_ids_all    = []
    trigger_dxs_all   = []
    segments_all      = []

    def fill_multihot(lab_arr, i, prefix, bucket):
        for sp in label_wide.get((*prefix, bucket), []):
            idx = specialty_vocab.get(sp)
            if idx and idx > 0:
                lab_arr[i, idx - 1] = 1.0

    skipped = 0
    for i, (key, ids) in enumerate(grouped.items()):
        member_id, trigger_date, trigger_dx, seg, t30, t60, t180 = key

        trigger_dates_all.append(trigger_date)
        member_ids_all.append(member_id)
        trigger_dxs_all.append(trigger_dx)
        segments_all.append(str(seg))

        is_t30_arr[i]  = bool(t30)
        is_t60_arr[i]  = bool(t60)
        is_t180_arr[i] = bool(t180)

        if len(ids) == 0:
            skipped += 1
            continue

        ids     = ids[-max_seq_len:]
        seq_len = len(ids)
        seq_lengths[i] = seq_len
        seq_matrix[i, max_seq_len - seq_len:] = ids

        prefix = (member_id, trigger_date, trigger_dx)
        fill_multihot(lab_t30,  i, prefix, "T0_30")
        fill_multihot(lab_t60,  i, prefix, "T30_60")
        fill_multihot(lab_t180, i, prefix, "T60_180")

    print(f"  Loop done — skipped: {skipped:,}")

    valid   = seq_lengths > 0
    n_valid = valid.sum()

    seq_matrix_f  = seq_matrix[valid]
    seq_lengths_f = seq_lengths[valid]
    lab_t30_f     = lab_t30[valid]
    lab_t60_f     = lab_t60[valid]
    lab_t180_f    = lab_t180[valid]
    is_t30_f      = is_t30_arr[valid]
    is_t60_f      = is_t60_arr[valid]
    is_t180_f     = is_t180_arr[valid]

    del seq_matrix, seq_lengths, lab_t30, lab_t60, lab_t180
    del is_t30_arr, is_t60_arr, is_t180_arr
    gc.collect()

    valid_list       = valid.tolist()
    trigger_dates_f  = np.array([trigger_dates_all[i] for i, v in enumerate(valid_list) if v])
    member_ids_f     = np.array([member_ids_all[i]    for i, v in enumerate(valid_list) if v])
    trigger_dxs_f    = np.array([trigger_dxs_all[i]   for i, v in enumerate(valid_list) if v])
    segments_f       = np.array([segments_all[i]       for i, v in enumerate(valid_list) if v])
    del trigger_dates_all, member_ids_all, trigger_dxs_all, segments_all
    gc.collect()

    any_label = (
        (lab_t30_f.sum(axis=1) + lab_t60_f.sum(axis=1) + lab_t180_f.sum(axis=1)) > 0
    )
    print(f"\n  {'='*50}")
    print(f"  Records: {n_valid:,} | Skipped: {skipped:,}")
    print(f"  Have labels    : {any_label.sum():,} ({any_label.mean()*100:.1f}%)")
    print(f"  T0_30  qual    : {is_t30_f.sum():,} | labeled: {(lab_t30_f.sum(axis=1) > 0).sum():,}")
    print(f"  T30_60 qual    : {is_t60_f.sum():,} | labeled: {(lab_t60_f.sum(axis=1) > 0).sum():,}")
    print(f"  T60_180 qual   : {is_t180_f.sum():,} | labeled: {(lab_t180_f.sum(axis=1) > 0).sum():,}")
    if any_label.sum() == 0:
        print("  CRITICAL: Zero records have labels")
    print(f"  {'='*50}\n")

    return {
        "seq_matrix":    seq_matrix_f,
        "seq_lengths":   seq_lengths_f,
        "lab_t30":       lab_t30_f,
        "lab_t60":       lab_t60_f,
        "lab_t180":      lab_t180_f,
        "is_t30":        is_t30_f,
        "is_t60":        is_t60_f,
        "is_t180":       is_t180_f,
        "trigger_dates": trigger_dates_f,
        "member_ids":    member_ids_f,
        "trigger_dxs":   trigger_dxs_f,
        "segments":      segments_f,
    }


class BERT4RecDataset(Dataset):
    """
    Training: randomly mask MASK_PROB of real items.
    Inference: mask last real position only.
    mask_for_inference=False → train | True → eval/test
    """
    def __init__(self, data, mask_for_inference=False):
        self.seq              = data["seq_matrix"]
        self.lengths          = data["seq_lengths"]
        self.lab_t30          = data["lab_t30"]
        self.lab_t60          = data["lab_t60"]
        self.lab_t180         = data["lab_t180"]
        self.is_t30           = data["is_t30"]
        self.is_t60           = data["is_t60"]
        self.is_t180          = data["is_t180"]
        self.mask_for_inference = mask_for_inference

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        seq     = self.seq[idx].copy()      # copy — modified in-place below
        seq_len = int(self.lengths[idx])
        L       = len(seq)
        pad_start = L - seq_len

        if self.mask_for_inference:
            # Mask last real position — deterministic
            mask_pos              = L - 1
            seq[mask_pos]         = MASK_IDX
            target_mask           = np.zeros(L, dtype=np.float32)
            target_mask[mask_pos] = 1.0
        else:
            # Vectorized random masking — no Python loop
            target_mask = np.zeros(L, dtype=np.float32)
            if seq_len > 0:
                rand_vals        = np.random.random(seq_len)
                mask_flags       = rand_vals < MASK_PROB
                # Guarantee at least one masked position
                if not mask_flags.any():
                    mask_flags[np.random.randint(seq_len)] = True
                real_positions   = np.arange(pad_start, L)
                masked_positions = real_positions[mask_flags]
                seq[masked_positions]         = MASK_IDX
                target_mask[masked_positions] = 1.0

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


# Build train data
data = build_records_fast(seq_df, label_df, specialty_vocab, MAX_SEQ_LEN, NUM_SPECIALTIES)

del seq_df, label_df
gc.collect()
print("seq_df and label_df freed")

# Temporal sort
print("Sorting by trigger_date...")
sort_idx = np.argsort(data["trigger_dates"])
for k in data:
    data[k] = data[k][sort_idx]

N     = data["seq_matrix"].shape[0]
n_val = max(1, int(N * 0.1))
split = N - n_val

train_data = {k: v[:split] for k, v in data.items()}
val_data   = {k: v[split:] for k, v in data.items()}
del data
gc.collect()
print("Unsplit data freed")

print(f"  Train: {split:,} | Val: {n_val:,}")
print(f"  Train date range: {train_data['trigger_dates'][0]} → {train_data['trigger_dates'][-1]}")
print(f"  Val   date range: {val_data['trigger_dates'][0]} → {val_data['trigger_dates'][-1]}")
if val_data["trigger_dates"][0] < train_data["trigger_dates"][-1]:
    print("  WARNING: Val dates overlap with train")
else:
    print("  Train/val split correct")

# Train: random masking | Val: mask last position
train_loader = DataLoader(
    BERT4RecDataset(train_data, mask_for_inference=False),
    batch_size=BATCH_SIZE, shuffle=True, **_loader_kwargs
)
val_loader = DataLoader(
    BERT4RecDataset(val_data, mask_for_inference=True),
    batch_size=BATCH_SIZE * 2, shuffle=False, **_loader_kwargs
)

print(f"Section 3 done — time={time.time()-t0:.1f}s")
display(Markdown(f"**Train:** {split:,} | **Val:** {n_val:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BERT4Rec MODEL
# Bidirectional attention — no causal mask
# Vocab: PAD=0, specialties=1..N, MASK=N+1
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 4 — BERT4Rec Model"))
print("Section 4 starting...")


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
        vocab_size         = mask_idx + 1   # 0..mask_idx inclusive
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

    def forward(self, seq, target_mask):
        B, L     = seq.shape
        pos      = torch.arange(L, device=seq.device).unsqueeze(0)
        x        = self.emb_drop(self.item_emb(seq) + self.pos_emb(pos))
        pad_mask = (seq == PAD_IDX)

        for block in self.blocks:
            x = block(x, pad_mask)
        x = self.norm(x)

        # Weighted average of masked positions → sequence representation
        tm_exp   = target_mask.unsqueeze(-1)                     # [B, L, 1]
        denom    = target_mask.sum(dim=1, keepdim=True)          # [B, 1]
        seq_repr = (x * tm_exp).sum(dim=1) / denom.clamp(min=1)  # [B, D]

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


model = BERT4Rec(
    num_specialties=NUM_SPECIALTIES,
    mask_idx=MASK_IDX,
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
print(f"Section 4 done — {n_params:,} parameters, time={time.time()-t0:.1f}s")
display(Markdown(f"**Parameters:** {n_params:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATION FUNCTIONS
# _disc_cache and _bce_eval at module level
# ══════════════════════════════════════════════════════════════════════════════
print("Section 5 — defining evaluation functions...")

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
    sums           = defaultdict(float)
    counts         = defaultdict(int)
    val_loss_sum   = 0.0
    val_loss_count = 0

    with torch.no_grad():
        for batch in loader:
            seq  = batch["sequence"].to(DEVICE, non_blocking=True)
            tm   = batch["target_mask"].to(DEVICE, non_blocking=True)
            l30  = batch["label_t30"].to(DEVICE, non_blocking=True)
            l60  = batch["label_t60"].to(DEVICE, non_blocking=True)
            l180 = batch["label_t180"].to(DEVICE, non_blocking=True)
            m30  = batch["is_t30"].to(DEVICE, non_blocking=True)
            m60  = batch["is_t60"].to(DEVICE, non_blocking=True)
            m180 = batch["is_t180"].to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                p30, p60, p180 = raw(seq, tm)

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
# SECTION 6 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 6 — Training"))
print(f"Section 6 — {EPOCHS} epochs, {len(train_loader)} batches/epoch")

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
        seq  = batch["sequence"].to(DEVICE, non_blocking=True)
        tm   = batch["target_mask"].to(DEVICE, non_blocking=True)
        l30  = batch["label_t30"].to(DEVICE, non_blocking=True)
        l60  = batch["label_t60"].to(DEVICE, non_blocking=True)
        l180 = batch["label_t180"].to(DEVICE, non_blocking=True)
        m30  = batch["is_t30"].to(DEVICE, non_blocking=True)
        m60  = batch["is_t60"].to(DEVICE, non_blocking=True)
        m180 = batch["is_t180"].to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            p30, p60, p180 = model(seq, tm)
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
                "model":           "BERT4Rec",
                "sample":          SAMPLE,
                "run_timestamp":   RUN_TIMESTAMP,
                "num_specialties": NUM_SPECIALTIES,
                "mask_idx":        MASK_IDX,
                "embedding_dim":   EMBEDDING_DIM,
                "max_seq_len":     MAX_SEQ_LEN,
                "num_heads":       NUM_HEADS,
                "num_blocks":      NUM_BLOCKS,
                "dropout":         DROPOUT,
                "pad_idx":         PAD_IDX,
                "mask_prob":       MASK_PROB,
            },
            "preprocessing": {
                "lookback_days": 365,
                "seq_ordering":  "visit_date DESC",
                "padding":       "left",
                "date_format":   "YYYY-MM-DD",
                "train_cutoff":  "2024-01-01",
                "inference_mask":"last real position",
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
                "mask_idx":        MASK_IDX,
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

print(f"Section 6 done — time={time.time()-t0:.1f}s")
display(Markdown(f"**Section 6 total:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 7 — Test Evaluation"))
print("Section 7 starting — loading checkpoint...")

ckpt    = torch.load(CHECKPOINT, weights_only=False)
cfg     = ckpt["config"]

model_cfg = {
    "num_specialties": cfg["num_specialties"],
    "mask_idx":        cfg["mask_idx"],
    "embedding_dim":   cfg["embedding_dim"],
    "max_seq_len":     cfg["max_seq_len"],
    "num_heads":       cfg["num_heads"],
    "num_blocks":      cfg["num_blocks"],
    "dropout":         cfg["dropout"],
    "num_classes":     cfg["num_specialties"],
}
t_model = BERT4Rec(**model_cfg).to(DEVICE)
t_model.load_state_dict(ckpt["model_state"])
idx_to_specialty = ckpt["idx_to_specialty"]
print(f"Checkpoint loaded — epoch {ckpt['epoch']+1}, NDCG: {ckpt['best_val_ndcg']:.4f}")
print(f"idx_to_specialty entries: {len(idx_to_specialty):,}")

print("Reading test sequences from BQ...")
test_seq_df = client.query(f"""
    SELECT
        member_id
        ,CAST(trigger_date AS STRING)                    AS trigger_date
        ,trigger_dx
        ,member_segment
        ,is_t30_qualified
        ,is_t60_qualified
        ,is_t180_qualified
        ,specialty_ctg_cd                                AS specialty
        ,recency_rank
    FROM `{DS}.A870800_gen_rec_test_sequences_{SAMPLE}`
    ORDER BY member_id, trigger_date, trigger_dx, recency_rank
""").to_dataframe()
test_seq_df["trigger_date"]  = test_seq_df["trigger_date"].astype(str).str[:10]
test_seq_df["specialty_id"]  = (
    test_seq_df["specialty"].map(specialty_vocab).fillna(PAD_IDX).astype(int)
)
print(f"Test sequences: {len(test_seq_df):,} rows")
qa_df(test_seq_df, "test_seq_df", check_cols=["recency_rank"])

oov = test_seq_df["specialty"].map(specialty_vocab).isna().sum()
print(f"OOV test specialties: {oov:,} ({oov/len(test_seq_df)*100:.1f}%) → mapped to PAD")

print("Reading test labels from BQ...")
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
print(f"Test labels: {len(test_label_df):,} rows")
qa_df(test_label_df, "test_label_df", check_cols=["time_bucket"])

t_seq_dates = set(test_seq_df["trigger_date"].unique())
t_lbl_dates = set(test_label_df["trigger_date"].unique())
t_overlap   = t_seq_dates & t_lbl_dates
print(f"Test trigger_date overlap: {len(t_overlap):,} / {len(t_seq_dates):,}")
if len(t_overlap) == 0:
    print("CRITICAL: No overlap — metrics will be zero")

print("Building test data...")
test_data = build_records_fast(
    test_seq_df, test_label_df, specialty_vocab, MAX_SEQ_LEN, NUM_SPECIALTIES
)
del test_seq_df, test_label_df
gc.collect()
print(f"Test data built — {test_data['seq_matrix'].shape[0]:,} records")

test_loader = DataLoader(
    BERT4RecDataset(test_data, mask_for_inference=True),
    batch_size=BATCH_SIZE * 2, shuffle=False, **_loader_kwargs
)

print(f"Running test evaluation...")
test_metrics = evaluate(test_loader, t_model)
print_metrics(test_metrics, "TEST RESULTS")

print(f"Section 7 done — time={time.time()-t0:.1f}s")
display(Markdown(f"**Section 7:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7B — SCORE TRIGGERS AND WRITE TO BIGQUERY
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 7B — Score Triggers"))
print("Section 7B — scoring test triggers...")

score_loader = DataLoader(
    BERT4RecDataset(test_data, mask_for_inference=True),
    batch_size=BATCH_SIZE * 2, shuffle=False, **_loader_kwargs
)

t_model.eval()
all_rows   = []
record_idx = 0

with torch.no_grad():
    for batch in score_loader:
        seq  = batch["sequence"].to(DEVICE, non_blocking=True)
        tm   = batch["target_mask"].to(DEVICE, non_blocking=True)
        m30  = batch["is_t30"].to(DEVICE, non_blocking=True)
        m60  = batch["is_t60"].to(DEVICE, non_blocking=True)
        m180 = batch["is_t180"].to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            p30, p60, p180 = t_model(seq, tm)

        p30  = torch.sigmoid(p30.float())
        p60  = torch.sigmoid(p60.float())
        p180 = torch.sigmoid(p180.float())

        bs = seq.size(0)
        for i in range(bs):
            ri = record_idx + i
            for window, pred, mask in [
                ("T0_30",   p30[i],  m30[i]),
                ("T30_60",  p60[i],  m60[i]),
                ("T60_180", p180[i], m180[i]),
            ]:
                if not mask.item():
                    continue

                top5_vals, top5_idx = torch.topk(pred, 5)
                top5_specs  = [idx_to_specialty.get(int(idx.item()), "UNK")
                               for idx in top5_idx]
                top5_scores = [round(float(v.item()), 4) for v in top5_vals]

                bucket_key  = {"T0_30": "lab_t30", "T30_60": "lab_t60", "T60_180": "lab_t180"}
                true_vec    = test_data[bucket_key[window]][ri]
                true_specs  = [idx_to_specialty[j]
                               for j in range(len(true_vec)) if true_vec[j] > 0
                               and j in idx_to_specialty]
                true_set    = set(true_specs)

                def hit_k(k):
                    return 1.0 if set(top5_specs[:k]) & true_set else 0.0

                def ndcg_k(k):
                    disc = [1.0 / np.log2(r + 2) for r in range(k)]
                    dcg  = sum(disc[r] for r, sp in enumerate(top5_specs[:k]) if sp in true_set)
                    n    = min(len(true_set), k)
                    idcg = sum(1.0 / np.log2(r + 2) for r in range(n))
                    return round(dcg / idcg, 4) if idcg > 0 else 0.0

                all_rows.append({
                    "member_id":        str(test_data["member_ids"][ri]),
                    "trigger_date":     str(test_data["trigger_dates"][ri]),
                    "trigger_dx":       str(test_data["trigger_dxs"][ri]),
                    "member_segment":   str(test_data["segments"][ri]),
                    "time_bucket":      window,
                    "true_labels":      "|".join(sorted(true_specs)),
                    "top5_predictions": "|".join(top5_specs),
                    "top5_scores":      "|".join(str(s) for s in top5_scores),
                    "hit_at_1":         hit_k(1),
                    "hit_at_3":         hit_k(3),
                    "hit_at_5":         hit_k(5),
                    "ndcg_at_1":        ndcg_k(1),
                    "ndcg_at_3":        ndcg_k(3),
                    "ndcg_at_5":        ndcg_k(5),
                    "model":            "BERT4Rec",
                    "sample":           SAMPLE,
                    "run_timestamp":    RUN_TIMESTAMP,
                })
        record_idx += bs

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

print(f"Section 7B done — {len(all_rows):,} rows, time={time.time()-t0:.1f}s")
display(Markdown(f"**7B:** {len(all_rows):,} trigger scores written | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — WRITE METRICS TO BIGQUERY
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 8 — Write Metrics"))
print("Section 8 — writing metrics...")

rows = [
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
print(f"Section 8 done — time={time.time()-t0:.1f}s")
display(Markdown(f"Written to `A870800_gen_rec_model_metrics` | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 9 — BERT4Rec Results"))
print("Section 9 — loading metrics from BQ...")

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
    print("WARNING: No metrics found — check Section 8 wrote correctly")
else:
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

print(f"Section 9 done — time={time.time()-t0:.1f}s")
display(Markdown(f"**Section 9:** {time.time()-t0:.1f}s"))
print("NB_09 BERT4Rec complete")
