# ============================================================
# NB_03 — build_provider_test_dataset.py  (vectorized)
# Purpose : Build provider test numpy arrays from BQ data
#           Run once per sample — shared by SASRec, BERT4Rec, HSTU scoring
# Sources : A870800_gen_rec_provider_test_sequences_{SAMPLE}
#           A870800_gen_rec_provider_model_test_{SAMPLE}
# Output  : ./cache_provider_{SAMPLE}/
#               test_seq_matrix.npy         (M, 20, 2) int32
#               test_delta_t_matrix.npy     (M, 20) int32
#               test_trigger_token.npy      (M, 1) int32
#               test_seq_lengths.npy        (M,) int32
#               test_lab_t30/t60/t180.npy   (M,) object — sparse, ALL providers
#               test_is_t30/t60/t180.npy    (M,) bool
#               test_member_ids/trigger_dates/trigger_dxs/segments.npy
# Key differences vs NB_02:
#   NO hard neg candidates — test only
#   NO val split — test only
#   ALL label providers kept including tail (test = real world)
#   tail providers stored as UNK int_id in labels — counted in metric denominator
#   Vocabs loaded from NB_01 cache — NO refitting
# ============================================================

import gc
import os
import pickle
import time

import numpy as np
import pandas as pd
from google.cloud import bigquery
from IPython.display import display, Markdown

print("Imports done")


def qa_df(df, label, sample_n=3, check_cols=None):
    print(f"\n{'='*60}")
    print(f"QA: {label}")
    print(f"  Shape    : {df.shape[0]:,} rows x {df.shape[1]} cols")
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    if len(null_cols):
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
    print(f"  Sample:\n{df.head(sample_n).to_string(index=False)}")
    print(f"{'='*60}\n")


# ── CONFIG ────────────────────────────────────────────────────────────────────
SAMPLE      = "5pct"
MAX_SEQ_LEN = 20
PAD_IDX     = 0
UNK_IDX     = 1

DS               = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
CACHE_DIR        = f"./cache_provider_{SAMPLE}"
os.makedirs(CACHE_DIR, exist_ok=True)
client           = bigquery.Client(project="anbc-hcb-dev")

print(f"Config — sample={SAMPLE}, max_seq_len={MAX_SEQ_LEN}")
display(Markdown(f"""
## Config
| Parameter | Value |
|---|---|
| Sample | {SAMPLE} |
| Max sequence length | {MAX_SEQ_LEN} |
| Cache dir | {CACHE_DIR} |
| Vocabs | loaded from NB_01 — no refitting |
"""))


# ── LOAD VOCABS (from NB_01 — no refitting) ───────────────────────────────────
print("Loading vocabs from NB_01...")
with open(f"{CACHE_DIR}/provider_vocab.pkl",  "rb") as f: provider_vocab  = pickle.load(f)
with open(f"{CACHE_DIR}/specialty_vocab.pkl", "rb") as f: specialty_vocab = pickle.load(f)
with open(f"{CACHE_DIR}/dx_vocab.pkl",        "rb") as f: dx_vocab        = pickle.load(f)

PROVIDER_VOCAB_SIZE = len(provider_vocab)
print(f"  provider={PROVIDER_VOCAB_SIZE:,}  specialty={len(specialty_vocab):,}  dx={len(dx_vocab):,}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PULL + ENCODE TEST SEQUENCES
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 1 — Pull Test Sequences"))

SEQ_CACHE = f"{CACHE_DIR}/raw_test_seq_df.parquet"

if os.path.exists(SEQ_CACHE):
    print("Loading from cache...")
    seq_df = pd.read_parquet(SEQ_CACHE)
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
            ,srv_prvdr_id
            ,specialty_ctg_cd
            ,delta_t_bucket
            ,recency_rank
        FROM `{DS}.A870800_gen_rec_provider_test_sequences_{SAMPLE}`
        ORDER BY member_id, trigger_date, trigger_dx, recency_rank
    """).to_dataframe()
    seq_df["trigger_date"] = seq_df["trigger_date"].astype(str).str[:10]
    qa_df(seq_df, "test seq_df raw", check_cols=["member_segment", "recency_rank"])
    seq_df.to_parquet(SEQ_CACHE, index=False)

seq_df["trigger_date"] = seq_df["trigger_date"].astype(str).str[:10]

# Vectorized encoding — no refitting, OOV → UNK
seq_df["provider_id"]  = seq_df["srv_prvdr_id"].map(provider_vocab).fillna(UNK_IDX).astype(np.int32)
seq_df["specialty_id"] = seq_df["specialty_ctg_cd"].map(specialty_vocab).fillna(UNK_IDX).astype(np.int32)
seq_df["delta_t_int"]  = seq_df["delta_t_bucket"].fillna(0).astype(np.int32)

unk_pct = (seq_df["provider_id"] == UNK_IDX).mean() * 100
oov_spec = (seq_df["specialty_id"] == UNK_IDX).mean() * 100
print(f"Provider UNK rate: {unk_pct:.1f}%  |  Specialty OOV rate: {oov_spec:.1f}%")

print(f"Section 1 done — {len(seq_df):,} rows | {time.time()-t0:.1f}s")
display(Markdown(f"**Test seq rows:** {len(seq_df):,} | **UNK:** {unk_pct:.1f}% | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PULL + ENCODE TEST LABELS
# ALL providers kept including tail — test = real world evaluation
# tail providers map to UNK — counted in metric denominator
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 2 — Pull Test Labels"))

LABEL_CACHE = f"{CACHE_DIR}/raw_test_label_df.parquet"

if os.path.exists(LABEL_CACHE):
    print("Loading from cache...")
    label_df = pd.read_parquet(LABEL_CACHE)
else:
    print("Reading from BQ...")
    label_df = client.query(f"""
        SELECT
            member_id
            ,CAST(trigger_date AS STRING)                AS trigger_date
            ,trigger_dx
            ,trigger_dx_clean
            ,member_segment
            ,is_t30_qualified
            ,is_t60_qualified
            ,is_t180_qualified
            ,label_provider
            ,time_bucket
        FROM `{DS}.A870800_gen_rec_provider_model_test_{SAMPLE}`
        WHERE label_provider IS NOT NULL
    """).to_dataframe()
    label_df["trigger_date"] = label_df["trigger_date"].astype(str).str[:10]
    qa_df(label_df, "test label_df raw", check_cols=["time_bucket"])
    label_df.to_parquet(LABEL_CACHE, index=False)

label_df["trigger_date"] = label_df["trigger_date"].astype(str).str[:10]

# Vectorized encoding
# ALL label providers encoded — tail → UNK_IDX
# Intentional: counted in metric denominator (model penalized for missing tail)
label_df["label_provider_id"] = label_df["label_provider"].map(provider_vocab).fillna(UNK_IDX).astype(np.int32)
label_df["trigger_dx_id"]     = label_df["trigger_dx_clean"].map(dx_vocab).fillna(UNK_IDX).astype(np.int32)

tail_pct = (label_df["label_provider_id"] == UNK_IDX).mean() * 100
print(f"Tail provider rate in labels: {tail_pct:.1f}% → stored as UNK, penalizes recall")

print(f"Section 2 done — {len(label_df):,} rows | {time.time()-t0:.1f}s")
display(Markdown(f"**Test label rows:** {len(label_df):,} | **Tail:** {tail_pct:.1f}% | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BUILD RECORDS (vectorized)
# Same approach as NB_02 — no hard negs, no val split
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 3 — Build Records"))

TEST_CACHE = f"{CACHE_DIR}/test_seq_matrix.npy"

if os.path.exists(TEST_CACHE):
    print("Test numpy cache exists — loading...")
    keys = ["seq_matrix", "delta_t_matrix", "trigger_token", "seq_lengths",
            "lab_t30", "lab_t60", "lab_t180",
            "is_t30", "is_t60", "is_t180",
            "member_ids", "trigger_dates", "trigger_dxs", "segments"]
    test_data = {k: np.load(f"{CACHE_DIR}/test_{k}.npy", allow_pickle=True) for k in keys}
    print(f"Loaded test={test_data['seq_matrix'].shape[0]:,}")
else:
    # ── 3a: Trigger metadata ───────────────────────────────────────────────────
    meta_cols = ["member_id", "trigger_date", "trigger_dx",
                 "trigger_dx_id", "member_segment",
                 "is_t30_qualified", "is_t60_qualified", "is_t180_qualified"]
    meta_df = (
        label_df[meta_cols]
        .drop_duplicates(["member_id", "trigger_date", "trigger_dx"])
        .reset_index(drop=True)
    )
    meta_df["trigger_idx"] = meta_df.index.astype(np.int32)
    M = len(meta_df)
    print(f"Unique test triggers: {M:,}")

    # ── 3b: Sparse labels via vectorized groupby ───────────────────────────────
    print("Building sparse labels...")
    label_grp = (
        label_df
        .groupby(["member_id", "trigger_date", "trigger_dx", "time_bucket"])["label_provider_id"]
        .apply(list)
    )
    lab_dict = {"T0_30": {}, "T30_60": {}, "T60_180": {}}
    for (member, tdate, tdx, bucket), ids in label_grp.items():
        lab_dict[bucket][(member, tdate, tdx)] = list(set(ids))

    # ── 3c: Vectorized seq_matrix build ───────────────────────────────────────
    print("Building seq_matrix (vectorized)...")
    t1 = time.time()

    seq_df2 = seq_df.merge(
        meta_df[["member_id", "trigger_date", "trigger_dx", "trigger_idx"]],
        on=["member_id", "trigger_date", "trigger_dx"],
        how="inner"
    )
    seq_df2 = seq_df2[seq_df2["recency_rank"] <= MAX_SEQ_LEN].copy()
    seq_df2["seq_pos"] = (MAX_SEQ_LEN - seq_df2["recency_rank"]).astype(np.int32)

    # Sequence lengths
    seq_lengths = np.zeros(M, dtype=np.int32)
    tmp = seq_df2.groupby("trigger_idx").size()
    seq_lengths[tmp.index.values] = np.minimum(tmp.values, MAX_SEQ_LEN).astype(np.int32)

    # Allocate + fill with numpy advanced indexing — no Python trigger loop
    seq_matrix  = np.zeros((M, MAX_SEQ_LEN, 2), dtype=np.int32)
    delta_t_mat = np.zeros((M, MAX_SEQ_LEN),    dtype=np.int32)

    trig_idx = seq_df2["trigger_idx"].values.astype(np.int64)
    seq_pos  = seq_df2["seq_pos"].values.astype(np.int64)
    prov_ids = seq_df2["provider_id"].values.astype(np.int32)
    spec_ids = seq_df2["specialty_id"].values.astype(np.int32)
    dt_vals  = seq_df2["delta_t_int"].values.astype(np.int32)

    seq_matrix[trig_idx, seq_pos, 0] = prov_ids
    seq_matrix[trig_idx, seq_pos, 1] = spec_ids
    delta_t_mat[trig_idx, seq_pos]   = dt_vals

    print(f"  seq_matrix: {seq_matrix.shape} | {time.time()-t1:.1f}s")

    del seq_df2, trig_idx, seq_pos, prov_ids, spec_ids, dt_vals
    gc.collect()

    # ── 3d: Trigger token + flags + metadata ──────────────────────────────────
    trigger_token     = meta_df["trigger_dx_id"].values.reshape(-1, 1).astype(np.int32)
    is_t30_arr        = meta_df["is_t30_qualified"].values.astype(bool)
    is_t60_arr        = meta_df["is_t60_qualified"].values.astype(bool)
    is_t180_arr       = meta_df["is_t180_qualified"].values.astype(bool)
    member_ids_arr    = meta_df["member_id"].values
    trigger_dates_arr = meta_df["trigger_date"].values
    trigger_dxs_arr   = meta_df["trigger_dx"].values
    segments_arr      = meta_df["member_segment"].fillna("Unknown").values

    # ── 3e: Sparse labels ─────────────────────────────────────────────────────
    lab_t30  = np.empty(M, dtype=object)
    lab_t60  = np.empty(M, dtype=object)
    lab_t180 = np.empty(M, dtype=object)

    for i in range(M):
        key = (member_ids_arr[i], trigger_dates_arr[i], trigger_dxs_arr[i])
        lab_t30[i]  = lab_dict["T0_30"].get(key, [])
        lab_t60[i]  = lab_dict["T30_60"].get(key, [])
        lab_t180[i] = lab_dict["T60_180"].get(key, [])

    del lab_dict, label_df, seq_df, meta_df
    gc.collect()

    # ── 3f: Filter (seq_len > 0) ──────────────────────────────────────────────
    valid     = seq_lengths > 0
    n_valid   = valid.sum()
    valid_idx = np.where(valid)[0]
    print(f"\nValid test triggers: {n_valid:,}/{M:,}")

    seq_matrix    = seq_matrix[valid_idx]
    delta_t_mat   = delta_t_mat[valid_idx]
    trigger_token = trigger_token[valid_idx]
    seq_lengths   = seq_lengths[valid_idx]
    lab_t30       = lab_t30[valid_idx]
    lab_t60       = lab_t60[valid_idx]
    lab_t180      = lab_t180[valid_idx]
    is_t30_arr    = is_t30_arr[valid_idx]
    is_t60_arr    = is_t60_arr[valid_idx]
    is_t180_arr   = is_t180_arr[valid_idx]
    member_ids_arr    = member_ids_arr[valid_idx]
    trigger_dates_arr = trigger_dates_arr[valid_idx]
    trigger_dxs_arr   = trigger_dxs_arr[valid_idx]
    segments_arr      = segments_arr[valid_idx]

    # ── 3g: Save ──────────────────────────────────────────────────────────────
    test_data = dict(
        seq_matrix=seq_matrix, delta_t_matrix=delta_t_mat,
        trigger_token=trigger_token, seq_lengths=seq_lengths,
        lab_t30=lab_t30, lab_t60=lab_t60, lab_t180=lab_t180,
        is_t30=is_t30_arr, is_t60=is_t60_arr, is_t180=is_t180_arr,
        member_ids=member_ids_arr, trigger_dates=trigger_dates_arr,
        trigger_dxs=trigger_dxs_arr, segments=segments_arr,
    )

    print("\nSaving test arrays...")
    for k, arr in test_data.items():
        np.save(f"{CACHE_DIR}/test_{k}.npy", arr)
        print(f"  test_{k}: {arr.shape}")

    gc.collect()

print(f"\nSection 3 done — {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — QA SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("---\n## Section 4 — QA Summary"))

M = test_data["seq_matrix"].shape[0]

# Overlap check — test dates must be >= 2024-01-01
min_date = test_data["trigger_dates"].min()
max_date = test_data["trigger_dates"].max()
print(f"Test trigger date range: {min_date} → {max_date}")
if min_date < "2024-01-01":
    print("  WARNING: test dates include pre-2024 triggers")

# Label density
has_label_t30 = sum(1 for i in range(M) if len(test_data["lab_t30"][i]) > 0)
print(f"Triggers with T30 labels: {has_label_t30:,}/{M:,} ({has_label_t30/M*100:.1f}%)")

# Sample
i = 0
print(f"\nSample test trigger[0]:")
print(f"  member_id      : {test_data['member_ids'][i]}")
print(f"  trigger_date   : {test_data['trigger_dates'][i]}")
print(f"  seq_length     : {test_data['seq_lengths'][i]}")
print(f"  trigger_token  : {test_data['trigger_token'][i]}")
print(f"  seq_matrix[-3:]: {test_data['seq_matrix'][i, -3:, :]}")
print(f"  lab_t30        : {test_data['lab_t30'][i][:10]}...")
print(f"  NOTE: lab_t30 includes tail providers as UNK={UNK_IDX}")

display(Markdown(f"""
## Test Dataset Summary
| Metric | Value |
|---|---|
| Test records | {M:,} |
| seq_matrix shape | {test_data['seq_matrix'].shape} |
| trigger_token shape | {test_data['trigger_token'].shape} |
| T30 qualified | {test_data['is_t30'].sum():,} |
| T60 qualified | {test_data['is_t60'].sum():,} |
| T180 qualified | {test_data['is_t180'].sum():,} |
| Date range | {min_date} → {max_date} |

Next: run NB_04 (SASRec model).
"""))
print("NB_03 complete")
