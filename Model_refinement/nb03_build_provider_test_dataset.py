# ============================================================
# NB_03 — build_provider_test_dataset.py  (BQ-first, vectorized)
# Purpose : Build provider test numpy arrays from BQ data
#           BQ does ALL heavy lifting — Python is a thin consumer
# Sources : A870800_gen_rec_provider_test_sequences_{SAMPLE}
#           A870800_gen_rec_provider_model_test_agg_{SAMPLE}
#             ONE ROW PER TRIGGER — labels pre-aggregated by BQ ARRAY_AGG
#             ALL providers kept including tail (test = real world)
# Output  : ./cache_provider_{SAMPLE}/test_*.npy
# No hard neg candidates — test only
# No val split — test only
# Vocabs loaded from NB_01 — no refitting
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

DS          = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
CACHE_DIR   = f"./cache_provider_{SAMPLE}"
os.makedirs(CACHE_DIR, exist_ok=True)
client      = bigquery.Client(project="anbc-hcb-dev")

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


# ── LOAD VOCABS (no refitting) ────────────────────────────────────────────────
with open(f"{CACHE_DIR}/provider_vocab.pkl",  "rb") as f: provider_vocab  = pickle.load(f)
with open(f"{CACHE_DIR}/specialty_vocab.pkl", "rb") as f: specialty_vocab = pickle.load(f)
with open(f"{CACHE_DIR}/dx_vocab.pkl",        "rb") as f: dx_vocab        = pickle.load(f)
PROVIDER_VOCAB_SIZE = len(provider_vocab)
print(f"Vocabs: provider={PROVIDER_VOCAB_SIZE:,}  specialty={len(specialty_vocab):,}  dx={len(dx_vocab):,}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PULL TEST LABELS (pre-aggregated — ONE ROW PER TRIGGER)
# ALL providers kept including tail — test = real world
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 1 — Pull Test Labels (pre-aggregated)"))

LABEL_CACHE = f"{CACHE_DIR}/raw_test_label_agg_df.parquet"

if os.path.exists(LABEL_CACHE):
    print("Loading from cache...")
    label_df = pd.read_parquet(LABEL_CACHE)
else:
    print("Reading pre-aggregated test labels from BQ...")
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
            ,lab_t30
            ,lab_t60
            ,lab_t180
        FROM `{DS}.A870800_gen_rec_provider_model_test_agg_{SAMPLE}`
        ORDER BY trigger_date, member_id, trigger_dx
    """).to_dataframe()
    label_df["trigger_date"] = label_df["trigger_date"].astype(str).str[:10]
    qa_df(label_df, "test label_df (one row per trigger)", check_cols=["member_segment"])
    label_df.to_parquet(LABEL_CACHE, index=False)

label_df["trigger_date"] = label_df["trigger_date"].astype(str).str[:10]
M = len(label_df)
print(f"Test triggers: {M:,} — one row each ✓")

# Vectorized encoding
label_df["trigger_dx_id"] = label_df["trigger_dx_clean"].map(dx_vocab).fillna(UNK_IDX).astype(np.int32)

def encode_label_list(lst, vocab):
    if lst is None or (hasattr(lst, '__len__') and len(lst) == 0):
        return []
    return [vocab.get(p, UNK_IDX) for p in lst]   # tail → UNK, counted in denominator

label_df["lab_t30_ids"]  = label_df["lab_t30"].apply(lambda x: encode_label_list(x, provider_vocab))
label_df["lab_t60_ids"]  = label_df["lab_t60"].apply(lambda x: encode_label_list(x, provider_vocab))
label_df["lab_t180_ids"] = label_df["lab_t180"].apply(lambda x: encode_label_list(x, provider_vocab))

tail_rate = np.mean([UNK_IDX in ids for ids in label_df["lab_t30_ids"] if len(ids) > 0])
print(f"T30 triggers with tail label: {tail_rate*100:.1f}% — counted in recall denominator")
print(f"Section 1 done — {time.time()-t0:.1f}s")
display(Markdown(f"**Test triggers:** {M:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PULL TEST SEQUENCES
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 2 — Pull Test Sequences"))

SEQ_CACHE = f"{CACHE_DIR}/raw_test_seq_df.parquet"

if os.path.exists(SEQ_CACHE):
    print("Loading from cache...")
    seq_df = pd.read_parquet(SEQ_CACHE)
else:
    print("Reading test sequences from BQ...")
    seq_df = client.query(f"""
        SELECT
            member_id
            ,CAST(trigger_date AS STRING)                AS trigger_date
            ,trigger_dx
            ,srv_prvdr_id
            ,specialty_ctg_cd
            ,delta_t_bucket
            ,recency_rank
        FROM `{DS}.A870800_gen_rec_provider_test_sequences_{SAMPLE}`
        ORDER BY member_id, trigger_date, trigger_dx, recency_rank
    """).to_dataframe()
    seq_df["trigger_date"] = seq_df["trigger_date"].astype(str).str[:10]
    qa_df(seq_df, "test seq_df raw", check_cols=["recency_rank"])
    seq_df.to_parquet(SEQ_CACHE, index=False)

seq_df["trigger_date"] = seq_df["trigger_date"].astype(str).str[:10]

seq_df["provider_id"]  = seq_df["srv_prvdr_id"].map(provider_vocab).fillna(UNK_IDX).astype(np.int32)
seq_df["specialty_id"] = seq_df["specialty_ctg_cd"].map(specialty_vocab).fillna(UNK_IDX).astype(np.int32)
seq_df["delta_t_int"]  = seq_df["delta_t_bucket"].fillna(0).astype(np.int32)

unk_pct = (seq_df["provider_id"] == UNK_IDX).mean() * 100
print(f"Provider UNK rate: {unk_pct:.1f}%")
print(f"Section 2 done — {len(seq_df):,} rows | {time.time()-t0:.1f}s")
display(Markdown(f"**Seq rows:** {len(seq_df):,} | **UNK:** {unk_pct:.1f}% | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BUILD NUMPY ARRAYS
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 3 — Build Numpy Arrays"))

TEST_CACHE = f"{CACHE_DIR}/test_seq_matrix.npy"

if os.path.exists(TEST_CACHE):
    print("Cache exists — loading...")
    keys = ["seq_matrix", "delta_t_matrix", "trigger_token", "seq_lengths",
            "lab_t30", "lab_t60", "lab_t180",
            "is_t30", "is_t60", "is_t180",
            "member_ids", "trigger_dates", "trigger_dxs", "segments"]
    test_data = {k: np.load(f"{CACHE_DIR}/test_{k}.npy", allow_pickle=True) for k in keys}
    print(f"Loaded test={test_data['seq_matrix'].shape[0]:,}")
else:
    label_df = label_df.reset_index(drop=True)
    label_df["trigger_idx"] = label_df.index.astype(np.int32)

    # ── 3a: Metadata ──────────────────────────────────────────────────────────
    trigger_token     = label_df["trigger_dx_id"].values.reshape(-1, 1).astype(np.int32)
    is_t30_arr        = label_df["is_t30_qualified"].values.astype(bool)
    is_t60_arr        = label_df["is_t60_qualified"].values.astype(bool)
    is_t180_arr       = label_df["is_t180_qualified"].values.astype(bool)
    member_ids_arr    = label_df["member_id"].values
    trigger_dates_arr = label_df["trigger_date"].values
    trigger_dxs_arr   = label_df["trigger_dx"].values
    segments_arr      = label_df["member_segment"].fillna("Unknown").values

    # ── 3b: Labels — direct from pre-aggregated BQ arrays ─────────────────────
    lab_t30  = label_df["lab_t30_ids"].values
    lab_t60  = label_df["lab_t60_ids"].values
    lab_t180 = label_df["lab_t180_ids"].values
    print("Labels assigned directly from BQ arrays — no groupby ✓")

    # ── 3c: seq_matrix — vectorized numpy advanced indexing ───────────────────
    print("Building seq_matrix (vectorized)...")
    t1 = time.time()

    seq_df2 = seq_df.merge(
        label_df[["member_id","trigger_date","trigger_dx","trigger_idx"]],
        on=["member_id","trigger_date","trigger_dx"],
        how="inner"
    )
    seq_df2 = seq_df2[seq_df2["recency_rank"] <= MAX_SEQ_LEN].copy()
    seq_df2["seq_pos"] = (MAX_SEQ_LEN - seq_df2["recency_rank"]).astype(np.int32)

    seq_lengths = np.zeros(M, dtype=np.int32)
    tmp = seq_df2.groupby("trigger_idx").size()
    seq_lengths[tmp.index.values] = np.minimum(tmp.values, MAX_SEQ_LEN).astype(np.int32)

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

    print(f"  seq_matrix {seq_matrix.shape} built in {time.time()-t1:.1f}s ✓")
    del seq_df2, trig_idx, seq_pos, prov_ids, spec_ids, dt_vals, seq_df, label_df
    gc.collect()

    # ── 3d: Filter + save ─────────────────────────────────────────────────────
    valid     = seq_lengths > 0
    valid_idx = np.where(valid)[0]
    print(f"Valid test triggers: {len(valid_idx):,}/{M:,}")

    test_data = dict(
        seq_matrix    = seq_matrix[valid_idx],
        delta_t_matrix= delta_t_mat[valid_idx],
        trigger_token = trigger_token[valid_idx],
        seq_lengths   = seq_lengths[valid_idx],
        lab_t30       = lab_t30[valid_idx],
        lab_t60       = lab_t60[valid_idx],
        lab_t180      = lab_t180[valid_idx],
        is_t30        = is_t30_arr[valid_idx],
        is_t60        = is_t60_arr[valid_idx],
        is_t180       = is_t180_arr[valid_idx],
        member_ids    = member_ids_arr[valid_idx],
        trigger_dates = trigger_dates_arr[valid_idx],
        trigger_dxs   = trigger_dxs_arr[valid_idx],
        segments      = segments_arr[valid_idx],
    )

    print("Saving test arrays...")
    for k, arr in test_data.items():
        np.save(f"{CACHE_DIR}/test_{k}.npy", arr)
        print(f"  test_{k}: {arr.shape}")

    gc.collect()

print(f"\nSection 3 done — {time.time()-t0:.1f}s")


# ── SUMMARY ───────────────────────────────────────────────────────────────────
display(Markdown("---\n## Summary"))
M_final = test_data["seq_matrix"].shape[0]
min_date = test_data["trigger_dates"].min()
max_date = test_data["trigger_dates"].max()

display(Markdown(f"""
## Test Dataset Summary
| Metric | Value |
|---|---|
| Test records | {M_final:,} |
| seq_matrix shape | {test_data['seq_matrix'].shape} |
| trigger_token shape | {test_data['trigger_token'].shape} |
| T30 qualified | {test_data['is_t30'].sum():,} |
| Date range | {min_date} → {max_date} |

**BQ did:** label aggregation (ARRAY_AGG) with ALL providers incl. tail
**Python did:** vocab encoding, numpy indexing

Next: run NB_04 (SASRec).
"""))
print("NB_03 complete")
