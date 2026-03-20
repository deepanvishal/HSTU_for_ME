
# ============================================================
# Model_03_build_test_dataset.py
# Purpose : Pull test sequences + labels from BQ
#           Run build_records_fast
#           Save numpy arrays to shared cache
#           Run once — shared by SASRec and BERT4Rec scoring
# Output  : ./cache_model_data_{SAMPLE}/
#               test_seq_matrix.npy
#               test_lab_t30/t60/t180.npy
#               test_seq_lengths.npy
#               test_is_t30/t60/t180.npy
#               test_trigger_dates/member_ids/trigger_dxs/segments.npy
# NOTE    : vocab.pkl must already exist from Model_00
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


# ── CONFIG ────────────────────────────────────────────────────────────────────
SAMPLE      = "1pct"
MAX_SEQ_LEN = 20
PAD_IDX     = 0

DS        = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
CACHE_DIR = f"./cache_model_data_{SAMPLE}"
os.makedirs(CACHE_DIR, exist_ok=True)

client = bigquery.Client(project="anbc-hcb-dev")

print(f"Config — sample={SAMPLE}, cache={CACHE_DIR}")

# Load vocab built in Model_00 — must exist
with open(f"{CACHE_DIR}/vocab.pkl", "rb") as f:
    specialty_vocab = pickle.load(f)
NUM_SPECIALTIES = len(specialty_vocab)
print(f"Vocab loaded — {NUM_SPECIALTIES:,} specialties")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PULL TEST SEQUENCES FROM BQ
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 1 — Pull Test Sequences"))
print("Section 1 starting...")

TEST_SEQ_CACHE = f"{CACHE_DIR}/raw_test_seq_df.parquet"

if os.path.exists(TEST_SEQ_CACHE):
    print("Loading test seq_df from cache...")
    test_seq_df = pd.read_parquet(TEST_SEQ_CACHE)
    print(f"Loaded {len(test_seq_df):,} rows")
else:
    print("Reading test sequences from BQ...")
    test_seq_df = client.query(f"""
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
        FROM `{DS}.A870800_gen_rec_test_sequences_{SAMPLE}`
        ORDER BY member_id, trigger_date, trigger_dx, recency_rank
    """).to_dataframe()
    print(f"BQ returned {len(test_seq_df):,} rows")
    qa_df(test_seq_df, "test_seq_df raw from BQ",
          check_cols=["member_segment", "is_t30_qualified"])

    # Map using existing vocab — OOV → PAD
    test_seq_df["specialty_id"] = (
        test_seq_df["specialty"].map(specialty_vocab).fillna(PAD_IDX).astype(int)
    )
    oov = test_seq_df["specialty"].map(specialty_vocab).isna().sum()
    print(f"OOV specialties: {oov:,} ({oov/len(test_seq_df)*100:.1f}%) → mapped to PAD")

    test_seq_df.to_parquet(TEST_SEQ_CACHE, index=False)
    print("Cached test seq_df")

test_seq_df["trigger_date"] = test_seq_df["trigger_date"].astype(str).str[:10]
qa_df(test_seq_df, "test_seq_df after trigger_date normalized", check_cols=["recency_rank"])
print(f"Section 1 done — {len(test_seq_df):,} rows, time={time.time()-t0:.1f}s")
display(Markdown(f"**Test seq rows:** {len(test_seq_df):,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PULL TEST LABELS FROM BQ
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 2 — Pull Test Labels"))
print("Section 2 starting...")

TEST_LABEL_CACHE = f"{CACHE_DIR}/raw_test_label_df.parquet"

if os.path.exists(TEST_LABEL_CACHE):
    print("Loading test label_df from cache...")
    test_label_df = pd.read_parquet(TEST_LABEL_CACHE)
    print(f"Loaded {len(test_label_df):,} rows")
else:
    print("Reading test labels from BQ...")
    test_label_df = client.query(f"""
        SELECT
            member_id
            ,CAST(trigger_date AS STRING)                AS trigger_date
            ,trigger_dx
            ,time_bucket
            ,ARRAY_AGG(DISTINCT label_specialty
                ORDER BY label_specialty)                AS true_label_set
        FROM `{DS}.A870800_gen_rec_model_test_{SAMPLE}`
        WHERE label_specialty IS NOT NULL
        GROUP BY member_id, trigger_date, trigger_dx, time_bucket
    """).to_dataframe()
    test_label_df.to_parquet(TEST_LABEL_CACHE, index=False)
    print(f"Cached {len(test_label_df):,} rows")

test_label_df["trigger_date"] = test_label_df["trigger_date"].astype(str).str[:10]
qa_df(test_label_df, "test_label_df after trigger_date normalized", check_cols=["time_bucket"])

# Overlap check
seq_dates = set(test_seq_df["trigger_date"].unique())
lbl_dates = set(test_label_df["trigger_date"].unique())
overlap   = seq_dates & lbl_dates
print(f"trigger_date overlap: seq={len(seq_dates):,} label={len(lbl_dates):,} overlap={len(overlap):,}")
if len(overlap) == 0:
    print("CRITICAL: No overlap — all labels will be empty")

print(f"Section 2 done — {len(test_label_df):,} rows, time={time.time()-t0:.1f}s")
display(Markdown(f"**Test label rows:** {len(test_label_df):,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BUILD TEST DATASET
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 3 — Build Test Dataset"))
print("Section 3 starting...")

TEST_CACHE = f"{CACHE_DIR}/test_seq_matrix.npy"

if os.path.exists(TEST_CACHE):
    print("Test numpy cache exists — loading...")
    test_data = {
        k: np.load(f"{CACHE_DIR}/test_{k}.npy", allow_pickle=True)
        for k in ["seq_matrix", "lab_t30", "lab_t60", "lab_t180",
                  "seq_lengths", "is_t30", "is_t60", "is_t180",
                  "trigger_dates", "member_ids", "trigger_dxs", "segments"]
    }
    print(f"Loaded — {test_data['seq_matrix'].shape[0]:,} records")
else:
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

        seq_matrix  = np.zeros((N, max_seq_len), dtype=np.int64)
        seq_lengths = np.zeros(N, dtype=np.int64)
        lab_t30     = np.zeros((N, num_specialties), dtype=np.float32)
        lab_t60     = np.zeros((N, num_specialties), dtype=np.float32)
        lab_t180    = np.zeros((N, num_specialties), dtype=np.float32)
        is_t30_arr  = np.zeros(N, dtype=bool)
        is_t60_arr  = np.zeros(N, dtype=bool)
        is_t180_arr = np.zeros(N, dtype=bool)

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

            if (i + 1) % 500_000 == 0:
                print(f"  Progress: {i+1:,}/{N:,}")

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
        print(f"  T0_30  qual    : {is_t30_f.sum():,} | labeled: {(lab_t30_f.sum(axis=1)>0).sum():,}")
        print(f"  T30_60 qual    : {is_t60_f.sum():,} | labeled: {(lab_t60_f.sum(axis=1)>0).sum():,}")
        print(f"  T60_180 qual   : {is_t180_f.sum():,} | labeled: {(lab_t180_f.sum(axis=1)>0).sum():,}")
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

    test_data = build_records_fast(
        test_seq_df, test_label_df, specialty_vocab, MAX_SEQ_LEN, NUM_SPECIALTIES
    )

    del test_seq_df, test_label_df
    gc.collect()
    print("test_seq_df and test_label_df freed")

    # Save test arrays
    print("Saving test arrays to disk...")
    for k, v in test_data.items():
        np.save(f"{CACHE_DIR}/test_{k}.npy", v)
        print(f"  Saved test_{k}.npy — {v.shape} {v.dtype}")

# Summary
N_test = test_data["seq_matrix"].shape[0]
print(f"\nTest dataset ready — {N_test:,} records")
print(f"  seq_matrix  : {test_data['seq_matrix'].shape}  {test_data['seq_matrix'].nbytes/1e9:.2f}GB")
print(f"  lab_t30     : {test_data['lab_t30'].shape}  {test_data['lab_t30'].nbytes/1e9:.2f}GB")
print(f"  Cache dir   : {CACHE_DIR}")
print(f"Section 3 done — time={time.time()-t0:.1f}s")
display(Markdown(f"""
## Test Dataset Built
| Key | Value |
|---|---|
| Test records | {N_test:,} |
| Specialties | {NUM_SPECIALTIES:,} |
| Seq matrix GB | {test_data['seq_matrix'].nbytes/1e9:.2f} |
| Labels GB (×3) | {test_data['lab_t30'].nbytes/1e9:.2f} each |
| Cache | {CACHE_DIR} |
| Time | {time.time()-t0:.1f}s |
"""))
print("Model_03_build_test_dataset complete")
