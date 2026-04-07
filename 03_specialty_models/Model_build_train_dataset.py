# ============================================================
# Model_00_build_train_dataset.py
# Purpose : Pull train sequences + labels from BQ
#           Run build_records_fast
#           Save numpy arrays + vocab to shared cache
#           Run once — shared by SASRec and BERT4Rec
# Output  : ./cache_model_data_{SAMPLE}/
#               train_seq_matrix.npy
#               train_lab_t30.npy
#               train_lab_t60.npy
#               train_lab_t180.npy
#               train_seq_lengths.npy
#               train_is_t30.npy / is_t60.npy / is_t180.npy
#               train_trigger_dates.npy
#               train_member_ids.npy
#               train_trigger_dxs.npy
#               train_segments.npy
#               vocab.pkl
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
SAMPLE      = "5pct"
MAX_SEQ_LEN = 20
PAD_IDX     = 0

DS          = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
CACHE_DIR   = f"./cache_train_data_{SAMPLE}"
os.makedirs(CACHE_DIR, exist_ok=True)

client = bigquery.Client(project="anbc-hcb-dev")

print(f"Config — sample={SAMPLE}, cache={CACHE_DIR}")
display(Markdown(f"""
## Config
| Parameter | Value |
|---|---|
| Sample | {SAMPLE} |
| Max sequence length | {MAX_SEQ_LEN} |
| Cache dir | {CACHE_DIR} |
"""))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PULL TRAIN SEQUENCES FROM BQ
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 1 — Pull Train Sequences"))
print("Section 1 starting...")

SEQ_CACHE   = f"{CACHE_DIR}/raw_seq_df.parquet"
VOCAB_CACHE = f"{CACHE_DIR}/vocab.pkl"

if os.path.exists(SEQ_CACHE) and os.path.exists(VOCAB_CACHE):
    print("Loading seq_df and vocab from cache...")
    seq_df = pd.read_parquet(SEQ_CACHE)
    with open(VOCAB_CACHE, "rb") as f:
        specialty_vocab = pickle.load(f)
    print(f"Loaded {len(seq_df):,} rows, vocab={len(specialty_vocab):,}")
else:
    print("Reading train sequences from BQ...")
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
    print("Cached seq_df and vocab to disk")

seq_df["trigger_date"] = seq_df["trigger_date"].astype(str).str[:10]
qa_df(seq_df, "seq_df after trigger_date normalized", check_cols=["recency_rank"])

NUM_SPECIALTIES = len(specialty_vocab)
print(f"Section 1 done — vocab={NUM_SPECIALTIES}, rows={len(seq_df):,}, time={time.time()-t0:.1f}s")
display(Markdown(f"**Vocab:** {NUM_SPECIALTIES:,} | **Rows:** {len(seq_df):,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PULL TRAIN LABELS FROM BQ
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 2 — Pull Train Labels"))
print("Section 2 starting...")

LABEL_CACHE = f"{CACHE_DIR}/raw_label_df.parquet"

if os.path.exists(LABEL_CACHE):
    print("Loading label_df from cache...")
    label_df = pd.read_parquet(LABEL_CACHE)
    print(f"Loaded {len(label_df):,} rows")
else:
    print("Reading train labels from BQ...")
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
    print(f"Cached {len(label_df):,} rows")

label_df["trigger_date"] = label_df["trigger_date"].astype(str).str[:10]
qa_df(label_df, "label_df after trigger_date normalized", check_cols=["time_bucket"])

seq_dates = set(seq_df["trigger_date"].unique())
lbl_dates = set(label_df["trigger_date"].unique())
overlap   = seq_dates & lbl_dates
print(f"trigger_date overlap: seq={len(seq_dates):,} label={len(lbl_dates):,} overlap={len(overlap):,}")
if len(overlap) == 0:
    print("CRITICAL: No trigger_date overlap — all labels will be empty")
elif len(overlap) < len(seq_dates) * 0.5:
    print(f"WARNING: <50% overlap")

print(f"Section 2 done — {len(label_df):,} rows, time={time.time()-t0:.1f}s")
display(Markdown(f"**Label rows:** {len(label_df):,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BUILD TRAIN DATASET
# Vectorized numpy — no list of dicts
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 3 — Build Train Dataset"))
print("Section 3 starting...")

TRAIN_CACHE = f"{CACHE_DIR}/train_seq_matrix.npy"

if os.path.exists(TRAIN_CACHE):
    print("Train numpy cache exists — loading...")
    train_data = {
        k: np.load(f"{CACHE_DIR}/train_{k}.npy", allow_pickle=True)
        for k in ["seq_matrix", "lab_t30", "lab_t60", "lab_t180",
                  "seq_lengths", "is_t30", "is_t60", "is_t180",
                  "trigger_dates", "member_ids", "trigger_dxs", "segments"]
    }
    print(f"Loaded — {train_data['seq_matrix'].shape[0]:,} records")
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

    raw_data = build_records_fast(
        seq_df, label_df, specialty_vocab, MAX_SEQ_LEN, NUM_SPECIALTIES
    )

    del seq_df, label_df
    gc.collect()
    print("seq_df and label_df freed")

    # Temporal sort
    print("Sorting by trigger_date...")
    sort_idx = np.argsort(raw_data["trigger_dates"])
    for k in raw_data:
        raw_data[k] = raw_data[k][sort_idx]

    # Train/val split
    N     = raw_data["seq_matrix"].shape[0]
    n_val = max(1, int(N * 0.1))
    split = N - n_val

    train_data = {k: v[:split] for k, v in raw_data.items()}
    val_data   = {k: v[split:] for k, v in raw_data.items()}
    del raw_data
    gc.collect()

    print(f"  Train: {split:,} | Val: {n_val:,}")
    print(f"  Train date range: {train_data['trigger_dates'][0]} → {train_data['trigger_dates'][-1]}")
    print(f"  Val   date range: {val_data['trigger_dates'][0]} → {val_data['trigger_dates'][-1]}")
    if val_data["trigger_dates"][0] < train_data["trigger_dates"][-1]:
        print("  WARNING: Val dates overlap with train")
    else:
        print("  Train/val split correct")

    # Save train arrays
    print("Saving train arrays to disk...")
    for k, v in train_data.items():
        np.save(f"{CACHE_DIR}/train_{k}.npy", v)
        print(f"  Saved train_{k}.npy — {v.shape} {v.dtype}")

    # Save val arrays
    print("Saving val arrays to disk...")
    for k, v in val_data.items():
        np.save(f"{CACHE_DIR}/val_{k}.npy", v)
        print(f"  Saved val_{k}.npy — {v.shape} {v.dtype}")

    del val_data
    gc.collect()

# Summary
N_train = train_data["seq_matrix"].shape[0]
print(f"\nTrain dataset ready — {N_train:,} records")
print(f"  seq_matrix  : {train_data['seq_matrix'].shape}  {train_data['seq_matrix'].nbytes/1e9:.2f}GB")
print(f"  lab_t30     : {train_data['lab_t30'].shape}  {train_data['lab_t30'].nbytes/1e9:.2f}GB")
print(f"  Vocab saved : {VOCAB_CACHE}")
print(f"  Cache dir   : {CACHE_DIR}")
print(f"Section 3 done — time={time.time()-t0:.1f}s")
display(Markdown(f"""
## Train Dataset Built
| Key | Value |
|---|---|
| Train records | {N_train:,} |
| Specialties | {NUM_SPECIALTIES:,} |
| Seq matrix GB | {train_data['seq_matrix'].nbytes/1e9:.2f} |
| Labels GB (×3) | {train_data['lab_t30'].nbytes/1e9:.2f} each |
| Cache | {CACHE_DIR} |
| Time | {time.time()-t0:.1f}s |
"""))
print("Model_00_build_train_dataset complete")
