# ============================================================
# NB_02 — build_provider_train_dataset.py  (vectorized)
# Purpose : Build provider train numpy arrays from BQ data
#           Run once per sample — shared by SASRec, BERT4Rec, HSTU
# Sources : A870800_gen_rec_provider_train_sequences_{SAMPLE}
#           A870800_gen_rec_provider_model_train_{SAMPLE}
#           A870800_gen_rec_provider_transitions
# Output  : ./cache_provider_{SAMPLE}/
#               train_seq_matrix.npy        (N, 20, 2) int32
#               train_delta_t_matrix.npy    (N, 20) int32
#               train_trigger_token.npy     (N, 1) int32
#               train_seq_lengths.npy       (N,) int32
#               train_lab_t30/t60/t180.npy  (N,) object — sparse int_id lists
#               train_hard_neg_candidates.pkl
#               train_is_t30/t60/t180.npy   (N,) bool
#               train_member_ids/trigger_dates/trigger_dxs/segments.npy
#               val_*.npy  (same keys — last 10% by trigger_date)
# Vectorization:
#   seq_matrix: built with numpy advanced indexing — no Python trigger loop
#   labels:     groupby+agg → dict lookup — no itertuples
#   transitions: groupby → nested dict — no itertuples over 41M rows
#   hard negs:  per-trigger loop unavoidable (depends on per-trigger positives)
#               but minimized with pre-built specialty lookup arrays
# ============================================================

import gc
import os
import pickle
import time
from collections import defaultdict

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
VAL_FRAC    = 0.10

DS          = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
CACHE_DIR   = f"./cache_provider_{SAMPLE}"
os.makedirs(CACHE_DIR, exist_ok=True)
client      = bigquery.Client(project="anbc-hcb-dev")

print(f"Config — sample={SAMPLE}, max_seq_len={MAX_SEQ_LEN}, val_frac={VAL_FRAC}")
display(Markdown(f"""
## Config
| Parameter | Value |
|---|---|
| Sample | {SAMPLE} |
| Max sequence length | {MAX_SEQ_LEN} |
| Val fraction | {VAL_FRAC} |
| Cache dir | {CACHE_DIR} |
"""))


# ── LOAD VOCABS ───────────────────────────────────────────────────────────────
print("Loading vocabs from NB_01...")
with open(f"{CACHE_DIR}/provider_vocab.pkl",       "rb") as f: provider_vocab        = pickle.load(f)
with open(f"{CACHE_DIR}/specialty_vocab.pkl",      "rb") as f: specialty_vocab       = pickle.load(f)
with open(f"{CACHE_DIR}/dx_vocab.pkl",             "rb") as f: dx_vocab              = pickle.load(f)
with open(f"{CACHE_DIR}/provider_specialty_map.pkl","rb") as f: provider_specialty_map = pickle.load(f)

PROVIDER_VOCAB_SIZE = len(provider_vocab)
print(f"  provider_vocab={PROVIDER_VOCAB_SIZE:,}  specialty={len(specialty_vocab):,}  "
      f"dx={len(dx_vocab):,}  prov_spec={len(provider_specialty_map):,}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PULL + ENCODE TRAIN SEQUENCES
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 1 — Pull Train Sequences"))

SEQ_CACHE = f"{CACHE_DIR}/raw_train_seq_df.parquet"

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
        FROM `{DS}.A870800_gen_rec_provider_train_sequences_{SAMPLE}`
        ORDER BY member_id, trigger_date, trigger_dx, recency_rank
    """).to_dataframe()
    seq_df["trigger_date"] = seq_df["trigger_date"].astype(str).str[:10]
    qa_df(seq_df, "seq_df raw", check_cols=["member_segment", "recency_rank"])
    seq_df.to_parquet(SEQ_CACHE, index=False)

seq_df["trigger_date"] = seq_df["trigger_date"].astype(str).str[:10]

# Vectorized encoding via pandas .map()
seq_df["provider_id"]  = seq_df["srv_prvdr_id"].map(provider_vocab).fillna(UNK_IDX).astype(np.int32)
seq_df["specialty_id"] = seq_df["specialty_ctg_cd"].map(specialty_vocab).fillna(UNK_IDX).astype(np.int32)
seq_df["delta_t_int"]  = seq_df["delta_t_bucket"].fillna(0).astype(np.int32)

unk_pct = (seq_df["provider_id"] == UNK_IDX).mean() * 100
print(f"Provider UNK rate: {unk_pct:.1f}% (tail providers → UNK)")
print(f"Section 1 done — {len(seq_df):,} rows | {time.time()-t0:.1f}s")
display(Markdown(f"**Seq rows:** {len(seq_df):,} | **UNK:** {unk_pct:.1f}% | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PULL + ENCODE TRAIN LABELS
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 2 — Pull Train Labels"))

LABEL_CACHE = f"{CACHE_DIR}/raw_train_label_df.parquet"

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
            ,from_provider
            ,member_segment
            ,is_t30_qualified
            ,is_t60_qualified
            ,is_t180_qualified
            ,label_provider
            ,time_bucket
        FROM `{DS}.A870800_gen_rec_provider_model_train_{SAMPLE}`
        WHERE label_provider IS NOT NULL
    """).to_dataframe()
    label_df["trigger_date"] = label_df["trigger_date"].astype(str).str[:10]
    qa_df(label_df, "label_df raw", check_cols=["time_bucket"])
    label_df.to_parquet(LABEL_CACHE, index=False)

label_df["trigger_date"] = label_df["trigger_date"].astype(str).str[:10]

# Vectorized encoding
# trigger_dx_clean for dx lookup — NOT trigger_dx (raw with dots)
label_df["label_provider_id"] = label_df["label_provider"].map(provider_vocab).fillna(UNK_IDX).astype(np.int32)
label_df["trigger_dx_id"]     = label_df["trigger_dx_clean"].map(dx_vocab).fillna(UNK_IDX).astype(np.int32)

print(f"Section 2 done — {len(label_df):,} label rows | {time.time()-t0:.1f}s")
display(Markdown(f"**Label rows:** {len(label_df):,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BUILD HARD NEGATIVE CANDIDATE LOOKUP FROM TRANSITIONS
# Vectorized: pandas groupby replaces itertuples over 41M rows
# Result: {from_provider -> {specialty -> np.array([to_provider_int_ids])}}
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 3 — Build Hard Negative Lookup"))

TRANS_CACHE  = f"{CACHE_DIR}/raw_transitions_df.parquet"
HARDNEG_LOOKUP_CACHE = f"{CACHE_DIR}/from_to_by_specialty.pkl"

if os.path.exists(HARDNEG_LOOKUP_CACHE):
    print("Loading hard neg lookup from cache...")
    with open(HARDNEG_LOOKUP_CACHE, "rb") as f:
        from_to_by_specialty = pickle.load(f)
else:
    if os.path.exists(TRANS_CACHE):
        print("Loading transitions from cache...")
        trans_df = pd.read_parquet(TRANS_CACHE)
    else:
        print("Reading transitions from BQ...")
        trans_df = client.query(f"""
            SELECT from_provider, to_provider, transition_count
            FROM `{DS}.A870800_gen_rec_provider_transitions`
            WHERE from_provider IS NOT NULL AND to_provider IS NOT NULL
        """).to_dataframe()
        trans_df.to_parquet(TRANS_CACHE, index=False)

    print(f"Transitions: {len(trans_df):,} rows — vectorizing...")
    t1 = time.time()

    # Vectorized: map to_provider to int_id and specialty in one shot
    top80_set = set(k for k in provider_vocab if k not in ("PAD", "UNK"))

    trans_df["to_int"] = trans_df["to_provider"].map(provider_vocab)
    trans_df["to_spec"] = trans_df["to_provider"].map(provider_specialty_map)

    # Filter: only to_providers in top80 with a known specialty
    trans_valid = trans_df.dropna(subset=["to_int", "to_spec"]).copy()
    trans_valid["to_int"] = trans_valid["to_int"].astype(np.int32)

    print(f"  Valid transitions (top80 + known specialty): {len(trans_valid):,}")

    # Vectorized groupby: {from_provider -> {specialty -> [to_int_ids]}}
    # groupby(['from_provider','to_spec'])['to_int'].apply(list) then pivot to nested dict
    grouped_trans = (
        trans_valid
        .groupby(["from_provider", "to_spec"])["to_int"]
        .apply(np.array)
        .reset_index()
    )

    # Build nested dict: {from_provider -> {specialty -> np.array}}
    from_to_by_specialty = {}
    for from_p, grp in grouped_trans.groupby("from_provider"):
        from_to_by_specialty[from_p] = dict(
            zip(grp["to_spec"], grp["to_int"])
        )

    print(f"  from_providers with transitions: {len(from_to_by_specialty):,} | {time.time()-t1:.1f}s")

    with open(HARDNEG_LOOKUP_CACHE, "wb") as f:
        pickle.dump(from_to_by_specialty, f)

    del trans_df, trans_valid, grouped_trans
    gc.collect()

print(f"Section 3 done — {time.time()-t0:.1f}s")
display(Markdown(f"**Hard neg lookup built** | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BUILD TRIGGER INDEX + METADATA (vectorized)
# Assign a dense integer trigger_idx to each unique (member, date, dx)
# All downstream arrays indexed by trigger_idx
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 4 — Build Trigger Index"))

TRAIN_CACHE = f"{CACHE_DIR}/train_seq_matrix.npy"

if os.path.exists(TRAIN_CACHE):
    print("Train numpy cache exists — loading...")
    keys = ["seq_matrix", "delta_t_matrix", "trigger_token", "seq_lengths",
            "lab_t30", "lab_t60", "lab_t180",
            "is_t30", "is_t60", "is_t180",
            "member_ids", "trigger_dates", "trigger_dxs", "segments"]
    train_data = {k: np.load(f"{CACHE_DIR}/train_{k}.npy", allow_pickle=True) for k in keys}
    val_data   = {k: np.load(f"{CACHE_DIR}/val_{k}.npy",   allow_pickle=True) for k in keys}
    with open(f"{CACHE_DIR}/train_hard_neg_candidates.pkl", "rb") as f:
        hard_neg_candidates = pickle.load(f)
    N_train = train_data["seq_matrix"].shape[0]
    N_val   = val_data["seq_matrix"].shape[0]
    print(f"Loaded train={N_train:,} | val={N_val:,}")
else:
    # ── 4a: Trigger metadata from label_df ────────────────────────────────────
    # One row per unique (member, trigger_date, trigger_dx)
    meta_cols = ["member_id", "trigger_date", "trigger_dx",
                 "trigger_dx_id", "from_provider", "member_segment",
                 "is_t30_qualified", "is_t60_qualified", "is_t180_qualified"]
    meta_df = (
        label_df[meta_cols]
        .drop_duplicates(["member_id", "trigger_date", "trigger_dx"])
        .reset_index(drop=True)
    )
    meta_df["trigger_idx"] = meta_df.index.astype(np.int32)
    N = len(meta_df)
    print(f"Unique triggers: {N:,}")

    # ── 4b: Build sparse labels vectorized ────────────────────────────────────
    # groupby on (member, date, dx, bucket) → list of label_provider_ids
    print("Building sparse label arrays...")
    t1 = time.time()

    label_grp = (
        label_df
        .groupby(["member_id", "trigger_date", "trigger_dx", "time_bucket"])["label_provider_id"]
        .apply(list)
    )

    # Build lookup dict per bucket — vectorized dict from multiindex
    lab_dict = {"T0_30": {}, "T30_60": {}, "T60_180": {}}
    for (member, tdate, tdx, bucket), ids in label_grp.items():
        lab_dict[bucket][(member, tdate, tdx)] = list(set(ids))

    print(f"  Label lookup built — {time.time()-t1:.1f}s")

    # ── 4c: Build seq_matrix FULLY vectorized ─────────────────────────────────
    # Key insight: assign trigger_idx to each seq_df row via merge
    # Then use numpy advanced indexing to fill seq_matrix in one shot
    print("Building seq_matrix (vectorized)...")
    t1 = time.time()

    # Merge trigger_idx into seq_df
    seq_df2 = seq_df.merge(
        meta_df[["member_id", "trigger_date", "trigger_dx", "trigger_idx"]],
        on=["member_id", "trigger_date", "trigger_dx"],
        how="inner"
    )

    # Group size per trigger → seq_lengths
    group_sizes = seq_df2.groupby("trigger_idx")["recency_rank"].transform("count").astype(np.int32)
    seq_lengths = np.zeros(N, dtype=np.int32)
    # Most recent group_size per trigger_idx
    tmp = seq_df2.groupby("trigger_idx").size()
    seq_lengths[tmp.index.values] = np.minimum(tmp.values, MAX_SEQ_LEN).astype(np.int32)

    # Position in seq_matrix:
    # recency_rank 1 = most recent → goes to position MAX_SEQ_LEN - 1
    # recency_rank k → goes to position MAX_SEQ_LEN - k  (if <= MAX_SEQ_LEN)
    # Only keep recency_rank <= MAX_SEQ_LEN (already filtered in SQL_03)
    seq_df2 = seq_df2[seq_df2["recency_rank"] <= MAX_SEQ_LEN].copy()
    seq_df2["seq_pos"] = (MAX_SEQ_LEN - seq_df2["recency_rank"]).astype(np.int32)

    # Allocate and fill with numpy advanced indexing — no Python loop
    seq_matrix  = np.zeros((N, MAX_SEQ_LEN, 2), dtype=np.int32)
    delta_t_mat = np.zeros((N, MAX_SEQ_LEN),    dtype=np.int32)

    trig_idx = seq_df2["trigger_idx"].values.astype(np.int64)
    seq_pos  = seq_df2["seq_pos"].values.astype(np.int64)
    prov_ids = seq_df2["provider_id"].values.astype(np.int32)
    spec_ids = seq_df2["specialty_id"].values.astype(np.int32)
    dt_vals  = seq_df2["delta_t_int"].values.astype(np.int32)

    seq_matrix[trig_idx, seq_pos, 0] = prov_ids   # provider_id
    seq_matrix[trig_idx, seq_pos, 1] = spec_ids   # specialty_id
    delta_t_mat[trig_idx, seq_pos]   = dt_vals

    print(f"  seq_matrix built: {seq_matrix.shape} | {time.time()-t1:.1f}s")

    del seq_df2, trig_idx, seq_pos, prov_ids, spec_ids, dt_vals
    gc.collect()

    # ── 4d: Trigger token + qualification flags + metadata ─────────────────────
    trigger_token = meta_df["trigger_dx_id"].values.reshape(-1, 1).astype(np.int32)
    is_t30_arr    = meta_df["is_t30_qualified"].values.astype(bool)
    is_t60_arr    = meta_df["is_t60_qualified"].values.astype(bool)
    is_t180_arr   = meta_df["is_t180_qualified"].values.astype(bool)
    member_ids_arr    = meta_df["member_id"].values
    trigger_dates_arr = meta_df["trigger_date"].values
    trigger_dxs_arr   = meta_df["trigger_dx"].values
    segments_arr      = meta_df["member_segment"].fillna("Unknown").values

    # ── 4e: Sparse labels ─────────────────────────────────────────────────────
    lab_t30  = np.empty(N, dtype=object)
    lab_t60  = np.empty(N, dtype=object)
    lab_t180 = np.empty(N, dtype=object)

    for i in range(N):
        key = (member_ids_arr[i], trigger_dates_arr[i], trigger_dxs_arr[i])
        lab_t30[i]  = lab_dict["T0_30"].get(key, [])
        lab_t60[i]  = lab_dict["T30_60"].get(key, [])
        lab_t180[i] = lab_dict["T60_180"].get(key, [])

    del lab_dict, label_df
    gc.collect()
    print(f"  Labels built (sparse ragged)")

    # ── 4f: Hard negative candidates ──────────────────────────────────────────
    # Per-trigger loop unavoidable — depends on per-trigger positives
    # Minimized: from_to_by_specialty pre-computed as numpy arrays
    print("Building hard neg candidates (minimal loop)...")
    t1 = time.time()

    hard_neg_candidates = {}
    from_prov_arr = meta_df["from_provider"].values

    for i in range(N):
        from_prov = from_prov_arr[i]
        if from_prov not in from_to_by_specialty:
            continue
        all_pos = set(
            lab_t30[i].tolist() if hasattr(lab_t30[i], 'tolist') else lab_t30[i]
        ) | set(
            lab_t60[i].tolist() if hasattr(lab_t60[i], 'tolist') else lab_t60[i]
        ) | set(
            lab_t180[i].tolist() if hasattr(lab_t180[i], 'tolist') else lab_t180[i]
        )
        if not all_pos:
            continue

        hard_negs = []
        for spec, cands in from_to_by_specialty[from_prov].items():
            # cands is already a numpy array — vectorized exclusion
            mask = ~np.isin(cands, list(all_pos))
            hard_negs.append(cands[mask])

        if hard_negs:
            combined = np.unique(np.concatenate(hard_negs))
            if len(combined) > 0:
                hard_neg_candidates[i] = combined

        if (i + 1) % 500_000 == 0:
            print(f"  Hard neg progress: {i+1:,}/{N:,}")

    print(f"  Hard negs built: {len(hard_neg_candidates):,} triggers | {time.time()-t1:.1f}s")

    del from_to_by_specialty, seq_df, meta_df
    gc.collect()

    # ── 4g: Filter, sort, split ────────────────────────────────────────────────
    # Filter: seq_len > 0
    valid = seq_lengths > 0
    n_valid = valid.sum()
    print(f"\nValid triggers: {n_valid:,}/{N:,}")

    any_label = np.array([len(lab_t30[i]) + len(lab_t60[i]) + len(lab_t180[i]) > 0
                          for i in range(N)])
    print(f"With labels: {any_label.sum():,} ({any_label.mean()*100:.1f}%)")

    valid_idx = np.where(valid)[0]

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

    # Remap hard_neg keys
    old_to_new = {old: new for new, old in enumerate(valid_idx)}
    hard_neg_candidates = {old_to_new[k]: v for k, v in hard_neg_candidates.items()
                           if k in old_to_new}

    # Temporal sort
    sort_idx = np.argsort(trigger_dates_arr, kind="stable")

    seq_matrix    = seq_matrix[sort_idx]
    delta_t_mat   = delta_t_mat[sort_idx]
    trigger_token = trigger_token[sort_idx]
    seq_lengths   = seq_lengths[sort_idx]
    lab_t30       = lab_t30[sort_idx]
    lab_t60       = lab_t60[sort_idx]
    lab_t180      = lab_t180[sort_idx]
    is_t30_arr    = is_t30_arr[sort_idx]
    is_t60_arr    = is_t60_arr[sort_idx]
    is_t180_arr   = is_t180_arr[sort_idx]
    member_ids_arr    = member_ids_arr[sort_idx]
    trigger_dates_arr = trigger_dates_arr[sort_idx]
    trigger_dxs_arr   = trigger_dxs_arr[sort_idx]
    segments_arr      = segments_arr[sort_idx]

    # Remap hard_neg after sort
    old_to_sorted = {old_sorted: new_sorted for new_sorted, old_sorted in enumerate(sort_idx)}
    hard_neg_candidates = {old_to_sorted[k]: v for k, v in hard_neg_candidates.items()
                           if k in old_to_sorted}

    # Train/val split — last VAL_FRAC by trigger_date
    N_total = seq_matrix.shape[0]
    n_val   = max(1, int(N_total * VAL_FRAC))
    n_train = N_total - n_val

    print(f"\nTrain: {n_train:,} | Val: {n_val:,}")
    print(f"Train dates: {trigger_dates_arr[0]} → {trigger_dates_arr[n_train-1]}")
    print(f"Val   dates: {trigger_dates_arr[n_train]} → {trigger_dates_arr[-1]}")

    # Package and save
    all_arrays = dict(
        seq_matrix=seq_matrix, delta_t_matrix=delta_t_mat,
        trigger_token=trigger_token, seq_lengths=seq_lengths,
        lab_t30=lab_t30, lab_t60=lab_t60, lab_t180=lab_t180,
        is_t30=is_t30_arr, is_t60=is_t60_arr, is_t180=is_t180_arr,
        member_ids=member_ids_arr, trigger_dates=trigger_dates_arr,
        trigger_dxs=trigger_dxs_arr, segments=segments_arr,
    )

    train_sl = slice(0, n_train)
    val_sl   = slice(n_train, None)
    train_hard_neg = {k: v for k, v in hard_neg_candidates.items() if k < n_train}

    train_data = {}
    val_data   = {}
    print("\nSaving arrays...")
    for k, arr in all_arrays.items():
        train_data[k] = arr[train_sl]
        val_data[k]   = arr[val_sl]
        np.save(f"{CACHE_DIR}/train_{k}.npy", train_data[k])
        np.save(f"{CACHE_DIR}/val_{k}.npy",   val_data[k])
        print(f"  train_{k}: {train_data[k].shape}  val_{k}: {val_data[k].shape}")

    with open(f"{CACHE_DIR}/train_hard_neg_candidates.pkl", "wb") as f:
        pickle.dump(train_hard_neg, f)
    print(f"  train_hard_neg_candidates: {len(train_hard_neg):,} triggers")

    del all_arrays, hard_neg_candidates
    gc.collect()

print(f"\nSection 4 done — {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — QA SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("---\n## Section 5 — QA Summary"))

N_train = train_data["seq_matrix"].shape[0]
N_val   = val_data["seq_matrix"].shape[0]

# Sample check on first trigger
i = 0
print(f"Sample trigger[0]:")
print(f"  member_id     : {train_data['member_ids'][i]}")
print(f"  trigger_date  : {train_data['trigger_dates'][i]}")
print(f"  seq_length    : {train_data['seq_lengths'][i]}")
print(f"  trigger_token : {train_data['trigger_token'][i]}")
print(f"  seq_matrix[-3:]: {train_data['seq_matrix'][i, -3:, :]}")
print(f"  lab_t30       : {train_data['lab_t30'][i]}")
print(f"  lab_t60       : {train_data['lab_t60'][i]}")

display(Markdown(f"""
## Train Dataset Summary
| Metric | Value |
|---|---|
| Train records | {N_train:,} |
| Val records | {N_val:,} |
| seq_matrix shape | {train_data['seq_matrix'].shape} |
| trigger_token shape | {train_data['trigger_token'].shape} |
| T30 qualified | {train_data['is_t30'].sum():,} |
| T60 qualified | {train_data['is_t60'].sum():,} |
| T180 qualified | {train_data['is_t180'].sum():,} |

Next: run NB_03 to build test dataset.
"""))
print("NB_02 complete")
