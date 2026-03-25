# ============================================================
# NB_02 — build_provider_train_dataset.py  (BQ-first, vectorized)
# Purpose : Build provider train numpy arrays from BQ data
#           BQ does ALL heavy lifting — Python is a thin consumer
# Sources : A870800_gen_rec_provider_train_sequences_{SAMPLE}
#             flat rows per (trigger, recency_rank) — SQL_03
#           A870800_gen_rec_provider_model_train_agg_{SAMPLE}
#             ONE ROW PER TRIGGER — labels pre-aggregated by BQ ARRAY_AGG
#           A870800_gen_rec_provider_hardneg_lookup
#             pre-aggregated hard neg candidates per (from_provider, specialty)
# Output  : ./cache_provider_{SAMPLE}/
#               train_seq_matrix.npy        (N, 20, 2) int32
#               train_delta_t_matrix.npy    (N, 20) int32
#               train_trigger_token.npy     (N, 1) int32
#               train_seq_lengths.npy       (N,) int32
#               train_lab_t30/t60/t180.npy  (N,) object — sparse int_id lists
#               train_hard_neg_candidates.pkl
#               train_is_t30/t60/t180.npy   (N,) bool
#               train_member_ids/trigger_dates/trigger_dxs/segments.npy
#               val_*.npy  (mirror — last 10% by trigger_date)
# What Python does:
#   1. Pull pre-aggregated label rows (1 row per trigger) — no groupby
#   2. Pull sequences (flat rows) — encode with vocab maps
#   3. Pull hard neg lookup — pre-aggregated by BQ
#   4. Encode provider/specialty/dx ids with vectorized .map()
#   5. Build seq_matrix with numpy advanced indexing
#   6. Assign labels directly from pre-aggregated arrays — no loop
#   7. Build hard neg candidates — small loop over triggers WITH hard negs
# ============================================================

import gc
import os
import pickle
import time
from functools import partial

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
| BQ does | label aggregation, hard neg lookup |
| Python does | vocab encoding, numpy assembly |
"""))


# ── LOAD VOCABS (from NB_01) ──────────────────────────────────────────────────
print("Loading vocabs from NB_01...")
with open(f"{CACHE_DIR}/provider_vocab.pkl",  "rb") as f: provider_vocab  = pickle.load(f)
with open(f"{CACHE_DIR}/specialty_vocab.pkl", "rb") as f: specialty_vocab = pickle.load(f)
with open(f"{CACHE_DIR}/dx_vocab.pkl",        "rb") as f: dx_vocab        = pickle.load(f)

PROVIDER_VOCAB_SIZE = len(provider_vocab)
print(f"  provider={PROVIDER_VOCAB_SIZE:,}  specialty={len(specialty_vocab):,}  dx={len(dx_vocab):,}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PULL TRAIN LABELS (pre-aggregated — ONE ROW PER TRIGGER)
# BQ ARRAY_AGG already groups labels per trigger per window
# Python has zero groupby work to do here
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 1 — Pull Train Labels (pre-aggregated)"))

LABEL_CACHE = f"{CACHE_DIR}/raw_train_label_agg_df.parquet"

if os.path.exists(LABEL_CACHE):
    print("Loading from cache...")
    label_df = pd.read_parquet(LABEL_CACHE)
else:
    print("Reading pre-aggregated labels from BQ...")
    # One row per trigger — BQ already did ARRAY_AGG
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
            ,lab_t30
            ,lab_t60
            ,lab_t180
        FROM `{DS}.A870800_gen_rec_provider_model_train_agg_{SAMPLE}`
        ORDER BY trigger_date, member_id, trigger_dx
    """).to_dataframe()
    label_df["trigger_date"] = label_df["trigger_date"].astype(str).str[:10]
    qa_df(label_df, "label_df (one row per trigger)", check_cols=["member_segment"])
    label_df.to_parquet(LABEL_CACHE, index=False)

label_df["trigger_date"] = label_df["trigger_date"].astype(str).str[:10]

N = len(label_df)
print(f"Triggers: {N:,} — one row each, labels pre-aggregated by BQ ✓")

# Vectorized: encode trigger_dx_clean for trigger token
label_df["trigger_dx_id"] = label_df["trigger_dx_clean"].map(dx_vocab).fillna(UNK_IDX).astype(np.int32)

# Vectorized: encode lab arrays — apply over pre-built lists
def encode_label_list(lst, vocab):
    if lst is None or (hasattr(lst, '__len__') and len(lst) == 0):
        return []
    return [vocab.get(p, UNK_IDX) for p in lst]

label_df["lab_t30_ids"]  = label_df["lab_t30"].apply(lambda x: encode_label_list(x, provider_vocab))
label_df["lab_t60_ids"]  = label_df["lab_t60"].apply(lambda x: encode_label_list(x, provider_vocab))
label_df["lab_t180_ids"] = label_df["lab_t180"].apply(lambda x: encode_label_list(x, provider_vocab))

print(f"Section 1 done — {N:,} triggers | {time.time()-t0:.1f}s")
display(Markdown(f"**Triggers:** {N:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PULL TRAIN SEQUENCES (flat rows per recency_rank)
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 2 — Pull Train Sequences"))

SEQ_CACHE = f"{CACHE_DIR}/raw_train_seq_df.parquet"

if os.path.exists(SEQ_CACHE):
    print("Loading from cache...")
    seq_df = pd.read_parquet(SEQ_CACHE)
else:
    print("Reading train sequences from BQ...")
    seq_df = client.query(f"""
        SELECT
            member_id
            ,CAST(trigger_date AS STRING)                AS trigger_date
            ,trigger_dx
            ,srv_prvdr_id
            ,specialty_ctg_cd
            ,delta_t_bucket
            ,recency_rank
        FROM `{DS}.A870800_gen_rec_provider_train_sequences_{SAMPLE}`
        ORDER BY member_id, trigger_date, trigger_dx, recency_rank
    """).to_dataframe()
    seq_df["trigger_date"] = seq_df["trigger_date"].astype(str).str[:10]
    qa_df(seq_df, "seq_df raw", check_cols=["recency_rank"])
    seq_df.to_parquet(SEQ_CACHE, index=False)

seq_df["trigger_date"] = seq_df["trigger_date"].astype(str).str[:10]

# Vectorized encoding
seq_df["provider_id"]  = seq_df["srv_prvdr_id"].map(provider_vocab).fillna(UNK_IDX).astype(np.int32)
seq_df["specialty_id"] = seq_df["specialty_ctg_cd"].map(specialty_vocab).fillna(UNK_IDX).astype(np.int32)
seq_df["delta_t_int"]  = seq_df["delta_t_bucket"].fillna(0).astype(np.int32)

unk_pct = (seq_df["provider_id"] == UNK_IDX).mean() * 100
print(f"Provider UNK rate: {unk_pct:.1f}%")
print(f"Section 2 done — {len(seq_df):,} rows | {time.time()-t0:.1f}s")
display(Markdown(f"**Seq rows:** {len(seq_df):,} | **UNK:** {unk_pct:.1f}% | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PULL HARD NEG LOOKUP (pre-aggregated by BQ)
# BQ already computed: {from_provider → specialty → [to_provider candidates]}
# Python just loads and encodes — no 41M row groupby
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 3 — Pull Hard Neg Lookup"))

HARDNEG_CACHE = f"{CACHE_DIR}/hardneg_lookup_df.parquet"

if os.path.exists(HARDNEG_CACHE):
    print("Loading from cache...")
    hn_df = pd.read_parquet(HARDNEG_CACHE)
else:
    print("Reading pre-aggregated hard neg lookup from BQ...")
    hn_df = client.query(f"""
        SELECT
            from_provider
            ,to_specialty
            ,to_provider_candidates
        FROM `{DS}.A870800_gen_rec_provider_hardneg_lookup`
        WHERE from_provider IS NOT NULL
    """).to_dataframe()
    qa_df(hn_df, "hard neg lookup", sample_n=3)
    hn_df.to_parquet(HARDNEG_CACHE, index=False)

# Build lookup: {from_provider -> {specialty -> np.array(int_ids)}}
# Vectorized: encode to_provider_candidates arrays
print("Encoding hard neg candidates...")
t1 = time.time()
from_to_by_specialty = {}
for row in hn_df.itertuples(index=False):
    fp   = row.from_provider
    spec = row.to_specialty
    # Encode candidate provider strings to int_ids — filter to known vocab
    cands_raw = row.to_provider_candidates
    if cands_raw is None or (hasattr(cands_raw, '__len__') and len(cands_raw) == 0):
        continue
    cands_int = np.array(
        [provider_vocab[p] for p in cands_raw if p in provider_vocab],
        dtype=np.int32
    )
    if len(cands_int) == 0:
        continue
    if fp not in from_to_by_specialty:
        from_to_by_specialty[fp] = {}
    from_to_by_specialty[fp][spec] = cands_int

print(f"  from_providers: {len(from_to_by_specialty):,} | {time.time()-t1:.1f}s")
print(f"  Note: itertuples over {len(hn_df):,} rows (much smaller than 41M transitions)")
print(f"Section 3 done — {time.time()-t0:.1f}s")
display(Markdown(f"**Hard neg lookup:** {len(from_to_by_specialty):,} from_providers | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BUILD NUMPY ARRAYS
# Python's only job: assemble pre-encoded data into numpy format
# seq_matrix: vectorized numpy advanced indexing
# labels: direct assignment from pre-aggregated BQ arrays — no groupby
# hard negs: per-trigger loop — unavoidable but minimized
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 4 — Build Numpy Arrays"))

TRAIN_CACHE = f"{CACHE_DIR}/train_seq_matrix.npy"

if os.path.exists(TRAIN_CACHE):
    print("Cache exists — loading...")
    keys = ["seq_matrix", "delta_t_matrix", "trigger_token", "seq_lengths",
            "lab_t30", "lab_t60", "lab_t180",
            "is_t30", "is_t60", "is_t180",
            "member_ids", "trigger_dates", "trigger_dxs", "segments",
            "from_provider_ids"]
    train_data = {k: np.load(f"{CACHE_DIR}/train_{k}.npy", allow_pickle=True) for k in keys}
    val_data   = {k: np.load(f"{CACHE_DIR}/val_{k}.npy",   allow_pickle=True) for k in keys}
    with open(f"{CACHE_DIR}/from_provider_to_cands.pkl", "rb") as f:
        from_provider_to_cands = pickle.load(f)
    print(f"Loaded train={train_data['seq_matrix'].shape[0]:,} | val={val_data['seq_matrix'].shape[0]:,}")
else:
    # label_df is already sorted by trigger_date (from BQ ORDER BY)
    # trigger_idx = row position in label_df
    label_df = label_df.reset_index(drop=True)
    label_df["trigger_idx"] = label_df.index.astype(np.int32)

    # ── 4a: Metadata arrays — vectorized from label_df ────────────────────────
    print("4a: Building metadata arrays (vectorized)...")
    trigger_token     = label_df["trigger_dx_id"].values.reshape(-1, 1).astype(np.int32)
    is_t30_arr        = label_df["is_t30_qualified"].values.astype(bool)
    is_t60_arr        = label_df["is_t60_qualified"].values.astype(bool)
    is_t180_arr       = label_df["is_t180_qualified"].values.astype(bool)
    member_ids_arr    = label_df["member_id"].values
    trigger_dates_arr = label_df["trigger_date"].values
    trigger_dxs_arr   = label_df["trigger_dx"].values
    segments_arr      = label_df["member_segment"].fillna("Unknown").values

    # ── 4b: Sparse label arrays — direct from pre-aggregated BQ data ──────────
    # NO groupby, NO loop over N triggers, NO dict lookup
    # BQ already aggregated — just assign the encoded lists
    print("4b: Assigning sparse labels (direct from BQ arrays)...")
    t1 = time.time()
    lab_t30  = label_df["lab_t30_ids"].values   # already list of int_ids per trigger
    lab_t60  = label_df["lab_t60_ids"].values
    lab_t180 = label_df["lab_t180_ids"].values
    print(f"  Labels assigned in {time.time()-t1:.1f}s — no Python groupby ✓")

    # ── 4c: seq_matrix — vectorized numpy advanced indexing ───────────────────
    print("4c: Building seq_matrix (vectorized numpy)...")
    t1 = time.time()

    # Assign trigger_idx to seq_df via merge on (member, date, dx)
    seq_df2 = seq_df.merge(
        label_df[["member_id","trigger_date","trigger_dx","trigger_idx"]],
        on=["member_id","trigger_date","trigger_dx"],
        how="inner"
    )
    seq_df2 = seq_df2[seq_df2["recency_rank"] <= MAX_SEQ_LEN].copy()
    seq_df2["seq_pos"] = (MAX_SEQ_LEN - seq_df2["recency_rank"]).astype(np.int32)

    # Sequence lengths per trigger
    seq_lengths = np.zeros(N, dtype=np.int32)
    tmp = seq_df2.groupby("trigger_idx").size()
    seq_lengths[tmp.index.values] = np.minimum(tmp.values, MAX_SEQ_LEN).astype(np.int32)

    # Fill with single numpy advanced indexing — no Python trigger loop
    seq_matrix  = np.zeros((N, MAX_SEQ_LEN, 2), dtype=np.int32)
    delta_t_mat = np.zeros((N, MAX_SEQ_LEN),    dtype=np.int32)

    trig_idx = seq_df2["trigger_idx"].values.astype(np.int64)
    seq_pos  = seq_df2["seq_pos"].values.astype(np.int64)
    prov_ids = seq_df2["provider_id"].values.astype(np.int32)
    spec_ids = seq_df2["specialty_id"].values.astype(np.int32)
    dt_vals  = seq_df2["delta_t_int"].values.astype(np.int32)

    seq_matrix[trig_idx, seq_pos, 0] = prov_ids
    seq_matrix[trig_idx, seq_pos, 1] = spec_ids
    delta_t_mat[trig_idx, seq_pos]   = dt_vals

    print(f"  seq_matrix {seq_matrix.shape} built in {time.time()-t1:.1f}s ✓")
    del seq_df2, trig_idx, seq_pos, prov_ids, spec_ids, dt_vals
    gc.collect()

    # ── 4d: Hard neg lookup — NO per-trigger loop ────────────────────────────
    # Exclusion of positives moves to collate_fn at training time
    # Here we only store:
    #   from_provider_ids: (N,) int32 — from_provider int_id per trigger
    #   from_provider_to_cands.pkl — {from_provider_int_id -> candidate_int_ids}
    # collate_fn does: cands[~np.isin(cands, positives)] per batch of 512
    print("4d: Building from_provider lookup (no per-trigger loop)...")
    t1 = time.time()

    # Encode from_provider string → int_id
    # Use provider_vocab for known providers, -1 for unknown
    from_prov_arr = label_df["from_provider"].values
    from_provider_ids = np.array(
        [provider_vocab.get(p, -1) for p in from_prov_arr],
        dtype=np.int32
    )

    # Build {from_provider_int_id -> np.array of all candidate int_ids}
    # One entry per unique from_provider — few thousand entries total
    # Merge all specialties for this from_provider into one array
    from_provider_to_cands = {}
    for from_prov_str, spec_dict in from_to_by_specialty.items():
        fp_int = provider_vocab.get(from_prov_str, -1)
        if fp_int < 0:
            continue
        all_cands = np.unique(np.concatenate(list(spec_dict.values())))
        from_provider_to_cands[fp_int] = all_cands

    print(f"  from_provider_to_cands: {len(from_provider_to_cands):,} entries | {time.time()-t1:.1f}s")
    print(f"  Positive exclusion deferred to collate_fn — 0s here ✓")

    del seq_df, label_df, from_to_by_specialty
    gc.collect()

    # ── 4e: Filter + temporal sort + train/val split ───────────────────────────
    valid     = seq_lengths > 0
    valid_idx = np.where(valid)[0]
    n_valid   = len(valid_idx)
    print(f"\nValid triggers: {n_valid:,}/{N:,}")

    any_label = np.array([len(lab_t30[i]) + len(lab_t60[i]) + len(lab_t180[i]) > 0
                          for i in valid_idx])
    print(f"With labels: {any_label.sum():,} ({any_label.mean()*100:.1f}%)")

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

    old_to_new = {old: new for new, old in enumerate(valid_idx)}
    hard_neg_candidates = {old_to_new[k]: v for k, v in hard_neg_candidates.items()
                           if k in old_to_new}

    # Already sorted by trigger_date from BQ — no sort needed
    # Verify
    assert all(trigger_dates_arr[i] <= trigger_dates_arr[i+1]
               for i in range(min(1000, len(trigger_dates_arr)-1))), \
        "WARNING: not sorted by date"
    print("Temporal order verified ✓ (sorted by BQ ORDER BY)")

    # Train/val split
    N_total = seq_matrix.shape[0]
    n_val   = max(1, int(N_total * VAL_FRAC))
    n_train = N_total - n_val

    print(f"Train: {n_train:,} | Val: {n_val:,}")
    print(f"Train: {trigger_dates_arr[0]} → {trigger_dates_arr[n_train-1]}")
    print(f"Val:   {trigger_dates_arr[n_train]} → {trigger_dates_arr[-1]}")

    train_hard_neg = None   # exclusion done in collate_fn — not stored per trigger

    all_arrays = dict(
        seq_matrix=seq_matrix, delta_t_matrix=delta_t_mat,
        trigger_token=trigger_token, seq_lengths=seq_lengths,
        lab_t30=lab_t30, lab_t60=lab_t60, lab_t180=lab_t180,
        is_t30=is_t30_arr, is_t60=is_t60_arr, is_t180=is_t180_arr,
        member_ids=member_ids_arr, trigger_dates=trigger_dates_arr,
        trigger_dxs=trigger_dxs_arr, segments=segments_arr,
        from_provider_ids=from_provider_ids,
    )

    train_data = {}; val_data = {}
    print("\nSaving arrays...")
    for k, arr in all_arrays.items():
        train_data[k] = arr[:n_train]
        val_data[k]   = arr[n_train:]
        np.save(f"{CACHE_DIR}/train_{k}.npy", train_data[k])
        np.save(f"{CACHE_DIR}/val_{k}.npy",   val_data[k])
        print(f"  {k}: train={train_data[k].shape} val={val_data[k].shape}")

    # Save from_provider arrays (used by collate_fn)
    np.save(f"{CACHE_DIR}/train_from_provider_ids.npy", all_arrays["from_provider_ids"][:n_train])
    np.save(f"{CACHE_DIR}/val_from_provider_ids.npy",   all_arrays["from_provider_ids"][n_train:])
    with open(f"{CACHE_DIR}/from_provider_to_cands.pkl", "wb") as f:
        pickle.dump(from_provider_to_cands, f)
    print(f"  from_provider_to_cands: {len(from_provider_to_cands):,} unique from_providers")

    del all_arrays
    gc.collect()

print(f"\nSection 4 done — {time.time()-t0:.1f}s")


# ── SUMMARY ───────────────────────────────────────────────────────────────────
display(Markdown("---\n## Summary"))
N_train = train_data["seq_matrix"].shape[0]
N_val   = val_data["seq_matrix"].shape[0]

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

**BQ did:** label aggregation (ARRAY_AGG), hard neg lookup, sequence capping
**Python did:** vocab encoding, numpy indexing, val split

Next: run NB_03 to build test dataset.
"""))
print("NB_02 complete")
