# ============================================================
# NB_01 — build_provider_vocab.py
# Purpose : Build all vocab files from full population (not sample)
#           Shared by NB_02, NB_03, NB_04, NB_05, NB_06, NB_07
#           Run once per sample size — vocab is sample-independent
# Sources : A870800_gen_rec_provider_vocab  (provider vocab)
#           A870800_gen_rec_visits           (specialty, dx, provider→specialty map)
# Output  : ./cache_provider_{SAMPLE}/
#               provider_vocab.pkl           {srv_prvdr_id -> int_id}
#               idx_to_provider.pkl          {int_id -> srv_prvdr_id}
#               specialty_vocab.pkl          {specialty_ctg_cd -> int_id}
#               dx_vocab.pkl                 {dx_clean -> int_id}
#               provider_specialty_map.pkl   {srv_prvdr_id -> most_common_specialty}
# Notes:
#   PAD = 0, UNK = 1 reserved in all vocabs
#   provider_vocab covers top80 providers only (is_top80 = TRUE)
#   specialty_vocab and dx_vocab built from FULL population — not just top80
#   provider_specialty_map covers all providers — used for hard neg mining
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
    print(f"  Sample:\n{df.head(sample_n).to_string(index=False)}")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
SAMPLE    = "5pct"                 # change to 1pct / 10pct as needed
PAD_IDX   = 0
UNK_IDX   = 1

DS        = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
CACHE_DIR = f"./cache_provider_{SAMPLE}"
os.makedirs(CACHE_DIR, exist_ok=True)

client = bigquery.Client(project="anbc-hcb-dev")

print(f"Config — sample={SAMPLE}, cache={CACHE_DIR}")
display(Markdown(f"""
## Config
| Parameter | Value |
|---|---|
| Sample | {SAMPLE} |
| Cache dir | {CACHE_DIR} |
| PAD idx | {PAD_IDX} |
| UNK idx | {UNK_IDX} |
"""))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PROVIDER VOCAB
# Pull top80 providers from A870800_gen_rec_provider_vocab
# Build provider_vocab.pkl and idx_to_provider.pkl
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 1 — Provider Vocab"))

PROV_CACHE = f"{CACHE_DIR}/provider_vocab.pkl"
IDX_CACHE  = f"{CACHE_DIR}/idx_to_provider.pkl"

if os.path.exists(PROV_CACHE) and os.path.exists(IDX_CACHE):
    print("Loading provider vocab from cache...")
    with open(PROV_CACHE, "rb") as f:
        provider_vocab = pickle.load(f)
    with open(IDX_CACHE, "rb") as f:
        idx_to_provider = pickle.load(f)
    print(f"Loaded provider_vocab={len(provider_vocab):,} | idx_to_provider={len(idx_to_provider):,}")
else:
    print("Reading top80 providers from BQ...")
    prov_df = client.query(f"""
        SELECT
            srv_prvdr_id
            ,provider_rank
            ,total_transition_count
        FROM `{DS}.A870800_gen_rec_provider_vocab`
        WHERE is_top80 = TRUE
        ORDER BY provider_rank
    """).to_dataframe()

    print(f"BQ returned {len(prov_df):,} providers")
    qa_df(prov_df, "provider vocab raw", check_cols=["srv_prvdr_id"])

    # Build vocab: PAD=0, UNK=1, providers start at 2
    # Ordered by provider_rank for consistency — highest-volume first
    provider_vocab  = {"PAD": PAD_IDX, "UNK": UNK_IDX}
    idx_to_provider = {PAD_IDX: "PAD", UNK_IDX: "UNK"}

    for i, row in enumerate(prov_df.itertuples(index=False), start=2):
        provider_vocab[row.srv_prvdr_id]  = i
        idx_to_provider[i] = row.srv_prvdr_id

    print(f"Provider vocab built — {len(provider_vocab):,} entries (incl PAD + UNK)")
    print(f"  Provider int_ids: 2 → {len(provider_vocab) - 1}")
    print(f"  Example: {prov_df.iloc[0].srv_prvdr_id} → {provider_vocab[prov_df.iloc[0].srv_prvdr_id]}")

    with open(PROV_CACHE, "wb") as f:
        pickle.dump(provider_vocab, f)
    with open(IDX_CACHE, "wb") as f:
        pickle.dump(idx_to_provider, f)
    print(f"Saved: {PROV_CACHE}")
    print(f"Saved: {IDX_CACHE}")

PROVIDER_VOCAB_SIZE = len(provider_vocab)
print(f"\nSection 1 done — PROVIDER_VOCAB_SIZE={PROVIDER_VOCAB_SIZE:,} | time={time.time()-t0:.1f}s")
display(Markdown(f"**Provider vocab:** {PROVIDER_VOCAB_SIZE:,} (incl PAD + UNK) | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SPECIALTY VOCAB
# Built from full population visits — NOT just sample or top80
# Ensures all specialties seen at test time have a known int_id
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 2 — Specialty Vocab"))

SPEC_CACHE = f"{CACHE_DIR}/specialty_vocab.pkl"

if os.path.exists(SPEC_CACHE):
    print("Loading specialty vocab from cache...")
    with open(SPEC_CACHE, "rb") as f:
        specialty_vocab = pickle.load(f)
    print(f"Loaded specialty_vocab={len(specialty_vocab):,}")
else:
    print("Reading distinct specialties from BQ (full population)...")
    spec_df = client.query(f"""
        SELECT DISTINCT
            specialty_ctg_cd
        FROM `{DS}.A870800_gen_rec_visits`
        WHERE specialty_ctg_cd IS NOT NULL
          AND specialty_ctg_cd != ''
        ORDER BY specialty_ctg_cd
    """).to_dataframe()

    print(f"BQ returned {len(spec_df):,} distinct specialties")
    qa_df(spec_df, "specialty list", sample_n=5)

    # PAD=0, UNK=1, specialties start at 2
    specialty_vocab = {"PAD": PAD_IDX, "UNK": UNK_IDX}
    for i, row in enumerate(spec_df.itertuples(index=False), start=2):
        specialty_vocab[row.specialty_ctg_cd] = i

    print(f"Specialty vocab built — {len(specialty_vocab):,} entries")
    with open(SPEC_CACHE, "wb") as f:
        pickle.dump(specialty_vocab, f)
    print(f"Saved: {SPEC_CACHE}")

SPEC_VOCAB_SIZE = len(specialty_vocab)
print(f"\nSection 2 done — SPEC_VOCAB_SIZE={SPEC_VOCAB_SIZE:,} | time={time.time()-t0:.1f}s")
display(Markdown(f"**Specialty vocab:** {SPEC_VOCAB_SIZE:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DX VOCAB
# Built from full population visits using dx_clean (no dots)
# trigger_dx_clean in NB_02/03 must be looked up here — not trigger_dx (raw)
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 3 — DX Vocab"))

DX_CACHE = f"{CACHE_DIR}/dx_vocab.pkl"

if os.path.exists(DX_CACHE):
    print("Loading dx vocab from cache...")
    with open(DX_CACHE, "rb") as f:
        dx_vocab = pickle.load(f)
    print(f"Loaded dx_vocab={len(dx_vocab):,}")
else:
    print("Reading distinct dx_clean from BQ (full population)...")
    dx_df = client.query(f"""
        SELECT DISTINCT
            dx_clean
        FROM `{DS}.A870800_gen_rec_visits`
        WHERE dx_clean IS NOT NULL
          AND dx_clean != ''
        ORDER BY dx_clean
    """).to_dataframe()

    print(f"BQ returned {len(dx_df):,} distinct dx codes")
    qa_df(dx_df, "dx list", sample_n=5)

    # PAD=0, UNK=1, dx codes start at 2
    dx_vocab = {"PAD": PAD_IDX, "UNK": UNK_IDX}
    for i, row in enumerate(dx_df.itertuples(index=False), start=2):
        dx_vocab[row.dx_clean] = i

    print(f"DX vocab built — {len(dx_vocab):,} entries")
    print(f"  Note: keyed on dx_clean (no dots). NB_02/03 use trigger_dx_clean.")
    with open(DX_CACHE, "wb") as f:
        pickle.dump(dx_vocab, f)
    print(f"Saved: {DX_CACHE}")

DX_VOCAB_SIZE = len(dx_vocab)
print(f"\nSection 3 done — DX_VOCAB_SIZE={DX_VOCAB_SIZE:,} | time={time.time()-t0:.1f}s")
display(Markdown(f"**DX vocab:** {DX_VOCAB_SIZE:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PROVIDER SPECIALTY MAP
# Most common specialty per srv_prvdr_id across full population
# Used in NB_02 hard negative mining:
#   transitions table has no specialty column for to_provider
#   need specialty of to_provider to find same-specialty hard negatives
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 4 — Provider Specialty Map"))

PROV_SPEC_CACHE = f"{CACHE_DIR}/provider_specialty_map.pkl"

if os.path.exists(PROV_SPEC_CACHE):
    print("Loading provider_specialty_map from cache...")
    with open(PROV_SPEC_CACHE, "rb") as f:
        provider_specialty_map = pickle.load(f)
    print(f"Loaded provider_specialty_map={len(provider_specialty_map):,} providers")
else:
    print("Reading most common specialty per provider from BQ (all providers)...")
    prov_spec_df = client.query(f"""
        WITH specialty_counts AS (
            SELECT
                srv_prvdr_id
                ,specialty_ctg_cd
                ,COUNT(*)                                AS visit_count
            FROM `{DS}.A870800_gen_rec_visits`
            WHERE srv_prvdr_id      IS NOT NULL
              AND specialty_ctg_cd  IS NOT NULL
              AND specialty_ctg_cd  != ''
            GROUP BY srv_prvdr_id, specialty_ctg_cd
        )
        SELECT
            srv_prvdr_id
            ,specialty_ctg_cd                            AS primary_specialty
        FROM specialty_counts
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY srv_prvdr_id
            ORDER BY visit_count DESC
        ) = 1
    """).to_dataframe()

    print(f"BQ returned {len(prov_spec_df):,} providers with specialty")
    qa_df(prov_spec_df, "provider_specialty_map", check_cols=["primary_specialty"], sample_n=5)

    # Build dict: {srv_prvdr_id -> specialty_ctg_cd}
    provider_specialty_map = dict(
        zip(prov_spec_df["srv_prvdr_id"], prov_spec_df["primary_specialty"])
    )

    # QA: check coverage for top80 providers
    top80_providers = set(k for k in provider_vocab if k not in ("PAD", "UNK"))
    covered = sum(1 for p in top80_providers if p in provider_specialty_map)
    print(f"\nTop80 provider coverage: {covered:,}/{len(top80_providers):,} "
          f"({covered/len(top80_providers)*100:.1f}%)")
    if covered < len(top80_providers) * 0.95:
        print("  WARNING: <95% of top80 providers have a specialty map entry")

    with open(PROV_SPEC_CACHE, "wb") as f:
        pickle.dump(provider_specialty_map, f)
    print(f"Saved: {PROV_SPEC_CACHE}")

print(f"\nSection 4 done — {len(provider_specialty_map):,} providers mapped | time={time.time()-t0:.1f}s")
display(Markdown(f"**Provider specialty map:** {len(provider_specialty_map):,} providers | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("---\n## Summary"))

print(f"\nAll vocab files saved to: {CACHE_DIR}/")
print(f"  provider_vocab.pkl        : {PROVIDER_VOCAB_SIZE:,} entries (PAD=0, UNK=1, providers start at 2)")
print(f"  idx_to_provider.pkl       : {PROVIDER_VOCAB_SIZE:,} entries")
print(f"  specialty_vocab.pkl       : {SPEC_VOCAB_SIZE:,} entries")
print(f"  dx_vocab.pkl              : {DX_VOCAB_SIZE:,} entries (keyed on dx_clean — no dots)")
print(f"  provider_specialty_map.pkl: {len(provider_specialty_map):,} providers")

display(Markdown(f"""
## Vocab Summary
| File | Entries | Notes |
|---|---|---|
| provider_vocab.pkl | {PROVIDER_VOCAB_SIZE:,} | top80 only, PAD=0 UNK=1 |
| idx_to_provider.pkl | {PROVIDER_VOCAB_SIZE:,} | reverse lookup for scoring |
| specialty_vocab.pkl | {SPEC_VOCAB_SIZE:,} | full population |
| dx_vocab.pkl | {DX_VOCAB_SIZE:,} | dx_clean, full population |
| provider_specialty_map.pkl | {len(provider_specialty_map):,} | all providers, for hard neg mining |

Next: run NB_02 to build train dataset.
"""))

print("NB_01 complete")
