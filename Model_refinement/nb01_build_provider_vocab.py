# ============================================================
# NB_01 — build_provider_vocab.py  (vectorized)
# Purpose : Build all vocab files from full population
#           Shared by NB_02, NB_03, NB_04, NB_05, NB_06, NB_07
#           Run once — vocab is sample-independent
# Sources : A870800_gen_rec_provider_vocab
#           A870800_gen_rec_visits
# Output  : ./cache_provider_{SAMPLE}/
#               provider_vocab.pkl        {srv_prvdr_id -> int_id}
#               idx_to_provider.pkl       {int_id -> srv_prvdr_id}
#               specialty_vocab.pkl       {specialty_ctg_cd -> int_id}
#               dx_vocab.pkl              {dx_clean -> int_id}
#               provider_specialty_map.pkl {srv_prvdr_id -> primary_specialty}
# Vectorization:
#   All vocab dicts built with dict(zip()) — no itertuples loops
#   provider_specialty_map entirely in BQ (QUALIFY), Python just assembles dict
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
    print(f"  Sample:\n{df.head(sample_n).to_string(index=False)}")
    print(f"{'='*60}\n")


# ── CONFIG ────────────────────────────────────────────────────────────────────
SAMPLE    = "5pct"
PAD_IDX   = 0
UNK_IDX   = 1
DS        = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
CACHE_DIR = f"./cache_provider_{SAMPLE}"
os.makedirs(CACHE_DIR, exist_ok=True)
client    = bigquery.Client(project="anbc-hcb-dev")

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
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 1 — Provider Vocab"))

PROV_CACHE = f"{CACHE_DIR}/provider_vocab.pkl"
IDX_CACHE  = f"{CACHE_DIR}/idx_to_provider.pkl"

if os.path.exists(PROV_CACHE) and os.path.exists(IDX_CACHE):
    print("Loading from cache...")
    with open(PROV_CACHE, "rb") as f: provider_vocab  = pickle.load(f)
    with open(IDX_CACHE,  "rb") as f: idx_to_provider = pickle.load(f)
else:
    print("Reading top80 providers from BQ...")
    prov_df = client.query(f"""
        SELECT srv_prvdr_id, provider_rank
        FROM `{DS}.A870800_gen_rec_provider_vocab`
        WHERE is_top80 = TRUE
        ORDER BY provider_rank
    """).to_dataframe()
    qa_df(prov_df, "provider vocab raw")

    # Vectorized: dict(zip()) — no itertuples
    ids            = range(2, len(prov_df) + 2)       # PAD=0, UNK=1, providers start at 2
    provider_vocab  = {"PAD": PAD_IDX, "UNK": UNK_IDX,
                       **dict(zip(prov_df["srv_prvdr_id"], ids))}
    idx_to_provider = {PAD_IDX: "PAD", UNK_IDX: "UNK",
                       **dict(zip(ids, prov_df["srv_prvdr_id"]))}

    with open(PROV_CACHE, "wb") as f: pickle.dump(provider_vocab,  f)
    with open(IDX_CACHE,  "wb") as f: pickle.dump(idx_to_provider, f)
    print(f"Saved provider_vocab ({len(provider_vocab):,})")

PROVIDER_VOCAB_SIZE = len(provider_vocab)
print(f"Section 1 done — PROVIDER_VOCAB_SIZE={PROVIDER_VOCAB_SIZE:,} | {time.time()-t0:.1f}s")
display(Markdown(f"**Provider vocab:** {PROVIDER_VOCAB_SIZE:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SPECIALTY VOCAB
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 2 — Specialty Vocab"))

SPEC_CACHE = f"{CACHE_DIR}/specialty_vocab.pkl"

if os.path.exists(SPEC_CACHE):
    print("Loading from cache...")
    with open(SPEC_CACHE, "rb") as f: specialty_vocab = pickle.load(f)
else:
    print("Reading distinct specialties from BQ (full population)...")
    spec_df = client.query(f"""
        SELECT DISTINCT specialty_ctg_cd
        FROM `{DS}.A870800_gen_rec_visits`
        WHERE specialty_ctg_cd IS NOT NULL AND specialty_ctg_cd != ''
        ORDER BY specialty_ctg_cd
    """).to_dataframe()
    qa_df(spec_df, "specialty list", sample_n=5)

    ids             = range(2, len(spec_df) + 2)
    specialty_vocab = {"PAD": PAD_IDX, "UNK": UNK_IDX,
                       **dict(zip(spec_df["specialty_ctg_cd"], ids))}

    with open(SPEC_CACHE, "wb") as f: pickle.dump(specialty_vocab, f)
    print(f"Saved specialty_vocab ({len(specialty_vocab):,})")

SPEC_VOCAB_SIZE = len(specialty_vocab)
print(f"Section 2 done — SPEC_VOCAB_SIZE={SPEC_VOCAB_SIZE:,} | {time.time()-t0:.1f}s")
display(Markdown(f"**Specialty vocab:** {SPEC_VOCAB_SIZE:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DX VOCAB  (keyed on dx_clean — no dots)
# NB_02/03 must use trigger_dx_clean not trigger_dx for lookup
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 3 — DX Vocab"))

DX_CACHE = f"{CACHE_DIR}/dx_vocab.pkl"

if os.path.exists(DX_CACHE):
    print("Loading from cache...")
    with open(DX_CACHE, "rb") as f: dx_vocab = pickle.load(f)
else:
    print("Reading distinct dx_clean from BQ (full population)...")
    dx_df = client.query(f"""
        SELECT DISTINCT dx_clean
        FROM `{DS}.A870800_gen_rec_visits`
        WHERE dx_clean IS NOT NULL AND dx_clean != ''
        ORDER BY dx_clean
    """).to_dataframe()
    qa_df(dx_df, "dx list", sample_n=5)

    ids      = range(2, len(dx_df) + 2)
    dx_vocab = {"PAD": PAD_IDX, "UNK": UNK_IDX,
                **dict(zip(dx_df["dx_clean"], ids))}

    with open(DX_CACHE, "wb") as f: pickle.dump(dx_vocab, f)
    print(f"Saved dx_vocab ({len(dx_vocab):,})")
    print(f"  Keyed on dx_clean (no dots). NB_02/03 use trigger_dx_clean.")

DX_VOCAB_SIZE = len(dx_vocab)
print(f"Section 3 done — DX_VOCAB_SIZE={DX_VOCAB_SIZE:,} | {time.time()-t0:.1f}s")
display(Markdown(f"**DX vocab:** {DX_VOCAB_SIZE:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PROVIDER SPECIALTY MAP
# Most common specialty per provider — built entirely in BQ
# Python only assembles result via dict(zip()) — no Python-side groupby
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 4 — Provider Specialty Map"))

PROV_SPEC_CACHE = f"{CACHE_DIR}/provider_specialty_map.pkl"

if os.path.exists(PROV_SPEC_CACHE):
    print("Loading from cache...")
    with open(PROV_SPEC_CACHE, "rb") as f: provider_specialty_map = pickle.load(f)
else:
    print("Reading primary specialty per provider from BQ...")
    prov_spec_df = client.query(f"""
        WITH specialty_counts AS (
            SELECT
                srv_prvdr_id
                ,specialty_ctg_cd
                ,COUNT(*)                                AS visit_count
            FROM `{DS}.A870800_gen_rec_visits`
            WHERE srv_prvdr_id     IS NOT NULL
              AND specialty_ctg_cd IS NOT NULL
              AND specialty_ctg_cd != ''
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

    qa_df(prov_spec_df, "provider_specialty_map", check_cols=["primary_specialty"], sample_n=5)

    # Vectorized: dict(zip()) — no itertuples
    provider_specialty_map = dict(
        zip(prov_spec_df["srv_prvdr_id"], prov_spec_df["primary_specialty"])
    )

    # QA: top80 coverage
    top80_set = set(k for k in provider_vocab if k not in ("PAD", "UNK"))
    covered   = sum(1 for p in top80_set if p in provider_specialty_map)
    pct       = covered / len(top80_set) * 100
    print(f"Top80 coverage: {covered:,}/{len(top80_set):,} ({pct:.1f}%)")
    if pct < 95:
        print("  WARNING: <95% coverage")

    with open(PROV_SPEC_CACHE, "wb") as f: pickle.dump(provider_specialty_map, f)
    print(f"Saved provider_specialty_map ({len(provider_specialty_map):,})")

print(f"Section 4 done — {len(provider_specialty_map):,} providers | {time.time()-t0:.1f}s")
display(Markdown(f"**Provider specialty map:** {len(provider_specialty_map):,} | **Time:** {time.time()-t0:.1f}s"))


# ── SUMMARY ───────────────────────────────────────────────────────────────────
display(Markdown(f"""
---
## Vocab Summary
| File | Entries | Notes |
|---|---|---|
| provider_vocab.pkl | {PROVIDER_VOCAB_SIZE:,} | top80 only, PAD=0 UNK=1 |
| idx_to_provider.pkl | {PROVIDER_VOCAB_SIZE:,} | reverse lookup for scoring |
| specialty_vocab.pkl | {SPEC_VOCAB_SIZE:,} | full population |
| dx_vocab.pkl | {DX_VOCAB_SIZE:,} | dx_clean (no dots), full population |
| provider_specialty_map.pkl | {len(provider_specialty_map):,} | all providers |

Next: run NB_02 to build train dataset.
"""))
print("NB_01 complete")
