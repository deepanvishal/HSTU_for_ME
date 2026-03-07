# ============================================================
# DATA PREP NOTEBOOK
# Builds full-population embedding aggregations in BQ,
# combines with labels, pulls to local npy + parquet cache.
# Run once for seq cache. Change TARGET for each label target.
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import bigquery

os.makedirs('./data', exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
TARGET   = 'specialty'   # 'specialty' | 'provider' | 'dx'

LABEL_COLS = {
    'specialty': ('specialties_30', 'specialties_60', 'specialties_180')
    ,'provider' : ('providers_30',   'providers_60',   'providers_180')
    ,'dx'       : ('dx_30',          'dx_60',          'dx_180')
}

assert TARGET in LABEL_COLS, f"TARGET must be one of {list(LABEL_COLS.keys())}"

COL_30, COL_60, COL_180 = LABEL_COLS[TARGET]

DS               = 'anbc-hcb-dev.provider_ds_netconf_data_hcb_dev'
SEQ_EMB_PATH     = './data/seq_emb_full.npy'
SEQ_META_PATH    = './data/seq_meta_full.parquet'
LABEL_CACHE_PATH = f'./data/labels_{TARGET}_full.parquet'
EMBEDDING_DIM    = 128
CHUNK_ROWS       = 1_000_000
NUM_RATINGS      = 16

SEQ_CACHED   = os.path.exists(SEQ_EMB_PATH) and os.path.exists(SEQ_META_PATH)
LABEL_CACHED = os.path.exists(LABEL_CACHE_PATH)

print(f"Target:       {TARGET}")
print(f"Seq cached:   {SEQ_CACHED}")
print(f"Label cached: {LABEL_CACHED}")

client = bigquery.Client(project='anbc-hcb-dev')

# ============================================================
# STEP 1: BUILD MODALITY AGG TABLES IN BQ (parallel)
# Submit all 4 jobs simultaneously — BQ runs them in parallel.
# Each does UNNEST + JOIN embedding table + AVG per visit.
# Skipped if seq cache already exists.
# ============================================================
if not SEQ_CACHED:
    t0 = time.time()
    print("\nStep 1: Submitting 4 modality agg jobs in parallel...")

    MODALITIES = [
        ('provider',  'A870800_full_provider_agg',  'provider_ids',    'A870800_provider_embeddings_bq',  'pe')
        ,('specialty', 'A870800_full_specialty_agg', 'specialty_codes', 'A870800_specialty_embeddings_bq', 'se')
        ,('dx',        'A870800_full_dx_agg',        'dx_list',         'A870800_dx_embeddings_bq',        'de')
        ,('procedure', 'A870800_full_procedure_agg', 'procedure_codes', 'A870800_procedure_embeddings_bq', 'pre')
    ]

    jobs = []
    for modality, out_table, arr_col, emb_table, prefix in MODALITIES:
        avg_cols = ', '.join([f'AVG(e.e{i}) AS {prefix}{i}' for i in range(32)])
        sql = f"""
            SELECT
                s.member_id
                ,s.visit_seq_num
                ,{avg_cols}
            FROM `{DS}.A870800_claims_gen_rec_visit_sequence` s
            WHERE s.recency_rank <= 20
            LEFT JOIN UNNEST(s.{arr_col}) AS code
            LEFT JOIN `{DS}.{emb_table}` e ON CAST(code AS STRING) = e.code
            GROUP BY s.member_id, s.visit_seq_num
        """
        cfg = bigquery.QueryJobConfig(
            destination       = f'{DS}.{out_table}'
            ,write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        job = client.query(sql, job_config=cfg)
        jobs.append((out_table, job))
        print(f"  Submitted: {out_table}")

    print("Waiting for all 4 jobs...")
    for out_table, job in jobs:
        job.result()
        print(f"  Done: {out_table}")

    print(f"Step 1 done — {time.time() - t0:.1f}s")

    # ============================================================
    # STEP 2: BUILD COMBINED TABLE IN BQ
    # Joins 4 agg tables + base sequence + labels.
    # No ORDER BY — unnecessary, Python handles ordering via cumsum.
    # ============================================================
    t0 = time.time()
    print("\nStep 2: Building combined table...")

    cpe  = ', '.join([f'COALESCE(pa.pe{i},   0.0) AS pe{i}'  for i in range(32)])
    cse  = ', '.join([f'COALESCE(sa.se{i},   0.0) AS se{i}'  for i in range(32)])
    cde  = ', '.join([f'COALESCE(da.de{i},   0.0) AS de{i}'  for i in range(32)])
    cpre = ', '.join([f'COALESCE(pra.pre{i}, 0.0) AS pre{i}' for i in range(32)])

    combined_sql = f"""
        SELECT
            s.member_id
            ,s.visit_seq_num
            ,LEAST(s.delta_t_bucket, {NUM_RATINGS - 1}) AS dt_bucket
            ,{cpe}
            ,{cse}
            ,{cde}
            ,{cpre}
            ,l.{COL_30}
            ,l.{COL_60}
            ,l.{COL_180}
        FROM `{DS}.A870800_claims_gen_rec_visit_sequence` s
        WHERE s.recency_rank <= 20
        LEFT JOIN `{DS}.A870800_full_provider_agg`  pa  ON s.member_id = pa.member_id  AND s.visit_seq_num = pa.visit_seq_num
        LEFT JOIN `{DS}.A870800_full_specialty_agg` sa  ON s.member_id = sa.member_id  AND s.visit_seq_num = sa.visit_seq_num
        LEFT JOIN `{DS}.A870800_full_dx_agg`        da  ON s.member_id = da.member_id  AND s.visit_seq_num = da.visit_seq_num
        LEFT JOIN `{DS}.A870800_full_procedure_agg` pra ON s.member_id = pra.member_id AND s.visit_seq_num = pra.visit_seq_num
        LEFT JOIN `{DS}.A870800_claims_gen_rec_label` l  ON s.member_id = l.member_id  AND s.visit_seq_num = l.visit_seq_num
    """

    cfg = bigquery.QueryJobConfig(
        destination       = f'{DS}.A870800_full_combined'
        ,write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
    )
    client.query(combined_sql, job_config=cfg).result()
    print(f"Step 2 done — {time.time() - t0:.1f}s")

    # ============================================================
    # STEP 3: PULL SEQ EMB + META TO LOCAL CACHE
    # memmap pre-allocated on disk — chunks written directly,
    # never loads full matrix into RAM.
    # Meta written incrementally via PyArrow ParquetWriter.
    # ============================================================
    t0 = time.time()
    print("\nStep 3: Pulling sequence embeddings + metadata...")

    total_rows = client.query(
        f"SELECT COUNT(*) AS n FROM `{DS}.A870800_full_combined`"
    ).to_dataframe().iloc[0]['n']
    print(f"Total rows: {total_rows:,}")

    emb_cols = (
        [f'pe{i}'  for i in range(32)]
        + [f'se{i}'  for i in range(32)]
        + [f'de{i}'  for i in range(32)]
        + [f'pre{i}' for i in range(32)]
    )

    emb_memmap = np.memmap(SEQ_EMB_PATH, dtype=np.float32, mode='w+', shape=(total_rows, EMBEDDING_DIM))
    row_offset = 0

    meta_schema = pa.schema([
        ('member_id',     pa.string())
        ,('visit_seq_num', pa.int32())
        ,('dt_bucket',     pa.int8())
        ,('visit_idx',     pa.int32())
    ])
    meta_writer = pq.ParquetWriter(SEQ_META_PATH, meta_schema)

    pull_sql  = f"SELECT member_id, visit_seq_num, dt_bucket, {', '.join(emb_cols)} FROM `{DS}.A870800_full_combined`"
    query_job = client.query(pull_sql)

    for chunk in query_job.result(page_size=CHUNK_ROWS).pages:
        df = chunk.to_dataframe()
        n  = len(df)

        emb_memmap[row_offset:row_offset + n] = df[emb_cols].values.astype(np.float32)

        meta_writer.write_table(pa.table({
            'member_id'     : df['member_id'].values
            ,'visit_seq_num': df['visit_seq_num'].values.astype(np.int32)
            ,'dt_bucket'    : df['dt_bucket'].values.astype(np.int8)
            ,'visit_idx'    : np.arange(row_offset, row_offset + n, dtype=np.int32)
        }))

        row_offset += n
        print(f"  {row_offset:,} / {total_rows:,}  {time.time() - t0:.1f}s")

    emb_memmap.flush()
    del emb_memmap
    meta_writer.close()

    print(f"Embeddings → {SEQ_EMB_PATH}")
    print(f"Metadata   → {SEQ_META_PATH}")
    print(f"Step 3 done — {time.time() - t0:.1f}s")

else:
    print("\nSeq cache exists — skipping Steps 1-3")

# ============================================================
# STEP 4: PULL LABELS TO LOCAL CACHE
# Separate from seq cache — one per target.
# Reads from A870800_full_combined which already exists.
# Written incrementally via ParquetWriter.
# ============================================================
if not LABEL_CACHED:
    t0 = time.time()
    print(f"\nStep 4: Pulling labels for target={TARGET}...")

    label_writer = pq.ParquetWriter(
        LABEL_CACHE_PATH
        ,pa.schema([
            ('member_id',     pa.string())
            ,('visit_seq_num', pa.int32())
            ,(COL_30,          pa.list_(pa.string()))
            ,(COL_60,          pa.list_(pa.string()))
            ,(COL_180,         pa.list_(pa.string()))
        ])
    )

    label_job = client.query(f"""
        SELECT member_id, visit_seq_num, {COL_30}, {COL_60}, {COL_180}
        FROM `{DS}.A870800_full_combined`
    """)

    for chunk in label_job.result(page_size=CHUNK_ROWS).pages:
        df = chunk.to_dataframe()
        label_writer.write_table(pa.table({
            'member_id'     : df['member_id'].values
            ,'visit_seq_num': df['visit_seq_num'].values.astype(np.int32)
            ,COL_30         : df[COL_30].tolist()
            ,COL_60         : df[COL_60].tolist()
            ,COL_180        : df[COL_180].tolist()
        }))

    label_writer.close()
    print(f"Labels → {LABEL_CACHE_PATH}")
    print(f"Step 4 done — {time.time() - t0:.1f}s")

else:
    print(f"\nLabel cache exists for {TARGET} — skipping Step 4")

print(f"\nData prep complete — target: {TARGET}")
print(f"  {SEQ_EMB_PATH}")
print(f"  {SEQ_META_PATH}")
print(f"  {LABEL_CACHE_PATH}")
print(f"\nTo build other targets: change TARGET and rerun.")
print(f"Steps 1-3 skipped automatically (seq cache reused).")
