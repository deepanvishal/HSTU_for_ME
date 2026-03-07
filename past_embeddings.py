# ============================================================
# NOTEBOOK 1: TRAIN EMBEDDINGS USING SPARSE SVD
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from google.cloud import bigquery

os.makedirs('./embeddings', exist_ok=True)

client = bigquery.Client(project='anbc-hcb-dev')

# ============================================================
# STEP 1: LOAD EDGE TABLES
# ============================================================
def load_edges(table, col1, col2):
    df = client.query(f"""
        SELECT {col1}, {col2}, weight
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.{table}`
    """).to_dataframe()
    print(f"{table} — {len(df):,} edges")
    return df

df_provider  = load_edges('A870800_provider_edges',  'provider_1',  'provider_2')
df_specialty = load_edges('A870800_specialty_edges', 'specialty_1', 'specialty_2')
df_dx        = load_edges('A870800_dx_edges',        'dx_1',        'dx_2')
df_procedure = load_edges('A870800_procedure_edges', 'procedure_1', 'procedure_2')

# ============================================================
# STEP 2: SVD EMBEDDINGS
# ============================================================
def svd_embeddings(df, col1, col2, dim, name):
    print(f"\nTraining {name} embeddings ({dim}-dim via SVD)...")

    # build node index
    nodes = pd.unique(df[[col1, col2]].values.ravel())
    vocab = {node: idx for idx, node in enumerate(nodes)}
    n     = len(nodes)

    # build sparse adjacency matrix (symmetric)
    row = [vocab[v] for v in df[col1]]
    col = [vocab[v] for v in df[col2]]
    dat = df['weight'].values.astype(np.float32)

    # symmetric — add both directions
    row_full = np.concatenate([row, col])
    col_full = np.concatenate([col, row])
    dat_full = np.concatenate([dat, dat])

    A = csr_matrix((dat_full, (row_full, col_full)), shape=(n, n))

    print(f"  {name} — nodes: {n:,}  sparse matrix: {A.nnz:,} entries")

    # SVD — k = embedding dim
    k = min(dim, n - 1)
    U, S, Vt = svds(A, k=k)

    # scale by singular values
    matrix = (U * np.sqrt(S)).astype(np.float32)

    # save
    np.save(f'./embeddings/{name}_embeddings.npy', matrix)
    with open(f'./embeddings/{name}_vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    print(f"  saved — vocab: {len(vocab):,}  shape: {matrix.shape}")
    return vocab, matrix

provider_vocab,  provider_matrix  = svd_embeddings(df_provider,  'provider_1',  'provider_2',  32, 'provider');  del df_provider
specialty_vocab, specialty_matrix = svd_embeddings(df_specialty, 'specialty_1', 'specialty_2', 32, 'specialty'); del df_specialty
dx_vocab,        dx_matrix        = svd_embeddings(df_dx,        'dx_1',        'dx_2',        32, 'dx');        del df_dx
procedure_vocab, procedure_matrix = svd_embeddings(df_procedure, 'procedure_1', 'procedure_2', 32, 'procedure'); del df_procedure

# ============================================================
# STEP 3: SAVE UNK EMBEDDINGS
# ============================================================
print("\nSaving UNK embeddings...")
unk_embeddings = {
    'provider'  : provider_matrix.mean(axis=0)
    ,'specialty' : specialty_matrix.mean(axis=0)
    ,'dx'        : dx_matrix.mean(axis=0)
    ,'procedure' : procedure_matrix.mean(axis=0)
}
with open('./embeddings/unk_embeddings.pkl', 'wb') as f:
    pickle.dump(unk_embeddings, f)

print("UNK embeddings saved")

# ============================================================
# STEP 4: SAVE SPECIALTY LABEL VOCAB
# ============================================================
print("Saving specialty label vocab...")
df_spec = client.query("""
    SELECT specialty, idx
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_specialty_label_vocab`
    ORDER BY idx
""").to_dataframe()

specialty_label_vocab = dict(zip(df_spec['specialty'], df_spec['idx']))
with open('./embeddings/specialty_label_vocab.pkl', 'wb') as f:
    pickle.dump(specialty_label_vocab, f)

print(f"Specialty label vocab — {len(specialty_label_vocab)} specialties")

# ============================================================
# STEP 5: VERIFY
# ============================================================
print("\nVerifying saved files...")
files = [
    './embeddings/provider_embeddings.npy'
    ,'./embeddings/provider_vocab.pkl'
    ,'./embeddings/specialty_embeddings.npy'
    ,'./embeddings/specialty_vocab.pkl'
    ,'./embeddings/dx_embeddings.npy'
    ,'./embeddings/dx_vocab.pkl'
    ,'./embeddings/procedure_embeddings.npy'
    ,'./embeddings/procedure_vocab.pkl'
    ,'./embeddings/unk_embeddings.pkl'
    ,'./embeddings/specialty_label_vocab.pkl'
]
for f in files:
    exists = os.path.exists(f)
    size   = f"{os.path.getsize(f) / 1e6:.2f} MB" if exists else ""
    print(f"  {'ok' if exists else 'MISSING'}  {f}  {size}")

print("\nNotebook 1 complete")

# ============================================================
# STEP 6: UPLOAD EMBEDDING TABLES TO BIGQUERY
# One-time operation. Creates 4 tables used by notebook 2
# Step 3 query to replace Python embedding build entirely.
# Schema: (code STRING, e0 FLOAT64 ... e31 FLOAT64)
# ============================================================
import time
from google.cloud import bigquery_storage
from google.api_core.exceptions import NotFound

DATASET  = 'provider_ds_netconf_data_hcb_dev'
PROJECT  = 'anbc-hcb-dev'
EMB_COLS = [f'e{i}' for i in range(32)]

job_config             = bigquery.LoadJobConfig()
job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

def upload_embedding_table(name, vocab, matrix):
    t0      = time.time()
    # invert vocab: idx → code
    idx_to_code = {idx: code for code, idx in vocab.items()}
    codes       = [idx_to_code[i] for i in range(len(idx_to_code))]

    df           = pd.DataFrame(matrix, columns=EMB_COLS)
    df.insert(0, 'code', codes)
    df['code']   = df['code'].astype(str)

    table_id = f'{PROJECT}.{DATASET}.A870800_{name}_embeddings_bq'
    job      = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()

    print(f"  {table_id} — {len(df):,} rows  {time.time() - t0:.1f}s")
    del df

print("\nUploading embedding tables to BQ...")
upload_embedding_table('provider',  provider_vocab,  provider_matrix)
upload_embedding_table('specialty', specialty_vocab, specialty_matrix)
upload_embedding_table('dx',        dx_vocab,        dx_matrix)
upload_embedding_table('procedure', procedure_vocab, procedure_matrix)
print("Embedding tables uploaded — notebook 2 Step 3 can now use BQ query path")
