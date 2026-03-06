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
    print(f"Loading {table}...")
    df = client.query(f"""
        SELECT {col1}, {col2}, weight
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.{table}`
    """).to_dataframe(create_bqstorage_client=True)
    print(f"  {len(df):,} edges")
    return df

df_provider  = load_edges('A870800_provider_edges',  'provider_1',  'provider_2')
df_specialty = load_edges('A870800_specialty_edges', 'specialty_1', 'specialty_2')
df_dx        = load_edges('A870800_dx_edges',        'dx_1',        'dx_2')
df_procedure = load_edges('A870800_procedure_edges', 'procedure_1', 'procedure_2')

# ============================================================
# STEP 2: SVD EMBEDDINGS
# ============================================================
def svd_embeddings(df, col1, col2, dim, name):
    print(f"\nTraining {name} embeddings ({dim}-dim)...")

    # build node index
    nodes = pd.unique(df[[col1, col2]].values.ravel())
    vocab = {node: idx for idx, node in enumerate(nodes)}
    n     = len(nodes)

    # build sparse upper triangle then symmetrize — avoids doubling arrays
    row     = np.array([vocab[v] for v in df[col1]], dtype=np.int32)
    col     = np.array([vocab[v] for v in df[col2]], dtype=np.int32)
    dat     = df['weight'].values.astype(np.float32)

    A_upper = csr_matrix((dat, (row, col)), shape=(n, n), dtype=np.float32)
    A       = (A_upper + A_upper.T).astype(np.float32)

    print(f"  nodes: {n:,}  nnz: {A.nnz:,}")

    # truncated SVD
    k        = min(dim, n - 1)
    U, S, Vt = svds(A, k=k)

    # scale by singular values
    matrix = (U * np.sqrt(S)).astype(np.float32)

    np.save(f'./embeddings/{name}_embeddings.npy', matrix)
    with open(f'./embeddings/{name}_vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    print(f"  saved — shape: {matrix.shape}")
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
# STEP 4: BUILD SPECIALTY LABEL VOCAB FROM LABEL TABLE
# ============================================================
print("\nBuilding specialty label vocab...")
df_spec = client.query("""
    SELECT DISTINCT specialty
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_label`
    CROSS JOIN UNNEST(specialties_180) AS specialty
    WHERE specialty IS NOT NULL
    ORDER BY specialty
""").to_dataframe(create_bqstorage_client=True)

specialty_label_vocab = {s: i for i, s in enumerate(df_spec['specialty'])}
with open('./embeddings/specialty_label_vocab.pkl', 'wb') as f:
    pickle.dump(specialty_label_vocab, f)
print(f"Specialty label vocab — {len(specialty_label_vocab):,} specialties")

# ============================================================
# STEP 5: BUILD PROVIDER LABEL VOCAB FROM LABEL TABLE
# ============================================================
print("\nBuilding provider label vocab...")
df_prov = client.query("""
    SELECT DISTINCT provider
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_label`
    CROSS JOIN UNNEST(providers_180) AS provider
    WHERE provider IS NOT NULL
    ORDER BY provider
""").to_dataframe(create_bqstorage_client=True)

provider_label_vocab = {p: i for i, p in enumerate(df_prov['provider'])}
with open('./embeddings/provider_label_vocab.pkl', 'wb') as f:
    pickle.dump(provider_label_vocab, f)
print(f"Provider label vocab — {len(provider_label_vocab):,} providers")

# ============================================================
# STEP 6: BUILD DX LABEL VOCAB FROM LABEL TABLE
# ============================================================
print("\nBuilding dx label vocab...")
df_dx_lab = client.query("""
    SELECT DISTINCT dx
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_label`
    CROSS JOIN UNNEST(dx_180) AS dx
    WHERE dx IS NOT NULL
    ORDER BY dx
""").to_dataframe(create_bqstorage_client=True)

dx_label_vocab = {d: i for i, d in enumerate(df_dx_lab['dx'])}
with open('./embeddings/dx_label_vocab.pkl', 'wb') as f:
    pickle.dump(dx_label_vocab, f)
print(f"DX label vocab — {len(dx_label_vocab):,} dx codes")

# ============================================================
# STEP 7: VERIFY
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
    ,'./embeddings/provider_label_vocab.pkl'
    ,'./embeddings/dx_label_vocab.pkl'
]
for f in files:
    exists = os.path.exists(f)
    size   = f"{os.path.getsize(f) / 1e6:.2f} MB" if exists else ""
    print(f"  {'ok' if exists else 'MISSING'}  {f}  {size}")

print("\nNotebook 1 complete")
