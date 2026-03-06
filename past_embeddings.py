# ============================================================
# NOTEBOOK 1: BUILD GRAPHS + TRAIN EMBEDDINGS
# ============================================================

import os
import pickle
import itertools
import numpy as np
import networkx as nx
import pandas as pd
from node2vec import Node2Vec
from multiprocessing import Pool
from google.cloud import bigquery

os.makedirs('./embeddings', exist_ok=True)

# ============================================================
# STEP 1: STREAM DATA FROM BIGQUERY — NO FULL LOAD
# ============================================================
client = bigquery.Client(project='anbc-hcb-dev')

sample_query = """
WITH random_sample AS (
    SELECT DISTINCT member_id
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_sequence`
    WHERE RAND() < 0.1
)
SELECT
    s.provider_ids
    ,s.specialty_codes
    ,s.dx_list
    ,s.procedure_codes
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_sequence` s
INNER JOIN random_sample r ON s.member_id = r.member_id
"""

print("Streaming data and building edge lists...")

provider_edges  = []
specialty_edges = []
dx_edges        = []
procedure_edges = []

def get_combinations(lst):
    if not lst or len(lst) < 2:
        return []
    return [tuple(sorted([str(a), str(b)])) for a, b in itertools.combinations(lst, 2)]

CHUNK_SIZE = 50_000
rows_processed = 0

job = client.query(sample_query)

for page in job.result(page_size=CHUNK_SIZE).pages:
    rows = list(page)
    if not rows:
        continue

    for row in rows:
        provider_edges.extend(get_combinations(row.provider_ids   or []))
        specialty_edges.extend(get_combinations(row.specialty_codes or []))
        dx_edges.extend(get_combinations(row.dx_list             or []))
        procedure_edges.extend(get_combinations(row.procedure_codes or []))

    rows_processed += len(rows)
    print(f"Rows processed: {rows_processed:,}", end='\r')

print(f"\nStreaming complete — {rows_processed:,} rows")

# ============================================================
# STEP 2: COUNT EDGES USING PANDAS — VECTORIZED
# ============================================================
def build_nx_graph(edge_list, name):
    print(f"Building {name} graph...")
    if not edge_list:
        print(f"  No edges for {name}")
        return nx.Graph()

    df = pd.DataFrame(edge_list, columns=['n1', 'n2'])
    df = df.groupby(['n1', 'n2']).size().reset_index(name='weight')

    G = nx.from_pandas_edgelist(df, 'n1', 'n2', edge_attr='weight')
    del df
    print(f"  {name} — nodes: {G.number_of_nodes():,}  edges: {G.number_of_edges():,}")
    return G

G_provider  = build_nx_graph(provider_edges,  'provider');  del provider_edges
G_specialty = build_nx_graph(specialty_edges, 'specialty'); del specialty_edges
G_dx        = build_nx_graph(dx_edges,        'dx');        del dx_edges
G_procedure = build_nx_graph(procedure_edges, 'procedure'); del procedure_edges

# ============================================================
# STEP 3: TRAIN NODE2VEC IN PARALLEL
# ============================================================
def train_and_save(args):
    G, dim, name = args
    print(f"Training {name} embeddings ({dim}-dim)...")

    node2vec = Node2Vec(
        G
        ,dimensions  = dim
        ,walk_length = 30
        ,num_walks   = 100
        ,workers     = 4
        ,weight_key  = 'weight'
        ,quiet       = True
    )
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    nodes  = model.wv.index_to_key
    vocab  = {node: idx for idx, node in enumerate(nodes)}
    matrix = np.array([model.wv[node] for node in nodes], dtype=np.float32)

    np.save(f'./embeddings/{name}_embeddings.npy', matrix)
    with open(f'./embeddings/{name}_vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    print(f"  {name} saved — vocab: {len(vocab):,}  shape: {matrix.shape}")
    return name

graph_configs = [
    (G_provider,  32, 'provider')
    ,(G_specialty, 32, 'specialty')
    ,(G_dx,        32, 'dx')
    ,(G_procedure, 32, 'procedure')
]

# node2vec uses workers internally so run sequentially
# parallel here would cause resource contention
for config in graph_configs:
    train_and_save(config)

print("\nAll embeddings saved to ./embeddings/")
print("Notebook 1 complete")
