# ============================================================
# NOTEBOOK 1: BUILD GRAPHS + TRAIN EMBEDDINGS (GPU)
# ============================================================

import os
import pickle
import itertools
import numpy as np
import cudf
import cugraph
from cuml.feature_extraction.text import HashingVectorizer
from gensim.models import Word2Vec
from collections import defaultdict
from google.cloud import bigquery

os.makedirs('./embeddings', exist_ok=True)

# ============================================================
# STEP 1: STREAM DATA + BUILD EDGE DICTS
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

provider_edges  = defaultdict(int)
dx_edges        = defaultdict(int)
specialty_edges = defaultdict(int)
procedure_edges = defaultdict(int)

def update_edges(edges, lst):
    if not lst or len(lst) < 2:
        return
    for a, b in itertools.combinations(lst, 2):
        key = (min(str(a), str(b)), max(str(a), str(b)))
        edges[key] += 1

rows_processed = 0
for page in client.query(sample_query).result(page_size=50_000).pages:
    rows = list(page)
    for row in rows:
        update_edges(provider_edges,  row.provider_ids    or [])
        update_edges(specialty_edges, row.specialty_codes or [])
        update_edges(dx_edges,        row.dx_list         or [])
        update_edges(procedure_edges, row.procedure_codes or [])
    rows_processed += len(rows)
    print(f"Rows processed: {rows_processed:,}", end='\r')

print(f"\nEdge counting complete")
print(f"Provider edges:  {len(provider_edges):,}")
print(f"Specialty edges: {len(specialty_edges):,}")
print(f"DX edges:        {len(dx_edges):,}")
print(f"Procedure edges: {len(procedure_edges):,}")

# ============================================================
# STEP 2: BUILD cuGRAPH GRAPHS + RUN RANDOM WALKS
# ============================================================
def build_cugraph_and_walk(edge_dict, name, walk_length=30, num_walks=100):
    print(f"\nBuilding {name} graph on GPU...")

    # node index mapping
    nodes = sorted(set(n for edge in edge_dict.keys() for n in edge))
    node_vocab = {node: idx for idx, node in enumerate(nodes)}

    src     = [node_vocab[e[0]] for e in edge_dict.keys()]
    dst     = [node_vocab[e[1]] for e in edge_dict.keys()]
    weights = list(edge_dict.values())

    # build cuGraph
    gdf = cudf.DataFrame({'src': src, 'dst': dst, 'weight': weights})
    G   = cugraph.Graph()
    G.from_cudf_edgelist(gdf, source='src', destination='dst', edge_attr='weight')

    print(f"  {name} — nodes: {G.number_of_vertices():,}  edges: {G.number_of_edges():,}")

    # random walks on GPU
    print(f"  Running random walks...")
    start_vertices = cudf.Series(list(range(G.number_of_vertices())) * num_walks)
    walks, _ = cugraph.node2vec(G, start_vertices, walk_length, use_padding=True)

    # convert walks to sentences for Word2Vec
    walks_np  = walks.to_pandas().values.reshape(-1, walk_length)
    idx_to_node = {idx: node for node, idx in node_vocab.items()}
    sentences = [
        [idx_to_node[int(n)] for n in walk if n >= 0]
        for walk in walks_np
    ]

    return sentences, node_vocab, nodes

def train_word2vec_and_save(sentences, nodes, name, dim=32):
    print(f"  Training Word2Vec for {name}...")
    model = Word2Vec(
        sentences
        ,vector_size = dim
        ,window      = 10
        ,min_count   = 1
        ,workers     = 8
        ,epochs      = 5
    )

    vocab  = {node: idx for idx, node in enumerate(nodes)}
    matrix = np.array(
        [model.wv[node] if node in model.wv else np.zeros(dim) for node in nodes]
        ,dtype=np.float32
    )

    np.save(f'./embeddings/{name}_embeddings.npy', matrix)
    with open(f'./embeddings/{name}_vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    print(f"  {name} saved — vocab: {len(vocab):,}  shape: {matrix.shape}")

# ============================================================
# STEP 3: PROCESS ALL 4 GRAPHS
# ============================================================
for edge_dict, name, dim in [
    (provider_edges,  'provider',  32)
    ,(specialty_edges, 'specialty', 32)
    ,(dx_edges,        'dx',        32)
    ,(procedure_edges, 'procedure', 32)
]:
    sentences, node_vocab, nodes = build_cugraph_and_walk(edge_dict, name)
    train_word2vec_and_save(sentences, nodes, name, dim)
    del sentences

print("\nAll embeddings saved to ./embeddings/")
print("Notebook 1 complete")
```

---

Run and share output. Expect:
```
Rows processed: ~2.6M
Provider edges: X
DX edges: X
...
