# ============================================================
# NOTEBOOK 2: HSTU TRAINING
# ============================================================

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from google.cloud import bigquery

sys.path.insert(0, './generative-recommenders')

os.makedirs('./checkpoints', exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
SAMPLE_PCT    = 0.005
MAX_SEQ_LEN   = 20
BATCH_SIZE    = 128
EPOCHS        = 5
EMBEDDING_DIM = 128
NUM_BLOCKS    = 2
NUM_HEADS     = 4
LINEAR_DIM    = 128
ATTENTION_DIM = 64
DROPOUT_RATE  = 0.2
LR            = 1e-4
EVAL_K        = [3, 5, 10]
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {DEVICE}")
print(f"Sample: {SAMPLE_PCT*100}%")

# ============================================================
# UTILITY
# ============================================================
def to_list(val):
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return [str(x) for x in val.tolist()]
    if isinstance(val, list):
        return [str(x) for x in val]
    return []

# ============================================================
# STEP 1: LOAD EMBEDDINGS
# ============================================================
print("\nLoading embeddings...")
provider_matrix  = np.load('./embeddings/provider_embeddings.npy')
specialty_matrix = np.load('./embeddings/specialty_embeddings.npy')
dx_matrix        = np.load('./embeddings/dx_embeddings.npy')
procedure_matrix = np.load('./embeddings/procedure_embeddings.npy')

with open('./embeddings/provider_vocab.pkl',       'rb') as f: provider_vocab        = pickle.load(f)
with open('./embeddings/specialty_vocab.pkl',      'rb') as f: specialty_vocab       = pickle.load(f)
with open('./embeddings/dx_vocab.pkl',             'rb') as f: dx_vocab              = pickle.load(f)
with open('./embeddings/procedure_vocab.pkl',      'rb') as f: procedure_vocab       = pickle.load(f)
with open('./embeddings/unk_embeddings.pkl',       'rb') as f: unk_emb               = pickle.load(f)
with open('./embeddings/specialty_label_vocab.pkl','rb') as f: specialty_label_vocab = pickle.load(f)

NUM_SPECIALTIES = len(specialty_label_vocab)
print(f"Specialties: {NUM_SPECIALTIES}")
print(f"Provider vocab:  {len(provider_vocab):,}")
print(f"DX vocab:        {len(dx_vocab):,}")
print(f"Procedure vocab: {len(procedure_vocab):,}")

# ============================================================
# STEP 2: VISIT EMBEDDING FUNCTION
# ============================================================
def get_visit_embedding(providers, specialties, dxs, procedures):
    def lookup(lst, matrix, vocab, unk):
        lst = to_list(lst)
        if not lst:
            return unk.copy()
        vecs = [matrix[vocab[x]] if x in vocab else unk for x in lst]
        return np.mean(vecs, axis=0).astype(np.float32)

    p  = lookup(providers,   provider_matrix,  provider_vocab,  unk_emb['provider'])
    s  = lookup(specialties, specialty_matrix, specialty_vocab, unk_emb['specialty'])
    d  = lookup(dxs,         dx_matrix,        dx_vocab,        unk_emb['dx'])
    pr = lookup(procedures,  procedure_matrix, procedure_vocab, unk_emb['procedure'])

    return np.concatenate([p, s, d, pr]).astype(np.float32)  # 128-dim

# ============================================================
# STEP 3: LOAD DATA FROM BIGQUERY
# ============================================================
client = bigquery.Client(project='anbc-hcb-dev')

sequence_query = f"""
WITH sampled_members AS (
    SELECT DISTINCT member_id
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_sequence`
    WHERE RAND() < {SAMPLE_PCT}
)
SELECT
    s.member_id
    ,s.visit_seq_num
    ,s.visit_date
    ,s.delta_t_bucket
    ,s.provider_ids
    ,s.specialty_codes
    ,s.dx_list
    ,s.procedure_codes
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_sequence` s
INNER JOIN sampled_members m ON s.member_id = m.member_id
ORDER BY s.member_id, s.visit_seq_num
"""

label_query = f"""
WITH sampled_members AS (
    SELECT DISTINCT member_id
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_sequence`
    WHERE RAND() < {SAMPLE_PCT}
)
SELECT
    l.member_id
    ,l.visit_seq_num
    ,l.specialties_30
    ,l.specialties_60
    ,l.specialties_180
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_label` l
INNER JOIN sampled_members m ON l.member_id = m.member_id
ORDER BY l.member_id, l.visit_seq_num
"""

print("\nLoading sequences...")
member_sequences = defaultdict(list)
member_labels    = defaultdict(list)

CHUNK_SIZE = 50_000

df_seq = client.query(sequence_query).to_dataframe()
print(f"Sequence rows: {len(df_seq):,}")

for start in range(0, len(df_seq), CHUNK_SIZE):
    chunk = df_seq.iloc[start:start + CHUNK_SIZE]
    for row in chunk.itertuples():
        emb = get_visit_embedding(
            row.provider_ids
            ,row.specialty_codes
            ,row.dx_list
            ,row.procedure_codes
        )
        member_sequences[row.member_id].append({
            'visit_seq_num'  : row.visit_seq_num
            ,'visit_date'    : row.visit_date
            ,'delta_t_bucket': int(row.delta_t_bucket)
            ,'embedding'     : emb
        })
    del chunk
del df_seq

print("Loading labels...")
df_lab = client.query(label_query).to_dataframe()
print(f"Label rows: {len(df_lab):,}")

for start in range(0, len(df_lab), CHUNK_SIZE):
    chunk = df_lab.iloc[start:start + CHUNK_SIZE]
    for row in chunk.itertuples():
        member_labels[row.member_id].append({
            'visit_seq_num'   : row.visit_seq_num
            ,'specialties_30' : to_list(row.specialties_30)
            ,'specialties_60' : to_list(row.specialties_60)
            ,'specialties_180': to_list(row.specialties_180)
        })
    del chunk
del df_lab

print(f"Members loaded: {len(member_sequences):,}")

# ============================================================
# STEP 4: TRAIN / VAL / TEST SPLIT
# ============================================================
train_data = []
val_data   = []
test_data  = []

for member_id, visits in member_sequences.items():
    n = len(visits)
    if n < 4:
        continue

    train_cut = int(n * 0.75)
    val_cut   = int(n * 0.875)
    labels    = {l['visit_seq_num']: l for l in member_labels.get(member_id, [])}

    train_data.append((member_id, visits[:train_cut],  labels))
    val_data.append((member_id,   visits[:val_cut],    labels))
    test_data.append((member_id,  visits[:val_cut],    labels))

print(f"Train: {len(train_data):,}  Val: {len(val_data):,}  Test: {len(test_data):,}")

# ============================================================
# STEP 5: DATASET
# ============================================================
def make_label_vector(specs):
    vec = np.zeros(NUM_SPECIALTIES, dtype=np.float32)
    for s in to_list(specs):
        if s in specialty_label_vocab:
            vec[specialty_label_vocab[s]] = 1.0
    return vec

class VisitDataset(Dataset):
    def __init__(self, data, max_seq_len):
        self.samples = []
        for member_id, visits, labels in data:
            visits = visits[-max_seq_len:]
            n      = len(visits)

            embeddings = np.stack([v['embedding'] for v in visits])
            delta_t    = np.array([v['delta_t_bucket'] for v in visits], dtype=np.int64)

            pad = max_seq_len - n
            if pad > 0:
                embeddings = np.vstack([embeddings, np.zeros((pad, EMBEDDING_DIM), dtype=np.float32)])
                delta_t    = np.concatenate([delta_t, np.zeros(pad, dtype=np.int64)])

            last_seq_num = visits[-1]['visit_seq_num']
            label        = labels.get(last_seq_num)
            if label is None:
                continue

            self.samples.append({
                'embeddings': torch.tensor(embeddings, dtype=torch.float32)
                ,'delta_t'  : torch.tensor(delta_t,   dtype=torch.long)
                ,'length'   : torch.tensor(n,         dtype=torch.long)
                ,'label_30' : torch.tensor(make_label_vector(label['specialties_30']),  dtype=torch.float32)
                ,'label_60' : torch.tensor(make_label_vector(label['specialties_60']),  dtype=torch.float32)
                ,'label_180': torch.tensor(make_label_vector(label['specialties_180']), dtype=torch.float32)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

train_dataset = VisitDataset(train_data, MAX_SEQ_LEN)
val_dataset   = VisitDataset(val_data,   MAX_SEQ_LEN)
test_dataset  = VisitDataset(test_data,  MAX_SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train batches: {len(train_loader):,}  Val batches: {len(val_loader):,}  Test batches: {len(test_loader):,}")

# ============================================================
# STEP 6: MODEL
# ============================================================
import torch
import torch.nn as nn

# patch all fbgemm ops used by HSTU
def asynchronous_complete_cumsum(lengths):
    zero = torch.zeros(1, dtype=lengths.dtype, device=lengths.device)
    return torch.cat([zero, torch.cumsum(lengths, dim=0)])

def dense_to_jagged(dense, offsets):
    # dense: [B, N, D] → jagged: [sum_N, D]
    results = []
    for i in range(dense.size(0)):
        n = int(offsets[0][i+1] - offsets[0][i]) if isinstance(offsets, list) else int(offsets[i+1] - offsets[i])
        results.append(dense[i, :n])
    return torch.cat(results, dim=0), offsets

def jagged_to_padded_dense(jagged, offsets, max_lengths, padding_value=0.0):
    # jagged: [sum_N, D] → dense: [B, N, D]
    offsets = offsets[0] if isinstance(offsets, list) else offsets
    max_len = max_lengths[0] if isinstance(max_lengths, list) else max_lengths
    B       = len(offsets) - 1
    D       = jagged.size(-1)
    out     = torch.full((B, max_len, D), padding_value, dtype=jagged.dtype, device=jagged.device)
    for i in range(B):
        start = int(offsets[i])
        end   = int(offsets[i+1])
        n     = min(end - start, max_len)
        out[i, :n] = jagged[start:start+n]
    return out

torch.ops.fbgemm.asynchronous_complete_cumsum = asynchronous_complete_cumsum
torch.ops.fbgemm.dense_to_jagged             = dense_to_jagged
torch.ops.fbgemm.jagged_to_padded_dense      = jagged_to_padded_dense

from generative_recommenders.research.modeling.sequential.hstu import HSTU
from generative_recommenders.research.modeling.sequential.embedding_modules import LocalEmbeddingModule
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor
from generative_recommenders.research.modeling.sequential.output_postprocessors import L2NormEmbeddingPostprocessor
from generative_recommenders.research.rails.similarities.dot_product_similarity_fn import DotProductSimilarity

HSTU_DIM = EMBEDDING_DIM + 32

class HSTUModel(nn.Module):
    def __init__(self, embedding_dim, num_specialties):
        super().__init__()

        self.hstu = HSTU(
            max_sequence_len               = MAX_SEQ_LEN
            ,max_output_len                = 1
            ,embedding_dim                 = HSTU_DIM
            ,num_blocks                    = NUM_BLOCKS
            ,num_heads                     = NUM_HEADS
            ,linear_dim                    = LINEAR_DIM
            ,attention_dim                 = ATTENTION_DIM
            ,normalization                 = 'rel_bias'
            ,linear_config                 = 'uvqk'
            ,linear_activation             = 'silu'
            ,linear_dropout_rate           = DROPOUT_RATE
            ,attn_dropout_rate             = DROPOUT_RATE
            ,embedding_module              = LocalEmbeddingModule(
                num_items           = 1
                ,item_embedding_dim = embedding_dim
            )
            ,similarity_module             = DotProductSimilarity()
            ,input_features_preproc_module = LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor(
                max_sequence_len      = MAX_SEQ_LEN
                ,item_embedding_dim   = embedding_dim
                ,dropout_rate         = DROPOUT_RATE
                ,rating_embedding_dim = 32
                ,num_ratings          = 10
            )
            ,output_postproc_module        = L2NormEmbeddingPostprocessor(
                embedding_dim = HSTU_DIM
            )
            ,verbose                       = False
        )

        self.head_30  = nn.Linear(HSTU_DIM, num_specialties)
        self.head_60  = nn.Linear(HSTU_DIM, num_specialties)
        self.head_180 = nn.Linear(HSTU_DIM, num_specialties)

        self.register_buffer('dummy_ids', torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long))

    def forward(self, embeddings, delta_t, lengths):
        past_ids = self.dummy_ids.expand(embeddings.size(0), -1)

        encoded  = self.hstu(
            past_lengths     = lengths
            ,past_ids        = past_ids
            ,past_embeddings = embeddings
            ,past_payloads   = {
                'ratings'    : delta_t
                ,'timestamps': torch.zeros_like(delta_t)
            }
        )

        idx      = (lengths - 1).clamp(0, MAX_SEQ_LEN - 1).view(-1, 1, 1).expand(-1, 1, encoded.size(-1))
        seq_repr = encoded.gather(1, idx).squeeze(1)

        return (
            torch.sigmoid(self.head_30(seq_repr))
            ,torch.sigmoid(self.head_60(seq_repr))
            ,torch.sigmoid(self.head_180(seq_repr))
        )

model = HSTUModel(EMBEDDING_DIM, NUM_SPECIALTIES).to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# STEP 7: METRICS
# ============================================================
def ndcg_at_k(pred, label, k):
    top_k     = torch.topk(pred, k, dim=1).indices
    gains     = label.gather(1, top_k).float()
    discounts = torch.log2(torch.arange(2, k + 2, dtype=torch.float32).to(pred.device))
    dcg       = (gains / discounts).sum(dim=1)
    ideal     = label.sum(dim=1).clamp(max=k)
    idcg      = torch.zeros_like(dcg)
    for i in range(1, k + 1):
        idcg += (ideal >= i).float() / np.log2(i + 1)
    return (dcg / idcg.clamp(min=1e-8)).mean().item()

def precision_at_k(pred, label, k):
    top_k = torch.topk(pred, k, dim=1).indices
    hits  = label.gather(1, top_k).sum(dim=1)
    return (hits / k).mean().item()

def recall_at_k(pred, label, k):
    top_k  = torch.topk(pred, k, dim=1).indices
    hits   = label.gather(1, top_k).sum(dim=1)
    actual = label.sum(dim=1).clamp(min=1)
    return (hits / actual).mean().item()

def hit_rate_at_k(pred, label, k):
    top_k = torch.topk(pred, k, dim=1).indices
    hits  = label.gather(1, top_k).sum(dim=1)
    return (hits > 0).float().mean().item()

def evaluate(loader, model, k_list):
    model.eval()
    metrics   = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            embeddings = batch['embeddings'].to(DEVICE)
            delta_t    = batch['delta_t'].to(DEVICE)
            lengths    = batch['length'].to(DEVICE)
            label_30   = batch['label_30'].to(DEVICE)
            label_60   = batch['label_60'].to(DEVICE)
            label_180  = batch['label_180'].to(DEVICE)

            pred_30, pred_60, pred_180 = model(embeddings, delta_t, lengths)

            for k in k_list:
                for window, pred, label in [
                    ('T30',  pred_30,  label_30)
                    ,('T60',  pred_60,  label_60)
                    ,('T180', pred_180, label_180)
                ]:
                    metrics[f'{window}_ndcg@{k}'].append(ndcg_at_k(pred, label, k))
                    metrics[f'{window}_prec@{k}'].append(precision_at_k(pred, label, k))
                    metrics[f'{window}_rec@{k}'].append(recall_at_k(pred, label, k))
                    metrics[f'{window}_hit@{k}'].append(hit_rate_at_k(pred, label, k))

    return {key: np.mean(vals) for key, vals in metrics.items()}

def print_metrics(metrics, split='Val'):
    print(f"\n{split} Metrics:")
    for window in ['T30', 'T60', 'T180']:
        print(f"  {window}:")
        for k in EVAL_K:
            ndcg = metrics.get(f'{window}_ndcg@{k}', 0)
            prec = metrics.get(f'{window}_prec@{k}', 0)
            rec  = metrics.get(f'{window}_rec@{k}',  0)
            hit  = metrics.get(f'{window}_hit@{k}',  0)
            print(f"    @{k:2d} — NDCG: {ndcg:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  Hit: {hit:.4f}")

# ============================================================
# STEP 8: TRAINING LOOP
# ============================================================
optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
bce_loss   = nn.BCELoss()
best_val   = float('inf')
patience   = 5
no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        embeddings = batch['embeddings'].to(DEVICE)
        delta_t    = batch['delta_t'].to(DEVICE)
        lengths    = batch['length'].to(DEVICE)
        label_30   = batch['label_30'].to(DEVICE)
        label_60   = batch['label_60'].to(DEVICE)
        label_180  = batch['label_180'].to(DEVICE)

        pred_30, pred_60, pred_180 = model(embeddings, delta_t, lengths)

        loss = (
            bce_loss(pred_30,  label_30)
            + bce_loss(pred_60,  label_60)
            + bce_loss(pred_180, label_180)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    val_metrics = evaluate(val_loader, model, EVAL_K)
    print(f"\nEpoch {epoch+1}/{EPOCHS} — Train Loss: {train_loss:.4f}")
    print_metrics(val_metrics, 'Val')

    if train_loss < best_val:
        best_val   = train_loss
        no_improve = 0
        torch.save({
            'epoch'                 : epoch
            ,'model_state_dict'     : model.state_dict()
            ,'optimizer_state_dict' : optimizer.state_dict()
            ,'specialty_label_vocab': specialty_label_vocab
            ,'best_val_loss'        : best_val
            ,'embedding_dim'        : EMBEDDING_DIM
            ,'num_specialties'      : NUM_SPECIALTIES
            ,'eval_k'               : EVAL_K
        }, './checkpoints/best_model.pt')
        print(f"  Model saved — loss: {best_val:.4f}")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# ============================================================
# STEP 9: TEST EVALUATION
# ============================================================
print("\nLoading best model for test evaluation...")
checkpoint = torch.load('./checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

test_metrics = evaluate(test_loader, model, EVAL_K)
print_metrics(test_metrics, 'Test')

print("\nNotebook 2 complete")
