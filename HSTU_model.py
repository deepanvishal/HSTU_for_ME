# ============================================================
# NOTEBOOK 2: HSTU TRAINING
# ============================================================
# CHANGE SAMPLE_PCT TO SCALE:
# 0.005 = 0.5% (~1 hour)
# 0.10  = 10%  (~overnight)
# 1.0   = 100% (~production)
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
# STEP 1: LOAD EMBEDDINGS
# ============================================================
print("\nLoading embeddings...")
provider_matrix  = np.load('./embeddings/provider_embeddings.npy')
specialty_matrix = np.load('./embeddings/specialty_embeddings.npy')
dx_matrix        = np.load('./embeddings/dx_embeddings.npy')
procedure_matrix = np.load('./embeddings/procedure_embeddings.npy')

with open('./embeddings/provider_vocab.pkl',  'rb') as f: provider_vocab  = pickle.load(f)
with open('./embeddings/specialty_vocab.pkl', 'rb') as f: specialty_vocab = pickle.load(f)
with open('./embeddings/dx_vocab.pkl',        'rb') as f: dx_vocab        = pickle.load(f)
with open('./embeddings/procedure_vocab.pkl', 'rb') as f: procedure_vocab = pickle.load(f)
with open('./embeddings/unk_embeddings.pkl',  'rb') as f: unk_emb         = pickle.load(f)
with open('./embeddings/specialty_label_vocab.pkl', 'rb') as f: specialty_label_vocab = pickle.load(f)

NUM_SPECIALTIES = len(specialty_label_vocab)
print(f"Specialties: {NUM_SPECIALTIES}")

# ============================================================
# STEP 2: VISIT EMBEDDING FUNCTION
# ============================================================
def get_visit_embedding(providers, specialties, dxs, procedures):
    def lookup(lst, matrix, vocab, unk):
        if not lst:
            return unk
        vecs = [matrix[vocab[x]] if x in vocab else unk for x in lst]
        return np.mean(vecs, axis=0)

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
            row.provider_ids    or []
            ,row.specialty_codes or []
            ,row.dx_list         or []
            ,row.procedure_codes or []
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
            ,'specialties_30' : list(row.specialties_30  or [])
            ,'specialties_60' : list(row.specialties_60  or [])
            ,'specialties_180': list(row.specialties_180 or [])
        })
    del chunk
del df_lab

print(f"Members loaded: {len(member_sequences):,}")

# ============================================================
# STEP 4: TRAIN / VAL / TEST SPLIT (TIME BASED PER MEMBER)
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

    labels = {l['visit_seq_num']: l for l in member_labels.get(member_id, [])}

    train_data.append((member_id, visits[:train_cut],      labels))
    val_data.append((member_id,   visits[:val_cut],        labels))
    test_data.append((member_id,  visits[:int(n * 0.875)], labels))

print(f"Train: {len(train_data):,}  Val: {len(val_data):,}  Test: {len(test_data):,}")

# ============================================================
# STEP 5: DATASET
# ============================================================
def make_label_vector(specs):
    vec = np.zeros(NUM_SPECIALTIES, dtype=np.float32)
    for s in (specs or []):
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

            # pad
            pad = max_seq_len - n
            if pad > 0:
                embeddings = np.vstack([embeddings, np.zeros((pad, EMBEDDING_DIM))])
                delta_t    = np.concatenate([delta_t, np.zeros(pad, dtype=np.int64)])

            # get label for last visit
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
from generative_recommenders.research.modeling.sequential.hstu import HSTU
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor
from generative_recommenders.research.modeling.sequential.features import SequentialFeatures

class HSTUModel(nn.Module):
    def __init__(self, embedding_dim, num_specialties):
        super().__init__()

        self.input_preproc = LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor(
            max_sequence_len      = MAX_SEQ_LEN
            ,item_embedding_dim   = embedding_dim
            ,dropout_rate         = DROPOUT_RATE
            ,rating_embedding_dim = 32
            ,num_ratings          = 10
        )

        self.hstu = HSTU(
            max_sequence_len  = MAX_SEQ_LEN
            ,embedding_dim    = embedding_dim
            ,num_blocks       = NUM_BLOCKS
            ,num_heads        = NUM_HEADS
            ,linear_dim       = LINEAR_DIM
            ,attention_dim    = ATTENTION_DIM
            ,dropout_rate     = DROPOUT_RATE
        )

        self.head_30  = nn.Linear(embedding_dim, num_specialties)
        self.head_60  = nn.Linear(embedding_dim, num_specialties)
        self.head_180 = nn.Linear(embedding_dim, num_specialties)

    def forward(self, embeddings, delta_t, lengths):
        B = embeddings.size(0)

        features = SequentialFeatures(
            past_lengths    = lengths
            ,past_ids       = torch.zeros(B, MAX_SEQ_LEN, dtype=torch.long).to(embeddings.device)
            ,past_embeddings= embeddings
            ,past_payloads  = {'ratings': delta_t, 'timestamps': torch.zeros_like(delta_t)}
        )

        x = self.input_preproc(features)
        x = self.hstu(x, features)

        # gather last valid token
        idx = (lengths - 1).clamp(0, MAX_SEQ_LEN - 1).view(-1, 1, 1).expand(-1, 1, x.size(-1))
        seq_repr = x.gather(1, idx).squeeze(1)

        return (
            torch.sigmoid(self.head_30(seq_repr))
            ,torch.sigmoid(self.head_60(seq_repr))
            ,torch.sigmoid(self.head_180(seq_repr))
        )

model = HSTUModel(EMBEDDING_DIM, NUM_SPECIALTIES).to(DEVICE)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# STEP 7: METRICS
# ============================================================
def ndcg_at_k(pred, label, k):
    top_k   = torch.topk(pred, k, dim=1).indices
    gains   = label.gather(1, top_k)
    discounts = torch.log2(torch.arange(2, k + 2, dtype=torch.float32).to(pred.device))
    dcg     = (gains / discounts).sum(dim=1)
    ideal   = label.sum(dim=1).clamp(max=k)
    idcg    = torch.zeros_like(dcg)
    for i in range(1, k + 1):
        idcg += (ideal >= i).float() / np.log2(i + 1)
    return (dcg / idcg.clamp(min=1e-8)).mean().item()

def precision_at_k(pred, label, k):
    top_k = torch.topk(pred, k, dim=1).indices
    hits  = label.gather(1, top_k).sum(dim=1)
    return (hits / k).mean().item()

def recall_at_k(pred, label, k):
    top_k    = torch.topk(pred, k, dim=1).indices
    hits     = label.gather(1, top_k).sum(dim=1)
    actual   = label.sum(dim=1).clamp(min=1)
    return (hits / actual).mean().item()

def hit_rate_at_k(pred, label, k):
    top_k = torch.topk(pred, k, dim=1).indices
    hits  = label.gather(1, top_k).sum(dim=1)
    return (hits > 0).float().mean().item()

def evaluate(loader, model, k_list):
    model.eval()
    metrics = defaultdict(lambda: defaultdict(float))
    n_batches = 0

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
                    metrics[f'{window}_ndcg@{k}'][n_batches]      = ndcg_at_k(pred, label, k)
                    metrics[f'{window}_precision@{k}'][n_batches] = precision_at_k(pred, label, k)
                    metrics[f'{window}_recall@{k}'][n_batches]    = recall_at_k(pred, label, k)
                    metrics[f'{window}_hit@{k}'][n_batches]       = hit_rate_at_k(pred, label, k)

            n_batches += 1

    return {
        key: np.mean(list(vals.values()))
        for key, vals in metrics.items()
    }

def print_metrics(metrics, split='Val'):
    print(f"\n{split} Metrics:")
    for window in ['T30', 'T60', 'T180']:
        print(f"  {window}:")
        for k in EVAL_K:
            ndcg = metrics.get(f'{window}_ndcg@{k}', 0)
            prec = metrics.get(f'{window}_precision@{k}', 0)
            rec  = metrics.get(f'{window}_recall@{k}', 0)
            hit  = metrics.get(f'{window}_hit@{k}', 0)
            print(f"    @{k:2d} — NDCG: {ndcg:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  Hit: {hit:.4f}")

# ============================================================
# STEP 8: TRAINING LOOP
# ============================================================
optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
bce_loss    = nn.BCELoss()
best_val    = float('inf')
patience    = 5
no_improve  = 0

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
    val_ndcg    = np.mean([val_metrics.get(f'T180_ndcg@{k}', 0) for k in EVAL_K])

    print(f"\nEpoch {epoch+1}/{EPOCHS} — Train Loss: {train_loss:.4f}")
    print_metrics(val_metrics, 'Val')

    if train_loss < best_val:
        best_val   = train_loss
        no_improve = 0
        torch.save({
            'epoch'                : epoch
            ,'model_state_dict'    : model.state_dict()
            ,'optimizer_state_dict': optimizer.state_dict()
            ,'specialty_label_vocab': specialty_label_vocab
            ,'best_val_loss'       : best_val
            ,'embedding_dim'       : EMBEDDING_DIM
            ,'num_specialties'     : NUM_SPECIALTIES
            ,'eval_k'              : EVAL_K
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
