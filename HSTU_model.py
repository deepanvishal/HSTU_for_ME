# ============================================================
# NOTEBOOK 2: HSTU TRAINING
# ============================================================

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from google.cloud import bigquery

sys.path.insert(0, '.')

os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./data',        exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
TARGET      = 'specialty'   # 'specialty' | 'provider' | 'dx'
SAMPLE_PCT  = 0.05

LABEL_COLS = {
    'specialty': ('specialties_30', 'specialties_60', 'specialties_180')
    ,'provider' : ('providers_30',   'providers_60',   'providers_180')
    ,'dx'       : ('dx_30',          'dx_60',          'dx_180')
}

LABEL_VOCAB_PATH = {
    'specialty': './embeddings/specialty_label_vocab.pkl'
    ,'provider' : './embeddings/provider_label_vocab.pkl'
    ,'dx'       : './embeddings/dx_label_vocab.pkl'
}

import pickle
assert TARGET in LABEL_COLS, f"TARGET must be one of {list(LABEL_COLS.keys())}"

COL_30, COL_60, COL_180 = LABEL_COLS[TARGET]
PCT                      = int(SAMPLE_PCT * 100)
CHECKPOINT_PATH          = f'./checkpoints/best_model_{TARGET}_{PCT}pct.pt'

# cache — no pickle; npy for matrices, parquet for tabular
SEQ_EMB_PATH    = f'./data/seq_emb_{PCT}pct.npy'
SEQ_META_PATH   = f'./data/seq_meta_{PCT}pct.parquet'
LABEL_CACHE_PATH = f'./data/member_labels_{TARGET}_{PCT}pct.parquet'
LOAD_FROM_CACHE  = False

MAX_SEQ_LEN   = 20
BATCH_SIZE    = 512
EPOCHS        = 5
EMBEDDING_DIM = 128
NUM_RATINGS   = 16
RATING_DIM    = 32
HSTU_DIM      = EMBEDDING_DIM + RATING_DIM
NUM_BLOCKS    = 2
NUM_HEADS     = 4
LINEAR_DIM    = 128
ATTENTION_DIM = 64
DROPOUT_RATE  = 0.2
LR            = 4e-4
EVAL_K        = [3, 5, 10]
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS      = torch.cuda.device_count()
NUM_WORKERS   = min(4 * NUM_GPUS, 16)

print(f"Target:     {TARGET}")
print(f"Sample:     {SAMPLE_PCT*100}%")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Device:     {DEVICE}  GPUs: {NUM_GPUS}")
print(f"Cache:      {LOAD_FROM_CACHE}")

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
t0 = time.time()

provider_matrix  = np.load('./embeddings/provider_embeddings.npy')
specialty_matrix = np.load('./embeddings/specialty_embeddings.npy')
dx_matrix        = np.load('./embeddings/dx_embeddings.npy')
procedure_matrix = np.load('./embeddings/procedure_embeddings.npy')

with open('./embeddings/provider_vocab.pkl',  'rb') as f: provider_vocab  = pickle.load(f)
with open('./embeddings/specialty_vocab.pkl', 'rb') as f: specialty_vocab = pickle.load(f)
with open('./embeddings/dx_vocab.pkl',        'rb') as f: dx_vocab        = pickle.load(f)
with open('./embeddings/procedure_vocab.pkl', 'rb') as f: procedure_vocab = pickle.load(f)
with open('./embeddings/unk_embeddings.pkl',  'rb') as f: unk_emb         = pickle.load(f)

with open(LABEL_VOCAB_PATH[TARGET], 'rb') as f:
    label_vocab = pickle.load(f)

NUM_CLASSES = len(label_vocab)
print(f"\nLabel vocab: {NUM_CLASSES:,}  Provider vocab: {len(provider_vocab):,}  DX vocab: {len(dx_vocab):,}")
print(f"Step 1 done — {time.time() - t0:.1f}s")

# ============================================================
# STEP 3: LOAD DATA
#
# Design:
#   emb_matrix [N_total, 128] — single npy, loaded read-only once,
#     shared across DataLoader workers via fork (no copy-on-write
#     since workers only read)
#   member_sequences[member_id] = (seq_nums, dt_arr, emb_idx_arr)
#     — three parallel int arrays, no Python dicts, no embedding copies
#   label_lookup[(member_id, visit_seq_num)] = (l30, l60, l180)
#     — flat dict of sparse code lists
#   Cache: npy + parquet only, no pickle
# ============================================================
t0 = time.time()

seq_cached   = LOAD_FROM_CACHE and os.path.exists(SEQ_EMB_PATH) and os.path.exists(SEQ_META_PATH)
label_cached = LOAD_FROM_CACHE and os.path.exists(LABEL_CACHE_PATH)

if not seq_cached or not label_cached:
    client = bigquery.Client(project='anbc-hcb-dev')
    print("\nSampling members in BigQuery...")
    client.query(f"""
        CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_sampled_members` AS
        SELECT DISTINCT member_id
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_sequence`
        WHERE RAND() < {SAMPLE_PCT}
    """).result()

    member_count = client.query("""
        SELECT COUNT(*) AS n
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_sampled_members`
    """).to_dataframe().iloc[0]['n']
    print(f"Sampled members: {member_count:,}")

    # single BQ query — UNNEST + JOIN embedding tables + AVG per modality per visit
    # + LEFT JOIN labels — one round trip, no Python embedding work
    print("Running combined embedding + label query in BQ...")
    pe  = ', '.join([f'AVG(pe.e{i})  AS pe{i}'  for i in range(32)])
    se  = ', '.join([f'AVG(se.e{i})  AS se{i}'  for i in range(32)])
    de  = ', '.join([f'AVG(de.e{i})  AS de{i}'  for i in range(32)])
    pre = ', '.join([f'AVG(pre.e{i}) AS pre{i}' for i in range(32)])
    cpe  = ', '.join([f'COALESCE(pa.pe{i},  0.0) AS pe{i}'  for i in range(32)])
    cse  = ', '.join([f'COALESCE(sa.se{i},  0.0) AS se{i}'  for i in range(32)])
    cde  = ', '.join([f'COALESCE(da.de{i},  0.0) AS de{i}'  for i in range(32)])
    cpre = ', '.join([f'COALESCE(pra.pre{i}, 0.0) AS pre{i}' for i in range(32)])
    DS   = 'anbc-hcb-dev.provider_ds_netconf_data_hcb_dev'

    df_combined = client.query(f"""
        WITH sampled AS (
            SELECT member_id FROM `{DS}.A870800_sampled_members`
        )
        ,provider_agg AS (
            SELECT s.member_id, s.visit_seq_num, {pe}
            FROM `{DS}.A870800_claims_gen_rec_visit_sequence` s
            INNER JOIN sampled m ON s.member_id = m.member_id
            LEFT JOIN UNNEST(s.provider_ids) AS pid
            LEFT JOIN `{DS}.A870800_provider_embeddings_bq` pe ON CAST(pid AS STRING) = pe.code
            GROUP BY s.member_id, s.visit_seq_num
        )
        ,specialty_agg AS (
            SELECT s.member_id, s.visit_seq_num, {se}
            FROM `{DS}.A870800_claims_gen_rec_visit_sequence` s
            INNER JOIN sampled m ON s.member_id = m.member_id
            LEFT JOIN UNNEST(s.specialty_codes) AS spid
            LEFT JOIN `{DS}.A870800_specialty_embeddings_bq` se ON CAST(spid AS STRING) = se.code
            GROUP BY s.member_id, s.visit_seq_num
        )
        ,dx_agg AS (
            SELECT s.member_id, s.visit_seq_num, {de}
            FROM `{DS}.A870800_claims_gen_rec_visit_sequence` s
            INNER JOIN sampled m ON s.member_id = m.member_id
            LEFT JOIN UNNEST(s.dx_list) AS dxid
            LEFT JOIN `{DS}.A870800_dx_embeddings_bq` de ON CAST(dxid AS STRING) = de.code
            GROUP BY s.member_id, s.visit_seq_num
        )
        ,procedure_agg AS (
            SELECT s.member_id, s.visit_seq_num, {pre}
            FROM `{DS}.A870800_claims_gen_rec_visit_sequence` s
            INNER JOIN sampled m ON s.member_id = m.member_id
            LEFT JOIN UNNEST(s.procedure_codes) AS prid
            LEFT JOIN `{DS}.A870800_procedure_embeddings_bq` pre ON CAST(prid AS STRING) = pre.code
            GROUP BY s.member_id, s.visit_seq_num
        )
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
        INNER JOIN sampled m       ON s.member_id = m.member_id
        LEFT JOIN provider_agg  pa  ON s.member_id = pa.member_id  AND s.visit_seq_num = pa.visit_seq_num
        LEFT JOIN specialty_agg sa  ON s.member_id = sa.member_id  AND s.visit_seq_num = sa.visit_seq_num
        LEFT JOIN dx_agg        da  ON s.member_id = da.member_id  AND s.visit_seq_num = da.visit_seq_num
        LEFT JOIN procedure_agg pra ON s.member_id = pra.member_id AND s.visit_seq_num = pra.visit_seq_num
        LEFT JOIN `{DS}.A870800_claims_gen_rec_label` l
            ON s.member_id = l.member_id AND s.visit_seq_num = l.visit_seq_num
        ORDER BY s.member_id, s.visit_seq_num
    """).to_dataframe(create_bqstorage_client=True)

    print(f"Combined query rows: {len(df_combined):,}")

    emb_cols       = [f'pe{i}' for i in range(32)] + [f'se{i}' for i in range(32)] + [f'de{i}' for i in range(32)] + [f'pre{i}' for i in range(32)]
    all_embeddings = df_combined[emb_cols].values.astype(np.float32)

    df_meta = pd.DataFrame({
        'member_id'     : df_combined['member_id'].values
        ,'visit_seq_num': df_combined['visit_seq_num'].values.astype(np.int32)
        ,'dt_bucket'    : df_combined['dt_bucket'].values.astype(np.int8)
        ,'visit_idx'    : np.arange(len(df_combined), dtype=np.int32)
    })
    df_labels = df_combined[['member_id', 'visit_seq_num', COL_30, COL_60, COL_180]].copy()
    del df_combined

    np.save(SEQ_EMB_PATH, all_embeddings)
    df_meta.to_parquet(SEQ_META_PATH,    index=False)
    df_labels.to_parquet(LABEL_CACHE_PATH, index=False)
    del all_embeddings, df_meta, df_labels
    print(f"Cache saved → {SEQ_EMB_PATH}, {SEQ_META_PATH}, {LABEL_CACHE_PATH}")

# ---- load emb_matrix + meta ----
print("\nLoading sequence cache...")
emb_matrix = np.load(SEQ_EMB_PATH)
emb_matrix.setflags(write=False)              # read-only — safe fork sharing in DataLoader workers
df_meta    = pd.read_parquet(SEQ_META_PATH)
df_meta    = df_meta.sort_values('member_id', kind='stable')   # ensure contiguous member blocks

member_ids_arr = df_meta['member_id'].values
seq_nums_all   = df_meta['visit_seq_num'].values
dt_all         = df_meta['dt_bucket'].values
emb_idx_all    = df_meta['visit_idx'].values
del df_meta

# build member_sequences — sort + cumsum, zero Python row loops
unique_members, counts = np.unique(member_ids_arr, return_counts=True)
offsets = np.concatenate([[0], np.cumsum(counts)])

member_sequences = {
    mid: (seq_nums_all[s:e], dt_all[s:e], emb_idx_all[s:e])
    for mid, s, e in zip(unique_members, offsets[:-1], offsets[1:])
}
del member_ids_arr, seq_nums_all, dt_all, emb_idx_all
print(f"Members: {len(member_sequences):,}  Total visits: {emb_matrix.shape[0]:,}")

# ---- load labels ----
print("Loading label cache...")
df_lab = pd.read_parquet(LABEL_CACHE_PATH)

# flat lookup: (member_id, visit_seq_num) → (l30, l60, l180) code lists
# zip over pandas series — no Python row loop
label_lookup = dict(zip(
    zip(df_lab['member_id'].tolist(), df_lab['visit_seq_num'].tolist())
    ,zip(
        df_lab[COL_30].tolist()
        ,df_lab[COL_60].tolist()
        ,df_lab[COL_180].tolist()
    )
))
del df_lab
print(f"Label entries: {len(label_lookup):,}")
print(f"\nStep 3 done — {time.time() - t0:.1f}s")

# ============================================================
# STEP 4: TRAIN / VAL / TEST SPLIT
# Array slices are numpy views — zero copy.
# ============================================================
t0 = time.time()

train_data = []
val_data   = []
test_data  = []

for member_id, (seq_nums, dt_vals, emb_idxs) in member_sequences.items():
    n = len(seq_nums)
    if n < 4:
        continue

    train_cut = int(n * 0.75)
    val_cut   = int(n * 0.875)

    train_data.append((member_id, seq_nums[:train_cut], dt_vals[:train_cut], emb_idxs[:train_cut]))
    val_data.append((member_id,   seq_nums[:val_cut],   dt_vals[:val_cut],   emb_idxs[:val_cut]))
    test_data.append((member_id,  seq_nums[:-1],        dt_vals[:-1],        emb_idxs[:-1]))

print(f"Train: {len(train_data):,}  Val: {len(val_data):,}  Test: {len(test_data):,}")
print(f"Step 4 done — {time.time() - t0:.1f}s")

# ============================================================
# STEP 5: DATASET
#
# Design:
#   self.samples stores (emb_idxs_padded, dt_padded, n, l30, l60, l180)
#   — index arrays + sparse code lists only, no pre-expanded embeddings
#   — no pre-expanded label vectors (critical for dx: 10K+ classes × 1.5M rows)
#   __getitem__ does emb_matrix[emb_idxs] — numpy fancy index, fast
#   emb_matrix is read-only; fork in DataLoader workers = zero memory copy
# ============================================================
t0 = time.time()

def codes_to_indices(codes):
    if codes is None:
        return np.empty(0, dtype=np.int32)
    if isinstance(codes, (np.ndarray, list)):
        if len(codes) == 0:
            return np.empty(0, dtype=np.int32)
    else:
        codes = [codes]
    idxs = [label_vocab[str(c)] for c in codes if str(c) in label_vocab]
    return np.array(idxs, dtype=np.int32) if idxs else np.empty(0, dtype=np.int32)

class VisitDataset(Dataset):
    def __init__(self, data, emb_matrix, label_lookup, max_seq_len):
        self.emb_matrix = emb_matrix
        self.samples    = []

        for member_id, seq_nums, dt_vals, emb_idxs in data:
            seq_nums = seq_nums[-max_seq_len:]
            dt_vals  = dt_vals[-max_seq_len:]
            emb_idxs = emb_idxs[-max_seq_len:]
            n        = len(seq_nums)

            entry = label_lookup.get((member_id, int(seq_nums[-1])))
            if entry is None:
                continue

            l30, l60, l180 = entry

            pad = max_seq_len - n
            if pad > 0:
                emb_idxs = np.concatenate([emb_idxs, np.zeros(pad, dtype=np.int32)])
                dt_vals  = np.concatenate([dt_vals,  np.zeros(pad, dtype=np.int8)])

            # convert label codes to index arrays once at init — no dict lookup in __getitem__
            self.samples.append((
                emb_idxs.astype(np.int32)
                ,dt_vals.astype(np.int64)
                ,np.int64(n)
                ,codes_to_indices(l30)
                ,codes_to_indices(l60)
                ,codes_to_indices(l180)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        emb_idxs, dt_vals, n, idx30, idx60, idx180 = self.samples[idx]

        embeddings     = self.emb_matrix[emb_idxs]  # fancy index → copy
        embeddings[n:] = 0.0

        l30 = np.zeros(NUM_CLASSES, dtype=np.float32); l30[idx30] = 1.0
        l60 = np.zeros(NUM_CLASSES, dtype=np.float32); l60[idx60] = 1.0
        l180 = np.zeros(NUM_CLASSES, dtype=np.float32); l180[idx180] = 1.0

        return {
            'embeddings': torch.from_numpy(embeddings)
            ,'delta_t'  : torch.from_numpy(dt_vals)
            ,'length'   : torch.tensor(n,   dtype=torch.long)
            ,'label_30' : torch.from_numpy(l30)
            ,'label_60' : torch.from_numpy(l60)
            ,'label_180': torch.from_numpy(l180)
        }

train_dataset = VisitDataset(train_data, emb_matrix, label_lookup, MAX_SEQ_LEN)
val_dataset   = VisitDataset(val_data,   emb_matrix, label_lookup, MAX_SEQ_LEN)
test_dataset  = VisitDataset(test_data,  emb_matrix, label_lookup, MAX_SEQ_LEN)

print(f"Train: {len(train_dataset):,}  Val: {len(val_dataset):,}  Test: {len(test_dataset):,}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

print(f"Train batches: {len(train_loader):,}  Val: {len(val_loader):,}  Test: {len(test_loader):,}")
print(f"Step 5 done — {time.time() - t0:.1f}s")

# ============================================================
# STEP 6: MODEL
# ============================================================
t0 = time.time()
from hstu_pytorch import PureHSTU

model = PureHSTU(
    max_seq_len        = MAX_SEQ_LEN
    ,embedding_dim     = EMBEDDING_DIM
    ,num_blocks        = NUM_BLOCKS
    ,num_heads         = NUM_HEADS
    ,linear_dim        = LINEAR_DIM
    ,attention_dim     = ATTENTION_DIM
    ,dropout_rate      = DROPOUT_RATE
    ,attn_dropout_rate = DROPOUT_RATE
    ,num_ratings       = NUM_RATINGS
    ,rating_dim        = RATING_DIM
    ,num_specialties   = NUM_CLASSES
)

if NUM_GPUS > 1:
    model = nn.DataParallel(model)
    print(f"Using {NUM_GPUS} GPUs")

model = model.to(DEVICE)

try:
    model = torch.compile(model)
    print("torch.compile enabled")
except Exception as e:
    print(f"torch.compile skipped: {e}")

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Step 6 done — {time.time() - t0:.1f}s")

# ============================================================
# STEP 7: METRICS
# ============================================================
t0 = time.time()

def ndcg_at_k(pred, label, k):
    top_k     = torch.topk(pred, k, dim=1).indices
    gains     = label.gather(1, top_k).float()
    discounts = torch.log2(torch.arange(2, k + 2, dtype=torch.float32, device=pred.device))
    dcg       = (gains / discounts).sum(dim=1)
    ideal     = label.sum(dim=1).clamp(max=k)
    ranks     = torch.arange(1, k + 1, dtype=torch.float32, device=pred.device)
    idcg      = (1.0 / torch.log2(ranks + 1)).unsqueeze(0) * (ranks.unsqueeze(0) <= ideal.unsqueeze(1)).float()
    return dcg / idcg.sum(dim=1).clamp(min=1e-8)

def precision_at_k(pred, label, k):
    top_k = torch.topk(pred, k, dim=1).indices
    return label.gather(1, top_k).sum(dim=1) / k

def recall_at_k(pred, label, k):
    top_k  = torch.topk(pred, k, dim=1).indices
    hits   = label.gather(1, top_k).sum(dim=1)
    actual = label.sum(dim=1).clamp(min=1)
    return hits / actual

def hit_rate_at_k(pred, label, k):
    top_k = torch.topk(pred, k, dim=1).indices
    return (label.gather(1, top_k).sum(dim=1) > 0).float()

def evaluate(loader, model, k_list):
    model.eval()
    metric_sums = defaultdict(float)
    n_samples   = 0

    with torch.no_grad():
        for batch in loader:
            embeddings = batch['embeddings'].to(DEVICE, non_blocking=True)
            delta_t    = batch['delta_t'].to(DEVICE,    non_blocking=True)
            lengths    = batch['length'].to(DEVICE,     non_blocking=True)
            label_30   = batch['label_30'].to(DEVICE,   non_blocking=True)
            label_60   = batch['label_60'].to(DEVICE,   non_blocking=True)
            label_180  = batch['label_180'].to(DEVICE,  non_blocking=True)

            with torch.cuda.amp.autocast():
                pred_30, pred_60, pred_180 = model(embeddings, delta_t, lengths)

            pred_30  = pred_30.float()
            pred_60  = pred_60.float()
            pred_180 = pred_180.float()

            n_samples += embeddings.size(0)

            for k in k_list:
                for window, pred, label in [
                    ('T30',  pred_30,  label_30)
                    ,('T60',  pred_60,  label_60)
                    ,('T180', pred_180, label_180)
                ]:
                    metric_sums[f'{window}_ndcg@{k}'] += ndcg_at_k(pred, label, k).sum().item()
                    metric_sums[f'{window}_prec@{k}'] += precision_at_k(pred, label, k).sum().item()
                    metric_sums[f'{window}_rec@{k}']  += recall_at_k(pred, label, k).sum().item()
                    metric_sums[f'{window}_hit@{k}']  += hit_rate_at_k(pred, label, k).sum().item()

    return {key: val / n_samples for key, val in metric_sums.items()}

def print_metrics(metrics, split='Val'):
    print(f"\n{split} Metrics [{TARGET}]:")
    for window in ['T30', 'T60', 'T180']:
        print(f"  {window}:")
        for k in EVAL_K:
            ndcg = metrics.get(f'{window}_ndcg@{k}', 0)
            prec = metrics.get(f'{window}_prec@{k}', 0)
            rec  = metrics.get(f'{window}_rec@{k}',  0)
            hit  = metrics.get(f'{window}_hit@{k}',  0)
            print(f"    @{k:2d} — NDCG: {ndcg:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  Hit: {hit:.4f}")

print(f"Step 7 done — {time.time() - t0:.1f}s")

# ============================================================
# STEP 8: TRAINING LOOP
# ============================================================
t0 = time.time()

optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler     = torch.cuda.amp.GradScaler()
bce_loss   = nn.BCELoss()
best_ndcg  = 0.0
patience   = 5
no_improve = 0

def get_raw_model(m):
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    if isinstance(m, nn.DataParallel):
        m = m.module
    return m

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        embeddings = batch['embeddings'].to(DEVICE, non_blocking=True)
        delta_t    = batch['delta_t'].to(DEVICE,    non_blocking=True)
        lengths    = batch['length'].to(DEVICE,     non_blocking=True)
        label_30   = batch['label_30'].to(DEVICE,   non_blocking=True)
        label_60   = batch['label_60'].to(DEVICE,   non_blocking=True)
        label_180  = batch['label_180'].to(DEVICE,  non_blocking=True)

        with torch.cuda.amp.autocast():
            pred_30, pred_60, pred_180 = model(embeddings, delta_t, lengths)

        loss = (
            bce_loss(pred_30.float(),  label_30)
            + bce_loss(pred_60.float(),  label_60)
            + bce_loss(pred_180.float(), label_180)
        )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    scheduler.step()
    train_loss  /= len(train_loader)
    val_metrics  = evaluate(val_loader, model, EVAL_K)
    val_ndcg     = np.mean([val_metrics.get(f'T180_ndcg@{k}', 0) for k in EVAL_K])

    print(f"\nEpoch {epoch+1}/{EPOCHS} — Train Loss: {train_loss:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}")
    print_metrics(val_metrics, 'Val')

    if val_ndcg > best_ndcg:
        best_ndcg  = val_ndcg
        no_improve = 0
        torch.save({
            'epoch'                : epoch
            ,'model_state_dict'    : get_raw_model(model).state_dict()
            ,'optimizer_state_dict': optimizer.state_dict()
            ,'scheduler_state_dict': scheduler.state_dict()
            ,'label_vocab'         : label_vocab
            ,'best_val_ndcg'       : best_ndcg
            ,'target'              : TARGET
            ,'config': {
                'embedding_dim' : EMBEDDING_DIM
                ,'hstu_dim'     : HSTU_DIM
                ,'num_classes'  : NUM_CLASSES
                ,'max_seq_len'  : MAX_SEQ_LEN
                ,'num_blocks'   : NUM_BLOCKS
                ,'num_heads'    : NUM_HEADS
                ,'linear_dim'   : LINEAR_DIM
                ,'attention_dim': ATTENTION_DIM
                ,'dropout_rate' : DROPOUT_RATE
                ,'num_ratings'  : NUM_RATINGS
                ,'rating_dim'   : RATING_DIM
                ,'eval_k'       : EVAL_K
                ,'sample_pct'   : SAMPLE_PCT
                ,'target'       : TARGET
            }
        }, CHECKPOINT_PATH)
        print(f"  Saved → {CHECKPOINT_PATH}  val NDCG@T180: {best_ndcg:.4f}")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"Step 8 done — {time.time() - t0:.1f}s")

# ============================================================
# STEP 9: TEST EVALUATION
# ============================================================
t0 = time.time()

print(f"\nLoading best model: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
config     = checkpoint['config']

base_model = PureHSTU(
    max_seq_len        = config['max_seq_len']
    ,embedding_dim     = config['embedding_dim']
    ,num_blocks        = config.get('num_blocks', 2)
    ,num_heads         = config.get('num_heads',  4)
    ,linear_dim        = config['linear_dim']
    ,attention_dim     = config['attention_dim']
    ,dropout_rate      = config['dropout_rate']
    ,attn_dropout_rate = config['dropout_rate']
    ,num_ratings       = config['num_ratings']
    ,rating_dim        = config['rating_dim']
    ,num_specialties   = NUM_CLASSES
).to(DEVICE)

base_model.load_state_dict(checkpoint['model_state_dict'])

test_metrics = evaluate(test_loader, base_model, EVAL_K)
print_metrics(test_metrics, 'Test')

print(f"Step 9 done — {time.time() - t0:.1f}s")
print(f"\nNotebook 2 complete — target: {TARGET}  checkpoint: {CHECKPOINT_PATH}")
