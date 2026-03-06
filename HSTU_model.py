# ============================================================
# NOTEBOOK 2: HSTU TRAINING
# ============================================================

import os
import sys
import pickle
import numpy as np
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

assert TARGET in LABEL_COLS, f"TARGET must be one of {list(LABEL_COLS.keys())}"

COL_30, COL_60, COL_180 = LABEL_COLS[TARGET]
CHECKPOINT_PATH          = f'./checkpoints/best_model_{TARGET}_{int(SAMPLE_PCT*100)}pct.pt'
SEQ_CACHE_PATH           = f'./data/member_sequences_{int(SAMPLE_PCT*100)}pct.pkl'
LABEL_CACHE_PATH         = f'./data/member_labels_{TARGET}_{int(SAMPLE_PCT*100)}pct.pkl'
LOAD_FROM_CACHE          = False

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

print(f"Target:      {TARGET}")
print(f"Sample:      {SAMPLE_PCT*100}%")
print(f"Checkpoint:  {CHECKPOINT_PATH}")
print(f"Device:      {DEVICE}")
print(f"GPUs:        {NUM_GPUS}")
print(f"Workers:     {NUM_WORKERS}")
print(f"Batch size:  {BATCH_SIZE}")
print(f"Cache:       {LOAD_FROM_CACHE}")

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

with open('./embeddings/provider_vocab.pkl',  'rb') as f: provider_vocab  = pickle.load(f)
with open('./embeddings/specialty_vocab.pkl', 'rb') as f: specialty_vocab = pickle.load(f)
with open('./embeddings/dx_vocab.pkl',        'rb') as f: dx_vocab        = pickle.load(f)
with open('./embeddings/procedure_vocab.pkl', 'rb') as f: procedure_vocab = pickle.load(f)
with open('./embeddings/unk_embeddings.pkl',  'rb') as f: unk_emb         = pickle.load(f)

with open(LABEL_VOCAB_PATH[TARGET], 'rb') as f:
    label_vocab = pickle.load(f)

NUM_CLASSES = len(label_vocab)
print(f"Target:          {TARGET}")
print(f"Label vocab:     {NUM_CLASSES:,}")
print(f"Provider vocab:  {len(provider_vocab):,}")
print(f"DX vocab:        {len(dx_vocab):,}")
print(f"Procedure vocab: {len(procedure_vocab):,}")

# ============================================================
# STEP 2: VECTORIZED EMBEDDING BUILD
# Replaces per-row itertuples() loop with bulk numpy scatter ops.
# Each modality: explode list column → map vocab → scatter-add
# into output matrix → divide by count → fill unknowns with unk.
# ============================================================

def _embed_column_vectorized(series, matrix, vocab, unk_vec):
    n_rows = len(series)
    dim    = matrix.shape[1]
    result = np.zeros((n_rows, dim), dtype=np.float32)
    counts = np.zeros(n_rows,       dtype=np.int32)

    for row_idx, codes in enumerate(series):
        if codes is None:
            continue
        if not isinstance(codes, (list, np.ndarray)):
            codes = [codes]
        indices = [vocab[str(c)] for c in codes if str(c) in vocab]
        if not indices:
            continue
        np.add.at(result, row_idx, matrix[indices].sum(axis=0))
        counts[row_idx] += len(indices)

    mask          = counts > 0
    result[mask]  /= counts[mask, np.newaxis]
    result[~mask]  = unk_vec
    return result


def build_embeddings_vectorized(df):
    """
    Input:  dataframe chunk with columns provider_ids, specialty_codes,
            dx_list, procedure_codes
    Output: float32 array [N, 128]
    """
    p_emb  = _embed_column_vectorized(df['provider_ids'],    provider_matrix,  provider_vocab,  unk_emb['provider'])
    s_emb  = _embed_column_vectorized(df['specialty_codes'], specialty_matrix, specialty_vocab, unk_emb['specialty'])
    d_emb  = _embed_column_vectorized(df['dx_list'],         dx_matrix,        dx_vocab,        unk_emb['dx'])
    pr_emb = _embed_column_vectorized(df['procedure_codes'], procedure_matrix, procedure_vocab, unk_emb['procedure'])
    return np.concatenate([p_emb, s_emb, d_emb, pr_emb], axis=1).astype(np.float32)



# ============================================================
# STEP 3: LOAD DATA
# ============================================================
seq_cached   = LOAD_FROM_CACHE and os.path.exists(SEQ_CACHE_PATH)
label_cached = LOAD_FROM_CACHE and os.path.exists(LABEL_CACHE_PATH)

if seq_cached and label_cached:
    print("\nLoading from cache...")
    with open(SEQ_CACHE_PATH,   'rb') as f: member_sequences = pickle.load(f)
    with open(LABEL_CACHE_PATH, 'rb') as f: member_labels    = pickle.load(f)
    print(f"Members loaded from cache: {len(member_sequences):,}")

else:
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

    sequence_query = """
    SELECT
        s.member_id
        ,s.visit_seq_num
        ,s.delta_t_bucket
        ,s.provider_ids
        ,s.specialty_codes
        ,s.dx_list
        ,s.procedure_codes
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_sequence` s
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_sampled_members` m
        ON s.member_id = m.member_id
    ORDER BY s.member_id, s.visit_seq_num
    """

    label_query = f"""
    SELECT
        l.member_id
        ,l.visit_seq_num
        ,l.{COL_30}
        ,l.{COL_60}
        ,l.{COL_180}
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_label` l
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_sampled_members` m
        ON l.member_id = m.member_id
    ORDER BY l.member_id, l.visit_seq_num
    """

    member_sequences = defaultdict(list)
    member_labels    = defaultdict(list)

    if not seq_cached:
        print("\nLoading sequences from BQ...")
        df_seq = client.query(sequence_query).to_dataframe(create_bqstorage_client=True)
        print(f"Sequence rows: {len(df_seq):,}")

        print("Building embeddings (vectorized)...")
        all_embeddings = build_embeddings_vectorized(df_seq)
        print("Embedding build complete — building member sequences...")

        # assign embeddings back and build member dict — pure numpy/pandas, no embedding work
        df_seq['visit_emb']  = list(all_embeddings)
        df_seq['dt_bucket']   = df_seq['delta_t_bucket'].clip(upper=NUM_RATINGS - 1).astype(int)

        for member_id, group in df_seq.groupby('member_id', sort=False):
            member_sequences[member_id] = [
                {
                    'visit_seq_num'  : row.visit_seq_num
                    ,'delta_t_bucket': row.dt_bucket
                    ,'embedding'     : row.visit_emb
                }
                for row in group.itertuples()
            ]

        del all_embeddings, df_seq

        print("Saving sequence cache...")
        with open(SEQ_CACHE_PATH, 'wb') as f:
            pickle.dump(dict(member_sequences), f)

    else:
        print("Loading sequence cache...")
        with open(SEQ_CACHE_PATH, 'rb') as f:
            member_sequences = pickle.load(f)

    print("\nLoading labels...")
    df_lab = client.query(label_query).to_dataframe(create_bqstorage_client=True)
    print(f"Label rows: {len(df_lab):,}")

    # label build — no embedding work, fast pandas groupby
    for row in df_lab.itertuples():
        member_labels[row.member_id].append({
            'visit_seq_num': row.visit_seq_num
            ,'label_30'    : to_list(getattr(row, COL_30))
            ,'label_60'    : to_list(getattr(row, COL_60))
            ,'label_180'   : to_list(getattr(row, COL_180))
        })
    del df_lab

    print("Saving label cache...")
    with open(LABEL_CACHE_PATH, 'wb') as f:
        pickle.dump(dict(member_labels), f)
    print(f"Cache saved — set LOAD_FROM_CACHE=True for next run")

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

    train_data.append((member_id, visits[:train_cut], labels))
    val_data.append((member_id,   visits[:val_cut],   labels))
    test_data.append((member_id,  visits[:-1],        labels))

print(f"Train: {len(train_data):,}  Val: {len(val_data):,}  Test: {len(test_data):,}")

# ============================================================
# STEP 5: DATASET
# ============================================================
def make_label_vector(codes):
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    for c in to_list(codes):
        if c in label_vocab:
            vec[label_vocab[c]] = 1.0
    return vec

class VisitDataset(Dataset):
    def __init__(self, data, max_seq_len):
        self.samples = []
        for member_id, visits, labels in data:
            visits = visits[-max_seq_len:]
            n      = len(visits)

            last_seq_num = visits[-1]['visit_seq_num']
            label        = labels.get(last_seq_num)
            if label is None:
                continue

            embeddings = np.stack([v['embedding'] for v in visits])
            delta_t    = np.array([v['delta_t_bucket'] for v in visits], dtype=np.int64)

            pad = max_seq_len - n
            if pad > 0:
                embeddings = np.vstack([embeddings, np.zeros((pad, EMBEDDING_DIM), dtype=np.float32)])
                delta_t    = np.concatenate([delta_t, np.zeros(pad, dtype=np.int64)])

            self.samples.append((
                embeddings
                ,delta_t
                ,np.int64(n)
                ,make_label_vector(label['label_30'])
                ,make_label_vector(label['label_60'])
                ,make_label_vector(label['label_180'])
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        emb, dt, n, l30, l60, l180 = self.samples[idx]
        return {
            'embeddings': torch.from_numpy(emb)
            ,'delta_t'  : torch.from_numpy(dt)
            ,'length'   : torch.tensor(n,   dtype=torch.long)
            ,'label_30' : torch.from_numpy(l30)
            ,'label_60' : torch.from_numpy(l60)
            ,'label_180': torch.from_numpy(l180)
        }

train_dataset = VisitDataset(train_data, MAX_SEQ_LEN)
val_dataset   = VisitDataset(val_data,   MAX_SEQ_LEN)
test_dataset  = VisitDataset(test_data,  MAX_SEQ_LEN)

print(f"Train samples: {len(train_dataset):,}")
print(f"Val samples:   {len(val_dataset):,}")
print(f"Test samples:  {len(test_dataset):,}")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
    ,num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
)
val_loader   = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
    ,num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
)
test_loader  = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
    ,num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
)

print(f"Train batches: {len(train_loader):,}  Val: {len(val_loader):,}  Test: {len(test_loader):,}")

# ============================================================
# STEP 6: MODEL
# ============================================================
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

# ============================================================
# STEP 7: METRICS
# ============================================================
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

            B          = embeddings.size(0)
            n_samples += B

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

# ============================================================
# STEP 8: TRAINING LOOP
# ============================================================
optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler     = torch.cuda.amp.GradScaler()
bce_loss   = nn.BCELoss()
best_ndcg  = 0.0
patience   = 5
no_improve = 0

def get_raw_model(model):
    raw = model
    if hasattr(raw, '_orig_mod'):
        raw = raw._orig_mod
    if isinstance(raw, nn.DataParallel):
        raw = raw.module
    return raw

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
            'epoch'               : epoch
            ,'model_state_dict'   : get_raw_model(model).state_dict()
            ,'optimizer_state_dict': optimizer.state_dict()
            ,'scheduler_state_dict': scheduler.state_dict()
            ,'label_vocab'         : label_vocab
            ,'best_val_ndcg'       : best_ndcg
            ,'target'              : TARGET
            ,'config': {
                'embedding_dim'  : EMBEDDING_DIM
                ,'hstu_dim'      : HSTU_DIM
                ,'num_classes'   : NUM_CLASSES
                ,'max_seq_len'   : MAX_SEQ_LEN
                ,'num_blocks'    : NUM_BLOCKS
                ,'num_heads'     : NUM_HEADS
                ,'linear_dim'    : LINEAR_DIM
                ,'attention_dim' : ATTENTION_DIM
                ,'dropout_rate'  : DROPOUT_RATE
                ,'num_ratings'   : NUM_RATINGS
                ,'rating_dim'    : RATING_DIM
                ,'eval_k'        : EVAL_K
                ,'sample_pct'    : SAMPLE_PCT
                ,'target'        : TARGET
            }
        }, CHECKPOINT_PATH)
        print(f"  Saved → {CHECKPOINT_PATH}  val NDCG@T180: {best_ndcg:.4f}")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# ============================================================
# STEP 9: TEST EVALUATION
# ============================================================
print(f"\nLoading best model: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)

base_model = PureHSTU(
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
).to(DEVICE)

base_model.load_state_dict(checkpoint['model_state_dict'])

test_metrics = evaluate(test_loader, base_model, EVAL_K)
print_metrics(test_metrics, 'Test')

print(f"\nNotebook 2 complete — target: {TARGET}  checkpoint: {CHECKPOINT_PATH}")
