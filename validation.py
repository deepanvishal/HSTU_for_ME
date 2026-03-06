# ============================================================
# NOTEBOOK 3: VALIDATION
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

sys.path.insert(0, '.')

os.makedirs('./validation', exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
TARGET      = 'specialty'   # 'specialty' | 'provider' | 'dx'
SAMPLE_PCT  = 0.05

LABEL_VOCAB_PATH = {
    'specialty': './embeddings/specialty_label_vocab.pkl'
    ,'provider' : './embeddings/provider_label_vocab.pkl'
    ,'dx'       : './embeddings/dx_label_vocab.pkl'
}

CHECKPOINT_PATH  = f'./checkpoints/best_model_{TARGET}_{int(SAMPLE_PCT*100)}pct.pt'
SEQ_CACHE_PATH   = f'./data/member_sequences_{int(SAMPLE_PCT*100)}pct.pkl'
LABEL_CACHE_PATH = f'./data/member_labels_{TARGET}_{int(SAMPLE_PCT*100)}pct.pkl'

MAX_SEQ_LEN   = 20
BATCH_SIZE    = 512
EMBEDDING_DIM = 128
NUM_RATINGS   = 16
EVAL_K        = [3, 5, 10]
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS      = torch.cuda.device_count()
NUM_WORKERS   = min(4 * NUM_GPUS, 16)

print(f"Target:      {TARGET}")
print(f"Checkpoint:  {CHECKPOINT_PATH}")
print(f"Device:      {DEVICE}")

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
# STEP 1: LOAD EMBEDDINGS + CHECKPOINT
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

print(f"Loading checkpoint: {CHECKPOINT_PATH}")
checkpoint   = torch.load(CHECKPOINT_PATH, weights_only=False)
config       = checkpoint['config']
label_vocab  = checkpoint['label_vocab']
idx_to_label = {v: k for k, v in label_vocab.items()}
NUM_CLASSES  = len(label_vocab)

print(f"Label vocab:   {NUM_CLASSES:,}")
print(f"Best val NDCG: {checkpoint['best_val_ndcg']:.4f}")

# ============================================================
# STEP 2: LOAD MODEL
# ============================================================
from hstu_pytorch import PureHSTU

model = PureHSTU(
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

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# STEP 3: LOAD CACHED DATA + BUILD TEST SET
# ============================================================
print("\nLoading cached data...")
with open(SEQ_CACHE_PATH,   'rb') as f: member_sequences = pickle.load(f)
with open(LABEL_CACHE_PATH, 'rb') as f: member_labels    = pickle.load(f)
print(f"Members: {len(member_sequences):,}")

def batch_lookup(codes, matrix, vocab, unk):
    codes   = to_list(codes)
    indices = [vocab[c] for c in codes if c in vocab]
    if not indices:
        return unk
    return matrix[indices].mean(axis=0).astype(np.float32)

def get_visit_embedding(providers, specialties, dxs, procedures):
    p  = batch_lookup(providers,   provider_matrix,  provider_vocab,  unk_emb['provider'])
    s  = batch_lookup(specialties, specialty_matrix, specialty_vocab, unk_emb['specialty'])
    d  = batch_lookup(dxs,         dx_matrix,        dx_vocab,        unk_emb['dx'])
    pr = batch_lookup(procedures,  procedure_matrix, procedure_vocab, unk_emb['procedure'])
    return np.concatenate([p, s, d, pr]).astype(np.float32)

def make_label_vector(codes):
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    for c in to_list(codes):
        if c in label_vocab:
            vec[label_vocab[c]] = 1.0
    return vec

# build test set — visits[:-1] as input, second-to-last visit label
test_data = []
for member_id, visits in member_sequences.items():
    if len(visits) < 4:
        continue
    labels = {l['visit_seq_num']: l for l in member_labels.get(member_id, [])}
    test_data.append((member_id, visits[:-1], labels, len(visits)))

print(f"Test members: {len(test_data):,}")

# ============================================================
# STEP 4: DATASET
# ============================================================
class TestDataset(Dataset):
    def __init__(self, data, max_seq_len):
        self.samples    = []
        self.member_ids = []
        self.seq_lens   = []
        self.actuals    = []

        for member_id, visits, labels, full_len in data:
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
            self.member_ids.append(member_id)
            self.seq_lens.append(full_len)
            self.actuals.append({
                'actual_30' : label['label_30']
                ,'actual_60' : label['label_60']
                ,'actual_180': label['label_180']
            })

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
            ,'idx'      : torch.tensor(idx, dtype=torch.long)
        }

test_dataset = TestDataset(test_data, MAX_SEQ_LEN)
test_loader  = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
    ,num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
)
print(f"Test samples: {len(test_dataset):,}  Batches: {len(test_loader):,}")

# ============================================================
# STEP 5: METRICS
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

def compute_metrics(pred, label, k_list):
    metrics = {}
    for k in k_list:
        metrics[f'ndcg@{k}'] = ndcg_at_k(pred, label, k)
        metrics[f'prec@{k}'] = precision_at_k(pred, label, k)
        metrics[f'rec@{k}']  = recall_at_k(pred, label, k)
        metrics[f'hit@{k}']  = hit_rate_at_k(pred, label, k)
    return metrics

def print_metrics(metrics, split='Test'):
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
# STEP 6: RUN INFERENCE ON TEST SET
# ============================================================
print("\nRunning inference on test set...")

all_pred_30  = []
all_pred_60  = []
all_pred_180 = []
all_lab_30   = []
all_lab_60   = []
all_lab_180  = []
all_attn     = []
all_indices  = []

with torch.no_grad():
    for batch in test_loader:
        embeddings = batch['embeddings'].to(DEVICE, non_blocking=True)
        delta_t    = batch['delta_t'].to(DEVICE,    non_blocking=True)
        lengths    = batch['length'].to(DEVICE,     non_blocking=True)
        label_30   = batch['label_30'].to(DEVICE,   non_blocking=True)
        label_60   = batch['label_60'].to(DEVICE,   non_blocking=True)
        label_180  = batch['label_180'].to(DEVICE,  non_blocking=True)

        with torch.cuda.amp.autocast():
            pred_30, pred_60, pred_180, attn = model(
                embeddings, delta_t, lengths, return_attention=True
            )

        all_pred_30.append(pred_30.float().cpu())
        all_pred_60.append(pred_60.float().cpu())
        all_pred_180.append(pred_180.float().cpu())
        all_lab_30.append(label_30.float().cpu())
        all_lab_60.append(label_60.float().cpu())
        all_lab_180.append(label_180.float().cpu())
        all_attn.append(attn.float().cpu())
        all_indices.append(batch['idx'])

all_pred_30  = torch.cat(all_pred_30,  dim=0)
all_pred_60  = torch.cat(all_pred_60,  dim=0)
all_pred_180 = torch.cat(all_pred_180, dim=0)
all_lab_30   = torch.cat(all_lab_30,   dim=0)
all_lab_60   = torch.cat(all_lab_60,   dim=0)
all_lab_180  = torch.cat(all_lab_180,  dim=0)
all_attn     = torch.cat(all_attn,     dim=0)  # [N, MAX_SEQ_LEN]
all_indices  = torch.cat(all_indices,  dim=0)

print(f"Inference complete — {all_pred_30.shape[0]:,} members")

# ============================================================
# STEP 7: OVERALL METRICS
# ============================================================
overall_metrics = {}
for window, pred, label in [
    ('T30',  all_pred_30,  all_lab_30)
    ,('T60',  all_pred_60,  all_lab_60)
    ,('T180', all_pred_180, all_lab_180)
]:
    m = compute_metrics(pred, label, EVAL_K)
    for key, val in m.items():
        overall_metrics[f'{window}_{key}'] = val.mean().item()

print_metrics(overall_metrics, 'Overall Test')

# ============================================================
# STEP 8: ATTENTION WEIGHTS
# ============================================================
print("\nBuilding attention analysis...")

attn_rows = []
for i, sample_idx in enumerate(all_indices.tolist()):
    member_id = test_dataset.member_ids[sample_idx]
    seq_len   = test_dataset.seq_lens[sample_idx]
    attn_vec  = all_attn[i].numpy()  # [MAX_SEQ_LEN]

    attn_rows.append({
        'member_id'         : member_id
        ,'seq_len'          : seq_len
        ,'attn_weights'     : attn_vec.tolist()
        ,'top_attn_position': int(np.argmax(attn_vec))
        ,'recency_bias'     : float(attn_vec[-1] / (attn_vec.sum() + 1e-8))
    })

df_attn      = pd.DataFrame(attn_rows)
attn_matrix  = np.stack(df_attn['attn_weights'].values)
mean_attn    = attn_matrix.mean(axis=0)  # [MAX_SEQ_LEN]

print("\nMean attention by position (0=oldest, 19=most recent):")
for pos, score in enumerate(mean_attn):
    bar = '|' * int(score * 200)
    print(f"  pos {pos:2d}: {score:.4f} {bar}")

# ============================================================
# STEP 9: FEATURE ABLATION
# ============================================================
print("\nRunning feature ablation...")

ABLATION_CONFIGS = {
    'baseline'     : (None, None)
    ,'no_provider' : (0,    32)
    ,'no_specialty': (32,   64)
    ,'no_dx'       : (64,   96)
    ,'no_procedure': (96,  128)
}

ablation_results = []

for ablation_name, (dim_start, dim_end) in ABLATION_CONFIGS.items():
    abl_sums  = defaultdict(float)
    n_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch['embeddings'].to(DEVICE, non_blocking=True)
            delta_t    = batch['delta_t'].to(DEVICE,    non_blocking=True)
            lengths    = batch['length'].to(DEVICE,     non_blocking=True)
            label_30   = batch['label_30'].to(DEVICE,   non_blocking=True)
            label_60   = batch['label_60'].to(DEVICE,   non_blocking=True)
            label_180  = batch['label_180'].to(DEVICE,  non_blocking=True)

            if dim_start is not None:
                embeddings = embeddings.clone()
                embeddings[:, :, dim_start:dim_end] = 0.0

            with torch.cuda.amp.autocast():
                pred_30, pred_60, pred_180 = model(embeddings, delta_t, lengths)

            pred_30  = pred_30.float()
            pred_60  = pred_60.float()
            pred_180 = pred_180.float()
            n_samples += embeddings.size(0)

            for k in EVAL_K:
                for window, pred, label in [
                    ('T30',  pred_30,  label_30)
                    ,('T60',  pred_60,  label_60)
                    ,('T180', pred_180, label_180)
                ]:
                    abl_sums[f'{window}_ndcg@{k}'] += ndcg_at_k(pred, label, k).sum().item()

    row = {'ablation': ablation_name}
    for key, val in abl_sums.items():
        row[key] = round(val / n_samples, 4)
    ablation_results.append(row)
    print(f"  {ablation_name:15s} — T180 NDCG@10: {row.get('T180_ndcg@10', 0):.4f}")

df_ablation = pd.DataFrame(ablation_results)

baseline_row = df_ablation[df_ablation['ablation'] == 'baseline'].iloc[0]
for col in [c for c in df_ablation.columns if 'ndcg' in c]:
    df_ablation[f'{col}_drop'] = round(baseline_row[col] - df_ablation[col], 4)

print("\nNDCG Drop vs Baseline (T180@10):")
print(df_ablation[['ablation', 'T180_ndcg@10', 'T180_ndcg@10_drop']].to_string(index=False))

# ============================================================
# STEP 10: SAVE RESULTS
# ============================================================
print("\nSaving validation results...")
suffix = f'{TARGET}_{int(SAMPLE_PCT*100)}pct'

# overall metrics
pd.DataFrame([overall_metrics]).to_csv(f'./validation/overall_metrics_{suffix}.csv', index=False)

# per-member predictions + actuals
pred_rows = []
for i, sample_idx in enumerate(all_indices.tolist()):
    member_id = test_dataset.member_ids[sample_idx]
    actuals   = test_dataset.actuals[sample_idx]

    def topk_decode(preds, k=10):
        top = torch.topk(preds, k)
        return (
            [idx_to_label.get(j, 'UNK') for j in top.indices.tolist()]
            ,[round(s, 4) for s in top.values.tolist()]
        )

    codes_30,  sc_30  = topk_decode(all_pred_30[i])
    codes_60,  sc_60  = topk_decode(all_pred_60[i])
    codes_180, sc_180 = topk_decode(all_pred_180[i])

    pred_rows.append({
        'member_id'   : member_id
        ,'t30_pred'   : codes_30
        ,'t30_scores' : sc_30
        ,'t60_pred'   : codes_60
        ,'t60_scores' : sc_60
        ,'t180_pred'  : codes_180
        ,'t180_scores': sc_180
        ,'actual_30'  : actuals['actual_30']
        ,'actual_60'  : actuals['actual_60']
        ,'actual_180' : actuals['actual_180']
    })

df_preds = pd.DataFrame(pred_rows)
df_preds.to_parquet(f'./validation/predictions_{suffix}.parquet', index=False)
df_preds.to_csv(f'./validation/predictions_{suffix}.csv',         index=False)

# attention summary (no raw weights in csv)
df_attn.drop(columns=['attn_weights']).to_csv(f'./validation/attention_summary_{suffix}.csv', index=False)
df_attn.to_parquet(f'./validation/attention_weights_{suffix}.parquet', index=False)

# mean attention by position
pd.DataFrame({
    'position'      : list(range(MAX_SEQ_LEN))
    ,'mean_attention': mean_attn.tolist()
}).to_csv(f'./validation/mean_attention_by_position_{suffix}.csv', index=False)

# ablation
df_ablation.to_csv(f'./validation/ablation_results_{suffix}.csv', index=False)

print("\nFiles saved:")
for f in sorted(os.listdir('./validation')):
    path = f'./validation/{f}'
    size = f"{os.path.getsize(path) / 1e6:.2f} MB"
    print(f"  {f}  {size}")

print(f"\nNotebook 3 complete — target: {TARGET}")
