# ============================================================
# NB_07 — scoring_provider.py
# Purpose : Score all trained models on test set
#           Compare SASRec, BERT4Rec, HSTU vs Markov baseline
#           Write per-trigger scores to BQ for post-hoc analysis
# Sources : ./cache_provider_{SAMPLE}/test_*.npy  (NB_03)
#           ./models/sasrec_provider_{SAMPLE}_ep*.pt
#           ./models/bert4rec_provider_{SAMPLE}_ep*.pt
#           ./models/hstu_provider_{SAMPLE}_ep*.pt
#           BQ: A870800_gen_rec_provider_markov_metrics_{SAMPLE}
# Output  : ./output/provider_model_comparison_{SAMPLE}.csv
#           BQ: A870800_gen_rec_provider_trigger_scores (APPEND)
# ============================================================

import gc
import os
import pickle
import time
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from google.cloud import bigquery
from IPython.display import display, Markdown

print("Imports done")
print(f"PyTorch: {torch.__version__}")

# ── CONFIG ────────────────────────────────────────────────────────────────────
SAMPLE       = "5pct"
MAX_SEQ_LEN  = 20
PAD_IDX      = 0
UNK_IDX      = 1
K_VALUES     = [1, 3, 5]
BATCH_SIZE   = 512       # smaller for scoring — score tensor (B, 31K) is large
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
RUN_TS       = datetime.now().strftime("%Y-%m-%d %H:%M")

DS           = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
CACHE_DIR    = f"./cache_provider_{SAMPLE}"
MODEL_DIR    = "./models"
OUTPUT_DIR   = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = bigquery.Client(project="anbc-hcb-dev")

print(f"Device: {DEVICE} | Sample: {SAMPLE}")
display(Markdown(f"""
## Config
| Parameter | Value |
|---|---|
| Sample | {SAMPLE} |
| Batch size (scoring) | {BATCH_SIZE} |
| K values | {K_VALUES} |
| Device | {DEVICE} |
"""))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD VOCABS + TEST DATA
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 1 — Load Test Data"))

with open(f"{CACHE_DIR}/provider_vocab.pkl",  "rb") as f: provider_vocab  = pickle.load(f)
with open(f"{CACHE_DIR}/specialty_vocab.pkl", "rb") as f: specialty_vocab = pickle.load(f)
with open(f"{CACHE_DIR}/dx_vocab.pkl",        "rb") as f: dx_vocab        = pickle.load(f)
with open(f"{CACHE_DIR}/idx_to_provider.pkl", "rb") as f: idx_to_provider = pickle.load(f)

PROVIDER_VOCAB_SIZE = len(provider_vocab)
SPEC_VOCAB_SIZE     = len(specialty_vocab)
DX_VOCAB_SIZE       = len(dx_vocab)

keys = ["seq_matrix", "delta_t_matrix", "trigger_token", "seq_lengths",
        "lab_t30", "lab_t60", "lab_t180",
        "is_t30", "is_t60", "is_t180",
        "member_ids", "trigger_dates", "trigger_dxs", "segments",
        "from_provider_ids"]

test_data = {k: np.load(f"{CACHE_DIR}/test_{k}.npy", allow_pickle=True) for k in keys}
M = test_data["seq_matrix"].shape[0]

print(f"Test triggers: {M:,}")
print(f"Provider vocab: {PROVIDER_VOCAB_SIZE:,}")
print(f"Date range: {test_data['trigger_dates'].min()} → {test_data['trigger_dates'].max()}")
print(f"Section 1 done — {time.time()-t0:.1f}s")
display(Markdown(f"**Test triggers:** {M:,} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATASET + HELPERS
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 2 — Dataset"))


def sparse_to_multihot(label_list, vocab_size):
    v = torch.zeros(vocab_size, dtype=torch.float32)
    if len(label_list) > 0:
        idx = torch.tensor(label_list, dtype=torch.long)
        idx = idx[idx >= 2]
        if len(idx) > 0:
            v.scatter_(0, idx, 1.0)
    return v


class TestDataset(Dataset):
    """Single dataset used for all three models — delta_t included for HSTU."""
    def __init__(self, data):
        self.seq_matrix    = data["seq_matrix"]
        self.delta_t_matrix= data["delta_t_matrix"]
        self.trigger_token = data["trigger_token"]
        self.seq_lengths   = data["seq_lengths"]
        self.lab_t30       = data["lab_t30"]
        self.lab_t60       = data["lab_t60"]
        self.lab_t180      = data["lab_t180"]
        self.is_t30        = data["is_t30"]
        self.is_t60        = data["is_t60"]
        self.is_t180       = data["is_t180"]
        self.member_ids    = data["member_ids"]
        self.trigger_dates = data["trigger_dates"]
        self.trigger_dxs   = data["trigger_dxs"]
        self.segments      = data["segments"]

    def __len__(self):
        return len(self.seq_lengths)

    def __getitem__(self, idx):
        return {
            "seq":          torch.from_numpy(self.seq_matrix[idx].copy()),
            "delta_t":      torch.from_numpy(self.delta_t_matrix[idx].copy()),
            "trigger":      torch.from_numpy(self.trigger_token[idx].copy()),
            "seq_len":      torch.tensor(int(self.seq_lengths[idx]), dtype=torch.long),
            "lab_t30":      self.lab_t30[idx],
            "lab_t60":      self.lab_t60[idx],
            "lab_t180":     self.lab_t180[idx],
            "is_t30":       torch.tensor(bool(self.is_t30[idx]),  dtype=torch.bool),
            "is_t60":       torch.tensor(bool(self.is_t60[idx]),  dtype=torch.bool),
            "is_t180":      torch.tensor(bool(self.is_t180[idx]), dtype=torch.bool),
            "member_id":    self.member_ids[idx],
            "trigger_date": self.trigger_dates[idx],
            "trigger_dx":   self.trigger_dxs[idx],
            "segment":      self.segments[idx],
        }


def collate_test(batch, vocab_size):
    return {
        "seq":          torch.stack([b["seq"]     for b in batch]),
        "delta_t":      torch.stack([b["delta_t"] for b in batch]),
        "trigger":      torch.stack([b["trigger"] for b in batch]),
        "seq_len":      torch.stack([b["seq_len"] for b in batch]),
        "lab_t30":      torch.stack([sparse_to_multihot(b["lab_t30"],  vocab_size) for b in batch]),
        "lab_t60":      torch.stack([sparse_to_multihot(b["lab_t60"],  vocab_size) for b in batch]),
        "lab_t180":     torch.stack([sparse_to_multihot(b["lab_t180"], vocab_size) for b in batch]),
        "is_t30":       torch.stack([b["is_t30"]  for b in batch]),
        "is_t60":       torch.stack([b["is_t60"]  for b in batch]),
        "is_t180":      torch.stack([b["is_t180"] for b in batch]),
        "member_ids":   [b["member_id"]    for b in batch],
        "trigger_dates":[b["trigger_date"] for b in batch],
        "trigger_dxs":  [b["trigger_dx"]   for b in batch],
        "segments":     [b["segment"]      for b in batch],
    }


test_loader = DataLoader(
    TestDataset(test_data),
    batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=partial(collate_test, vocab_size=PROVIDER_VOCAB_SIZE),
    num_workers=4, pin_memory=(DEVICE == "cuda"), persistent_workers=True,
)

print(f"Test loader: {len(test_loader)} batches")
print(f"Section 2 done — {time.time()-t0:.1f}s")
display(Markdown(f"**Test batches:** {len(test_loader)} | **Time:** {time.time()-t0:.1f}s"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL DEFINITIONS
# Minimal definitions needed for loading checkpoints
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("---\n## Section 3 — Model Definitions"))


class PointWiseFeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim*4, dim), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)


# ── SASRec ────────────────────────────────────────────────────────────────────
class SASRecBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff    = PointWiseFeedForward(dim, dropout)
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)
        self.drop  = nn.Dropout(dropout)
    def forward(self, x, causal, pad):
        a, _ = self.attn(x, x, x, attn_mask=causal, key_padding_mask=pad, need_weights=False)
        x = self.norm1(x + self.drop(a)); x = self.norm2(x + self.ff(x)); return x

class SASRecProvider(nn.Module):
    def __init__(self, provider_vocab_size, spec_vocab_size, dx_vocab_size,
                 d_model, max_seq_len, num_heads, num_blocks, dropout):
        super().__init__()
        self.d_model = d_model
        self.provider_emb = nn.Embedding(provider_vocab_size, d_model, padding_idx=PAD_IDX)
        self.spec_emb     = nn.Embedding(spec_vocab_size,     d_model, padding_idx=PAD_IDX)
        self.dx_emb       = nn.Embedding(dx_vocab_size,       d_model, padding_idx=PAD_IDX)
        self.pos_emb      = nn.Embedding(max_seq_len+1,       d_model)
        self.emb_drop     = nn.Dropout(dropout)
        self.blocks       = nn.ModuleList([SASRecBlock(d_model, num_heads, dropout) for _ in range(num_blocks)])
        self.norm         = nn.LayerNorm(d_model)
    def encode(self, seq, trigger, seq_len):
        B, L, _ = seq.shape
        x  = self.provider_emb(seq[:,:,0]) + self.spec_emb(seq[:,:,1])
        x  = torch.cat([x, self.dx_emb(trigger)], dim=1)
        TL = L + 1
        x  = self.emb_drop(x + self.pos_emb(torch.arange(TL, device=seq.device).unsqueeze(0)))
        pad_mask   = torch.cat([seq[:,:,0]==PAD_IDX, torch.zeros(B,1,dtype=torch.bool,device=seq.device)], dim=1)
        causal_mask= torch.triu(torch.ones(TL,TL,device=seq.device,dtype=torch.bool), diagonal=1)
        for block in self.blocks: x = block(x, causal_mask, pad_mask)
        return self.norm(x)[:,-1,:]
    def forward(self, seq, trigger, seq_len):
        u   = self.encode(seq, trigger, seq_len)
        ids = torch.arange(2, self.provider_emb.num_embeddings, device=seq.device)
        return u @ self.provider_emb(ids).T, u


# ── BERT4Rec ──────────────────────────────────────────────────────────────────
class BERT4RecBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff    = PointWiseFeedForward(dim, dropout)
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)
        self.drop  = nn.Dropout(dropout)
    def forward(self, x, pad):
        a, _ = self.attn(x, x, x, key_padding_mask=pad, need_weights=False)
        x = self.norm1(x + self.drop(a)); x = self.norm2(x + self.ff(x)); return x

class BERT4RecProvider(nn.Module):
    def __init__(self, provider_vocab_size, spec_vocab_size, dx_vocab_size,
                 d_model, max_seq_len, num_heads, num_blocks, dropout):
        super().__init__()
        self.d_model      = d_model
        self.max_seq_len  = max_seq_len
        self.provider_emb = nn.Embedding(provider_vocab_size+1, d_model, padding_idx=PAD_IDX)
        self.spec_emb     = nn.Embedding(spec_vocab_size,       d_model, padding_idx=PAD_IDX)
        self.dx_emb       = nn.Embedding(dx_vocab_size,         d_model, padding_idx=PAD_IDX)
        self.pos_emb      = nn.Embedding(max_seq_len+1,         d_model)
        self.emb_drop     = nn.Dropout(dropout)
        self.blocks       = nn.ModuleList([BERT4RecBlock(d_model, num_heads, dropout) for _ in range(num_blocks)])
        self.norm         = nn.LayerNorm(d_model)
    def forward(self, seq, trigger, target_mask):
        B, L, _ = seq.shape
        x  = self.provider_emb(seq[:,:,0]) + self.spec_emb(seq[:,:,1])
        x  = torch.cat([x, self.dx_emb(trigger)], dim=1)
        TL = L + 1
        x  = self.emb_drop(x + self.pos_emb(torch.arange(TL, device=seq.device).unsqueeze(0)))
        pad= torch.cat([(seq[:,:,0]==PAD_IDX)&(target_mask[:,:L]==0),
                        torch.zeros(B,1,dtype=torch.bool,device=seq.device)], dim=1)
        for block in self.blocks: x = block(x, pad)
        x = self.norm(x)
        mask_pos  = target_mask[:,:L].float().argmax(dim=1)
        user_repr = x[torch.arange(B,device=seq.device), mask_pos, :]
        ids       = torch.arange(2, self.provider_emb.num_embeddings-1, device=seq.device)
        return user_repr @ self.provider_emb(ids).T, user_repr


# ── HSTU ─────────────────────────────────────────────────────────────────────
class RelativeBucketedTimeAndPositionBias(nn.Module):
    def __init__(self, max_seq_len, num_buckets=64):
        super().__init__()
        self._max_seq_len = max_seq_len; self._num_buckets = num_buckets
        self._ts_w  = nn.Parameter(torch.empty(num_buckets+1).normal_(0,0.02))
        self._pos_w = nn.Parameter(torch.empty(2*max_seq_len-1).normal_(0,0.02))
        self._cache = None
    def _pos(self, dev):
        N=self._max_seq_len
        t=F.pad(self._pos_w[:2*N-1],[0,N]).repeat(N)
        t=t[:-N].reshape(N,3*N-2); r=(2*N-1)//2
        return t[:,r:-r].unsqueeze(0)
    def forward(self, ts):
        B,N=ts.shape
        if self.training or self._cache is None: self._cache=self._pos(ts.device)
        pb=self._cache.to(ts.device)
        ext=torch.cat([ts,ts[:,N-1:N]],dim=1)
        diff=ext[:,1:].unsqueeze(2)-ext[:,:-1].unsqueeze(1)
        bk=torch.clamp((torch.log(diff.abs().clamp(min=1))/0.301).long(),0,self._num_buckets).detach()
        return pb+self._ts_w[bk.view(-1)].view(B,N,N)

class HSTUBlock(nn.Module):
    def __init__(self, embedding_dim, linear_dim, attention_dim, num_heads,
                 dropout_rate, attn_dropout_rate, max_seq_len, num_buckets=64):
        super().__init__()
        self._num_heads=num_heads; self._linear_dim=linear_dim; self._attn_dim=attention_dim
        self._uvqk=nn.Parameter(torch.empty(embedding_dim,linear_dim*2*num_heads+attention_dim*num_heads*2).normal_(0,0.02))
        self._o=nn.Linear(linear_dim*num_heads,embedding_dim)
        self._norm_x=nn.LayerNorm(embedding_dim); self._norm_attn=nn.LayerNorm(linear_dim*num_heads)
        self._rel_bias=RelativeBucketedTimeAndPositionBias(max_seq_len,num_buckets)
        self._dropout=nn.Dropout(dropout_rate); self._attn_drop=attn_dropout_rate
        nn.init.xavier_uniform_(self._o.weight)
    def forward(self,x,ts,cm,pm):
        B,N,D=x.shape; H,Dv,Dq=self._num_heads,self._linear_dim,self._attn_dim
        out=F.silu(self._norm_x(x).reshape(B*N,D)@self._uvqk)
        u,v,q,k=torch.split(out,[H*Dv,H*Dv,H*Dq,H*Dq],dim=-1)
        q=q.view(B,N,H,Dq).permute(0,2,1,3).reshape(B*H,N,Dq)
        k=k.view(B,N,H,Dq).permute(0,2,1,3).reshape(B*H,N,Dq)
        v=v.view(B,N,H,Dv).permute(0,2,1,3).reshape(B*H,N,Dv)
        qk=torch.bmm(q,k.transpose(1,2)).view(B,H,N,N)+self._rel_bias(ts).unsqueeze(1)
        qk=F.silu(qk)/N
        qk=qk.masked_fill(cm.unsqueeze(0).unsqueeze(0),0.0).masked_fill(pm.unsqueeze(1).unsqueeze(2),0.0)
        qk=F.dropout(qk,p=self._attn_drop,training=self.training)
        ao=torch.bmm(qk.reshape(B*H,N,N),v).view(B,H,N,Dv).permute(0,2,1,3).reshape(B,N,H*Dv)
        ao=self._norm_attn(ao); u=u.view(B,N,H*Dv)
        return (self._o(self._dropout(u*ao))+x).masked_fill(pm.unsqueeze(-1),0.0)

class HSTUProvider(nn.Module):
    def __init__(self, provider_vocab_size, spec_vocab_size, dx_vocab_size,
                 d_model, delta_t_dim, num_ratings, hstu_dim,
                 linear_dim, attention_dim, max_seq_len,
                 num_heads, num_blocks, dropout, attn_dropout, num_buckets):
        super().__init__()
        self.d_model=d_model; self.hstu_dim=hstu_dim; self.max_seq_len=max_seq_len
        self.provider_emb =nn.Embedding(provider_vocab_size,d_model,padding_idx=PAD_IDX)
        self.spec_emb     =nn.Embedding(spec_vocab_size,    d_model,padding_idx=PAD_IDX)
        self.dx_emb       =nn.Embedding(dx_vocab_size,      d_model,padding_idx=PAD_IDX)
        self.delta_t_emb  =nn.Embedding(num_ratings,delta_t_dim)
        self.pos_emb      =nn.Embedding(max_seq_len+1,hstu_dim)
        self.input_dropout=nn.Dropout(dropout)
        self.blocks=nn.ModuleList([HSTUBlock(hstu_dim,linear_dim,attention_dim,num_heads,
                                             dropout,attn_dropout,max_seq_len+1,num_buckets)
                                   for _ in range(num_blocks)])
        self.out_norm=nn.LayerNorm(hstu_dim)
        self.register_buffer('_causal_mask',
            torch.triu(torch.ones(max_seq_len+1,max_seq_len+1,dtype=torch.bool),diagonal=1))
    def encode(self,seq,delta_t,trigger,seq_len):
        B,L,_=seq.shape
        item_emb=self.provider_emb(seq[:,:,0])+self.spec_emb(seq[:,:,1])
        dt_capped=delta_t.clamp(0,self.delta_t_emb.num_embeddings-1)
        x_seq=torch.cat([item_emb,self.delta_t_emb(dt_capped)],dim=-1)
        dx_pad=F.pad(self.dx_emb(trigger),(0,self.hstu_dim-self.d_model))
        x=torch.cat([x_seq,dx_pad],dim=1)
        pos=torch.arange(L+1,device=seq.device).unsqueeze(0)
        x=x*math.sqrt(self.hstu_dim)+self.pos_emb(pos)
        x=self.input_dropout(x)
        ts=torch.cat([dt_capped,torch.zeros(B,1,dtype=torch.long,device=seq.device)],dim=1).long()
        pm=torch.cat([(seq[:,:,0]==PAD_IDX),torch.zeros(B,1,dtype=torch.bool,device=seq.device)],dim=1)
        x=x.masked_fill(pm.unsqueeze(-1),0.0)
        for block in self.blocks: x=block(x,ts,self._causal_mask,pm)
        return self.out_norm(x)[:,-1,:]
    def forward(self,seq,delta_t,trigger,seq_len):
        u=self.encode(seq,delta_t,trigger,seq_len)
        ids=torch.arange(2,self.provider_emb.num_embeddings,device=seq.device)
        return u[:,:self.d_model]@self.provider_emb(ids).T, u


print("Model definitions loaded ✓")
print(f"Section 3 done — {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — METRICS HELPER
# ══════════════════════════════════════════════════════════════════════════════

def metrics_at_k(scores, labels_multihot, k):
    topk_idx  = scores.topk(k, dim=1).indices
    topk_hits = labels_multihot.gather(1, topk_idx)
    hit       = (topk_hits.sum(1) > 0).float().mean().item()
    prec      = topk_hits.sum(1).float().div(k).mean().item()
    n_pos     = labels_multihot.sum(1).clamp(min=1)
    rec       = topk_hits.sum(1).float().div(n_pos).mean().item()
    positions = torch.arange(1, k+1, dtype=torch.float32, device=scores.device)
    discounts = 1.0 / torch.log2(positions + 1)
    dcg       = (topk_hits.float() * discounts).sum(1)
    ideal     = torch.zeros(scores.shape[0], k, device=scores.device)
    for i in range(scores.shape[0]):
        ideal[i, :min(int(n_pos[i].item()), k)] = 1.0
    idcg      = (ideal * discounts).sum(1).clamp(min=1e-8)
    ndcg      = (dcg / idcg).mean().item()
    return {"hit": hit, "prec": prec, "rec": rec, "ndcg": ndcg}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SCORING FUNCTION
# Runs one model over full test set, returns metrics + per-trigger rows for BQ
# ══════════════════════════════════════════════════════════════════════════════

def score_model(model, loader, model_name, mask_idx=None):
    """
    Score a model on the test set.
    Returns (metrics_dict, rows_for_bq)
    rows_for_bq: list of dicts with top5 predictions per trigger per window
    """
    model.eval()
    buckets    = {"T0_30":  ("is_t30","lab_t30"),
                  "T30_60": ("is_t60","lab_t60"),
                  "T60_180":("is_t180","lab_t180")}
    all_metrics = {b: {k: {"hit":[],"prec":[],"rec":[],"ndcg":[]}
                        for k in K_VALUES} for b in buckets}
    bq_rows    = []
    MASK_IDX   = mask_idx  # only for BERT4Rec

    is_bert    = (model_name == "BERT4Rec")
    is_hstu    = (model_name == "HSTU")

    with torch.no_grad():
        for batch in loader:
            seq     = batch["seq"].to(DEVICE,     non_blocking=True)
            delta_t = batch["delta_t"].to(DEVICE, non_blocking=True)
            trigger = batch["trigger"].to(DEVICE, non_blocking=True)
            seq_len = batch["seq_len"].to(DEVICE, non_blocking=True)
            B       = seq.shape[0]

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                if is_bert:
                    # Mask last real position for inference
                    L = seq.shape[1]
                    target_mask = torch.zeros(B, L+1, device=DEVICE)
                    for i in range(B):
                        sl = int(seq_len[i].item())
                        if sl > 0:
                            seq[i, L - 1, 0] = MASK_IDX
                            target_mask[i, L - 1] = 1.0
                        else:
                            target_mask[i, 0] = 1.0
                    scores, _ = model(seq, trigger, target_mask)
                elif is_hstu:
                    scores, _ = model(seq, delta_t, trigger, seq_len)
                else:  # SASRec
                    scores, _ = model(seq, trigger, seq_len)

            # Pad scores to full vocab for label alignment
            pad_scores  = torch.zeros(B, 2, device=scores.device)
            full_scores = torch.cat([pad_scores, scores], dim=1)  # (B, V)

            # ── Metrics ───────────────────────────────────────────────────────
            for bucket, (flag_key, lab_key) in buckets.items():
                mask = batch[flag_key].to(DEVICE)
                if mask.sum() == 0: continue
                labs = batch[lab_key].to(DEVICE)
                s_m  = full_scores[mask]
                l_m  = labs[mask]
                for k in K_VALUES:
                    m = metrics_at_k(s_m, l_m, k)
                    for metric, val in m.items():
                        all_metrics[bucket][k][metric].append(val)

            # ── BQ rows — top5 per trigger per window ─────────────────────────
            top5_scores_cpu = full_scores.topk(5, dim=1)
            top5_idx_cpu    = top5_scores_cpu.indices.cpu().numpy()
            top5_vals_cpu   = top5_scores_cpu.values.cpu().numpy()

            for i in range(B):
                for bucket, (flag_key, lab_key) in buckets.items():
                    if not batch[flag_key][i].item():
                        continue
                    top5_providers = [str(idx_to_provider.get(idx, idx))
                                      for idx in top5_idx_cpu[i].tolist()]
                    top5_sc        = [round(float(v), 4) for v in top5_vals_cpu[i]]
                    true_ids       = batch[lab_key][i].nonzero(as_tuple=True)[0].tolist()
                    true_providers = [str(idx_to_provider.get(idx, idx)) for idx in true_ids]
                    top5_providers_str = [str(p) for p in top5_providers]

                    bq_rows.append({
                        "member_id":        str(batch["member_ids"][i]),
                        "trigger_date":     str(batch["trigger_dates"][i]),
                        "trigger_dx":       str(batch["trigger_dxs"][i]),
                        "member_segment":   str(batch["segments"][i]),
                        "time_bucket":      str(bucket),
                        "true_labels":      "|".join(sorted(str(p) for p in true_providers)),
                        "top5_predictions": "|".join(str(p) for p in top5_providers_str),
                        "top5_scores":      "|".join(str(s) for s in top5_sc),
                        "model":            str(model_name),
                        "sample":           str(SAMPLE),
                        "run_timestamp":    str(RUN_TS),
                    })

    # Aggregate metrics
    results = {}
    for bucket in buckets:
        for k in K_VALUES:
            for metric in ["hit","prec","rec","ndcg"]:
                vals = all_metrics[bucket][k][metric]
                if vals:
                    results[f"{bucket}_{metric}@{k}"] = round(np.mean(vals), 4)
    return results, bq_rows


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — LOAD + SCORE EACH MODEL
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 6 — Score Models"))

all_model_metrics = {}
all_bq_rows       = []


def load_latest_checkpoint(pattern):
    """Find most recent checkpoint matching pattern."""
    import glob
    files = glob.glob(f"{MODEL_DIR}/{pattern}")
    if not files:
        return None
    return sorted(files)[-1]   # latest epoch


# ── SASRec ────────────────────────────────────────────────────────────────────
ckpt_path = load_latest_checkpoint(f"sasrec_provider_{SAMPLE}_ep*.pt")
if ckpt_path:
    print(f"\nLoading SASRec: {ckpt_path}")
    ckpt   = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg    = ckpt["config"]
    sasrec = SASRecProvider(
        provider_vocab_size = cfg["provider_vocab_size"],
        spec_vocab_size     = cfg["spec_vocab_size"],
        dx_vocab_size       = cfg["dx_vocab_size"],
        d_model             = cfg["d_model"],
        max_seq_len         = cfg["max_seq_len"],
        num_heads           = cfg["num_heads"],
        num_blocks          = cfg["num_blocks"],
        dropout             = cfg["dropout"],
    ).to(DEVICE)
    sasrec.load_state_dict(ckpt["model_state"])
    t1 = time.time()
    metrics_s, rows_s = score_model(sasrec, test_loader, "SASRec")
    all_model_metrics["SASRec"] = metrics_s
    all_bq_rows.extend(rows_s)
    print(f"SASRec scored — {time.time()-t1:.0f}s | T30 NDCG@3: {metrics_s.get('T0_30_ndcg@3',0):.4f}")
    del sasrec; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
else:
    print("SASRec checkpoint not found — skipping")


# ── BERT4Rec ──────────────────────────────────────────────────────────────────
ckpt_path = load_latest_checkpoint(f"bert4rec_provider_{SAMPLE}_ep*.pt")
if ckpt_path:
    print(f"\nLoading BERT4Rec: {ckpt_path}")
    ckpt    = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg     = ckpt["config"]
    bert4rec = BERT4RecProvider(
        provider_vocab_size = cfg["provider_vocab_size"],
        spec_vocab_size     = cfg["spec_vocab_size"],
        dx_vocab_size       = cfg["dx_vocab_size"],
        d_model             = cfg["d_model"],
        max_seq_len         = cfg["max_seq_len"],
        num_heads           = cfg["num_heads"],
        num_blocks          = cfg["num_blocks"],
        dropout             = cfg["dropout"],
    ).to(DEVICE)
    bert4rec.load_state_dict(ckpt["model_state"])
    MASK_IDX = cfg["mask_idx"]
    t1 = time.time()
    metrics_b, rows_b = score_model(bert4rec, test_loader, "BERT4Rec", mask_idx=MASK_IDX)
    all_model_metrics["BERT4Rec"] = metrics_b
    all_bq_rows.extend(rows_b)
    print(f"BERT4Rec scored — {time.time()-t1:.0f}s | T30 NDCG@3: {metrics_b.get('T0_30_ndcg@3',0):.4f}")
    del bert4rec; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
else:
    print("BERT4Rec checkpoint not found — skipping")


# ── HSTU ─────────────────────────────────────────────────────────────────────
ckpt_path = load_latest_checkpoint(f"hstu_provider_{SAMPLE}_ep*.pt")
if ckpt_path:
    print(f"\nLoading HSTU: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg  = ckpt["config"]
    hstu = HSTUProvider(
        provider_vocab_size = cfg["provider_vocab_size"],
        spec_vocab_size     = cfg["spec_vocab_size"],
        dx_vocab_size       = cfg["dx_vocab_size"],
        d_model             = cfg["d_model"],
        delta_t_dim         = cfg["delta_t_dim"],
        num_ratings         = cfg["num_ratings"],
        hstu_dim            = cfg["hstu_dim"],
        linear_dim          = cfg["linear_dim"],
        attention_dim       = cfg["attention_dim"],
        max_seq_len         = cfg["max_seq_len"],
        num_heads           = cfg["num_heads"],
        num_blocks          = cfg["num_blocks"],
        dropout             = cfg["dropout"],
        attn_dropout        = cfg["attn_dropout"],
        num_buckets         = cfg["num_buckets"],
    ).to(DEVICE)
    hstu.load_state_dict(ckpt["model_state"])
    t1 = time.time()
    metrics_h, rows_h = score_model(hstu, test_loader, "HSTU")
    all_model_metrics["HSTU"] = metrics_h
    all_bq_rows.extend(rows_h)
    print(f"HSTU scored — {time.time()-t1:.0f}s | T30 NDCG@3: {metrics_h.get('T0_30_ndcg@3',0):.4f}")
    del hstu; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
else:
    print("HSTU checkpoint not found — skipping")

print(f"\nSection 6 done — {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PULL MARKOV BASELINE FROM BQ
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 7 — Markov Baseline"))

print("Reading Markov metrics from BQ...")
markov_df = client.query(f"""
    SELECT
        time_bucket
        ,k
        ,member_segment
        ,n_evaluated
        ,hit_at_k
        ,precision_at_k
        ,recall_at_k
        ,ndcg_at_k
    FROM `{DS}.A870800_gen_rec_provider_markov_metrics_{SAMPLE}`
    WHERE member_segment = 'ALL'
    ORDER BY time_bucket, k
""").to_dataframe()

# Convert Markov BQ format to same dict format as deep learning models
markov_metrics = {}
bucket_map = {"T0_30": "T0_30", "T30_60": "T30_60", "T60_180": "T60_180"}
for _, row in markov_df.iterrows():
    b = bucket_map.get(row["time_bucket"], row["time_bucket"])
    k = int(row["k"])
    markov_metrics[f"{b}_hit@{k}"]  = round(float(row["hit_at_k"]),  4)
    markov_metrics[f"{b}_prec@{k}"] = round(float(row["precision_at_k"]), 4)
    markov_metrics[f"{b}_rec@{k}"]  = round(float(row["recall_at_k"]),  4)
    markov_metrics[f"{b}_ndcg@{k}"] = round(float(row["ndcg_at_k"]),  4)

all_model_metrics["Markov"] = markov_metrics
print(f"Markov metrics loaded | T30 NDCG@3: {markov_metrics.get('T0_30_ndcg@3',0):.4f}")
print(f"Section 7 done — {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 8 — Model Comparison"))

model_order  = ["Markov"] + [m for m in ["SASRec","BERT4Rec","HSTU"] if m in all_model_metrics]
windows      = ["T0_30", "T30_60", "T60_180"]
metrics_list = ["hit", "ndcg", "rec", "prec"]

# Build flat comparison dataframe
rows = []
for model_name in model_order:
    m = all_model_metrics[model_name]
    row = {"model": model_name}
    for bucket in windows:
        for metric in metrics_list:
            for k in K_VALUES:
                col = f"{bucket}_{metric}@{k}"
                row[col] = m.get(col, None)
    rows.append(row)

comparison_df = pd.DataFrame(rows)

# Save to CSV
csv_path = f"{OUTPUT_DIR}/provider_model_comparison_{SAMPLE}.csv"
comparison_df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

# Display focused summary — T0_30 NDCG and Hit
print(f"\n{'='*70}")
print(f"T0_30 WINDOW SUMMARY")
print(f"{'='*70}")
print(f"{'Model':<12}", end="")
for k in K_VALUES:
    print(f"  Hit@{k}  NDCG@{k}  Rec@{k}", end="")
print()
print("-" * 70)
for model_name in model_order:
    m = all_model_metrics.get(model_name, {})
    print(f"{model_name:<12}", end="")
    for k in K_VALUES:
        hit  = m.get(f"T0_30_hit@{k}",  0)
        ndcg = m.get(f"T0_30_ndcg@{k}", 0)
        rec  = m.get(f"T0_30_rec@{k}",  0)
        print(f"  {hit:.4f}  {ndcg:.4f}  {rec:.4f}", end="")
    print()

print(f"\n{'='*70}")
print(f"T30_60 WINDOW SUMMARY")
print(f"{'='*70}")
print(f"{'Model':<12}", end="")
for k in K_VALUES:
    print(f"  Hit@{k}  NDCG@{k}  Rec@{k}", end="")
print()
print("-" * 70)
for model_name in model_order:
    m = all_model_metrics.get(model_name, {})
    print(f"{model_name:<12}", end="")
    for k in K_VALUES:
        hit  = m.get(f"T30_60_hit@{k}",  0)
        ndcg = m.get(f"T30_60_ndcg@{k}", 0)
        rec  = m.get(f"T30_60_rec@{k}",  0)
        print(f"  {hit:.4f}  {ndcg:.4f}  {rec:.4f}", end="")
    print()

display(Markdown(f"""
## Summary
| Model | T30 Hit@3 | T30 NDCG@3 | T30 Rec@5 | T60 NDCG@3 | T180 NDCG@3 |
|-------|-----------|------------|-----------|------------|-------------|
""" + "\n".join([
    f"| {m} | "
    f"{all_model_metrics.get(m,{}).get('T0_30_hit@3',0):.4f} | "
    f"{all_model_metrics.get(m,{}).get('T0_30_ndcg@3',0):.4f} | "
    f"{all_model_metrics.get(m,{}).get('T0_30_rec@5',0):.4f} | "
    f"{all_model_metrics.get(m,{}).get('T30_60_ndcg@3',0):.4f} | "
    f"{all_model_metrics.get(m,{}).get('T60_180_ndcg@3',0):.4f} |"
    for m in model_order
])))

print(f"Section 8 done — {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — WRITE TRIGGER SCORES TO BQ
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 9 — Write Scores to BQ"))

print(f"Writing {len(all_bq_rows):,} trigger-window rows to BQ...")

schema = [
    bigquery.SchemaField("member_id",        "STRING"),
    bigquery.SchemaField("trigger_date",     "STRING"),
    bigquery.SchemaField("trigger_dx",       "STRING"),
    bigquery.SchemaField("member_segment",   "STRING"),
    bigquery.SchemaField("time_bucket",      "STRING"),
    bigquery.SchemaField("true_labels",      "STRING"),
    bigquery.SchemaField("top5_predictions", "STRING"),
    bigquery.SchemaField("top5_scores",      "STRING"),
    bigquery.SchemaField("model",            "STRING"),
    bigquery.SchemaField("sample",           "STRING"),
    bigquery.SchemaField("run_timestamp",    "STRING"),
]
job_cfg = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND", schema=schema)

BATCH_BQ = 100_000
for start in range(0, len(all_bq_rows), BATCH_BQ):
    chunk = pd.DataFrame(all_bq_rows[start:start + BATCH_BQ])
    client.load_table_from_dataframe(
        chunk,
        f"{DS}.A870800_gen_rec_provider_trigger_scores",
        job_config=job_cfg,
    ).result()
    print(f"  Written {start:,} → {min(start+BATCH_BQ, len(all_bq_rows)):,}")

print(f"Section 9 done — {time.time()-t0:.1f}s")
display(Markdown(f"**BQ rows written:** {len(all_bq_rows):,} | **Table:** A870800_gen_rec_provider_trigger_scores"))


# ── CLEANUP ───────────────────────────────────────────────────────────────────
import gc
for var in ["test_data","all_bq_rows","comparison_df"]:
    if var in dir(): del globals()[var]
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU freed — {torch.cuda.memory_allocated()/1e9:.2f}GB")
print("NB_07 complete")
