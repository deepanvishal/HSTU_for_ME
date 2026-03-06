import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RelativeBucketedTimeAndPositionBias(nn.Module):
    def __init__(self, max_seq_len: int, num_buckets: int = 128) -> None:
        super().__init__()
        self._max_seq_len  = max_seq_len
        self._num_buckets  = num_buckets
        self._ts_w  = nn.Parameter(torch.empty(num_buckets + 1).normal_(0, 0.02))
        self._pos_w = nn.Parameter(torch.empty(2 * max_seq_len - 1).normal_(0, 0.02))

    def forward(self, all_timestamps: torch.Tensor) -> torch.Tensor:
        # all_timestamps: [B, N] int64 — we use delta_t_buckets as timestamps
        B = all_timestamps.size(0)
        N = self._max_seq_len

        # positional bias [1, N, N]
        t   = F.pad(self._pos_w[:2 * N - 1], [0, N]).repeat(N)
        t   = t[:-N].reshape(N, 3 * N - 2)
        r   = (2 * N - 1) // 2
        pos_bias = t[:, r:-r].unsqueeze(0)  # [1, N, N]

        # time bias [B, N, N]
        ext = torch.cat([all_timestamps, all_timestamps[:, N-1:N]], dim=1)  # [B, N+1]
        diff = ext[:, 1:].unsqueeze(2) - ext[:, :-1].unsqueeze(1)          # [B, N, N]
        buckets = torch.clamp(
            (torch.log(torch.abs(diff).clamp(min=1)) / 0.301).long()
            ,min=0, max=self._num_buckets
        ).detach()
        ts_bias = self._ts_w[buckets.view(-1)].view(B, N, N)

        return pos_bias + ts_bias  # [B, N, N]


class HSTUBlock(nn.Module):
    def __init__(
        self
        ,embedding_dim: int
        ,linear_dim:    int
        ,attention_dim: int
        ,num_heads:     int
        ,dropout_rate:  float
        ,attn_dropout_rate: float
        ,max_seq_len:   int
        ,num_buckets:   int = 128
    ) -> None:
        super().__init__()

        self._embedding_dim  = embedding_dim
        self._linear_dim     = linear_dim
        self._attention_dim  = attention_dim
        self._num_heads      = num_heads
        self._dropout_rate   = dropout_rate
        self._attn_dropout   = attn_dropout_rate

        # uvqk projection — single fused linear
        self._uvqk = nn.Parameter(
            torch.empty(
                embedding_dim
                ,linear_dim * 2 * num_heads + attention_dim * num_heads * 2
            ).normal_(0, 0.02)
        )

        # output projection
        self._o = nn.Linear(linear_dim * num_heads, embedding_dim)
        nn.init.xavier_uniform_(self._o.weight)

        # layer norms
        self._norm_x   = nn.LayerNorm(embedding_dim)
        self._norm_attn= nn.LayerNorm(linear_dim * num_heads)

        # relative bias
        self._rel_bias = RelativeBucketedTimeAndPositionBias(max_seq_len, num_buckets)

        self._dropout  = nn.Dropout(dropout_rate)

    def forward(
        self
        ,x:           torch.Tensor   # [B, N, D]
        ,timestamps:  torch.Tensor   # [B, N] int64
        ,causal_mask: torch.Tensor   # [N, N] bool — True = masked
        ,pad_mask:    torch.Tensor   # [B, N] bool — True = padding
    ) -> torch.Tensor:
        B, N, D = x.shape
        H  = self._num_heads
        Dv = self._linear_dim
        Dq = self._attention_dim

        # layer norm input
        normed = self._norm_x(x)  # [B, N, D]

        # fused uvqk projection
        out = F.silu(normed.reshape(B * N, D) @ self._uvqk)  # [B*N, 2*H*Dv + 2*H*Dq]
        u, v, q, k = torch.split(out, [H * Dv, H * Dv, H * Dq, H * Dq], dim=-1)

        # reshape for multi-head attention
        u = u.view(B, N, H, Dv)
        v = v.view(B, N, H, Dv)
        q = q.view(B, N, H, Dq)
        k = k.view(B, N, H, Dq)

        # attention scores [B, H, N, N]
        qk = torch.einsum('bnhd,bmhd->bhnm', q, k)

        # relative bias [B, N, N] → [B, 1, N, N]
        rel_bias = self._rel_bias(timestamps).unsqueeze(1)
        qk = qk + rel_bias

        # silu attention (HSTU uses silu not softmax)
        qk = F.silu(qk) / N

        # apply causal mask [N, N] → [1, 1, N, N]
        qk = qk.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), 0.0)

        # apply padding mask [B, N] → [B, 1, 1, N]
        qk = qk.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), 0.0)

        # attention dropout
        qk = F.dropout(qk, p=self._attn_dropout, training=self.training)

        # weighted sum [B, N, H, Dv]
        attn_out = torch.einsum('bhnm,bmhd->bnhd', qk, v).reshape(B, N, H * Dv)

        # norm + gate
        attn_out = self._norm_attn(attn_out)
        u_flat   = u.reshape(B, N, H * Dv)
        o_input  = u_flat * attn_out

        # output projection + residual
        out = self._o(self._dropout(o_input)) + x

        # zero out padding positions
        out = out.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        return out


class PureHSTU(nn.Module):
    def __init__(
        self
        ,max_seq_len:    int
        ,embedding_dim:  int
        ,num_blocks:     int
        ,num_heads:      int
        ,linear_dim:     int
        ,attention_dim:  int
        ,dropout_rate:   float
        ,attn_dropout_rate: float
        ,num_ratings:    int
        ,rating_dim:     int
        ,num_specialties:int
    ) -> None:
        super().__init__()

        self._max_seq_len  = max_seq_len
        self._embedding_dim= embedding_dim
        hstu_dim           = embedding_dim + rating_dim

        # rating embedding
        self._rating_emb = nn.Embedding(num_ratings, rating_dim)

        # positional embedding
        self._pos_emb = nn.Embedding(max_seq_len, hstu_dim)

        # input dropout
        self._input_dropout = nn.Dropout(dropout_rate)

        # HSTU blocks
        self._blocks = nn.ModuleList([
            HSTUBlock(
                embedding_dim   = hstu_dim
                ,linear_dim     = linear_dim
                ,attention_dim  = attention_dim
                ,num_heads      = num_heads
                ,dropout_rate   = dropout_rate
                ,attn_dropout_rate = attn_dropout_rate
                ,max_seq_len    = max_seq_len
            )
            for _ in range(num_blocks)
        ])

        # L2 norm output
        self._out_norm = nn.LayerNorm(hstu_dim)

        # prediction heads
        self.head_30  = nn.Linear(hstu_dim, num_specialties)
        self.head_60  = nn.Linear(hstu_dim, num_specialties)
        self.head_180 = nn.Linear(hstu_dim, num_specialties)

        # causal mask — upper triangle = True (masked)
        self.register_buffer(
            '_causal_mask'
            ,torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self._rating_emb.weight, 0, 0.02)
        nn.init.normal_(self._pos_emb.weight,    0, 0.02)
        nn.init.xavier_normal_(self.head_30.weight)
        nn.init.xavier_normal_(self.head_60.weight)
        nn.init.xavier_normal_(self.head_180.weight)

    def forward(
        self
        ,embeddings: torch.Tensor  # [B, N, D]
        ,delta_t:    torch.Tensor  # [B, N] int64
        ,lengths:    torch.Tensor  # [B]    int64
    ):
        B, N, D = embeddings.shape
        device  = embeddings.device

        # padding mask [B, N] — True = padding position
        pad_mask = torch.arange(N, device=device).unsqueeze(0) >= lengths.unsqueeze(1)

        # rating embedding concat → [B, N, D + rating_dim]
        x = torch.cat([embeddings, self._rating_emb(delta_t)], dim=-1)

        # scale + positional embedding
        pos   = torch.arange(N, device=device).unsqueeze(0)  # [1, N]
        x     = x * math.sqrt(x.size(-1)) + self._pos_emb(pos)
        x     = self._input_dropout(x)

        # zero padding
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        # HSTU blocks
        for block in self._blocks:
            x = block(x, delta_t, self._causal_mask, pad_mask)

        # L2 norm
        x = self._out_norm(x)

        # gather last valid token per sequence
        idx      = (lengths - 1).clamp(0, N - 1).view(-1, 1, 1).expand(-1, 1, x.size(-1))
        seq_repr = x.gather(1, idx).squeeze(1)  # [B, hstu_dim]

        return (
            torch.sigmoid(self.head_30(seq_repr))
            ,torch.sigmoid(self.head_60(seq_repr))
            ,torch.sigmoid(self.head_180(seq_repr))
        )
