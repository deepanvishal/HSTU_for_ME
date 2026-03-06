import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativeBucketedTimeAndPositionBias(nn.Module):
    def __init__(self, max_seq_len: int, num_buckets: int = 128) -> None:
        super().__init__()
        self._max_seq_len = max_seq_len
        self._num_buckets = num_buckets
        self._ts_w  = nn.Parameter(torch.empty(num_buckets + 1).normal_(0, 0.02))
        self._pos_w = nn.Parameter(torch.empty(2 * max_seq_len - 1).normal_(0, 0.02))
        self._cached_pos_bias = None

    def _compute_pos_bias(self, device):
        N = self._max_seq_len
        t = F.pad(self._pos_w[:2 * N - 1], [0, N]).repeat(N)
        t = t[:-N].reshape(N, 3 * N - 2)
        r = (2 * N - 1) // 2
        return t[:, r:-r].unsqueeze(0)  # [1, N, N]

    def forward(self, all_timestamps: torch.Tensor) -> torch.Tensor:
        B      = all_timestamps.size(0)
        N      = self._max_seq_len
        device = all_timestamps.device

        # recompute pos_bias during training (params change), cache during eval
        if self.training or self._cached_pos_bias is None:
            self._cached_pos_bias = self._compute_pos_bias(device)
        pos_bias = self._cached_pos_bias.to(device)  # [1, N, N]

        # time bias [B, N, N]
        ext     = torch.cat([all_timestamps, all_timestamps[:, N-1:N]], dim=1)
        diff    = ext[:, 1:].unsqueeze(2) - ext[:, :-1].unsqueeze(1)
        buckets = torch.clamp(
            (torch.log(torch.abs(diff).clamp(min=1)) / 0.301).long()
            ,min=0, max=self._num_buckets
        ).detach()
        ts_bias = self._ts_w[buckets.view(-1)].view(B, N, N)

        return pos_bias + ts_bias  # [B, N, N]


class HSTUBlock(nn.Module):
    def __init__(
        self
        ,embedding_dim:     int
        ,linear_dim:        int
        ,attention_dim:     int
        ,num_heads:         int
        ,dropout_rate:      float
        ,attn_dropout_rate: float
        ,max_seq_len:       int
        ,num_buckets:       int = 128
    ) -> None:
        super().__init__()

        self._embedding_dim = embedding_dim
        self._linear_dim    = linear_dim
        self._attention_dim = attention_dim
        self._num_heads     = num_heads
        self._dropout_rate  = dropout_rate
        self._attn_dropout  = attn_dropout_rate

        self._uvqk = nn.Parameter(
            torch.empty(
                embedding_dim
                ,linear_dim * 2 * num_heads + attention_dim * num_heads * 2
            ).normal_(0, 0.02)
        )

        self._o         = nn.Linear(linear_dim * num_heads, embedding_dim)
        self._norm_x    = nn.LayerNorm(embedding_dim)
        self._norm_attn = nn.LayerNorm(linear_dim * num_heads)
        self._rel_bias  = RelativeBucketedTimeAndPositionBias(max_seq_len, num_buckets)
        self._dropout   = nn.Dropout(dropout_rate)

        nn.init.xavier_uniform_(self._o.weight)

    def forward(
        self
        ,x:           torch.Tensor   # [B, N, D]
        ,timestamps:  torch.Tensor   # [B, N] int64
        ,causal_mask: torch.Tensor   # [N, N] bool
        ,pad_mask:    torch.Tensor   # [B, N] bool
    ) -> torch.Tensor:
        B, N, D = x.shape
        H  = self._num_heads
        Dv = self._linear_dim
        Dq = self._attention_dim

        normed = self._norm_x(x)

        # fused uvqk projection + silu
        out = F.silu(normed.reshape(B * N, D) @ self._uvqk)
        u, v, q, k = torch.split(out, [H * Dv, H * Dv, H * Dq, H * Dq], dim=-1)

        # reshape for bmm: [B*H, N, Dq/Dv]
        q = q.view(B, N, H, Dq).permute(0, 2, 1, 3).reshape(B * H, N, Dq)
        k = k.view(B, N, H, Dq).permute(0, 2, 1, 3).reshape(B * H, N, Dq)
        v = v.view(B, N, H, Dv).permute(0, 2, 1, 3).reshape(B * H, N, Dv)

        # attention scores via bmm [B, H, N, N]
        qk = torch.bmm(q, k.transpose(1, 2)).view(B, H, N, N)

        # relative bias [B, N, N] → [B, 1, N, N]
        rel_bias = self._rel_bias(timestamps).unsqueeze(1)
        qk = qk + rel_bias

        # silu attention (not softmax)
        qk = F.silu(qk) / N

        # causal + padding masks
        qk = qk.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
        qk = qk.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2),    0.0)
        qk = F.dropout(qk, p=self._attn_dropout, training=self.training)

        # weighted sum via bmm
        attn_out = torch.bmm(qk.reshape(B * H, N, N), v).view(B, H, N, Dv)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, H * Dv)

        # norm + gate + output projection + residual
        attn_out = self._norm_attn(attn_out)
        u_flat   = u.view(B, N, H * Dv)
        out      = self._o(self._dropout(u_flat * attn_out)) + x

        return out.masked_fill(pad_mask.unsqueeze(-1), 0.0)


class PureHSTU(nn.Module):
    def __init__(
        self
        ,max_seq_len:       int
        ,embedding_dim:     int
        ,num_blocks:        int
        ,num_heads:         int
        ,linear_dim:        int
        ,attention_dim:     int
        ,dropout_rate:      float
        ,attn_dropout_rate: float
        ,num_ratings:       int
        ,rating_dim:        int
        ,num_specialties:   int
    ) -> None:
        super().__init__()

        self._max_seq_len = max_seq_len
        hstu_dim          = embedding_dim + rating_dim

        self._rating_emb    = nn.Embedding(num_ratings, rating_dim)
        self._pos_emb       = nn.Embedding(max_seq_len, hstu_dim)
        self._input_dropout = nn.Dropout(dropout_rate)

        self._blocks = nn.ModuleList([
            HSTUBlock(
                embedding_dim      = hstu_dim
                ,linear_dim        = linear_dim
                ,attention_dim     = attention_dim
                ,num_heads         = num_heads
                ,dropout_rate      = dropout_rate
                ,attn_dropout_rate = attn_dropout_rate
                ,max_seq_len       = max_seq_len
            )
            for _ in range(num_blocks)
        ])

        self._out_norm = nn.LayerNorm(hstu_dim)
        self.head_30   = nn.Linear(hstu_dim, num_specialties)
        self.head_60   = nn.Linear(hstu_dim, num_specialties)
        self.head_180  = nn.Linear(hstu_dim, num_specialties)

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

        pad_mask = torch.arange(N, device=device).unsqueeze(0) >= lengths.unsqueeze(1)

        x   = torch.cat([embeddings, self._rating_emb(delta_t)], dim=-1)
        pos = torch.arange(N, device=device).unsqueeze(0)
        x   = x * math.sqrt(x.size(-1)) + self._pos_emb(pos)
        x   = self._input_dropout(x)
        x   = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        for block in self._blocks:
            x = block(x, delta_t, self._causal_mask, pad_mask)

        x        = self._out_norm(x)
        idx      = (lengths - 1).clamp(0, N - 1).view(-1, 1, 1).expand(-1, 1, x.size(-1))
        seq_repr = x.gather(1, idx).squeeze(1)

        return (
            torch.sigmoid(self.head_30(seq_repr))
            ,torch.sigmoid(self.head_60(seq_repr))
            ,torch.sigmoid(self.head_180(seq_repr))
        )
