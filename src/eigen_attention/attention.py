"""EigenAttention: multi-head attention with self-suppressed logits.

The Q-K logit is the Lorentz displacement d(q, k) = ||q|| ||k|| - q . k,
which is >= 0 and zero exactly on parallel pairs. Softmax over these
logits puts least mass on self / parallel-duplicate keys: attention is
diversity-seeking by construction, with no mask.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from eigen_attention.similarity import lorentz_displacement


class EigenAttention(nn.Module):
    """Multi-head attention using Lorentz displacement logits.

    Logits are d(q, k) = ||q|| ||k|| - q . k, scaled by 1/sqrt(head_dim)
    to match the magnitude of standard scaled dot-product attention.

    Args:
        dim: model dimension. Must be divisible by num_heads.
        num_heads: number of attention heads.
        bias: include bias in Q/K/V/O projections.
        dropout: attention dropout probability.
        causal: apply causal mask (no attending to future positions).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        bias: bool = True,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_q = nn.Linear(dim, dim, bias=bias)
        self.W_k = nn.Linear(dim, dim, bias=bias)
        self.W_v = nn.Linear(dim, dim, bias=bias)
        self.W_o = nn.Linear(dim, dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, _, L, _ = x.shape
        return x.transpose(1, 2).reshape(B, L, self.dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Self-attention over x.

        Args:
            x: (B, L, D) input.
            attn_mask: optional additive mask, broadcastable to (B, H, L, L).
                Use -inf for masked positions.

        Returns:
            out: (B, L, D) attended output.
            attn: (B, H, L, L) attention weights.
        """
        if x.dim() != 3:
            raise ValueError(f"expected (B, L, D), got {tuple(x.shape)}")
        B, L, D = x.shape
        if D != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {D}")

        q = self._split_heads(self.W_q(x))
        k = self._split_heads(self.W_k(x))
        v = self._split_heads(self.W_v(x))

        # logits = d(q, k) * scale. Non-negative; self-row diagonal is 0.
        logits = lorentz_displacement(q, k) * self.scale

        if self.causal:
            causal_mask = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            logits = logits.masked_fill(causal_mask, float("-inf"))

        if attn_mask is not None:
            logits = logits + attn_mask

        attn = F.softmax(logits, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = self._merge_heads(out)
        return self.W_o(out), attn
