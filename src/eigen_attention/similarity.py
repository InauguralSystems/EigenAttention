"""Lorentz-projected attention primitives.

Embed n-dim vectors in (n+1)-Minkowski space as (u, ||u||). Every embedded
vector is lightlike by construction:

    <(u, ||u||), (u, ||u||)>_L = ||u||^2 - ||u||^2 = 0

The Lorentz inner product between two such embeddings is

    <(q, ||q||), (k, ||k||)>_L = q . k - ||q|| ||k||                 (1)

which is <= 0 by Cauchy-Schwarz, with equality iff q is parallel to k
(including q == k). Its negation is the displacement

    d(q, k) = ||q|| ||k|| - q . k = -<(q, ||q||), (k, ||k||)>_L     (2)

which is >= 0 and zero exactly on parallel pairs. d is the geometric
quantity worth attending to: self gets minimum logit by construction, so
self-attention is suppressed without an explicit mask.
"""

from __future__ import annotations

import torch


def lorentz_inner(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Lorentz inner product of (q, ||q||) and (k, ||k||) embeddings.

    Equation (1) above. Result is <= 0 with equality iff q is parallel
    to k in the same direction.

    Args:
        q: (..., L_q, D) query tensor.
        k: (..., L_k, D) key tensor with matching leading dims.

    Returns:
        (..., L_q, L_k) tensor of Lorentz inner products.
    """
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(
            f"feature dim mismatch: q={q.shape[-1]}, k={k.shape[-1]}"
        )

    qk = torch.matmul(q, k.transpose(-2, -1))
    q_norm = torch.linalg.norm(q, dim=-1, keepdim=True)
    k_norm = torch.linalg.norm(k, dim=-1, keepdim=True)
    norm_outer = torch.matmul(q_norm, k_norm.transpose(-2, -1))
    return qk - norm_outer


def lorentz_displacement(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Cauchy-Schwarz displacement ||q|| ||k|| - q . k.

    Equation (2) above. Non-negative; zero iff q is parallel to k
    (so self pairs and parallel duplicates both map to 0).

    Args:
        q: (..., L_q, D) query tensor.
        k: (..., L_k, D) key tensor with matching leading dims.

    Returns:
        (..., L_q, L_k) non-negative displacement tensor.
    """
    return -lorentz_inner(q, k)
