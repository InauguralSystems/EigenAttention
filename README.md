# EigenAttention

Diversity-seeking multi-head attention via the Lorentz displacement
primitive. Self-attention is suppressed by construction — no mask, no
learned bias, no diagonal subtraction.

```python
from eigen_attention import EigenAttention
import torch

layer = EigenAttention(dim=512, num_heads=8)
x = torch.randn(2, 100, 512)
out, attn = layer(x)
# attn[..., i, i] is geometrically minimized; softmax mass goes elsewhere.
```

## The primitive

Embed each n-dim vector `u` in (n+1)-Minkowski space as `(u, ||u||)`.
Every such embedding is **lightlike** by construction:

```
⟨(u, ||u||), (u, ||u||)⟩_L  =  ||u||² − ||u||²  =  0
```

For two distinct vectors the Lorentz inner product is

```
⟨(q, ||q||), (k, ||k||)⟩_L  =  q · k  −  ||q|| ||k||
```

which is ≤ 0 by Cauchy-Schwarz, with equality iff `q ∥ k` in the same
direction. Its negation is the **displacement**

```
d(q, k)  =  ||q|| ||k||  −  q · k  =  −⟨(q, ||q||), (k, ||k||)⟩_L
```

which is ≥ 0 and zero exactly on parallel pairs. Use `d(q, k)` as the
attention logit and self-suppression falls out of the geometry: the
diagonal of `softmax(d(Q, Kᵀ))` is automatically the minimum of each
row, and softmax mass goes to **dissimilar** keys.

## What it's for

- **Diversity-promoting attention** — encourages each query to attend
  to keys that differ from it geometrically, instead of collapsing to
  self / near-duplicate keys.
- **Fixed-point avoidance** in iterative refinement loops, where
  standard cosine attention amplifies whatever state the system is
  already in.
- **Retrieval with explicit anti-duplication**, where you want the
  top-k to span the embedding space rather than cluster.

This is **not** a drop-in replacement for standard scaled-dot-product
attention. Standard attention pulls similar tokens together;
EigenAttention pulls dissimilar tokens together. Use it when that's
what you want.

## Install

```bash
pip install -e .
pip install -e ".[test]"   # for tests
pytest
```

## API

```python
from eigen_attention import EigenAttention, lorentz_displacement, lorentz_inner

# nn.Module — multi-head self-attention layer.
EigenAttention(dim, num_heads, *, bias=True, dropout=0.0, causal=False)
#   forward(x: (B, L, D), attn_mask=None) -> (out: (B, L, D), attn: (B, H, L, L))

# Bare primitive — d(q, k) = ||q|| ||k|| - q . k, non-negative, zero on parallel.
lorentz_displacement(q, k)   # (..., L_q, D), (..., L_k, D) -> (..., L_q, L_k)

# The signed form: <(q, ||q||), (k, ||k||)>_L = -d(q, k), non-positive.
lorentz_inner(q, k)
```

## Lineage

This primitive was extracted from [EigenFunction], which originally
explored the Lorentz / lightlike construction as a foundation for the
[EigenScript] observability model (every measurement changes state,
self-observation contributes nothing). The attention primitive is the
piece of that exploration that carries weight on its own: it's a real
geometric construction with a real effect on softmax mass distribution.

[EigenFunction]: https://github.com/InauguralPhysicist/EigenFunction
[EigenScript]: https://github.com/InauguralSystems/EigenScript

## License

MIT.
