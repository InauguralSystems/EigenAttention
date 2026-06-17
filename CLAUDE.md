# CLAUDE.md

Guidance for working in this repository.

## What this is

EigenAttention is a single PyTorch attention layer plus its underlying
geometric primitive. Logits are the **Lorentz displacement**
`d(q, k) = ||q|| ||k|| - q · k`, which is non-negative by
Cauchy-Schwarz with equality iff `q ∥ k`. Self-attention rows have
diagonal = 0 by construction, so softmax never amplifies self.

Use it when you want diversity-seeking attention. It is **not** a
drop-in replacement for standard scaled dot-product attention — the
polarity is inverted (dissimilar keys attend, not similar ones).

Lineage: extracted from [EigenFunction] in 2026. The original repo
mixed the primitive with a spacetime-feedback model, learned
"interval" MLPs, and consciousness framing. This repo keeps only the
attention primitive, with the math stated honestly and no benchmarks
that aren't apples-to-apples.

[EigenFunction]: https://github.com/InauguralPhysicist/EigenFunction

## Run

```bash
pip install -e ".[test]"
pytest
```

Tests cover: self / parallel / orthogonal / antiparallel displacement
values, non-negativity, shape correctness across 2D-4D tensors,
attention row sums, self-diagonal suppression, causal mask, additive
mask, and gradient flow.

## Layout

| Path | Role |
|---|---|
| `src/eigen_attention/similarity.py` | `lorentz_inner`, `lorentz_displacement` primitives |
| `src/eigen_attention/attention.py`  | `EigenAttention` nn.Module |
| `tests/test_similarity.py` | primitive properties + shapes |
| `tests/test_attention.py` | layer forward/backward + masks |

## Hard-won rules

- **Polarity matters.** This is diversity-seeking attention, not
  similarity-seeking. Don't quietly flip the sign of the logits to
  match standard attention — that would make the self-suppression
  property meaningless (self would dominate again).
- **No rigged benchmarks.** Any comparison vs. standard attention must
  be parameter-matched and use a success criterion that both models
  could in principle satisfy. The original EigenFunction repo got
  this wrong (4/4 vs 0/4 was structural, not empirical) — don't
  reintroduce that style here.
- **Geometric statement first, code second.** The README/docstrings
  derive the formula from the Minkowski embedding. Keep that grounding
  visible — it's the reason the primitive exists.
