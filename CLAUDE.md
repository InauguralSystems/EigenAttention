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

## Evidence path

The primitive is well-tested code with no empirical result behind it
yet. Until one of these lands, the repo is a construction, not a
contribution. Two candidate first experiments, ordered by cost-to-
evidence ratio:

1. **Anti-duplication retrieval bench.** Pick a small image or text
   retrieval task. Two identical encoders, one with standard
   attention pooling, one with `EigenAttention` pooling over
   patches/tokens. Measure **both** relevance (precision@k) **and**
   inter-result diversity (pairwise distance among top-k).
   Hypothesis: EigenAttention preserves relevance while increasing
   diversity. If true, that's a real citable contribution; if false,
   the primitive doesn't carry weight and the repo should be archived.

2. **Mode-collapse stress test in a tiny VAE / autoencoder.** Same
   idea, different surface. Train identical AEs except for the
   attention layer; measure latent-space coverage (entropy of nearest-
   neighbor assignment over a held-out set). Easier to get a clean
   signal than retrieval — fewer confounders.

Either experiment must be **parameter-matched** with a **symmetric
success criterion** both models could in principle satisfy. The
EigenFunction repo's 4/4-vs-0/4 result failed both of those — don't
reintroduce that style.

**Amplification moves deferred until there's a result to amplify:**
arXiv / blog writeup, Hugging Face model card, dropping into
iLambdaAi's LM pipeline (polarity mismatch — diversity-seeking
attention is wrong for language modeling). Don't queue any of these
before an experiment lands.

## This design is pre-v1 and mostly written by older models — question it

**90% or more of this repo, and of EigenScript and the AOT it runs on, was
designed by Claude sessions running models many generations old.** Newer
models keep arriving that are substantially stronger, the ecosystem is
**pre-v1**, and there are no external consumers to break. A decision you
find in the tree — here or upstream — carries **no authority from
seniority**.

That applies in both directions, and the second one is the point of a
consumer repo. When you hit a rough edge in the runtime, the standing rule
is already to surface it as an upstream issue rather than work around it
silently. Add to that: ask whether the thing you hit is a **law** of the
language or an **earlier decision**. The tell is writing, or thinking,
*"X must be true because the runtime does Y."*

Bought 2026-08-28 (ouroboros#127 / DMG). The AOT compiles a program's main
file but emits `load_file` as a runtime call, so loaded modules are
interpreted by the linked VM. A real bug in that seam was found, minimised,
fixed and verified — and reported as "unlocking the AOT multiplier for
DMG". Measured on being challenged: DMG is 3,288 lines, 818 compiled and
2,470 interpreted, including the 128-function opcode dispatch. Every
emulated instruction runs interpreted, so the fix makes it *run* and cannot
make it *faster*. A whole investigation cycle had treated that design as
terrain, and the capability to do it the other way already existed upstream
for another purpose.
