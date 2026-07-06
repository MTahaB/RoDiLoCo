# Research journal

Dated technical notes on what was built and what ran.
Rule: **a non-reproducible result does not exist** — every entry names seed, config, commit.

---

## 2026-07-03 — repo scaffolded

Full codebase skeleton for all phases:

- `model.py` — from-scratch decoder Transformer (RMSNorm, RoPE, hand-written causal MHA,
  SwiGLU, weight tying). Tests cover shapes, attention-vs-reference equivalence (atol
  1e-4), causality (future tokens don't leak), RoPE norm preservation, tiny-batch overfit.
- `optim.py` — AdamW from scratch (decoupled weight decay) + cosine-with-warmup. Reused as
  the DiLoCo inner optimizer.
- `diloco.py` — sequential worker simulation, Δ_k = θ_before − θ_after, outer Nesterov,
  analytic communication accounting; plus `run_synchronous` (the reference-high baseline).
- `aggregators.py` — the single `aggregate(deltas) -> Tensor` interface: mean / trimmed /
  krum / geometric_median / trust_weighted.
- `attacks.py` — sign_flip, scaled_noise, targeted_drift (inner poisoning via label perm).

**Smoke checks (CPU, tiny scale — NOT results):**
- Phase 1: loss 5.58 → 0.01 in 150 steps; samples reproduce the training text. ✔
- DiLoCo baseline (mean, f=0): val ppl → ~1.03. ✔
- Attack (mean, sign_flip λ=10, f=1): val ppl → ~4261 — a single of 8 workers destroys the
  run. The plot-#1 story ("one liar in eight kills pretraining") already appears in
  miniature. ✔
- 23/23 unit + integration tests pass.

## 2026-07-03 — synchronous baseline + paper draft

- **Synchronous baseline** (`run_synchronous`): trains one model on the pooled corpus for the
  same token budget as DiLoCo but "communicates" every step — the reference-high curve for
  the perplexity-vs-communication plot. Wired into `reproduce_phase2_baseline.sh` + a `comm`
  plot mode. Verified end-to-end at tiny scale (sync ppl ~1.04, high comm).
- **Paper** `paper/rodiloco.tex` — full 6-section arXiv-style draft; compiles clean (5 pp
  with `article`, → ~8 with figures). Every number is a `\todo{}` mapped to a run output.
  Bibliography resolves (14 refs, bibtex clean).
- **Novelty positioning** captured in `docs/related_work.md`: the intersection
  "Byzantine-robust DiLoCo outer optimizer" is unpublished; nearest neighbours (Decoupled
  DiLoCo = failures not adversaries; DeMoA = federated not DiLoCo; outer-optimizer local SGD
  = no adversaries). Framing: *resilient (failures) ≠ robust (adversaries)*.

Next (needs compute): real runs on TinyStories / FineWeb-Edu subset at 10–20 M params →
fill the `\todo{}`s → drop PDF figures into `paper/figures/` → uncomment `\includegraphics`.

---

## Template for future entries

```
## YYYY-MM-DD — <phase / milestone>
- what changed / what ran (config, seed, commit)
- result (number + plot path)
- decision / next
```
