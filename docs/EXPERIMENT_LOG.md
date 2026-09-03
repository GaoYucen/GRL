# GRL Experiment Log

This file is a concise index of experiments worth remembering. Raw logs, checkpoints, copied worktrees, and large outputs remain in their project-local result directories.

## 2026-09-01 to 2026-09-02 — Marginal-gain predictability validation
**Goal:** Determine whether the model genuinely learns conditional marginal gain Δ(v|S), especially candidate ranking under a fixed seed set/state.

**Current retained conclusion:** Marginal-gain prediction is sufficiently promising to remain the core learning target, but strict unseen-state validation and state-conditioning diagnostics remain important before treating the predictor as a reliable oracle.

**Related work:** NetHEPT marginal-gain predictability tests and strict state-conditioning checks documented in the GRL Notion workspace.

## 2026-09-02 — ICLR 2027 route pre-experiments
**Goal:** Test components around a learning-augmented certified marginal-gain oracle, including uncertainty/trust logic and ensemble-style pretests.

**Status:** Active / exploratory. Multiple project-local pre-experiment directories exist under `/workspace/GRL`. Exact metrics should be promoted into this log only after the corresponding result/config artifact has been checked.

## 2026-09-03 — Strict state-conditioning diagnostics
**Goal:** Determine whether high candidate-ranking accuracy actually reflects dependence on the current seed set S.

**Verified finding:** Strict unseen-state ranking is strong, but zero/shuffled seed-mask and controlled-overlap diagnostics show that the original predictor relies heavily on candidate-intrinsic strength. State-sensitive difficult-state supervision plus same-candidate delta loss restores strong cross-state sensitivity without destroying ordinary candidate ranking. A candidate-conditioned residual variant improves state-response amplitude further, but candidate-specific drop-magnitude calibration remains unresolved.

**Decision:** Do not spend the current phase on small predictor-metric tuning. Keep state-aware supervision as the default learning direction and move to end-to-end sequential IM evaluation.

## 2026-09-03 — First end-to-end learning-augmented IM prototype
**Protocol:** NetHEPT; shared fixed candidate pool of 128 nodes; seed budget 10; selection MC 40; final spread MC 1000. Full-MC, learned-only, and selective methods use the same candidate pool. Selective baseline scores all candidates with the learned state-aware predictor and exactly refines only the learned Top-M candidates at each step.

**Results:**
- Full-MC greedy: spread 444.911; 1,235 exact candidate evaluations; 49,400 MC candidate samples.
- Learned strict: spread 325.880 (73.25% of Full-MC); 0 exact candidate evaluations.
- Learned state-aware: spread 321.526 (72.27%); 0 exact candidate evaluations.
- Selective Top-8: spread 412.186 (92.64%); 80 exact candidate evaluations = 6.48% of Full-MC.
- Selective Top-16: spread 414.101 (93.08%); 160 exact candidate evaluations = 12.96% of Full-MC.
- Selective Top-32: spread 422.776 (95.02%); 320 exact candidate evaluations = 25.91% of Full-MC.
- Degree baseline: spread 299.100 (67.23%); Degree Discount: 302.340 (67.96%).

**Interpretation:** This is a usable framework/result anchor, not the final paper result. Selective correction materially closes the quality gap while using much fewer exact candidate evaluations. The next algorithmic priority is to improve learned shortlist recall and replace fixed Top-M with adaptive trust/certification; full-graph/RIS scaling comes after that.

**Compact artifact:** `docs/results/nethept_end_to_end_20260903.json`.


## 2026-09-03 — Sequential shortlist and adaptive certification
**Protocol:** NetHEPT, 128-candidate shared pool, budget 10, MC40 selection oracle, MC1000 final spread evaluation.

**Sequential shortlist diagnostic:** true Full-MC winner ranks under the state-aware predictor are `[1, 57, 87, 2, 2, 42, 1, 8, 1, 32]` (Top-32 recall 0.70; Top-64 recall 0.90). This explains the fixed Top-M quality ceiling.

**Adaptive result:** residual-envelope adaptive refinement with `beta=0.5` obtains spread **443.626** versus Full-MC **444.911** (ratio **0.9971**) using **512** exact candidate evaluations versus **1235** (fraction **0.4146**). With per-step MC-world reuse it uses 400 live-edge worlds and takes 32.4s in the current NetworkX prototype.

**Interpretation:** the main predictor metric for the next phase is sequential hard-state Top-K recall, and the main algorithmic direction is adaptive certification/fallback rather than a fixed shortlist. See `docs/results/nethept_adaptive_certification_20260903.json`.

## 2026-09-03 — Two-level adaptive certification with progressive MC

Starting from the validated adaptive shortlist rule, we separated two decisions: (1) how many candidates need exact verification, and (2) how many common-random-number MC worlds are needed inside the current shortlist. The final implementation uses progressive MC budgets 5→10→20→40 and caches candidate/world gains across rounds.

On NetHEPT (128-node candidate pool, budget=10, final spread MC=1000), the recommended configuration is residual_beta=0.5, confidence_z=0.5, bootstrap_mc=10. It selects the same seeds and obtains the same measured spread as the fixed adaptive beta=0.5 baseline: 443.626 (99.71% of Full-MC 444.911). Exact candidate evaluations are 504 versus 1235 for Full-MC; MC candidate-samples are 18,280 versus 49,400 for Full-MC and 20,480 for fixed adaptive. Selection time is 23.59s in this implementation. The per-step verified shortlist sizes are [16,16,16,72,64,112,96,80,16,16], with final MC budgets [40,10,10,20,40,20,5,40,20,40].

Bootstrap ablation 10/20/40 produced identical selected seeds and spread. Bootstrap=10 was best on compute: 18,280 samples and 320 generated live-edge worlds, versus 18,920/360 for bootstrap=20 and 18,920/400 for bootstrap=40.
