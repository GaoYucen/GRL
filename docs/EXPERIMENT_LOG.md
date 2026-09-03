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
