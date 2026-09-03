# GRL Next Steps

Last updated: 2026-09-03

## Current ICLR 2027 route
1. State-aware conditional marginal-gain predictor Δ(v|S).
2. Learning-augmented marginal oracle that scores broadly with the learned model and selectively invokes a stronger exact/MC/RIS oracle.
3. Sequential influence maximization driven by that oracle; RL is not a required main component at this stage.
4. Adaptive trust / certification / fallback to obtain a quality-versus-oracle-cost tradeoff and later consistency/robustness guarantees.

## P0 — End-to-end prototype: completed baseline
- [x] Strict unseen-state validation.
- [x] Correct-mask vs zero/shuffled-mask diagnostics.
- [x] Within-state ranking metrics and controlled overlap/state-conditioning tests.
- [x] Modular learned oracle + batched MC oracle + sequential greedy framework.
- [x] First NetHEPT Full-MC / learned-only / selective / degree baselines.
- [x] First quality-versus-exact-oracle-budget curve (Top-8/16/32).

## P0 — Next engineering/research milestone
- [ ] Measure per-step **shortlist recall**: whether the Full-MC best candidate is inside learned Top-K, for K={8,16,32,64}; separate predictor recall failure from MC noise.
- [ ] Replace fixed Top-M refinement with an **adaptive trust/certification rule** based initially on prediction gaps / shortlist stability, then uncertainty if needed.
- [ ] Improve training/data only where it directly increases sequential shortlist recall; prioritize hard evolving seed states and candidate-state interactions over small global MAE gains.
- [ ] Obtain a reference point near Full-MC quality with materially fewer exact candidate evaluations, then repeat across random seeds.

## P1 — Scale toward paper-quality evaluation
- [ ] Expand beyond the fixed 128-candidate prototype to larger/all-node candidate sets using scalable candidate generation or RIS-style oracle support.
- [ ] Evaluate additional IM datasets and graph settings.
- [ ] Compare against strong classical IM baselines and relevant learning-based IM methods.
- [ ] Report influence spread, quality ratio, exact oracle calls / simulations, runtime, and memory.
- [ ] Add ablations for state-aware supervision, trust/certification, and fallback.

## P1 — Theory / paper framing after empirical mechanism stabilizes
- [ ] Formalize the learned-oracle interface and prediction-error/trust assumptions.
- [ ] Derive consistency when predictions are reliable and robustness/fallback behavior when they fail.
- [ ] Keep RL as an optional later extension only if it adds measurable value beyond sequential greedy with the learned oracle.

## Session-resume instruction
A new session should start by reading `AGENTS.md`, `docs/RESEARCH_STATE.md`, `docs/EXPERIMENT_LOG.md`, and this file. The immediate next target is shortlist recall + adaptive certification, not further micro-tuning of regression metrics.


## P0 — After the first adaptive end-to-end result
- [x] Measure step-level sequential shortlist recall and identify hard states.
- [x] Implement adaptive candidate refinement with a robust all-candidate fallback.
- [x] Reuse common MC live-edge worlds within each greedy step.
- [ ] Improve predictor training specifically on sequential hard states (large true-winner rank / regret), rather than optimizing global MAE.
- [ ] Replace the empirical residual envelope with calibrated uncertainty / confidence bounds and test coverage.
- [ ] Add adaptive MC sample allocation so both candidate count and simulations per candidate are refined only when needed.
- [ ] Run robustness stress tests with degraded/noisy/shuffled predictors and verify automatic fallback.
- [ ] Extend beyond the fixed 128-candidate prototype to larger/all-node candidate settings and then RIS-style scalable oracles.

## P0 after progressive MC: robustness / consistency stress test

1. Freeze the current recommended configuration (residual_beta=0.5, confidence_z=0.5, bootstrap_mc=10).
2. Deliberately degrade learned scores with controlled noise / score mixing and a shuffled-ranking endpoint.
3. For each predictor-quality level measure final spread, exact candidate evaluations, MC candidate-samples, verified candidates per step, and fallback frequency.
4. Success criterion for the learning-augmented story: as prediction quality worsens, oracle/sample cost should rise automatically while spread degrades much more slowly and eventually approaches the classical-oracle fallback behavior.
5. After this mechanism-level stress test, move from the fixed 128-node pool toward scalable all-node/RIS evaluation and multiple IM datasets.

## P0 — Compose trust audit with progressive fast path

1. Keep tau=0.3 as the initial trust threshold; tau=0.4/0.5 were more conservative without improving the clean case.
2. Replace the current trusted MC40 residual path with the validated progressive-v3 path (bootstrap=10, confidence_z=0.5, residual_beta=0.5, MC budgets 5→10→20→40).
3. Reuse audit MC samples through the same ProgressiveMonteCarloOracle cache.
4. Tune only the audit overhead first: audit_mc in {10,20} and sentinel_count in {4,8}, on alpha in {0,0.75,1.0}.
5. Target clean cost approximately 18k–22k samples with spread >=99.5% of Full-MC, while alpha=1 must still trigger 10/10 fallback and recover Full-MC cost/quality.
6. After mechanism tuning, rerun the full corruption curve with multiple random seeds and then move to additional IM datasets / scalable candidate sets.
