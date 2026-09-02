# GRL Next Steps

Last updated: 2026-09-02

## P0 — Consolidate the current pre-experiment evidence
- [ ] Identify the latest valid P1/P2/ensemble result artifacts under `/workspace/GRL`.
- [ ] Record the exact configs, commands, and comparable metrics for those runs.
- [ ] Update `EXPERIMENT_LOG.md` with only verified results and remove ambiguity between smoke tests and reference experiments.

## P0 — Validate the learned marginal-gain oracle rigorously
- [ ] Evaluate strict unseen-state generalization.
- [ ] Compare correct state/seed-mask conditioning with zeroed or shuffled state information.
- [ ] Report ranking metrics within each state in addition to MAE/MSE-type regression metrics.
- [ ] Measure downstream influence-maximization quality when using predictions for candidate selection.

## P0 — Complete the certified / trust-aware decision mechanism
- [ ] Define the uncertainty/trust signal used to decide when predictions are accepted.
- [ ] Define fallback behavior when confidence is insufficient.
- [ ] Measure quality-versus-oracle-query/computation tradeoffs.

## P1 — Paper-oriented evaluation
- [ ] Evaluate robustness across graph settings / datasets available in the project.
- [ ] Run ablations separating predictor quality, uncertainty/trust mechanism, and fallback/certification components.
- [ ] Maintain a reproducible table mapping paper claims to scripts/configs/results.

## Session-resume instruction
A new Codex session should start by reading `AGENTS.md`, `docs/RESEARCH_STATE.md`, and this file, then inspect the latest result artifacts before deciding which unchecked P0 item to execute.
