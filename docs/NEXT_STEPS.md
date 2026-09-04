# GRL Next Steps

Last updated: 2026-09-04

## Current ICLR 2027 route
1. State-aware conditional marginal-gain predictor Δ(v|S).
2. Learning-augmented sequential greedy: learned scores propose/rank candidates.
3. Progressive candidate/sample verification: adapt both shortlist size and MC budget.
4. Online trust audit with sentinel candidates.
5. Classical-oracle fallback when learned ranking is not trustworthy.
6. Main paper target: **near-classical influence quality with oracle cost that decreases when predictions are useful and returns toward classical cost when predictions fail**.

RL is not a required main component and should not be added during the current validation phase.

## P0 — Completed mechanism milestones
- [x] Strict unseen-state validation and seed-mask/state-conditioning diagnostics.
- [x] Modular learned oracle + MC oracle + sequential greedy framework.
- [x] First NetHEPT Full-MC / learned-only / fixed Top-M baselines.
- [x] Step-level sequential shortlist recall diagnosis.
- [x] Adaptive candidate refinement and common-world reuse.
- [x] Progressive MC allocation (5→10→20→40) with cached candidate/world samples.
- [x] Controlled predictor-corruption stress test exposing false-confidence failure.
- [x] Sentinel-audited trust gate and classical Full-MC fallback.
- [x] Trust audit composed with the progressive fast path.
- [x] Five-repeat multi-seed robustness/calibration pilot with same-seed Full-MC references.

## P0 — Immediate next milestone: held-out statistical trust calibration
The multi-seed pilot shows that robustness is strong but clean-case efficiency varies materially across MC/audit seeds. Do **not** continue hand-tuning a single `tau` on the evaluation runs.

1. Collect a calibration set of per-step audit statistics from independent runs:
   - learned-vs-exact audit Spearman;
   - sentinel-surprise indicator and margin;
   - head-vs-sentinel exact gap;
   - progressive verification outcome / required MC budget;
   - whether the trusted decision agrees with the stronger reference decision.
2. Split calibration and evaluation randomness explicitly. Calibration seeds must not be reused to report final evaluation performance.
3. Define a statistically grounded trust rule, initially using a held-out quantile / false-trust or false-distrust target rather than manually selecting `tau` from the test curves.
4. Keep sentinel-surprise as a conservative guard until held-out evidence shows it can be relaxed safely.
5. Re-evaluate on fresh seeds with the frozen calibrated rule.
6. Success criterion:
   - clean predictor retains near-Full-MC quality with materially lower and less variable oracle cost;
   - severe corruption still drives cost close to Full-MC and prevents quality collapse.

## Current multi-seed anchor
Five independent NetHEPT repeats, shared 128-candidate pool, budget 10:
- Clean: quality ratio 1.0005±0.0082; sample fraction **52.6%**; fallback 4.4±1.5/10.
- Alpha=.5: quality ratio 1.0030±0.0093; sample fraction **49.7%**; fallback 4.0±1.6/10.
- Alpha=.75: quality ratio 1.0019±0.0045; sample fraction **83.3%**; fallback 7.6±0.5/10.
- Random alpha=1: quality ratio 0.9998±0.0005; sample fraction **95.5%**; fallback 9.4±0.9/10.

The clean alpha=0 versus alpha=.5 cost ordering is not strictly monotone in this small five-run sample. Treat this as another reason to calibrate on held-out statistics and expand repeats before making a monotonicity claim.

## P1 — Multiple budgets after calibrated trust rule freezes
- [ ] Evaluate `k={5,10,20}` first; add `k=50` only after runtime/scalability is acceptable.
- [ ] Use multiple independent seeds and same-protocol references.
- [ ] Measure spread, quality ratio, candidate evaluations, MC candidate-samples, fallback frequency, runtime, and memory.
- [ ] Check whether the trust/cost behavior changes as sequential depth grows.

## P1 — Candidate-scale and graph-scale expansion
- [ ] Expand the shared candidate pool from 128 to 256 and 512.
- [ ] Identify the point where the current NetworkX/MC prototype becomes the bottleneck.
- [ ] Move toward scalable candidate generation / RIS-style oracle support for all-node evaluation.
- [ ] Add at least 2–3 additional standard IM graph datasets.
- [ ] Compare against strong classical IM methods and relevant learning-based IM methods under harmonized candidate/oracle settings.

## P2 — Paper ablations
- [ ] Full-MC greedy.
- [ ] Learned-only.
- [ ] Fixed Top-8/16/32 refinement.
- [ ] Progressive verification without trust gate.
- [ ] Trust gate without progressive sampling.
- [ ] Full trust + progressive method.
- [ ] Predictor state-awareness ablation (original vs state-aware / same-candidate delta supervision).
- [ ] Controlled predictor-quality corruption curve.

## P2 — Theory / framing
- [ ] Formalize the learned-oracle interface and the audit/fallback mechanism.
- [ ] Separate two claims:
  - **consistency/efficiency:** useful predictions reduce oracle work;
  - **robustness:** unreliable predictions are detected sufficiently often that the method returns toward classical computation/quality.
- [ ] Do not claim a formal robustness theorem until audit coverage/error assumptions are explicitly derived.
- [ ] Consider conformal/quantile residual bounds or randomized sentinel coverage as the bridge from the current empirical trust rule to a theorem.

## P3 — Paper-quality evidence
- [ ] Freeze all main configurations before the final benchmark sweep.
- [ ] Report mean ± std over independent runs.
- [ ] Main figure: **Influence Quality vs Oracle Cost**.
- [ ] Robustness figure: predictor quality / audit reliability vs fallback frequency and oracle cost.
- [ ] Ensure all result tables can be reproduced from repository scripts/configs.

## Do not prioritize now
- Do not add complex RL.
- Do not chase small MAE/Spearman improvements unless they improve sequential shortlist/trust behavior.
- Do not keep manually adjusting `tau` on the evaluation data.
- Do not over-interpret quality ratios slightly above 1; they are finite-MC / trajectory effects.

## Session-resume instruction
A new session should start by reading `AGENTS.md`, `docs/RESEARCH_STATE.md`, `docs/EXPERIMENT_LOG.md`, and this file. The held-out calibration diagnosis is complete: Spearman trust was rejected and the audited-residual gate is now the frozen main prototype. The immediate target is **candidate-pool 256/512 scaling**, followed by multiple budgets and graph/RIS scaling.


## Latest override — audited residual gate frozen
- [x] Held-out trust calibration diagnosis completed; local audit Spearman rejected as the primary trust statistic.
- [x] Audited residual upper-bound gate implemented and validated over five independent 128-candidate repeats.
- [x] Frozen prototype: `residual_q=1.0`, `residual_beta=0`, Top-16 + 8 sentinels, audit MC20, progressive 5→10→20→40 fast path.
- [ ] Immediate: candidate pool 256 and 512 scaling under the frozen rule.
- [ ] Then: multiple budgets and RIS/all-node scaling; do not return to manual trust-threshold tuning unless scaling exposes a new failure.

## 2026-09-04 scaling override
- [x] Stress current audited-residual method at candidate pools 256 and 512.
- [x] Test naive scale-aware Top-K/sentinel expansion; reject it as structurally non-monotone.
- [x] Diagnose exact-winner learned ranks across pools 128/256/512.
- [ ] **P0 now:** large-candidate sequential hard-negative fine-tuning of the state-aware marginal predictor.
- [ ] First success criterion: materially reduce pool512 winner mean-rank from 71.6 and raise Top64 recall from 0.5 toward >=0.8 without losing state-conditioning sensitivity.
- [ ] Only after proposal improves: redesign a population-aware outsider certificate; do not resume ad hoc Top-K tuning.
