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

## 2026-09-04 — Online trust gate repairs robustness failure

A controlled predictor-corruption stress test exposed a failure of the residual-envelope-only heuristic: when learned scores were fully randomized (clean-vs-corrupted Spearman approximately 0), the ungated method became falsely confident, used only 224 exact candidate evaluations / 5,560 MC candidate-samples, and final spread fell to 369.541 (83.06% of the Full-MC reference).

We therefore added an online trust audit. At every greedy step it evaluates the predicted Top-16 plus eight rank-spaced sentinel candidates using MC20. The predictor is trusted only if audit learned-vs-exact Spearman exceeds a threshold and no sentinel candidate unexpectedly beats the audited head. Untrusted steps ignore learned ranking and fall back to Full-MC40 over all available candidates.

On NetHEPT (128 candidates, budget=10), tau=0.3 gives the best first-pass consistency/robustness tradeoff. Clean predictor: spread 443.591 (99.70% of Full-MC), 802 exact candidate evaluations, 31,440 MC candidate-samples, 3/10 fallback steps. Moderate corruption alpha=0.5: spread 446.189 under the finite evaluation protocol, 769 exact evaluations, 30,060 samples, 5/10 fallbacks. Strong corruption alpha=0.75: spread 444.911, 1,044 exact evaluations, 41,480 samples, 8/10 fallbacks. Fully randomized predictor alpha=1.0: all 10 steps fall back, exactly recovering the Full-MC reference cost (1,235 candidates / 49,400 samples) and measured spread 444.911.

Values slightly above the Full-MC reference at intermediate corruption levels are finite-MC / trajectory effects and must not be interpreted as outperforming exact greedy. The important result is graceful fallback: as prediction trust collapses, the method approaches classical Full-MC behavior instead of becoming cheaper and catastrophically worse.

## 2026-09-04 — Multi-seed trust calibration: robustness trend survives independent randomness

**Protocol:** NetHEPT; shared 128-candidate pool; budget 10; five independent repeats. Predictor corruption, MC-oracle worlds, and final-spread evaluation use separate repeat-specific random seeds. Each repeat has its own same-seed Full-MC40 reference. The frozen trust-progressive configuration is `tau=0.3`, audit MC20, Top-16 audit head, four rank-spaced sentinels, and the validated progressive 5→10→20→40 fast path.

**Aggregate results:**
- Full-MC reference spread: **445.154 ± 3.154**, 49,400 MC candidate-samples.
- Clean predictor (`alpha=0`): mean quality ratio **1.0005 ± 0.0082**, mean **26,004 samples = 52.6%** of Full-MC, **4.4 ± 1.5 fallback steps**.
- Moderate corruption (`alpha=0.5`, predictor Spearman ≈0.624): quality ratio **1.0030 ± 0.0093**, **24,542 samples = 49.7%**, **4.0 ± 1.6 fallbacks**.
- Strong corruption (`alpha=0.75`, predictor Spearman ≈0.248): quality ratio **1.0019 ± 0.0045**, **41,171 samples = 83.3%**, **7.6 ± 0.5 fallbacks**.
- Randomized predictor (`alpha=1`, predictor Spearman ≈−0.040): quality ratio **0.9998 ± 0.0005**, **47,161 samples = 95.5%**, **9.4 ± 0.9 fallbacks**.

**Interpretation:** the desired learning-augmented behavior is now visible across independent randomness: solution quality remains essentially at the Full-MC reference while oracle effort rises sharply as predictor quality collapses. A randomized predictor no longer causes the earlier catastrophic 83.06% quality failure; the solver approaches classical Full-MC cost automatically. It is not necessary for every random-predictor step to trigger literal full fallback, because trusted progressive verification can occasionally certify a decision safely.

**Current bottleneck:** clean-case **efficiency variance / false distrust**, not robustness. Across the five clean repeats, fallback count ranges from 2/10 to 6/10 and sample fraction from 41.4% to 64.2%. Therefore the next step is statistical/held-out calibration of the audit trust score, not another hand-tuned threshold sweep.

**Caveat:** quality ratios slightly above 1 are finite-MC / trajectory effects, not evidence of outperforming Full-MC greedy. This remains a 128-candidate prototype; full-graph/RIS and multi-dataset evidence are still required.

**Compact artifact:** `docs/results/nethept_trust_calibration_multiseed_20260904.json`.

## 2026-09-04 — Audited residual gate replaces Spearman trust score

Held-out calibration showed that audit Spearman is not a reliable trust statistic: unsafe steps can have higher local learned-vs-exact Spearman than safe steps, and the conservative calibrated threshold drove clean-case fallback close to Full-MC cost. We therefore replaced the Spearman gate with an **audited residual gate**. The audit evaluates the learned head plus rank-spaced sentinel candidates, estimates the upper tail of `exact - learned` residuals, and trusts the learned ranking only when the audited best head candidate dominates the learned best outsider after adding this audited residual upper bound; sentinel surprise still forces distrust.

The frozen prototype configuration is `residual_q=1.0`, `residual_beta=0`, audit Top-16 + 8 sentinels, audit MC20, followed by the validated progressive 5→10→20→40 fast path. Extra residual safety margins (`beta=0.5/1.0`) were too conservative and rapidly approached Full-MC cost.

Five independent NetHEPT repeats (128-candidate shared pool, budget 10) give:
- Clean predictor: quality ratio **1.0070 ± 0.0070**, **20,715 ± 1,997 samples = 41.9% ± 4.0%** of Full-MC, **0.8 ± 0.45 fallbacks**.
- `alpha=0.5`: quality ratio **1.0079 ± 0.0063**, **18,944 ± 3,756 samples = 38.3% ± 7.6%**, **1.2 ± 1.3 fallbacks**.
- `alpha=0.75`: quality ratio **1.0006 ± 0.0145**, **28,621 ± 5,010 samples = 57.9% ± 10.1%**, **3.8 ± 1.3 fallbacks**.
- Random predictor `alpha=1`: quality ratio **1.0042 ± 0.0051**, **41,893 ± 3,821 samples = 84.8% ± 7.7%**, **7.8 ± 1.3 fallbacks**.

Ratios slightly above 1 are finite-MC/trajectory noise and are not interpreted as outperforming Full-MC greedy. The important result is the cost response: useful predictions retain a large oracle saving, while degraded/random predictions automatically consume much more oracle work and preserve solution quality. The earlier Spearman gate is now retained as a failed/diagnostic ablation rather than the main method.

**Compact artifact:** `docs/results/nethept_audited_residual_multiseed_20260904.json`.

## 2026-09-04 — Candidate-scale stress test exposes proposal + certificate scaling failures

The audited-residual gate was next tested beyond the original 128-candidate prototype. This test is intentionally diagnostic rather than a paper-ready scalability claim.

### Fixed audit rule does not scale
Using the frozen 128-pool rule (Top-16 + 8 rank-spaced sentinels, audit MC20, residual q=1, beta=0):
- Pool 256, two repeats: clean quality ratio **0.9895 ± 0.0047** at **36.6% ± 10.4%** of Full-MC samples; alpha=.75 remains ~1.0002 at 64.5% cost; random remains ~0.9971 at 82.6% cost.
- Pool 512, one repeat: clean **0.9173** quality at 35.0% cost; alpha=.75 **0.8857** at 20.4% cost; random **0.9591** at 64.5% cost.

Thus the 128-candidate success must not be extrapolated to larger candidate sets.

### Simply enlarging the audit head is not a fix
A separate scale-aware audit-budget pilot tested larger heads/sentinel counts. It failed structurally:
- pool256 Top32+8: clean ratio **0.9804**, random **0.9547**.
- pool512 Top32+16: clean **0.8916**, random **0.9803**.
- pool512 Top64+16: clean **0.9110**, random **0.9299**.

The current certificate compares the audited best against the first learned outsider after the head. Enlarging the head moves this outsider deeper to a lower learned score and can therefore make the certificate *easier* to satisfy. Audit size tuning alone is not a principled solution.

### Clean proposal ranking also degrades with candidate count
Along exact Full-MC trajectories, the learned rank of the true winner changes sharply with pool size:
- pool128: ranks `[1,57,87,2,2,42,1,8,1,32]`, mean **23.3**, max **87**, Top64 recall **0.9**.
- pool256: `[1,70,79,141,2,47,2,9,2,46]`, mean **39.9**, max **141**, Top64 recall **0.7**.
- pool512: `[1,144,81,173,2,48,13,160,11,83]`, mean **71.6**, max **173**, Top64 recall **0.5**.

Several large-pool hard states have learned-top1 regret above 40–50 marginal-influence units. Therefore the scaling failure is **two coupled problems**: (1) the learned proposal itself becomes weaker on large candidate sets, and (2) the current audited residual certificate lacks population-aware coverage of the unseen outsider set.

**Decision:** stop manual Top-K/sentinel tuning. The next P0 is large-candidate sequential hard-negative training for the proposal, followed separately by a population-aware statistical certificate whose confidence becomes stricter as the number of unseen outsiders grows.

## 2026-09-05 — Full-graph RR screening + 128 shortlist 闭环通过 {#FULLGRAPH_RR_SCREENING_20260905}

当前主线从“人为 Top-degree 128”升级为完整图输入：`NetHEPT 15,233 nodes -> independent RR screening -> Top-128 shortlist -> state-aware marginal proposal -> audited residual + progressive verification -> Full-MC fallback`。

3 个独立 repeat（screening RR 与 full-graph RR baseline 使用独立 50k RR samples）结果：
- independent RR-greedy seed recall in Top-128: **1.0000 ± 0.0000**；
- full-graph independent RR-sketch greedy spread: **506.7407 ± 3.5330**；
- screened Full-MC40 spread: **508.4677 ± 1.9645**；
- audited/progressive spread: **500.9403 ± 2.4410**；
- audited / screened Full-MC quality: **0.9852 ± 0.0035**；
- audited / full-graph RR-sketch quality: **0.9886 ± 0.0109**；
- expensive MC sample fraction: **0.5908 ± 0.0846**（约 40.9% reduction）；
- fallback: **4.0 ± 1.63 / 10 steps**；
- pairwise shortlist Jaccard: **0.6954 ± 0.0092**。

Interpretation: 128 不再是人为 toy pool，而是完整图上的 cheap screening budget。不同 RR 随机种子得到的 shortlist 并不完全相同，但独立 RR-greedy 的强 seed 在 3/3 repeats 中均被覆盖。当前可以认为 full-graph-input learning-augmented closure 已经跑通。

Caveat: 这里的 50k RR baseline 是 prototype RR-sketch greedy，不等同于正式 IMM/OPIM-C 等带理论采样量的 paper-grade RIS baseline；不能据此声称优于 RIS。下一步先验证 shortlist budget M={64,128,256}，再做 k={5,10,20} 和多图/正式 RIS baseline。
