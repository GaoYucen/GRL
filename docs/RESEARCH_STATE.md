# GRL Research State

Last updated: 2026-09-04

## Scope
The project targets **Influence Maximization** rather than generic combinatorial optimization.

## Current repository baseline
- Primary workspace: `/workspace/GRL`
- Git remote: `GaoYucen/GRL`
- Main branch: `main`
- The repository contains a structured baseline with `configs/`, `docs/`, `scripts/`, `src/grl/`, and `tests/`.

## Confirmed research findings
- Predicting conditional **marginal gain** is a more promising learning target than directly predicting the final solution/set.
- Validation should emphasize whether the model correctly ranks candidate nodes under a fixed current seed set/state, not only global regression error.
- The current ICLR 2027-oriented route is a **learning-augmented sequential IM solver**: state-aware marginal prediction → progressive candidate/sample verification → online trust audit → classical-oracle fallback.
- The predictor should be treated as a proposal/ranking mechanism rather than a complete replacement for MC/RIS marginal-gain estimation.

## Current technical direction
1. Learn / estimate conditional marginal gain Δ(v|S) with explicit state-aware supervision.
2. Use learned scores to propose/rank candidates at each sequential greedy step.
3. Adapt both the number of verified candidates and the MC samples allocated to them.
4. Audit learned ranking online using predicted-head and sentinel candidates.
5. Increase oracle effort automatically when trust degrades; approach classical Full-MC/RIS behavior when prediction is unreliable.
6. Evaluate downstream influence spread, oracle/sample savings, robustness, and generalization—not prediction accuracy alone.

## Current workspace note
Several GRL-specific pre-experiment/worktree directories currently live inside `/workspace/GRL`. They are intentionally left untouched. Future GRL temporary artifacts should remain inside the GRL project directory rather than `/workspace` root.

## Source of truth
- Code/config/reproducible artifacts: GitHub + `/workspace/GRL`
- Human-readable research plan and decisions: Notion GRL Research
- Current machine-independent project state: this file plus `DECISIONS.md`, `EXPERIMENT_LOG.md`, and `NEXT_STEPS.md`

## 2026-09-03 verified update — state-aware oracle and first end-to-end prototype
- Strict disjoint unseen-state validation confirms that marginal-gain prediction remains a viable core learning target, but seed-mask ablations and overlap stress exposed a candidate-strength shortcut in the original predictor.
- State-sensitive difficult-state training plus same-candidate delta supervision substantially restores conditionality: same-candidate cross-state Spearman improved from 0.267 to 0.908 while preserving strong ordinary candidate ranking.
- A candidate-conditioned residual ablation further demonstrated that explicit candidate–seed interactions can recover state-response amplitude, but candidate-specific drop calibration is not yet solved; this is not the current bottleneck to framework integration.
- The first end-to-end sequential IM framework is implemented with learned-oracle, batched-MC-oracle, full-greedy, learned-greedy, and selective-refinement components.
- NetHEPT first prototype (fixed 128-candidate pool, budget 10, MC=40 for selection, MC=1000 for final spread): Full-MC spread 444.911; learned-only state-aware spread 321.526 (72.3% of Full-MC); selective Top-8/16/32 reaches 92.6%/93.1%/95.0% of Full-MC while using 6.5%/13.0%/25.9% of Full-MC exact candidate evaluations.
- Interpretation: the end-to-end learning-augmented route is operational, but learned-only ranking is not reliable enough for sequential IM. Selective verification is essential.

## 2026-09-03 adaptive certification milestone
- Sequential diagnosis on the Full-MC trajectory shows the learned ranks of the true best candidate are `[1, 57, 87, 2, 2, 42, 1, 8, 1, 32]`; offline marginal-ranking metrics therefore overstate fixed-shortlist reliability.
- Adaptive residual-envelope refinement (`beta=0.5`) reaches **443.626 spread = 99.71% of Full-MC** with **512 / 1235 = 41.46%** of exact candidate evaluations.
- Reusing common live-edge worlds within each greedy step reduces adaptive live-edge generation from 2560 to 400 and prototype selection time from 139.7s to 32.4s without changing quality.
- Current certification is empirical/operational; formal uncertainty calibration and a robustness guarantee remain open.

## 2026-09-03 update — progressive two-dimensional oracle adaptation
The learned marginal oracle proposes a ranking; adaptive residual-envelope certification determines how many candidates receive exact verification; progressive common-random-number MC allocates 5→10→20→40 worlds only as needed. Candidate/world results are cached.

Current recommended fast-path prototype (`residual_beta=0.5`, `confidence_z=0.5`, `bootstrap_mc=10`) reaches spread 443.626 = 99.71% of Full-MC using 504/1235 exact candidate evaluations and 18,280/49,400 MC candidate-samples.

## 2026-09-04 update — trust-aware robustness mechanism
A randomized-predictor stress test showed that progressive/residual certification alone can become falsely confident: it used only 5,560 candidate-samples and collapsed to 83.06% of Full-MC quality. This invalidated a naive robustness claim.

The repair is an online trust audit over predicted-head and rank-spaced sentinel candidates. Untrusted steps ignore learned ranking and fall back to all-candidate MC40; trusted steps use the progressive fast path. This yields the desired architectural behavior: good predictions reduce oracle work, bad predictions trigger more exact work.

## 2026-09-04 verified update — multi-seed consistency/robustness pilot
The frozen trust-progressive configuration was evaluated over five independent repeats with separate predictor-corruption, exact-MC, and final-evaluation random seeds. Each repeat used a same-seed Full-MC reference. NetHEPT, shared 128-candidate pool, budget 10.

Aggregate results:
- Full-MC spread: **445.154 ± 3.154**, 49,400 candidate-samples.
- Clean predictor: quality ratio **1.0005 ± 0.0082**, **26,004 samples = 52.6%** of Full-MC, **4.4 ± 1.5 fallback steps**.
- Moderate corruption (`alpha=0.5`, predictor Spearman ≈0.624): quality ratio **1.0030 ± 0.0093**, **24,542 samples = 49.7%**, **4.0 ± 1.6 fallbacks**.
- Strong corruption (`alpha=0.75`, predictor Spearman ≈0.248): quality ratio **1.0019 ± 0.0045**, **41,171 samples = 83.3%**, **7.6 ± 0.5 fallbacks**.
- Random predictor (`alpha=1`, predictor Spearman ≈−0.040): quality ratio **0.9998 ± 0.0005**, **47,161 samples = 95.5%**, **9.4 ± 0.9 fallbacks**.

**Current interpretation:** the central empirical learning-augmented story now survives independent randomness: final quality remains essentially at Full-MC while oracle effort rises strongly as prediction quality collapses. The random-predictor endpoint no longer catastrophically fails and instead approaches classical Full-MC cost.

**Current bottleneck:** clean-case trust efficiency is variable. Across clean repeats, fallback count ranges 2–6/10 and sample fraction 41.4%–64.2%. Robustness is therefore no longer the first problem to solve; the next methodological task is **held-out/statistical trust calibration to reduce false distrust and efficiency variance without weakening the severe-failure fallback**.

**Important caveat:** ratios above 1 are finite-MC / trajectory noise and must not be framed as outperforming Full-MC greedy. All current end-to-end results are still on a fixed 128-candidate prototype pool, not full-graph/RIS paper-scale evaluation.

## 2026-09-04 verified update — audited residual gate becomes the main trust mechanism
Held-out analysis invalidated local audit Spearman as the primary trust statistic: high local rank correlation does not guarantee that a strong outsider was not omitted. The main trust mechanism is therefore now an **audited residual upper-bound test** over predicted-head plus rank-spaced sentinel candidates.

Frozen prototype rule: `residual_q=1.0`, `residual_beta=0`, Top-16 + 8 sentinels at MC20, followed by progressive 5→10→20→40 verification when trusted. Five independent NetHEPT repeats on the 128-candidate prototype show sample fraction rising from **41.9%±4.0%** for a clean predictor to **57.9%±10.1%** at strong corruption and **84.8%±7.7%** for a random predictor, while final spread remains within finite-MC variation of the same-seed Full-MC references.

**Current interpretation:** the core consistency–robustness architecture is now empirically credible at the 128-candidate prototype scale. The next bottleneck is no longer trust-threshold tuning; it is **candidate-scale / graph-scale validation** and eventual replacement of the NetworkX MC prototype with a scalable RIS/all-node oracle.
