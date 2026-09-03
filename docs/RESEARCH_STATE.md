# GRL Research State

Last updated: 2026-09-03

## Scope
The project targets **Influence Maximization** rather than generic combinatorial optimization.

## Current repository baseline
- Primary workspace: `/workspace/GRL`
- Git remote: `GaoYucen/GRL`
- Main branch observed on 2026-09-02: `main`
- Main commit observed before this state-document setup: `a554560`
- The repository already contains a structured baseline with `configs/`, `docs/`, `scripts/`, `src/grl/`, and `tests/`.

## Confirmed research findings
- Predicting conditional **marginal gain** is a more promising learning target than directly predicting the final solution/set.
- Validation should emphasize whether the model correctly ranks candidate nodes under a fixed current seed set/state, not only global regression error.
- The current ICLR 2027-oriented route is to build on the marginal-gain predictor toward a **learning-augmented / certified marginal-gain oracle** for sequential influence-maximization decisions.

## Current technical direction
1. Learn / estimate conditional marginal gain Δ(v|S).
2. Validate generalization under strict unseen-state conditions and diagnose whether state information is genuinely used.
3. Use prediction uncertainty / certification / fallback logic to determine when learned estimates are trusted versus when an exact or stronger oracle is invoked.
4. Evaluate downstream influence-maximization quality, query/computation savings, robustness, and generalization—not prediction accuracy alone.

## Current workspace note
Several GRL-specific pre-experiment/worktree directories currently live inside `/workspace/GRL`. They are intentionally left untouched by this setup. Future GRL temporary artifacts should remain inside the GRL project directory rather than `/workspace` root.

## Source of truth
- Code/config/reproducible artifacts: GitHub + `/workspace/GRL`
- Human-readable research plan and decisions: Notion GRL Research
- Current machine-independent project state: this file plus `DECISIONS.md`, `EXPERIMENT_LOG.md`, and `NEXT_STEPS.md`

## 2026-09-03 verified update — state-aware oracle and first end-to-end prototype
- Strict disjoint unseen-state validation confirms that marginal-gain prediction remains a viable core learning target, but seed-mask ablations and overlap stress exposed a candidate-strength shortcut in the original predictor.
- State-sensitive difficult-state training plus same-candidate delta supervision substantially restores conditionality: same-candidate cross-state Spearman improved from 0.267 to 0.908 while preserving strong ordinary candidate ranking.
- A candidate-conditioned residual ablation further demonstrated that explicit candidate–seed interactions can recover state-response amplitude, but candidate-specific drop calibration is not yet solved; this is not the current bottleneck to framework integration.
- The first end-to-end sequential IM framework is now implemented with explicit learned-oracle, batched-MC-oracle, full-greedy, learned-greedy, and selective-refinement components.
- NetHEPT first prototype (fixed 128-candidate pool, budget 10, MC=40 for selection, MC=1000 for final spread): Full-MC spread 444.911; learned-only state-aware spread 321.526 (72.3% of Full-MC); selective Top-8/16/32 reaches 92.6%/93.1%/95.0% of Full-MC while using 6.5%/13.0%/25.9% of Full-MC exact candidate evaluations.
- Interpretation: the end-to-end learning-augmented route is operational and useful, but learned-only ranking over a broader sequential candidate pool is not yet strong enough. The immediate bottleneck is shortlist recall / certification quality, not whether the overall sequential framework works.


## 2026-09-03 adaptive certification milestone
- Sequential diagnosis on the Full-MC trajectory shows the learned ranks of the true best candidate are `[1, 57, 87, 2, 2, 42, 1, 8, 1, 32]`; therefore offline marginal-ranking metrics substantially overstate fixed-shortlist reliability.
- Fixed Top-8/16/32 refinement reaches 92.64% / 93.08% / 95.02% of Full-MC spread while using 6.48% / 12.96% / 25.91% of exact candidate evaluations.
- Adaptive residual-envelope refinement (`beta=0.5`) reaches **443.626 spread = 99.71% of Full-MC** with **512 / 1235 = 41.46%** of exact candidate evaluations.
- Per-step exact verification is `[16, 16, 16, 80, 64, 112, 96, 80, 16, 16]`, demonstrating the intended easy-state / hard-state adaptive behavior.
- Reusing common live-edge worlds within each greedy step reduces adaptive live-edge generation from 2560 to 400 and prototype selection time from 139.7s to 32.4s without changing quality.
- Current certification is empirical/operational; formal uncertainty calibration and a robustness guarantee remain open.

## 2026-09-03 update — progressive two-dimensional oracle adaptation

The current end-to-end method now adapts along two axes. The learned marginal oracle first proposes a ranking; adaptive residual-envelope certification determines how many candidates receive exact verification; progressive common-random-number MC then allocates 5→10→20→40 worlds only as needed. Candidate/world results are cached, so refinement does not repeat completed simulations.

Current recommended NetHEPT prototype: residual_beta=0.5, confidence_z=0.5, bootstrap_mc=10. It reaches spread 443.626 = 99.71% of Full-MC while using 504/1235 exact candidate evaluations and 18,280/49,400 MC candidate-samples. This is the strongest current evidence for the learning-augmented certified-oracle framing. These are still single-protocol prototype results; robustness under deliberately degraded predictions is the next required validation before making a general consistency/robustness claim.
