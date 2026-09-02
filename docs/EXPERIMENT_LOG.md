# GRL Experiment Log

This file is a concise index of experiments worth remembering. Raw logs, checkpoints, copied worktrees, and large outputs remain in their project-local result directories.

## 2026-09-01 to 2026-09-02 — Marginal-gain predictability validation
**Goal:** Determine whether the model genuinely learns conditional marginal gain Δ(v|S), especially candidate ranking under a fixed seed set/state.

**Current retained conclusion:** Marginal-gain prediction is sufficiently promising to remain the core learning target, but strict unseen-state validation and state-conditioning diagnostics remain important before treating the predictor as a reliable oracle.

**Related work:** NetHEPT marginal-gain predictability tests and strict state-conditioning checks documented in the GRL Notion workspace.

## 2026-09-02 — ICLR 2027 route pre-experiments
**Goal:** Test components around a learning-augmented certified marginal-gain oracle, including uncertainty/trust logic and ensemble-style pretests.

**Status:** Active / exploratory. Multiple project-local pre-experiment directories exist under `/workspace/GRL`. Exact metrics should be promoted into this log only after the corresponding result/config artifact has been checked.
