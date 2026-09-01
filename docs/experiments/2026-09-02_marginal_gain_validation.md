# NetHEPT Marginal Gain Predictability Validation

Date: 2026-09-02  
Source GRL commit: `a554560992018fa556eca7d063da0c7ef7a2dcb1`  
4090 runner: `81f635a75946`  
Server-control run: `33531405398`

## Research question

Can the current predictor learn the conditional marginal gain

\[
\Delta(v\mid S)=\sigma(S\cup\{v\})-\sigma(S)
\]

well enough to support sequential node selection?

For the downstream influence-maximization decision, the important question is not only the numerical regression error. The model should rank candidate nodes correctly **for the same current seed set `S`**.

## First grouped validation

Dataset: NetHEPT

- Nodes: 15,233
- Edges: 32,235
- Embedding: Node2Vec
- Predictor: `MarginalGainPredictor`
- Candidate count per state: 16
- Train / validation / test states: 60 / 12 / 20
- MC runs: 10 / 20 / 40
- Model random seeds: 11, 22, 33
- Labels for candidates under the same state share common live-edge worlds.

### Aggregate results

| Metric | Mean |
| --- | ---: |
| MAE | 2.2892 |
| RMSE | 3.8748 |
| Pearson | 0.9805 |
| Global Spearman | 0.9383 |
| **Conditional Spearman** | **0.9314** |
| **Pairwise ranking accuracy** | **0.9163** |
| **Top-1 accuracy** | **0.9000** |
| **Top-3 recall** | **0.9111** |
| **Selected true gain / candidate oracle gain** | **0.9943** |
| Mean regret | 0.1825 |
| Degree selected gain / candidate oracle gain | 0.4011 |

The three model seeds are stable. Conditional Spearman is 0.9306 / 0.9299 / 0.9337, and Top-1 accuracy is 0.90 for all three runs.

## Current interpretation

The result is a strong positive signal that direct marginal-gain prediction is substantially easier and more decision-aligned than predicting total spread and taking two predicted values' difference.

The most decision-relevant number is the selected-gain ratio: within the evaluated candidate sets, the node selected by the predictor achieves 99.43% of the true gain of the candidate oracle on average.

However, this result is **not yet final generalization evidence**.

## Important limitation

The first state generator mixes random states with deterministic high-degree and degree-discount prefixes. Therefore, some seed sets can repeat across train/validation/test. This can make the first result optimistic.

The next validation must enforce disjoint seed sets:

\[
S_{train}\cap S_{val}=S_{train}\cap S_{test}=S_{val}\cap S_{test}=\varnothing.
\]

## Required strict validation

Use:

```bash
python scripts/experiments/evaluate_marginal_strict.py
```

The strict experiment contains three checks.

### 1. Unseen-state generalization

Train, validation and test seed sets are unique and disjoint. Test labels use more Monte Carlo samples.

### 2. Seed-mask ablation

Evaluate the same trained model and the same test candidate sets under:

- correct seed set: `f(S, v)`
- zero seed set: `f(empty, v)`
- shuffled/wrong seed set: `f(S_wrong, v)`

This distinguishes true conditional learning from simply learning candidate intrinsic influence.

### 3. Overlap stress test

For the same candidate, compare a seed set that overlaps with the candidate's local influence neighborhood against a far/random seed set. The model should reproduce the reduction in true marginal gain caused by overlap.

## Decision rule for the next stage

Do **not** start claiming that the marginal predictor is validated solely from the first grouped result.

A useful working criterion for proceeding is:

- unseen-state Conditional Spearman remains high (target: at least 0.8 as a strong signal);
- selected true gain / candidate oracle gain remains high (target: at least 0.9);
- the correct seed mask is meaningfully better than zero/shuffled masks;
- overlap stress shows that the predictor responds to diminishing returns rather than only candidate strength.

If these conditions hold, the next algorithmic question becomes whether reinforcement learning adds value beyond a simple predicted-marginal greedy policy.

## Reproducibility files

- `scripts/experiments/evaluate_marginal_predictability.py`: reproduces the first grouped validation.
- `scripts/experiments/evaluate_marginal_strict.py`: strict unseen-state and ablation validation.
- `outputs/marginal_predictability/nethept_grouped/summary.json`: compact record of the first 4090 results.

Large Node2Vec embeddings, checkpoints and temporary labeled datasets are intentionally not committed.
