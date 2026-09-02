# Preliminary 3-Seed Ensemble Result

This is a smoke-scale result for the ICLR 2027 marginal-gain oracle route.

## Setup

- Dataset: NetHEPT
- Ensemble members: 3 random seeds (1, 2, 3)
- Training: 12 marginal-gain samples, 1 epoch, 3 Monte Carlo label runs
- Conditional evaluation: 2 test ranking groups, 4 candidates per group
- Confidence scale: 1.96
- Code commit: `2472682e8b04db6a5fc5417df1556fbfecefe79d`
- Server-control run: [33639274371](https://github.com/GaoYucen/server-control/actions/runs/33639274371)

## Results

| Metric | Ensemble mean | Conservative lower bound |
|---|---:|---:|
| Conditional Spearman | 0.3291 | 0.5873 |
| Conditional Kendall | 0.2500 | 0.4167 |
| Top-1 accuracy | 0.5000 | 1.0000 |
| Mean regret | 2.5000 | 0.0000 |
| Recall@1 | 0.5000 | 1.0000 |

Uncertainty diagnostics:

- Mean absolute error: 1.9120
- Mean ensemble standard deviation: 0.0929
- Uncertainty-error Spearman: -0.1905
- Empirical 1.96-std coverage: 0.0000
- Evaluated candidates: 8

## Interpretation

The conservative score happens to improve top-1 selection on this tiny test split, but the uncertainty-error correlation is negative and interval coverage is zero. Therefore, the current uncertainty estimate is not calibrated; the apparent ranking gain is only a path-validation signal.

The next valid experiment is to increase the number of held-out ranking groups, use higher-MC labels, and sweep the confidence scale before integrating conservative fallback into sequential selection.
