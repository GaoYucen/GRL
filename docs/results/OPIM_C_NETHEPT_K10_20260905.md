# Official OPIM-C vs GRL — NetHEPT k=10 same-protocol check

Date: 2026-09-05

## Protocol
- Dataset: NetHEPT, 15,233 nodes / 32,235 edges.
- Diffusion: IC, edge probabilities loaded from `data/NetHEPT.txt`.
- Baseline: official OPIM-C from `tangj90/OPIM`, source commit `344cc3d5eaa13d8cdf9a9e75722e49d341981e8d`.
- OPIM-C: `k=10`, `eps=0.1`, default `delta=1/n`, minimum-upper-bound mode.
- Final spread evaluation: GRL `estimate_spread`, MC=1000, seeds `1900401`, `1901422`, `1902443`, exactly matching the three full-graph closure evaluation seeds.

## Quality
| Method | Spread mean ± std |
| --- | ---: |
| Official OPIM-C | **502.116 ± 1.430** |
| GRL audited learning-augmented | **500.940 ± 2.441** |
| Independent 50k-RR greedy | **506.741 ± 3.533** |
| Screened Full-MC40 | **508.468 ± 1.965** |

OPIM-C spreads by evaluation seed: `504.017`, `501.763`, `500.568`. Audited / OPIM-C = `0.9976586`. These values should be treated as essentially the same quality at current finite-MC resolution.

## Runtime
- GRL 50k-RR screening: `0.482 ± 0.116 s`.
- GRL audited sequential selection: `24.502 ± 1.633 s`.
- GRL screening + audited selection: about `24.984 s` mean.
- OPIM-C official optimization executable: `0.602 s`.
- OPIM-C format/reverse-graph step: `0.069 s`; output is successfully generated even though the formatter process returns code 1.
- OPIM-C format + optimization: about `0.672 s`.

Current GRL prototype is therefore roughly `37×` slower on the optimization path while delivering comparable spread.

## Decision implication
This is a P0 paper-value risk, not a routine parameter-tuning issue. Do not continue broad benchmark expansion until the paper has a defensible setting in which fallible learned advice provides value beyond mature RIS (e.g. expensive/nonstandard black-box oracle, amortized repeated-query setting, or a clearly justified safe-advice problem formulation).
