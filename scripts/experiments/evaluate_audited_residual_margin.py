from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from evaluate_adaptive_certification import build_context
from evaluate_audited_residual_gate import run_level
from evaluate_trust_calibration_multiseed import run_full_reference


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-size", type=int, default=128)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--eval-mc", type=int, default=1000)
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.75, 1.0])
    p.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    p.add_argument("--residual-q", type=float, default=1.0)
    p.add_argument("--audit-mc", type=int, default=20)
    p.add_argument("--sentinels", type=int, default=8)
    p.add_argument("--corruption-seed", type=int, default=940401)
    p.add_argument("--exact-seed", type=int, default=950401)
    p.add_argument("--eval-seed", type=int, default=960401)
    args = p.parse_args()

    graph_data, graph, device, embeddings, norm_degrees, model, candidate_pool = build_context(args.pool_size)
    print(
        f"device={device} nodes={graph_data.num_nodes} edges={graph_data.num_edges} "
        f"pool={len(candidate_pool)} budget={args.budget}", flush=True
    )

    ref = run_full_reference(
        graph, candidate_pool, args.budget, args.eval_mc, args.exact_seed, args.eval_seed
    )
    full_samples = int(ref["oracle_stats"]["mc_candidate_samples"])
    print(f"REF spread={ref['final_spread_mean']:.3f} samples={full_samples}", flush=True)

    methods = {}
    for beta in args.betas:
        for alpha in args.alphas:
            item = run_level(
                graph, candidate_pool, args.budget, args.eval_mc,
                model, embeddings, norm_degrees, device,
                alpha, args.residual_q, beta,
                args.audit_mc, args.sentinels,
                args.corruption_seed, args.exact_seed, args.eval_seed,
            )
            item["quality_ratio_vs_full_mc"] = float(item["final_spread_mean"] / ref["final_spread_mean"])
            item["sample_fraction_vs_full_mc"] = float(item["oracle_stats"]["mc_candidate_samples"] / full_samples)
            key = f"beta{beta:g}_alpha{alpha:g}"
            methods[key] = item
            print(
                key,
                f"pred_rho={item['mean_clean_corrupt_spearman']:.3f}",
                f"fallback={item['fallback_steps']}/{args.budget}",
                f"spread={item['final_spread_mean']:.3f}",
                f"ratio={item['quality_ratio_vs_full_mc']:.4f}",
                f"samples={item['oracle_stats']['mc_candidate_samples']}",
                f"frac={item['sample_fraction_vs_full_mc']:.3f}",
                flush=True,
            )

    out = ROOT / "outputs" / "end_to_end" / "audited_residual_margin" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": "NetHEPT",
        "scope": "single-seed audited-residual safety-margin pilot",
        "config": vars(args),
        "full_mc_reference": ref,
        "methods": methods,
    }, indent=2), encoding="utf-8")
    print(f"saved={out}", flush=True)


if __name__ == "__main__":
    main()
