from __future__ import annotations

"""Multi-trajectory full-candidate listwise tuning for scalable IM proposals.

This follows the first hard-negative pilot.  Instead of supervising a small
mined subset around one winner, every training state contains the full
256/512 candidate ranking under a Full-MC trajectory.  The objective combines
state-normalized regression, listwise distribution matching, and a small
anchor to the state-aware checkpoint.

If the held-out 512-candidate proposal reaches the predefined rank target, the
script immediately runs a clean/random audited-residual end-to-end pilot using
the tuned proposal so one job can test the complete loop.
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from evaluate_adaptive_certification import build_context
from evaluate_audited_residual_scaleaware import run_level as run_gate_level
from evaluate_large_candidate_hard_negative import fast_scores, rank_diagnostic
from evaluate_trust_calibration_multiseed import run_full_reference
from grl.oracle import BatchedMonteCarloMarginalOracle

OUT = ROOT / "outputs" / "marginal_predictability" / "large_candidate_listwise"
OUT.mkdir(parents=True, exist_ok=True)


def collect_full_trajectory(
    graph,
    candidate_pool,
    budget,
    exact_mc,
    exact_seed,
    model,
    embeddings,
    norm_degrees,
):
    exact = BatchedMonteCarloMarginalOracle(graph, int(exact_mc), random_seed=int(exact_seed))
    selected = []
    states = []
    model.eval()
    for step in range(int(budget)):
        available = [v for v in candidate_pool if v not in set(selected)]
        truth = exact.score(selected, available, step=step)
        with torch.no_grad():
            base = fast_scores(model, embeddings, norm_degrees, selected, available)
        base_list = [float(x) for x in base.detach().cpu().tolist()]
        winner = max(available, key=lambda v: (truth[v], -v))
        learned_ranked = [
            v for _, v in sorted(
                zip(base_list, available), key=lambda x: (x[0], -x[1]), reverse=True
            )
        ]
        rank = learned_ranked.index(winner) + 1
        states.append({
            "step": int(step),
            "seeds": [int(x) for x in selected],
            "candidates": [int(v) for v in available],
            "labels": [float(truth[v]) for v in available],
            "base_predictions": base_list,
            "winner": int(winner),
            "winner_rank_before": int(rank),
        })
        selected.append(int(winner))
        print(
            f"collect-full seed={exact_seed} step={step+1} avail={len(available)} "
            f"winner={winner} learned_rank={rank}", flush=True
        )
    return states


def tune_listwise(
    base_model,
    embeddings,
    norm_degrees,
    states,
    epochs,
    lr,
    shape_weight,
    list_weight,
    raw_weight,
    distill_weight,
    temperature,
):
    tuned = copy.deepcopy(base_model)
    tuned.train()
    opt = torch.optim.Adam(tuned.parameters(), lr=float(lr), weight_decay=1e-5)
    rng = random.Random(280901)

    for epoch in range(int(epochs)):
        order = list(range(len(states)))
        rng.shuffle(order)
        totals = {"loss": 0.0, "shape": 0.0, "list": 0.0, "raw": 0.0, "distill": 0.0}
        for idx in order:
            row = states[idx]
            cand = row["candidates"]
            labels = torch.tensor(row["labels"], dtype=torch.float32, device=embeddings.device)
            base_pred = torch.tensor(row["base_predictions"], dtype=torch.float32, device=embeddings.device)
            pred = fast_scores(tuned, embeddings, norm_degrees, row["seeds"], cand)

            label_mean = labels.mean()
            label_std = labels.std(unbiased=False).clamp_min(1.0)
            pred_std = pred.std(unbiased=False).clamp_min(1e-3)
            label_z = (labels - label_mean) / label_std
            pred_z = (pred - pred.mean()) / pred_std

            shape = F.smooth_l1_loss(pred_z, label_z)
            raw = F.smooth_l1_loss((pred - labels) / label_std, torch.zeros_like(pred))

            t = float(temperature)
            target_prob = torch.softmax(label_z / t, dim=0).detach()
            list_loss = -(target_prob * torch.log_softmax(pred_z / t, dim=0)).sum()

            # Keep the absolute scale from drifting too far; the later residual
            # certificate uses learned scores as a calibrated proposal signal.
            distill = F.smooth_l1_loss((pred - base_pred) / label_std, torch.zeros_like(pred))
            loss = (
                float(shape_weight) * shape
                + float(list_weight) * list_loss
                + float(raw_weight) * raw
                + float(distill_weight) * distill
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tuned.parameters(), 5.0)
            opt.step()

            totals["loss"] += float(loss.item())
            totals["shape"] += float(shape.item())
            totals["list"] += float(list_loss.item())
            totals["raw"] += float(raw.item())
            totals["distill"] += float(distill.item())

        if epoch == 0 or (epoch + 1) % 3 == 0 or epoch + 1 == int(epochs):
            print(
                "list-ft epoch={} loss={:.4f} shape={:.4f} list={:.4f} raw={:.4f} distill={:.4f}".format(
                    epoch + 1,
                    totals["loss"], totals["shape"], totals["list"],
                    totals["raw"], totals["distill"],
                ),
                flush=True,
            )
    tuned.eval()
    return tuned


def gate_pilot(
    graph,
    pool,
    budget,
    eval_mc,
    tuned,
    embeddings,
    norm_degrees,
    device,
    audit_top_k,
    sentinels,
):
    exact_seed = 1180401
    eval_seed = 1190401
    corruption_seed = 1170401
    ref = run_full_reference(graph, pool, budget, eval_mc, exact_seed, eval_seed)
    full_samples = sum(len(pool) - step for step in range(budget)) * 40
    methods = {}
    for alpha in (0.0, 1.0):
        item = run_gate_level(
            graph, pool, budget, eval_mc,
            tuned, embeddings, norm_degrees, device,
            alpha, audit_top_k, sentinels, 20, 1.0, 0.0,
            corruption_seed, exact_seed, eval_seed,
        )
        item["quality_ratio_vs_full_mc"] = float(item["final_spread_mean"] / ref["final_spread_mean"])
        item["sample_fraction_vs_full_mc"] = float(item["oracle_stats"]["mc_candidate_samples"] / full_samples)
        methods[f"alpha_{alpha:g}"] = item
        print(
            f"GATE alpha={alpha:g} spread={item['final_spread_mean']:.3f} "
            f"ratio={item['quality_ratio_vs_full_mc']:.4f} "
            f"samples_frac={item['sample_fraction_vs_full_mc']:.3f} "
            f"fallback={item['fallback_steps']}/{budget}",
            flush=True,
        )
    return {"reference": ref, "methods": methods}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--train-mc", type=int, default=20)
    p.add_argument("--eval-mc", type=int, default=40)
    p.add_argument("--train-seeds-256", nargs="+", type=int, default=[281101, 281503])
    p.add_argument("--train-seeds-512", nargs="+", type=int, default=[282101, 282503])
    p.add_argument("--eval-seed", type=int, default=260903)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--shape-weight", type=float, default=1.0)
    p.add_argument("--list-weight", type=float, default=0.6)
    p.add_argument("--raw-weight", type=float, default=0.25)
    p.add_argument("--distill-weight", type=float, default=0.02)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--gate-audit-top-k", type=int, default=16)
    p.add_argument("--gate-sentinels", type=int, default=8)
    args = p.parse_args()

    graph_data, graph, device, embeddings, norm_degrees, base_model, full_pool = build_context(512)
    print(
        f"device={device} nodes={graph_data.num_nodes} edges={graph_data.num_edges} pool=512",
        flush=True,
    )

    baseline = {}
    for pool_size in (256, 512):
        pool = full_pool[:pool_size]
        _, s = rank_diagnostic(
            graph, pool, args.budget, args.eval_mc, args.eval_seed,
            base_model, embeddings, norm_degrees,
        )
        baseline[str(pool_size)] = s
        print(f"BASE pool={pool_size} {json.dumps(s, sort_keys=True)}", flush=True)

    states = []
    for seed in args.train_seeds_256:
        states.extend(collect_full_trajectory(
            graph, full_pool[:256], args.budget, args.train_mc, seed,
            base_model, embeddings, norm_degrees,
        ))
    for seed in args.train_seeds_512:
        states.extend(collect_full_trajectory(
            graph, full_pool[:512], args.budget, args.train_mc, seed,
            base_model, embeddings, norm_degrees,
        ))
    print(f"full_listwise_training_states={len(states)}", flush=True)

    tuned = tune_listwise(
        base_model, embeddings, norm_degrees, states,
        args.epochs, args.lr,
        args.shape_weight, args.list_weight, args.raw_weight,
        args.distill_weight, args.temperature,
    )

    tuned_diag = {}
    for pool_size in (256, 512):
        pool = full_pool[:pool_size]
        rows, s = rank_diagnostic(
            graph, pool, args.budget, args.eval_mc, args.eval_seed,
            tuned, embeddings, norm_degrees,
        )
        tuned_diag[str(pool_size)] = {"summary": s, "steps": rows}
        print(f"TUNED pool={pool_size} {json.dumps(s, sort_keys=True)}", flush=True)

    s512 = tuned_diag["512"]["summary"]
    proposal_pass = bool(s512["top_64_recall"] >= 0.8 and s512["mean_rank"] < 40.0)
    print(f"PROPOSAL_PASS={proposal_pass}", flush=True)

    gate = None
    if proposal_pass:
        print("=== END-TO-END GATE PILOT WITH TUNED PROPOSAL ===", flush=True)
        gate = gate_pilot(
            graph, full_pool[:512], args.budget, 1000,
            tuned, embeddings, norm_degrees, device,
            args.gate_audit_top_k, args.gate_sentinels,
        )
    else:
        print("gate pilot skipped because proposal target was not met", flush=True)

    report = {
        "dataset": "NetHEPT",
        "scope": "multi-trajectory full-candidate listwise proposal tuning",
        "config": vars(args),
        "baseline": baseline,
        "tuned": tuned_diag,
        "proposal_pass": proposal_pass,
        "gate_pilot": gate,
    }
    out = OUT / "report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({"state_dict": tuned.state_dict(), "config": vars(args)}, OUT / "model_listwise.pt")
    print(f"saved={out}", flush=True)
    print(f"saved_model={OUT / 'model_listwise.pt'}", flush=True)


if __name__ == "__main__":
    main()
