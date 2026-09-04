from __future__ import annotations

"""Large-candidate sequential hard-negative fine-tuning for the marginal proposal.

This is a focused scaling experiment.  It starts from the state-aware NetHEPT
checkpoint, collects held-out sequential Full-MC trajectories on 256/512
candidate pools, mines false positives / true high-gain candidates, and tunes
within-state ranking without changing the model architecture.

The primary success criterion is proposal quality on a held-out 512-candidate
trajectory (winner rank / Top-K recall), not end-to-end certification.
"""

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from evaluate_adaptive_certification import build_context
from grl.oracle import BatchedMonteCarloMarginalOracle


OUT = ROOT / "outputs" / "marginal_predictability" / "large_candidate_hard_negative"
OUT.mkdir(parents=True, exist_ok=True)


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def fast_scores(model, embeddings, norm_degrees, seeds, candidates):
    """Vectorized candidate scoring while encoding the seed set only once."""
    device = embeddings.device
    features = torch.cat([embeddings, norm_degrees], dim=-1)
    mask = torch.zeros((features.shape[0], 1), dtype=features.dtype, device=device)
    if seeds:
        mask[list(seeds), 0] = 1.0
    seed_repr = model.seed_encoder(features, mask)
    cand = torch.as_tensor(candidates, dtype=torch.long, device=device)
    cand_repr = model.candidate_encoder(features[cand])
    sr = seed_repr.expand(cand_repr.shape[0], -1)
    interaction = sr * cand_repr
    difference = (sr - cand_repr).abs()
    return model.head(torch.cat([sr, cand_repr, interaction, difference], dim=-1)).squeeze(-1)


def spaced_background(ranked, excluded, count):
    remain = [v for v in ranked if v not in excluded]
    if not remain or count <= 0:
        return []
    if len(remain) <= count:
        return remain
    out = []
    for i in range(count):
        pos = round(i * (len(remain) - 1) / max(1, count - 1))
        out.append(remain[int(pos)])
    return list(dict.fromkeys(out))


def collect_trajectory(
    graph,
    candidate_pool,
    budget,
    exact_mc,
    exact_seed,
    model,
    embeddings,
    norm_degrees,
    learned_top_k,
    true_top_k,
    background_k,
):
    exact = BatchedMonteCarloMarginalOracle(graph, int(exact_mc), random_seed=int(exact_seed))
    selected = []
    states = []
    model.eval()
    for step in range(int(budget)):
        selected_set = set(selected)
        available = [v for v in candidate_pool if v not in selected_set]
        truth = exact.score(selected, available, step=step)
        with torch.no_grad():
            pred_tensor = fast_scores(model, embeddings, norm_degrees, selected, available)
        pred = {v: float(x) for v, x in zip(available, pred_tensor.detach().cpu().tolist())}

        truth_ranked = sorted(available, key=lambda v: (truth[v], -v), reverse=True)
        learned_ranked = sorted(available, key=lambda v: (pred[v], -v), reverse=True)
        winner = truth_ranked[0]
        winner_rank = learned_ranked.index(winner) + 1

        hard = list(dict.fromkeys(
            truth_ranked[: int(true_top_k)] + learned_ranked[: int(learned_top_k)]
        ))
        background = spaced_background(learned_ranked, set(hard), int(background_k))
        train_candidates = list(dict.fromkeys(hard + background))
        if winner not in train_candidates:
            train_candidates.insert(0, winner)

        states.append({
            "step": int(step),
            "seeds": [int(x) for x in selected],
            "winner": int(winner),
            "winner_rank_before": int(winner_rank),
            "train_candidates": [int(v) for v in train_candidates],
            "labels": [float(truth[v]) for v in train_candidates],
            "base_predictions": [float(pred[v]) for v in train_candidates],
            "available_count": int(len(available)),
        })
        selected.append(int(winner))
        print(
            f"collect seed={exact_seed} step={step+1} avail={len(available)} "
            f"winner={winner} learned_rank={winner_rank} train_cands={len(train_candidates)}",
            flush=True,
        )
    return states


def fit_hard_negative(
    model,
    embeddings,
    norm_degrees,
    states,
    epochs,
    lr,
    rank_weight,
    regression_weight,
    distill_weight,
):
    tuned = copy.deepcopy(model)
    tuned.train()
    opt = torch.optim.Adam(tuned.parameters(), lr=float(lr), weight_decay=1e-5)
    rng = random.Random(270901)

    for epoch in range(int(epochs)):
        order = list(range(len(states)))
        rng.shuffle(order)
        total = reg_total = rank_total = distill_total = 0.0
        for idx in order:
            row = states[idx]
            candidates = row["train_candidates"]
            labels = torch.tensor(row["labels"], dtype=torch.float32, device=embeddings.device)
            base_pred = torch.tensor(row["base_predictions"], dtype=torch.float32, device=embeddings.device)
            pred = fast_scores(tuned, embeddings, norm_degrees, row["seeds"], candidates)

            label_scale = labels.std(unbiased=False).clamp_min(1.0)
            reg = F.smooth_l1_loss((pred - labels) / label_scale, torch.zeros_like(pred))

            winner_idx = candidates.index(int(row["winner"]))
            winner_pred = pred[winner_idx]
            winner_label = labels[winner_idx]
            neg_mask = torch.arange(len(candidates), device=pred.device) != winner_idx
            neg_pred = pred[neg_mask]
            neg_label = labels[neg_mask]
            gaps = (winner_label - neg_label).clamp_min(0.0)
            useful = gaps > 0.25
            if useful.any():
                # Larger true regret receives more ranking weight, capped for stability.
                weights = (gaps[useful] / label_scale).clamp(0.25, 4.0).detach()
                pair = F.softplus(-(winner_pred - neg_pred[useful]) / 2.0)
                rank_loss = (weights * pair).mean()
            else:
                rank_loss = pred.sum() * 0.0

            # Small anchor term limits destructive drift of the calibrated state-aware model.
            distill = F.smooth_l1_loss((pred - base_pred) / label_scale, torch.zeros_like(pred))
            loss = (
                float(regression_weight) * reg
                + float(rank_weight) * rank_loss
                + float(distill_weight) * distill
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tuned.parameters(), 5.0)
            opt.step()

            total += float(loss.item())
            reg_total += float(reg.item())
            rank_total += float(rank_loss.item())
            distill_total += float(distill.item())

        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == int(epochs):
            print(
                f"hn-ft epoch={epoch+1} loss={total:.4f} reg={reg_total:.4f} "
                f"rank={rank_total:.4f} distill={distill_total:.4f}",
                flush=True,
            )
    tuned.eval()
    return tuned


def rank_diagnostic(
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
    rows = []
    cutoffs = (1, 8, 16, 32, 64, 128)
    model.eval()
    for step in range(int(budget)):
        available = [v for v in candidate_pool if v not in set(selected)]
        truth = exact.score(selected, available, step=step)
        with torch.no_grad():
            scores = fast_scores(model, embeddings, norm_degrees, selected, available)
        pred = {v: float(x) for v, x in zip(available, scores.detach().cpu().tolist())}
        ranked = sorted(available, key=lambda v: (pred[v], -v), reverse=True)
        winner = max(available, key=lambda v: (truth[v], -v))
        rank = ranked.index(winner) + 1
        top1 = ranked[0]
        rows.append({
            "step": int(step + 1),
            "winner": int(winner),
            "rank": int(rank),
            "top1": int(top1),
            "top1_regret": float(truth[winner] - truth[top1]),
        })
        selected.append(int(winner))

    ranks = [r["rank"] for r in rows]
    summary = {
        "mean_rank": mean(ranks),
        "max_rank": int(max(ranks)),
        "ranks": ranks,
        "mean_top1_regret": mean([r["top1_regret"] for r in rows]),
        "max_top1_regret": float(max(r["top1_regret"] for r in rows)),
    }
    for k in cutoffs:
        summary[f"top_{k}_recall"] = mean([float(r <= k) for r in ranks])
    return rows, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-pools", nargs="+", type=int, default=[256, 512])
    p.add_argument("--eval-pools", nargs="+", type=int, default=[256, 512])
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--train-mc", type=int, default=30)
    p.add_argument("--eval-mc", type=int, default=40)
    p.add_argument("--train-seeds", nargs="+", type=int, default=[271101, 271503])
    p.add_argument("--eval-seed", type=int, default=260903)
    p.add_argument("--learned-top-k", type=int, default=48)
    p.add_argument("--true-top-k", type=int, default=16)
    p.add_argument("--background-k", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank-weight", type=float, default=2.0)
    p.add_argument("--regression-weight", type=float, default=1.0)
    p.add_argument("--distill-weight", type=float, default=0.05)
    args = p.parse_args()

    max_pool = max(max(args.train_pools), max(args.eval_pools))
    graph_data, graph, device, embeddings, norm_degrees, base_model, full_pool = build_context(max_pool)
    print(
        f"device={device} nodes={graph_data.num_nodes} edges={graph_data.num_edges} "
        f"max_pool={len(full_pool)}",
        flush=True,
    )

    # Baseline held-out diagnostics before any fine-tuning.
    baseline = {}
    for pool_size in args.eval_pools:
        pool = full_pool[: int(pool_size)]
        _, summary = rank_diagnostic(
            graph, pool, args.budget, args.eval_mc, args.eval_seed,
            base_model, embeddings, norm_degrees,
        )
        baseline[str(pool_size)] = summary
        print(f"BASE pool={pool_size} {json.dumps(summary, sort_keys=True)}", flush=True)

    train_states = []
    for i, pool_size in enumerate(args.train_pools):
        seed = args.train_seeds[i % len(args.train_seeds)]
        pool = full_pool[: int(pool_size)]
        train_states.extend(collect_trajectory(
            graph, pool, args.budget, args.train_mc, seed,
            base_model, embeddings, norm_degrees,
            args.learned_top_k, args.true_top_k, args.background_k,
        ))
    print(f"training_states={len(train_states)}", flush=True)

    tuned = fit_hard_negative(
        base_model, embeddings, norm_degrees, train_states,
        args.epochs, args.lr, args.rank_weight,
        args.regression_weight, args.distill_weight,
    )

    tuned_diag = {}
    for pool_size in args.eval_pools:
        pool = full_pool[: int(pool_size)]
        rows, summary = rank_diagnostic(
            graph, pool, args.budget, args.eval_mc, args.eval_seed,
            tuned, embeddings, norm_degrees,
        )
        tuned_diag[str(pool_size)] = {"summary": summary, "steps": rows}
        print(f"TUNED pool={pool_size} {json.dumps(summary, sort_keys=True)}", flush=True)

    report = {
        "dataset": "NetHEPT",
        "scope": "large-candidate sequential hard-negative proposal tuning",
        "config": vars(args),
        "baseline": baseline,
        "tuned": tuned_diag,
        "success_target": {
            "pool512_top64_recall": 0.8,
            "pool512_mean_rank_below": 40.0,
        },
    }
    out = OUT / "report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({"state_dict": tuned.state_dict(), "config": vars(args)}, OUT / "model_hard_negative.pt")
    print(f"saved={out}", flush=True)
    print(f"saved_model={OUT / 'model_hard_negative.pt'}", flush=True)


if __name__ == "__main__":
    main()
