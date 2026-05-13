"""
Multi-step degradation metrics.

Measures how CVR accuracy, restart behaviour, and error location
vary as reasoning depth (number of required steps) increases.
"""

from __future__ import annotations

from collections import defaultdict


def compute_multistep_metrics(results: list[dict]) -> dict:
    """
    Compute multi-step metrics grouped by reasoning depth.

    Each result record should have:
        depth             int   — number of chained reasoning steps
        is_correct        bool
        is_collapse       bool  — True if no answer was extracted
        total_restarts    int   — restarts across all chains for this problem
        chains            list  — per-chain chain result dicts (optional, for step metrics)

    Returns:
        by_depth: dict mapping depth → metrics dict
        overall: aggregate metrics
    """
    by_depth: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_depth[r.get("depth", 0)].append(r)

    depth_metrics = {}
    for depth, recs in sorted(by_depth.items()):
        n = len(recs)
        n_correct = sum(1 for r in recs if r.get("is_correct"))
        n_collapse = sum(1 for r in recs if r.get("is_collapse"))
        avg_restarts = (
            sum(r.get("total_restarts", 0) for r in recs) / n if n else 0.0
        )

        # Restart success rate: among problems that had restarts, how many succeeded?
        had_restarts = [r for r in recs if r.get("total_restarts", 0) > 0]
        restart_success = (
            sum(1 for r in had_restarts if r.get("is_correct")) / len(had_restarts)
            if had_restarts else None
        )

        depth_metrics[depth] = {
            "accuracy": round(n_correct / n, 4) if n else 0.0,
            "n_correct": n_correct,
            "n_total": n,
            "collapse_rate": round(n_collapse / n, 4) if n else 0.0,
            "avg_restarts_per_problem": round(avg_restarts, 3),
            "restart_success_rate": round(restart_success, 4) if restart_success is not None else None,
            "n_problems_with_restarts": len(had_restarts),
        }

    # Overall
    all_correct = sum(1 for r in results if r.get("is_correct"))
    all_collapse = sum(1 for r in results if r.get("is_collapse"))
    n_all = len(results)

    overall = {
        "accuracy": round(all_correct / n_all, 4) if n_all else 0.0,
        "n_correct": all_correct,
        "n_total": n_all,
        "collapse_rate": round(all_collapse / n_all, 4) if n_all else 0.0,
        "avg_restarts_per_problem": round(
            sum(r.get("total_restarts", 0) for r in results) / n_all, 3
        ) if n_all else 0.0,
    }

    return {"by_depth": depth_metrics, "overall": overall}


def compute_depth_ceiling(depth_metrics: dict, baseline_key: int = 1, threshold: float = 0.30) -> int | None:
    """
    Find the first depth k where accuracy drops >threshold relative to depth 1 (or baseline_key).
    Returns None if accuracy never drops that much.
    """
    baseline_acc = depth_metrics.get(baseline_key, {}).get("accuracy")
    if baseline_acc is None or baseline_acc == 0:
        return None

    for depth in sorted(depth_metrics.keys()):
        if depth <= baseline_key:
            continue
        acc = depth_metrics[depth].get("accuracy", 0.0)
        relative_drop = (baseline_acc - acc) / baseline_acc
        if relative_drop > threshold:
            return depth
    return None
