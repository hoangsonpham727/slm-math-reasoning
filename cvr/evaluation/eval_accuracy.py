"""
General accuracy metrics for CVR results.

Works on any list of result records that have 'is_correct' and optional 'depth' fields.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional


def compute_accuracy(results: list[dict], group_by: Optional[str] = None) -> dict:
    """
    Compute accuracy overall and optionally grouped by a field (e.g. 'depth', 'model').

    Returns:
        overall_accuracy  float
        n_correct         int
        n_total           int
        by_group          dict  — only present if group_by is set
    """
    n_correct = sum(1 for r in results if r.get("is_correct"))
    n_total = len(results)
    overall = n_correct / n_total if n_total else 0.0

    out: dict = {
        "overall_accuracy": round(overall, 4),
        "n_correct": n_correct,
        "n_total": n_total,
    }

    if group_by:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in results:
            groups[r.get(group_by, "unknown")].append(r)
        by_group = {}
        for key, recs in sorted(groups.items(), key=lambda x: str(x[0])):
            nc = sum(1 for r in recs if r.get("is_correct"))
            nt = len(recs)
            by_group[key] = {
                "accuracy": round(nc / nt, 4) if nt else 0.0,
                "n_correct": nc,
                "n_total": nt,
            }
        out["by_group"] = by_group

    return out


def compare_methods(results_by_method: dict[str, list[dict]]) -> dict:
    """
    Compare accuracy across multiple methods on the same dataset.

    Args:
        results_by_method: {method_name: [result_record, ...]}

    Returns:
        dict with per-method accuracy summary
    """
    comparison = {}
    for method, results in results_by_method.items():
        comparison[method] = compute_accuracy(results)
    return comparison
