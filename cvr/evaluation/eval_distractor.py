"""
Distractor robustness metrics.

Compares CVR performance on clean vs. distractor-injected problems.
Key metric: accuracy drop = clean_accuracy - distractor_accuracy.
Also tracks how often the relevance check triggered.
"""

from __future__ import annotations


def compute_distractor_metrics(
    clean_results: list[dict],
    distractor_results: list[dict],
) -> dict:
    """
    Compute distractor robustness metrics.

    Args:
        clean_results: results on original (no distractor) problems
        distractor_results: results on distractor-injected problems

    Each result record should have: is_correct (bool), and optionally
    relevance_triggered (bool) if relevance check was logged.

    Returns:
        clean_accuracy          float
        distractor_accuracy     float
        accuracy_drop           float  — clean - distractor (positive = hurt by distractors)
        accuracy_drop_pct       float  — relative drop as % of clean accuracy
        relevance_trigger_rate  float  — fraction of distractor problems where relevance check fired
        n_clean                 int
        n_distractor            int
    """
    def _acc(records: list[dict]) -> float:
        if not records:
            return 0.0
        return sum(1 for r in records if r.get("is_correct")) / len(records)

    clean_acc = _acc(clean_results)
    dist_acc = _acc(distractor_results)
    drop = clean_acc - dist_acc
    drop_pct = (drop / clean_acc * 100) if clean_acc > 0 else 0.0

    # Relevance trigger rate: fraction where any chain had a relevance failure
    triggered = sum(
        1 for r in distractor_results
        if r.get("relevance_triggered") or r.get("relevance_failures", 0) > 0
    )
    trigger_rate = triggered / len(distractor_results) if distractor_results else 0.0

    return {
        "clean_accuracy": round(clean_acc, 4),
        "distractor_accuracy": round(dist_acc, 4),
        "accuracy_drop": round(drop, 4),
        "accuracy_drop_pct": round(drop_pct, 2),
        "relevance_trigger_rate": round(trigger_rate, 4),
        "n_clean": len(clean_results),
        "n_distractor": len(distractor_results),
    }


def compute_distractor_metrics_by_type(
    distractor_results: list[dict],
) -> dict:
    """
    Break down accuracy and trigger rate by distractor type (TYPE_A/B/C/D).
    """
    from collections import defaultdict
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in distractor_results:
        dtype = r.get("distractor_type", "UNKNOWN")
        by_type[dtype].append(r)

    out = {}
    for dtype, records in sorted(by_type.items()):
        n_correct = sum(1 for r in records if r.get("is_correct"))
        n_triggered = sum(
            1 for r in records
            if r.get("relevance_triggered") or r.get("relevance_failures", 0) > 0
        )
        out[dtype] = {
            "accuracy": round(n_correct / len(records), 4) if records else 0.0,
            "n_total": len(records),
            "relevance_trigger_rate": round(n_triggered / len(records), 4) if records else 0.0,
        }
    return out
