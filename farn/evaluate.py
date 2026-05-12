"""
evaluate.py — Evaluation metrics and comparison tables for FARN.

This module provides three evaluation levels:

  Level 1 — Per-problem metrics:
      Was the final answer correct? How many steps did it take?
      What actions did the navigator select?

  Level 2 — Aggregate metrics across a result file:
      Accuracy, average steps, action distribution,
      per-model breakdown.

  Level 3 — Comparison tables:
      Given result files from multiple conditions (direct baseline,
      CoT baseline, FARN-PRM, FARN-CCQA), produce a formatted
      comparison table ready for pasting into a paper.

Usage:
    # Compare all results in a directory
    python evaluate.py --results_dir results/ --output_dir tables/

    # Evaluate a single result file
    python evaluate.py --result_file results/inference_qwen25_math_1.5b.json
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Metric Helpers ───────────────────────────────────────────────────

def accuracy(results: List[Dict]) -> float:
    """Fraction of problems answered correctly."""
    if not results:
        return 0.0
    return np.mean([r["correct"] for r in results])


def mean_steps(results: List[Dict]) -> float:
    """Average number of reasoning steps taken."""
    if not results:
        return 0.0
    return np.mean([r["num_steps"] for r in results])


def action_distribution(results: List[Dict]) -> Dict[str, float]:
    """Fraction of actions from each logic block (summing to 1.0)."""
    counts = defaultdict(int)
    total = 0
    for r in results:
        for a in r.get("actions", []):
            counts[a] += 1
            total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in sorted(counts.items())}


def accuracy_by_step_count(results: List[Dict]) -> Dict[int, float]:
    """Accuracy broken down by the number of steps used.

    Useful for understanding whether the navigator is terminating
    too early (truncated chains) or too late (over-spending on easy problems).
    """
    by_steps = defaultdict(list)
    for r in results:
        by_steps[r["num_steps"]].append(r["correct"])
    return {k: float(np.mean(v)) for k, v in sorted(by_steps.items())}


def accuracy_on_hard_vs_easy(
    results: List[Dict],
    hard_ids: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Accuracy split by problem difficulty.

    If hard_ids is provided, uses those. Otherwise, uses step count
    as a proxy (problems requiring >= 5 steps are considered "hard").
    """
    if hard_ids is not None:
        hard = [r for r in results if r.get("index") in hard_ids]
        easy = [r for r in results if r.get("index") not in hard_ids]
    else:
        hard = [r for r in results if r["num_steps"] >= 5]
        easy = [r for r in results if r["num_steps"] < 5]

    return {
        "hard": accuracy(hard),
        "easy": accuracy(easy),
        "n_hard": len(hard),
        "n_easy": len(easy),
    }


def compute_metrics(result_data: Dict) -> Dict:
    """Compute all evaluation metrics for a single result file.

    Args:
        result_data: Dict as returned by inference.py (loaded from JSON).

    Returns:
        Dict with all computed metrics.
    """
    problems = result_data.get("problems", [])
    if not problems:
        return {"error": "No problems found in result data"}

    metrics = {
        "model": result_data.get("model", "unknown"),
        "condition": _infer_condition(result_data),
        "n": len(problems),
        "accuracy": accuracy(problems),
        "avg_steps": mean_steps(problems),
        "action_dist": action_distribution(problems),
        "acc_by_steps": accuracy_by_step_count(problems),
        "acc_hard_easy": accuracy_on_hard_vs_easy(problems),
    }

    # Feature-level metrics (if feature traces are present)
    if problems[0].get("feature_trace"):
        metrics["feature_means"] = _average_feature_traces(problems)

    return metrics


def _infer_condition(result_data: Dict) -> str:
    """Infer the condition name from a result dict."""
    if "mode" in result_data:
        return f"baseline_{result_data['mode']}"
    if "navigator_path" in result_data:
        path = result_data["navigator_path"]
        if "prm" in path.lower():
            return "farn_prm"
        if "ccqa" in path.lower():
            return "farn_ccqa"
        return "farn"
    return "unknown"


def _average_feature_traces(problems: List[Dict]) -> Dict[str, float]:
    """Average each feature across all steps and all problems.

    Returns a dict mapping feature name → mean value.
    """
    feature_sums = defaultdict(float)
    feature_counts = defaultdict(int)

    for p in problems:
        for step_features in p.get("feature_trace", []):
            for fname, fval in step_features.items():
                feature_sums[fname] += fval
                feature_counts[fname] += 1

    return {
        k: feature_sums[k] / feature_counts[k]
        for k in feature_sums
        if feature_counts[k] > 0
    }


# ── Comparison Tables ─────────────────────────────────────────────────

def build_comparison_table(
    result_files: List[str],
) -> Tuple[List[str], List[List]]:
    """Build a comparison table from multiple result files.

    The rows are models, the columns are conditions (direct, CoT, FARN-PRM,
    FARN-CCQA). Each cell shows accuracy.

    Args:
        result_files: Paths to JSON result files from inference.py.

    Returns:
        (headers, rows) — headers is a list of column names, rows is a
        list of lists (one row per model).
    """
    # Load all results
    all_metrics = []
    for fpath in result_files:
        with open(fpath) as f:
            data = json.load(f)
        metrics = compute_metrics(data)
        all_metrics.append(metrics)

    # Collect all models and conditions
    models = sorted(set(m["model"] for m in all_metrics))
    conditions = ["baseline_direct", "baseline_cot", "farn_prm", "farn_ccqa", "farn"]

    # Resolve which conditions are actually present
    present_conditions = [
        c for c in conditions
        if any(m["condition"] == c for m in all_metrics)
    ]

    # Build lookup: (model, condition) -> accuracy
    lookup = {}
    for m in all_metrics:
        lookup[(m["model"], m["condition"])] = m["accuracy"]

    # Pretty column headers
    condition_labels = {
        "baseline_direct": "Direct",
        "baseline_cot": "CoT",
        "farn_prm": "FARN-PRM",
        "farn_ccqa": "FARN-CCQA",
        "farn": "FARN",
    }
    headers = ["Model"] + [condition_labels.get(c, c) for c in present_conditions]

    # Build rows
    rows = []
    for model in models:
        row = [model]
        for cond in present_conditions:
            acc = lookup.get((model, cond))
            row.append(f"{acc:.1%}" if acc is not None else "—")
        rows.append(row)

    return headers, rows


def format_table_plain(headers: List[str], rows: List[List]) -> str:
    """Format comparison table as plain text."""
    col_widths = [max(len(str(headers[i])), max(len(str(r[i])) for r in rows))
                  for i in range(len(headers))]

    lines = []
    header_line = "  ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))

    for row in rows:
        lines.append("  ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))

    return "\n".join(lines)


def format_table_latex(headers: List[str], rows: List[List]) -> str:
    """Format comparison table as a LaTeX tabular block.

    Produces a table ready to paste into a paper. The best result
    in each row (excluding the model name column) is bolded.
    """
    n_cols = len(headers)
    col_spec = "l" + "c" * (n_cols - 1)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Accuracy comparison across baselines and FARN variants}",
        r"\label{tab:main_results}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]

    for row in rows:
        # Find the best accuracy in this row (columns 1 onwards)
        numeric_vals = []
        for v in row[1:]:
            try:
                numeric_vals.append(float(v.strip("%")) / 100)
            except (ValueError, AttributeError):
                numeric_vals.append(-1.0)

        best_val = max(numeric_vals) if numeric_vals else -1.0

        # Format cells, bolding the best
        cells = [row[0]]
        for i, v in enumerate(row[1:]):
            try:
                val = float(v.strip("%")) / 100
                if abs(val - best_val) < 1e-9 and best_val >= 0:
                    cells.append(r"\textbf{" + v + "}")
                else:
                    cells.append(v)
            except (ValueError, AttributeError):
                cells.append(str(v))

        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── Per-Model Summary ─────────────────────────────────────────────────

def print_summary(metrics: Dict) -> None:
    """Print a human-readable summary of one result's metrics."""
    print(f"\n{'=' * 55}")
    print(f"  Model:     {metrics['model']}")
    print(f"  Condition: {metrics['condition']}")
    print(f"  N:         {metrics['n']}")
    print(f"{'=' * 55}")
    print(f"  Accuracy:      {metrics['accuracy']:.1%}")
    print(f"  Avg steps:     {metrics['avg_steps']:.2f}")

    if metrics.get("acc_hard_easy"):
        he = metrics["acc_hard_easy"]
        print(f"  Acc (hard):    {he['hard']:.1%}  (n={he['n_hard']})")
        print(f"  Acc (easy):    {he['easy']:.1%}  (n={he['n_easy']})")

    if metrics.get("action_dist"):
        print("\n  Action distribution:")
        for action, frac in metrics["action_dist"].items():
            bar = "█" * int(frac * 30)
            print(f"    {action:<12} {frac:.1%}  {bar}")

    if metrics.get("acc_by_steps"):
        print("\n  Accuracy by step count:")
        for steps, acc in metrics["acc_by_steps"].items():
            print(f"    {steps} step(s): {acc:.1%}")

    if metrics.get("feature_means"):
        print("\n  Mean feature values across all steps:")
        for fname, fmean in sorted(metrics["feature_means"].items()):
            bar = "█" * int(fmean * 20)
            print(f"    {fname:<30} {fmean:.3f}  {bar}")


# ── CLI Entry Point ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FARN results and build comparison tables"
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default=None,
        help="Single result JSON file to evaluate",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory containing multiple result JSON files for comparison",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tables",
        help="Directory to save comparison tables",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Also output LaTeX table format",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Single file mode ──
    if args.result_file:
        with open(args.result_file) as f:
            data = json.load(f)
        metrics = compute_metrics(data)
        print_summary(metrics)
        return

    # ── Directory mode: build comparison table ──
    if args.results_dir:
        result_files = [
            os.path.join(args.results_dir, f)
            for f in os.listdir(args.results_dir)
            if f.endswith(".json")
        ]

        if not result_files:
            print(f"No JSON files found in {args.results_dir}")
            return

        print(f"Found {len(result_files)} result files.")

        # Print individual summaries
        for fpath in sorted(result_files):
            with open(fpath) as f:
                data = json.load(f)
            metrics = compute_metrics(data)
            print_summary(metrics)

        # Build comparison table
        headers, rows = build_comparison_table(result_files)

        print("\n\n" + "=" * 55)
        print("COMPARISON TABLE")
        print("=" * 55)
        plain = format_table_plain(headers, rows)
        print(plain)

        # Save plain text
        plain_path = os.path.join(args.output_dir, "comparison_table.txt")
        with open(plain_path, "w") as f:
            f.write(plain)
        print(f"\nPlain table saved to: {plain_path}")

        # Save LaTeX
        if args.latex:
            latex = format_table_latex(headers, rows)
            latex_path = os.path.join(args.output_dir, "comparison_table.tex")
            with open(latex_path, "w") as f:
                f.write(latex)
            print(f"LaTeX table saved to: {latex_path}")
            print("\nLaTeX output:")
            print(latex)

        return

    print("Provide --result_file or --results_dir")


if __name__ == "__main__":
    main()
