"""
Experiment 3 Analysis

Reads JSONL result files from Experiment3/results/ and prints:
  - Overall accuracy per model
  - Accuracy split by filter success vs. filter failure (fallback to full problem)
  - Accuracy split by distractor type
  - Average number of irrelevant clauses filtered
    - Comparison bar chart against Experiment 1 accuracy

Usage:
    python analysis.py [--results_dir results]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def accuracy(records: list[dict]) -> float:
    if not records:
        return float("nan")
    return sum(r["is_correct"] for r in records) / len(records)


def analyse_model(records: list[dict], model_name: str) -> None:
    total = len(records)
    filter_model = records[0].get("filter_model", "unknown") if records else "unknown"
    print(f"\n{'─'*55}")
    print(f"  Solver: {model_name}   Filter: {filter_model}   (n={total})")
    print(f"{'─'*55}")

    print(f"  Overall accuracy          : {accuracy(records):.4f}")

    clean = [r for r in records if not r.get("filter_failed")]
    failed = [r for r in records if r.get("filter_failed")]
    if clean:
        print(f"  Filter succeeded (n={len(clean):>3})   : {accuracy(clean):.4f}")
    if failed:
        print(f"  Filter failed    (n={len(failed):>3})   : {accuracy(failed):.4f}  "
              f"(fell back to full problem)")

    fail_rate = len(failed) / total if total else 0.0
    print(f"  Filter fail rate          : {fail_rate:.4f}")

    avg_irr = sum(len(r.get("irrelevant_clauses", [])) for r in records) / total
    avg_rel = sum(len(r.get("relevant_clauses", [])) for r in records) / total
    print(f"  Avg relevant clauses kept : {avg_rel:.2f}")
    print(f"  Avg irrelevant filtered   : {avg_irr:.2f}")

    # Breakdown by distractor type
    by_type: dict[str, list] = defaultdict(list)
    for r in records:
        by_type[r.get("distractor_type", "UNKNOWN")].append(r)
    if len(by_type) > 1:
        print(f"\n  Accuracy by distractor type:")
        for dtype in sorted(by_type):
            grp = by_type[dtype]
            print(f"    {dtype:<20} n={len(grp):>3}   acc={accuracy(grp):.4f}")


DISTRACTOR_LABELS = {
    "TYPE_A": "Scope Confusion",
    "TYPE_B": "Wrong Formula",
    "TYPE_C": "Unit Trap",
    "TYPE_D": "Temporal Misdirection",
}


def _distractor_correctly_filtered(record: dict) -> bool:
    """
    Return True if the filter placed the distractor text inside irrelevant_clauses.
    Matching is done as a case-insensitive substring check so minor tokenisation
    differences between the injected distractor and the split clause do not cause
    false negatives.
    """
    distractor = str(record.get("distractor", "")).strip().lower()
    if not distractor:
        return False
    irrelevant = [str(c).strip().lower() for c in record.get("irrelevant_clauses", [])]
    return any(distractor in clause or clause in distractor for clause in irrelevant)


def print_filter_detection_table(records: list[dict]) -> None:
    """
    Print a table of filter detection rate grouped by distractor type.

    Detection rate = fraction of problems where the filter correctly placed
    the injected distractor into irrelevant_clauses.
    """
    by_type: dict[str, list] = defaultdict(list)
    for r in records:
        by_type[r.get("distractor_type", "UNKNOWN")].append(r)

    print(f"\n{'─'*60}")
    print("  Filter Detection Rate by Distractor Type")
    print(f"{'─'*60}")
    print(f"  {'Type':<10}  {'Description':<26}  {'n':>4}  {'Detected':>10}  {'Rate':>8}")
    print(f"  {'-'*10}  {'-'*26}  {'-'*4}  {'-'*10}  {'-'*8}")

    totals_n = totals_detected = 0
    for dtype in sorted(by_type):
        grp       = by_type[dtype]
        n         = len(grp)
        detected  = sum(_distractor_correctly_filtered(r) for r in grp)
        rate      = detected / n if n else float("nan")
        label     = DISTRACTOR_LABELS.get(dtype, dtype)
        print(f"  {dtype:<10}  {label:<26}  {n:>4}  {detected:>10}  {rate:>7.1%}")
        totals_n         += n
        totals_detected  += detected

    overall = totals_detected / totals_n if totals_n else float("nan")
    print(f"  {'-'*10}  {'-'*26}  {'-'*4}  {'-'*10}  {'-'*8}")
    print(f"  {'OVERALL':<10}  {'':26}  {totals_n:>4}  {totals_detected:>10}  {overall:>7.1%}")
    print(f"{'─'*60}\n")


def load_experiment1_accuracies(input_path: Path) -> dict[str, float]:
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    enhanced = payload.get("enhanced", {})
    if not enhanced:
        raise ValueError("Experiment 1 JSON must contain an 'enhanced' result block.")

    return {model_name: float(metrics["accuracy"]) for model_name, metrics in enhanced.items()}


def plot_experiment_comparison(
    exp1_accuracies: dict[str, float],
    exp3_accuracies: dict[str, float],
    out_path: Path,
) -> None:
    models = sorted(set(exp1_accuracies) & set(exp3_accuracies))
    if not models:
        raise ValueError("No overlapping model keys found between Experiment 1 and Experiment 3.")

    exp1_values = np.array([exp1_accuracies[model] for model in models], dtype=float)
    exp3_values = np.array([exp3_accuracies[model] for model in models], dtype=float)

    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_exp1 = ax.bar(x - width / 2, exp1_values, width=width, label="Baselines", color="#4C72B0")
    bars_exp3 = ax.bar(x + width / 2, exp3_values, width=width, label="Our Method", color="#DD8452")

    for bar in list(bars_exp1) + list(bars_exp3):
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Experiment 3 Analysis")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument(
        "--experiment1_json",
        type=str,
        default="../Experiment1/inference_results.json",
        help="Path to Experiment 1 inference results JSON.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="figures",
        help="Directory to save comparison figures.",
    )
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(results_dir.glob("*_distractor_filter.jsonl"))

    if not files:
        print(f"No result files found in {results_dir}/ matching *_distractor_filter.jsonl")
        return

    print(f"\n{'='*55}")
    print(f"  EXPERIMENT 3 — Distractor Filtering")
    print(f"{'='*55}")

    all_records = []
    exp3_accuracies: dict[str, float] = {}

    for fpath in files:
        records = load_jsonl(fpath)
        model_name = fpath.stem.replace("_distractor_filter", "")
        analyse_model(records, model_name)
        all_records.extend(records)
        exp3_accuracies[model_name] = accuracy(records)

    if len(files) > 1:
        print(f"\n{'─'*55}")
        print(f"  All models combined  (n={len(all_records)})")
        print(f"  Overall accuracy: {accuracy(all_records):.4f}")

    # Filter detection rate grouped by distractor type (uses combined records
    # so each problem is counted once regardless of how many solver models ran)
    first_model_records = load_jsonl(files[0])
    print_filter_detection_table(first_model_records)

    exp1_path = Path(args.experiment1_json).resolve()
    exp1_accuracies = load_experiment1_accuracies(exp1_path)
    comparison_out = output_dir / "exp1_vs_exp3_accuracy.png"
    plot_experiment_comparison(exp1_accuracies, exp3_accuracies, comparison_out)
    print(f"Saved comparison bar chart: {comparison_out}")

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()
