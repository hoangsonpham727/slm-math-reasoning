"""
Experiment 4 Analysis

Reads JSONL result files from Experiment4/results/ and prints:
  - Overall accuracy per model
  - Accuracy by reasoning depth (1–8)
  - Execution error rate by depth
  - Valid execution rate (successful code runs / k)

Usage:
    python analysis.py [--results_dir results]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


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
    k = records[0]["k_samples"] if records else 1

    print(f"\n{'─'*60}")
    print(f"  Model: {model_name}   (n={total}, k={k})")
    print(f"{'─'*60}")
    print(f"  Overall accuracy: {accuracy(records):.4f}")

    # Collapse rate (no valid executions at all)
    collapses = sum(1 for r in records if r["valid_executions"] == 0)
    print(f"  Full exec failure (all k failed): "
          f"{collapses}/{total} = {collapses/total:.4f}")

    total_samples = sum(r["k_samples"] for r in records)
    total_errors  = sum(r["execution_errors"] for r in records)
    total_valid   = sum(r["valid_executions"] for r in records)
    print(f"  Exec error rate (across all samples): "
          f"{total_errors}/{total_samples} = {total_errors/total_samples:.4f}")
    print(f"  Valid exec rate: "
          f"{total_valid}/{total_samples} = {total_valid/total_samples:.4f}")

    # Per-depth breakdown
    depth_groups: dict[int, list] = defaultdict(list)
    for r in records:
        depth_groups[r["depth"]].append(r)

    print(f"\n  {'Depth':<8} {'N':<6} {'Acc':<10} {'Valid%':<12} {'ErrRate'}")
    print(f"  {'-'*50}")
    for depth in sorted(depth_groups):
        grp = depth_groups[depth]
        acc = accuracy(grp)
        grp_samples = sum(r["k_samples"] for r in grp)
        grp_valid   = sum(r["valid_executions"] for r in grp)
        grp_errors  = sum(r["execution_errors"] for r in grp)
        valid_pct   = grp_valid / grp_samples if grp_samples else 0.0
        err_rate    = grp_errors / grp_samples if grp_samples else 0.0
        print(f"  {depth:<8} {len(grp):<6} {acc:<10.4f} {valid_pct:<12.4f} {err_rate:.4f}")


def main():
    p = argparse.ArgumentParser(description="Experiment 4 Analysis")
    p.add_argument("--results_dir", type=str, default="results")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    files = sorted(results_dir.glob("*_pot_k*.jsonl"))

    if not files:
        print(f"No result files found in {results_dir}/ matching *_pot_k*.jsonl")
        return

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 4 — PoT + Execution Verification")
    print(f"{'='*60}")

    for fpath in files:
        records = load_jsonl(fpath)
        model_name = fpath.stem
        analyse_model(records, model_name)

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
