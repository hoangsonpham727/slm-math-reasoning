"""
Experiment 3 Analysis

Reads JSONL result files from Experiment3/results/ and prints:
  - Overall accuracy per model
  - Accuracy split by filter success vs. filter failure (fallback to full problem)
  - Accuracy split by distractor type
  - Average number of irrelevant clauses filtered

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
    print(f"\n{'─'*55}")
    print(f"  Model: {model_name}   (n={total})")
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


def main():
    p = argparse.ArgumentParser(description="Experiment 3 Analysis")
    p.add_argument("--results_dir", type=str, default="results")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    files = sorted(results_dir.glob("*_distractor_filter.jsonl"))

    if not files:
        print(f"No result files found in {results_dir}/ matching *_distractor_filter.jsonl")
        return

    print(f"\n{'='*55}")
    print(f"  EXPERIMENT 3 — Distractor Filtering")
    print(f"{'='*55}")

    all_records = []
    for fpath in files:
        records = load_jsonl(fpath)
        model_name = fpath.stem.replace("_distractor_filter", "")
        analyse_model(records, model_name)
        all_records.extend(records)

    if len(files) > 1:
        print(f"\n{'─'*55}")
        print(f"  All models combined  (n={len(all_records)})")
        print(f"  Overall accuracy: {accuracy(all_records):.4f}")

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()
