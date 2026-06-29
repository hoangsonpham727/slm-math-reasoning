"""
GSM8K Data Preparation

Downloads the GSM8K test split from HuggingFace, estimates step count for each
problem by counting <<...>> arithmetic markers in the gold answer, filters to
problems with >= MIN_STEPS steps, parses the numeric gold answer, and saves
the result to data/gsm8k_test.json.

Requires:
    pip install datasets

Usage:
    python prepare_gsm8k.py [--min_steps 3] [--output data/gsm8k_test.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Gold-answer parser (self-contained, no Experiment1 dependency)
# ---------------------------------------------------------------------------

_RE_GSM_FINAL = re.compile(r"####\s*([\-\d,\.]+)")


def parse_gsm_gold(answer_text: str) -> float | None:
    """
    Extract the numeric answer from a GSM8K gold answer string.
    GSM8K format: step-by-step reasoning ending with '#### <number>'.
    Returns None if no valid number is found.
    """
    matches = _RE_GSM_FINAL.findall(answer_text)
    if not matches:
        return None
    raw = matches[-1].replace(",", "").strip()
    try:
        return float(raw)
    except ValueError:
        return None


def count_steps(answer_text: str) -> int:
    """
    Estimate the number of arithmetic steps in a GSM8K answer by counting
    <<expression=result>> calculation markers.  These appear for every
    explicit calculation in the gold solution.
    """
    return len(re.findall(r"<<[^>]+>>", answer_text))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_MIN_STEPS = 6
DEFAULT_OUTPUT    = "data/gsm8k_test.json"


def parse_args():
    p = argparse.ArgumentParser(
        description="Download GSM8K test split, filter by step count, save JSON"
    )
    p.add_argument("--min_steps", type=int, default=DEFAULT_MIN_STEPS,
                   help=f"Minimum number of arithmetic steps (default: {DEFAULT_MIN_STEPS})")
    p.add_argument("--output",    type=str, default=DEFAULT_OUTPUT,
                   help=f"Output JSON file path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--no_filter", action="store_true",
                   help="Save all problems regardless of step count")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: 'datasets' library not found. Install it with:")
        print("    pip install datasets")
        sys.exit(1)

    print("Downloading GSM8K test split from HuggingFace...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"  Downloaded {len(ds)} problems.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    problems = []
    skipped_no_answer = 0
    skipped_few_steps = 0

    for i, item in enumerate(ds):
        question    = item["question"]
        answer_raw  = item["answer"]

        # Parse gold answer
        ground_truth = parse_gsm_gold(answer_raw)
        if ground_truth is None:
            skipped_no_answer += 1
            continue

        # Count arithmetic steps
        estimated_steps = count_steps(answer_raw)

        # Filter
        if not args.no_filter and estimated_steps < args.min_steps:
            skipped_few_steps += 1
            continue

        problems.append({
            "problem_id":      f"gsm8k_{i:04d}",
            "question":        question,
            "answer":          answer_raw,        # raw GSM8K field (steps + #### N)
            "ground_truth":    ground_truth,
            "estimated_steps": estimated_steps,
        })

    print(f"\nFiltering summary:")
    print(f"  Total downloaded   : {len(ds)}")
    print(f"  Skipped (no answer): {skipped_no_answer}")
    if not args.no_filter:
        print(f"  Skipped (<{args.min_steps} steps): {skipped_few_steps}")
    print(f"  Kept               : {len(problems)}")

    if problems:
        step_counts = [p["estimated_steps"] for p in problems]
        print(f"  Step count range   : {min(step_counts)} – {max(step_counts)}")
        print(f"  Median steps       : {sorted(step_counts)[len(step_counts)//2]}")

    with open(out_path, "w") as f:
        json.dump(problems, f, indent=2)

    print(f"\nSaved {len(problems)} problems to '{out_path}'")

    # Spot-check
    print("\n--- Spot-check (first 2 problems) ---")
    for p in problems[:2]:
        print(f"\n  ID      : {p['problem_id']}")
        print(f"  Steps   : {p['estimated_steps']}")
        print(f"  GT      : {p['ground_truth']}")
        print(f"  Question: {p['question'][:120]}...")


if __name__ == "__main__":
    main()
