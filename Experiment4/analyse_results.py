"""
Analysis functions for Experiment 4 — Chunked Incremental Solve.
"""

import json
from collections import defaultdict
from pathlib import Path


def analyse_exp4(path: str) -> None:
    with open(path) as f:
        results = json.load(f)

    def accuracy(r):
        return sum(x["correct"] for x in r) / len(r) if r else 0.0

    print("=" * 60)
    print("EXPERIMENT 4 — Chunked Incremental Solve")
    print("=" * 60)
    print(f"Total problems   : {len(results)}")
    print(f"Overall accuracy : {accuracy(results):.4f}")
    print(f"Chunk size       : {results[0].get('chunk_size', '?')}")

    depth_groups: dict[int, list] = defaultdict(list)
    for r in results:
        depth_groups[r["depth"]].append(r)

    print(
        f"\n{'Depth':<8}{'N':<8}{'Acc':<10}"
        f"{'Corr/prob':<14}{'Unverif/prob':<14}{'ExtFail/prob'}"
    )
    print("-" * 70)
    for d in sorted(depth_groups.keys()):
        g    = depth_groups[d]
        acc  = accuracy(g)
        corr = sum(r["corrections_applied"]  for r in g) / len(g)
        unv  = sum(r["unverified_fallbacks"] for r in g) / len(g)
        ef   = sum(r["extraction_failures"]  for r in g) / len(g)
        print(
            f"{d:<8}{len(g):<8}{acc:<10.4f}"
            f"{corr:<14.2f}{unv:<14.2f}{ef:.2f}"
        )

    corrected = [r for r in results if r["corrections_applied"] > 0]
    if corrected:
        print(f"\nProblems with at least one correction : {len(corrected)}")
        print(f"Accuracy on those problems            : {accuracy(corrected):.4f}")
        print(
            f"Accuracy on uncorrected problems      : "
            f"{accuracy([r for r in results if r['corrections_applied'] == 0]):.4f}"
        )

    failed = [r for r in results if r["extraction_failures"] > 0]
    if failed:
        print(f"\nProblems with extraction failures     : {len(failed)}")
        print(f"Accuracy on those problems            : {accuracy(failed):.4f}")

    print(
        f"\nTotal expressions verified  : "
        f"{sum(r['total_expressions_checked'] for r in results)}"
    )
    print(
        f"Total corrections applied   : "
        f"{sum(r['corrections_applied'] for r in results)}"
    )
    print(
        f"Total unverified fallbacks  : "
        f"{sum(r['unverified_fallbacks'] for r in results)}"
    )
    print(
        f"Total extraction failures   : "
        f"{sum(r['extraction_failures'] for r in results)}"
    )


if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Analyse Experiment 4 results")
    p.add_argument(
        "path", nargs="?",
        default=str(Path(__file__).resolve().parent.parent / "results" / "exp4_chunked_qwen25_math_1.5b.json"),
        help="Path to exp4 results JSON",
    )
    args = p.parse_args()

    if not Path(args.path).exists():
        print(f"File not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    analyse_exp4(args.path)
