"""
Generate Exp2-style depth-controlled training problems with non-overlapping seeds.

Experiment 2's test set uses problem_idx 0..199 for each depth (seed = depth*10000 + idx).
This script generates problem_idx 200..399 — entirely different seeds, no overlap.

Output: RL_SLM/data/train_exp2/problems_depth{01..08}.json  (200 problems each, 1600 total)
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Experiment2"))
from dataset_generator import build_problem

OUTPUT_DIR = REPO_ROOT / "RL_SLM" / "data" / "train_exp2"
DEPTHS = list(range(1, 9))
N_PER_DEPTH = 200
IDX_OFFSET = 200   # test set uses 0..199; training uses 200..399


def generate():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for depth in DEPTHS:
        problems = []
        for i in range(N_PER_DEPTH):
            idx = IDX_OFFSET + i
            p = build_problem(depth, idx)
            # Relabel so IDs are clearly from the training split
            p.problem_id = f"train_depth{depth:02d}_prob{i:04d}"
            assert p.ground_truth > 0, f"Non-positive GT at {p.problem_id}"
            problems.append(p)

        out_path = OUTPUT_DIR / f"problems_depth{depth:02d}.json"
        with open(out_path, "w") as f:
            json.dump([asdict(p) for p in problems], f, indent=2)

        gt_vals = [p.ground_truth for p in problems]
        print(f"  Depth {depth}: {len(problems)} problems. "
              f"GT range [{min(gt_vals):.2f}, {max(gt_vals):.2f}]")
        total += len(problems)

    print(f"\nTraining dataset saved to '{OUTPUT_DIR}'. Total: {total} problems.")


if __name__ == "__main__":
    generate()
