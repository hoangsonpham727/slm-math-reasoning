"""
Multi-seed Orchestrator for Experiment 2 + CISV

Generates 3 dataset variants with different seeds, runs all inference
(Direct + CoT for Experiment 2, CISV for Experiment 4) for each seed,
then prints a summary.

Seed strategy
─────────────
  seed=None  → legacy formula seed_val = depth*10000 + idx
               (reproduces the existing data/ directory exactly)
  seed=int   → new formula seed_val = seed*100000 + depth*10000 + idx
               (guaranteed distinct problem sets for each seed value)

The three chosen seeds [42, 123, 456] are arbitrary but fixed so results
are reproducible across runs.

Usage
─────
  # Generate datasets + run Exp2 + run CISV for all seeds
  python run_multi_seed.py --device cuda:0

  # Only Exp2 (no CISV), custom seeds
  python run_multi_seed.py --seeds 10,20,30 --no_cisv

  # Single seed (useful for debugging)
  python run_multi_seed.py --seeds 42
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent         # Experiment2/
ROOT = HERE.parent                             # repo root

for _p in [str(ROOT), str(HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Imports ──────────────────────────────────────────────────────────────────

from dataset_generator import generate_dataset
from run_experiment2 import load_problems, run_model
from models import get_all_configs

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_SEEDS   = [42, 123, 456]
DEFAULT_DEPTHS  = list(range(1, 9))
DEFAULT_REGIMES = ["direct", "cot"]
N_PER_DEPTH     = 200


# ── Helpers ──────────────────────────────────────────────────────────────────

def _banner(msg: str, char: str = "=", width: int = 65):
    print(f"\n{char * width}")
    print(f"  {msg}")
    print(f"{char * width}")


def run_cisv_for_seed(seed: int, device: str):
    """Import and run CISV pipeline for one seed variant."""
    cisv_dir = ROOT / "CISV"
    if str(cisv_dir) not in sys.path:
        sys.path.insert(0, str(cisv_dir))

    # Import here to avoid issues if CISV deps aren't installed during dry-run
    from chunked_solve import run_all_models  # type: ignore

    _banner(f"CISV — seed={seed}", char="─")
    run_all_models(
        device=device,
        seed=seed,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-seed orchestrator: generate datasets + run all inference"
    )
    p.add_argument(
        "--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS),
        help=f"Comma-separated seed values (default: {DEFAULT_SEEDS})",
    )
    p.add_argument("--models",  type=str, default="all",
                   help="Comma-separated model short_names or 'all'")
    p.add_argument("--device",  type=str, default="cuda:0")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--no_cisv", action="store_true",
                   help="Skip CISV runs (only Experiment 2 Direct + CoT)")
    p.add_argument("--no_exp2", action="store_true",
                   help="Skip Experiment 2 runs (only CISV)")
    p.add_argument("--resume",  action="store_true", default=True,
                   help="Resume interrupted runs (default: True)")
    p.add_argument("--no_resume", action="store_true",
                   help="Disable resume; re-run everything from scratch")
    return p.parse_args()


def main():
    args   = parse_args()
    seeds  = [int(s.strip()) for s in args.seeds.split(",")]
    resume = not args.no_resume

    all_configs = get_all_configs()
    if args.models == "all":
        configs_to_run = all_configs
    else:
        wanted = set(args.models.split(","))
        configs_to_run = [c for c in all_configs if c.short_name in wanted]

    _banner(f"Multi-seed pipeline  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    print(f"  Seeds   : {seeds}")
    print(f"  Models  : {[c.short_name for c in configs_to_run]}")
    print(f"  Run Exp2: {not args.no_exp2}")
    print(f"  Run CISV: {not args.no_cisv}")
    print(f"  Resume  : {resume}")

    for seed in seeds:
        _banner(f"SEED = {seed}")

        # ── Step 1: Generate dataset ─────────────────────────────────────────
        data_dir   = HERE / "data"   / f"seed_{seed}"
        output_dir = HERE / "results" / f"seed_{seed}"

        _banner(f"Generating dataset — seed={seed}", char="─")
        generate_dataset(
            depths=DEFAULT_DEPTHS,
            n_per_depth=N_PER_DEPTH,
            output_dir=str(data_dir),
            seed=seed,
        )

        # ── Step 2: Run Experiment 2 (Direct + CoT) ──────────────────────────
        if not args.no_exp2:
            _banner(f"Experiment 2 inference — seed={seed}", char="─")
            dataset = load_problems(str(data_dir), DEFAULT_DEPTHS, N_PER_DEPTH)

            for cfg in configs_to_run:
                print(f"\n  ── Model: {cfg.short_name} ──")
                run_model(
                    config=cfg,
                    dataset=dataset,
                    regimes=DEFAULT_REGIMES,
                    output_dir=str(output_dir),
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                    resume=resume,
                )

        # ── Step 3: Run CISV ─────────────────────────────────────────────────
        if not args.no_cisv:
            run_cisv_for_seed(seed, args.device)

    _banner(f"All seeds complete  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    print("  Results written to:")
    for seed in seeds:
        print(f"    Experiment2/results/seed_{seed}/")
        print(f"    CISV/results/seed_{seed}/")


if __name__ == "__main__":
    main()
