"""
Multi-seed Orchestrator — CoT + CISV on Synthetic Datasets

Generates 3 synthetic dataset variants with different seeds, then runs:
  1. CoT inference  (Experiment 2 pipeline) on each seed's dataset
  2. CISV inference (this CISV pipeline)    on each seed's dataset

Only the three core SLMs are evaluated:
  - Qwen2.5-Math-1.5B-Instruct   (qwen25_math_1.5b)
  - Gemma 4 E2B                  (gemma4_e2b)
  - Phi-4-mini                   (phi4_mini)

Seed strategy
─────────────
  seed=None  →  legacy formula: seed_val = depth*10000 + idx
                (reproduces existing Experiment2/data/ exactly — not run here)
  seed=int   →  new formula:    seed_val = seed*100000 + depth*10000 + idx
                (guaranteed distinct problem sets for each seed value)

Output layout
─────────────
  Experiment2/data/seed_{seed}/          — generated datasets
  Experiment2/results/seed_{seed}/       — CoT (and direct) JSONL results
  CISV/results/seed_{seed}/              — CISV JSON results

Usage
─────
  # Full run: generate datasets + CoT + CISV for seeds [42, 123, 456]
  python CISV/run_multi_seed.py --device cuda:0

  # CoT only (skip CISV)
  python CISV/run_multi_seed.py --no_cisv --device cuda:0

  # CISV only (datasets + CoT already done)
  python CISV/run_multi_seed.py --no_cot --device cuda:0

  # Custom seeds
  python CISV/run_multi_seed.py --seeds 10,20,30 --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent          # CISV/
ROOT = HERE.parent                              # repo root
EXP2 = ROOT / "Experiment2"

for _p in [str(ROOT), str(HERE), str(EXP2)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Imports ───────────────────────────────────────────────────────────────────

from dataset_generator import generate_dataset                    # type: ignore
from run_experiment2   import load_problems, run_model            # type: ignore
from models            import get_all_configs                     # type: ignore
from chunked_solve     import run_exp4, _MODEL_CHUNK_SIZE         # type: ignore
from llm_wrapper       import init_model                          # type: ignore
import llm_wrapper as _lw                                         # type: ignore

# ── Experiment constants ──────────────────────────────────────────────────────

# Exactly three SLMs used in the paper — qwen25_7b is excluded.
EXPERIMENT_MODELS = ["qwen25_math_1.5b", "gemma4_e2b", "phi4_mini"]

DEFAULT_SEEDS   = [42, 123, 456]
DEFAULT_DEPTHS  = list(range(1, 9))
COT_REGIMES     = ["cot"]     # run CoT only on multi-seed (direct already in Exp2)
N_PER_DEPTH     = 200

# Field mappings for run_exp4 (match the synthetic dataset's JSON keys)
_FIELD_DEFAULTS = dict(
    problem_field = "question",
    answer_field  = "ground_truth",
    depth_field   = "depth",
    steps_field   = None,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(msg: str, char: str = "=", width: int = 65):
    print(f"\n{char * width}")
    print(f"  {msg}")
    print(f"{char * width}")


def _get_exp_configs(model_names: list[str]):
    """Return ModelConfig objects for only the experiment models."""
    all_cfg = {c.short_name: c for c in get_all_configs()}
    missing = [m for m in model_names if m not in all_cfg]
    if missing:
        raise ValueError(f"Unknown model short_names: {missing}")
    return [all_cfg[m] for m in model_names]


# ── Per-seed steps ────────────────────────────────────────────────────────────

def step_generate_dataset(seed: int, data_dir: Path):
    """Generate 1600 synthetic problems (8 depths × 200) for one seed."""
    _banner(f"Generating dataset  seed={seed}", char="─")
    generate_dataset(
        depths=DEFAULT_DEPTHS,
        n_per_depth=N_PER_DEPTH,
        output_dir=str(data_dir),
        seed=seed,
    )


def step_run_cot(seed: int, data_dir: Path, cot_dir: Path,
                 configs, device: str, max_new_tokens: int, resume: bool):
    """Run CoT inference for all three models on one seed's dataset."""
    _banner(f"CoT inference  seed={seed}", char="─")
    dataset = load_problems(str(data_dir), DEFAULT_DEPTHS, N_PER_DEPTH)
    print(f"  Loaded {sum(len(v) for v in dataset.values())} problems.\n")

    for cfg in configs:
        print(f"\n  ── {cfg.short_name} ──")
        run_model(
            config=cfg,
            dataset=dataset,
            regimes=COT_REGIMES,
            output_dir=str(cot_dir),
            device=device,
            max_new_tokens=max_new_tokens,
            resume=resume,
        )


def step_run_cisv(seed: int, data_dir: Path, cisv_dir: Path,
                  model_names: list[str], device: str):
    """Run CISV for all three models on one seed's dataset."""
    _banner(f"CISV inference  seed={seed}", char="─")
    cisv_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = str(data_dir / "problems_all.json")

    for short_name in model_names:
        output_path = str(cisv_dir / f"exp4_chunked_{short_name}.json")

        print(f"\n  ── {short_name}  —  {datetime.now():%H:%M:%S} ──")
        init_model(short_name, device=device)

        chunk_size = _MODEL_CHUNK_SIZE.get(short_name, 2)
        print(f"  chunk_size: {chunk_size}")

        run_exp4(
            dataset_path=dataset_path,
            output_path=output_path,
            llm_fn=_lw.llm,
            chunk_size=chunk_size,
            **_FIELD_DEFAULTS,
        )

        _lw._wrapper.unload()
        print(f"  [{short_name}] done — {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-seed orchestrator: generate datasets + CoT + CISV"
    )
    p.add_argument(
        "--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS),
        help=f"Comma-separated seed values (default: {DEFAULT_SEEDS})",
    )
    p.add_argument("--device",         type=str, default="cuda:0")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--no_cot",  action="store_true",
                   help="Skip CoT inference (datasets still generated)")
    p.add_argument("--no_cisv", action="store_true",
                   help="Skip CISV inference")
    p.add_argument("--no_generate", action="store_true",
                   help="Skip dataset generation (assumes data already exists)")
    p.add_argument("--no_resume", action="store_true",
                   help="Disable resume; re-run everything from scratch")
    return p.parse_args()


def main():
    os.environ.setdefault("HF_HOME",               "/mnt/data/hf")
    os.environ.setdefault("TRANSFORMERS_CACHE",     "/mnt/data/hf/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE",  "/mnt/data/hf/hub")
    os.environ.setdefault("TMPDIR",                 "/mnt/data/tmp")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES",   "0")

    args   = parse_args()
    seeds  = [int(s.strip()) for s in args.seeds.split(",")]
    resume = not args.no_resume
    configs = _get_exp_configs(EXPERIMENT_MODELS)

    _banner(f"Multi-seed pipeline  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    print(f"  Seeds        : {seeds}")
    print(f"  Models       : {EXPERIMENT_MODELS}")
    print(f"  Generate data: {not args.no_generate}")
    print(f"  Run CoT      : {not args.no_cot}")
    print(f"  Run CISV     : {not args.no_cisv}")
    print(f"  Resume       : {resume}")
    print(f"  Device       : {args.device}")

    for seed in seeds:
        _banner(f"SEED = {seed}")

        data_dir  = EXP2  / "data"    / f"seed_{seed}"
        cot_dir   = EXP2  / "results" / f"seed_{seed}"
        cisv_dir  = HERE  / "results" / f"seed_{seed}"

        # Step 1 — Generate dataset
        if not args.no_generate:
            step_generate_dataset(seed, data_dir)
        else:
            if not (data_dir / "problems_all.json").exists():
                raise FileNotFoundError(
                    f"Dataset not found: {data_dir / 'problems_all.json'}\n"
                    "Run without --no_generate to create it first."
                )
            print(f"  [skip generate] using existing data in {data_dir}")

        # Step 2 — CoT inference
        if not args.no_cot:
            step_run_cot(seed, data_dir, cot_dir, configs,
                         args.device, args.max_new_tokens, resume)
        else:
            print(f"  [skip CoT]")

        # Step 3 — CISV inference
        if not args.no_cisv:
            step_run_cisv(seed, data_dir, cisv_dir, EXPERIMENT_MODELS, args.device)
        else:
            print(f"  [skip CISV]")

    _banner(f"All seeds complete  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    print("  Results written to:")
    for seed in seeds:
        print(f"    Experiment2/results/seed_{seed}/   (CoT)")
        print(f"    CISV/results/seed_{seed}/           (CISV)")


if __name__ == "__main__":
    main()
