"""
Experiment 2: Reasoning Depth — Inference Runner

Iterates over all (model × depth × regime) cells and saves raw results to JSONL.
One model is loaded at a time and unloaded before the next, to respect GPU memory.

Usage:
    python run_inference.py [--models all] [--depths 1,2,3,4,5,6,7,8]
                            [--regime direct,cot] [--data_dir data]
                            [--output_dir results] [--device auto]
                            [--max_new_tokens 512] [--n_per_depth 200]
"""

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime
import os

try:
    from .dataset_generator import (
        MathProblem,
        build_prompt_direct,
        build_prompt_cot,
        parse_answer,
        check_step_accuracy,
        SYSTEM_DIRECT,
        SYSTEM_COT,
    )
except ImportError:
    from dataset_generator import (
        MathProblem,
        build_prompt_direct,
        build_prompt_cot,
        parse_answer,
        check_step_accuracy,
        SYSTEM_DIRECT,
        SYSTEM_COT,
    )

try:
    from ..models import get_all_configs, get_model_wrapper
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from models import get_all_configs, get_model_wrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_problems(data_dir: str, depths: list, n_per_depth: int) -> dict:
    """Load pre-generated problems from JSON files."""
    dataset = {}
    for depth in depths:
        fpath = Path(data_dir) / f"problems_depth{depth:02d}.json"
        if not fpath.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {fpath}\n"
                f"Run `python dataset_generator.py` first."
            )
        with open(fpath) as f:
            raw = json.load(f)
        problems = [MathProblem(**r) for r in raw[:n_per_depth]]
        dataset[depth] = problems
    return dataset


def result_record(
    model_name: str,
    regime: str,
    problem: MathProblem,
    response: str,
    predicted: float | None,
    elapsed_s: float,
) -> dict:
    """Build a flat result dictionary for one (model, regime, problem) triple."""
    is_correct = (
        predicted is not None
        and abs(predicted - problem.ground_truth) / (abs(problem.ground_truth) + 1e-9) < 0.01
    )
    first_error_step = check_step_accuracy(response, problem)

    return {
        "model":             model_name,
        "regime":            regime,
        "problem_id":        problem.problem_id,
        "depth":             problem.depth,
        "question":          problem.question,
        "ground_truth":      problem.ground_truth,
        "response":          response,
        "predicted":         predicted,
        "is_correct":        is_correct,
        "is_collapse":       predicted is None,
        "first_error_step":  first_error_step,
        "elapsed_s":         round(elapsed_s, 3),
    }


def save_results(records: list, output_dir: str, model_name: str, regime: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = Path(output_dir) / f"{model_name}_{regime}.jsonl"
    with open(fname, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return fname


# ---------------------------------------------------------------------------
# Per-model inference loop
# ---------------------------------------------------------------------------

def run_model(
    config,
    dataset: dict,
    regimes: list,
    output_dir: str,
    device: str,
    max_new_tokens: int,
    resume: bool = True,
):
    """
    Load one model, run all (depth × regime) cells, unload, save results.
    Supports resuming: already-saved problem_ids in the output file are skipped.
    """
    wrapper = get_model_wrapper(config, device)

    for regime in regimes:
        out_path = Path(output_dir) / f"{config.short_name}_{regime}.jsonl"

        # Build set of already-completed problem_ids for resume support
        completed_ids = set()
        if resume and out_path.exists():
            with open(out_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        completed_ids.add(rec["problem_id"])
                    except json.JSONDecodeError:
                        pass
            if completed_ids:
                print(f"    [resume] {len(completed_ids)} problems already done "
                      f"for {config.short_name}/{regime}.")

        system_prompt = SYSTEM_COT if regime == "cot" else SYSTEM_DIRECT
        prompt_fn     = build_prompt_cot if regime == "cot" else build_prompt_direct

        depths_sorted = sorted(dataset.keys())
        total_probs = sum(len(dataset[d]) for d in depths_sorted)
        done = 0

        # Lazy-load: load model only on first actual problem that needs running
        model_loaded = False

        for depth in depths_sorted:
            problems = dataset[depth]
            batch_records = []

            for problem in problems:
                if problem.problem_id in completed_ids:
                    done += 1
                    continue

                # Load model on first needed call
                if not model_loaded:
                    wrapper.load()
                    model_loaded = True

                user_prompt = prompt_fn(problem)

                t0 = time.time()
                try:
                    response = wrapper.generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_new_tokens=max_new_tokens,
                    )
                except Exception as e:
                    response = f"[ERROR: {e}]"
                elapsed = time.time() - t0

                predicted = parse_answer(response)
                rec = result_record(
                    config.short_name, regime, problem,
                    response, predicted, elapsed,
                )
                batch_records.append(rec)
                done += 1

                # Print live progress
                correct_symbol = "✓" if rec["is_correct"] else ("∅" if rec["is_collapse"] else "✗")
                fail_info = (
                    f" [fail@step{rec['first_error_step']}]"
                    if rec["first_error_step"] is not None and not rec["is_correct"]
                    else ""
                )
                print(f"    [{config.short_name}|{regime}|d{depth}] "
                      f"{done}/{total_probs}  "
                      f"GT={problem.ground_truth:>8.2f}  "
                      f"Pred={str(predicted):>8}  "
                      f"{correct_symbol}{fail_info}  "
                      f"({elapsed:.1f}s)")

            # Flush batch to disk after each depth level
            if batch_records:
                save_results(batch_records, output_dir, config.short_name, regime)

    if model_loaded:
        wrapper.unload()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 2 Reasoning Depth Inference Runner")
    parser.add_argument("--models",    type=str, default="all",
                        help="Comma-separated short_names or 'all'")
    parser.add_argument("--depths",    type=str, default="1,2,3,4,5,6,7,8",
                        help="Comma-separated depth levels to evaluate")
    parser.add_argument("--regime",    type=str, default="direct,cot",
                        help="Comma-separated regimes: direct, cot")
    parser.add_argument("--data_dir",  type=str, default="data")
    parser.add_argument("--output_dir",type=str, default="results")
    parser.add_argument("--device",    type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--n_per_depth",    type=int, default=200)
    parser.add_argument("--no_resume", action="store_true",
                        help="Disable resume; re-run everything from scratch")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed variant for multi-seed experiments. "
                             "When provided, data_dir defaults to data/seed_{seed}/ "
                             "and output_dir defaults to results/seed_{seed}/.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve seed-specific paths (only when --seed is given and user hasn't
    # explicitly overridden data_dir / output_dir away from their defaults).
    data_dir   = args.data_dir
    output_dir = args.output_dir
    if args.seed is not None:
        if data_dir == "data":
            data_dir = f"data/seed_{args.seed}"
        if output_dir == "results":
            output_dir = f"results/seed_{args.seed}"

    depths  = [int(d) for d in args.depths.split(",")]
    regimes = [r.strip() for r in args.regime.split(",")]
    all_configs = get_all_configs()

    if args.models == "all":
        configs_to_run = all_configs
    else:
        wanted = set(args.models.split(","))
        configs_to_run = [c for c in all_configs if c.short_name in wanted]

    seed_label = f"seed={args.seed}" if args.seed is not None else "legacy"
    print(f"\n{'='*60}")
    print(f"Experiment 2 — Reasoning Depth Inference [{seed_label}]")
    print(f"  Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Models:     {[c.short_name for c in configs_to_run]}")
    print(f"  Depths:     {depths}")
    print(f"  Regimes:    {regimes}")
    print(f"  N/depth:    {args.n_per_depth}")
    print(f"  Data dir:   {data_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*60}\n")

    # Load dataset once
    dataset = load_problems(data_dir, depths, args.n_per_depth)
    print(f"Dataset loaded: {sum(len(v) for v in dataset.values())} total problems.\n")

    for cfg in configs_to_run:
        print(f"\n{'─'*50}")
        print(f"  Model: {cfg.short_name}  ({cfg.model_id})")
        print(f"{'─'*50}")
        run_model(
            config=cfg,
            dataset=dataset,
            regimes=regimes,
            output_dir=output_dir,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            resume=not args.no_resume,
        )

    print(f"\n{'='*60}")
    print(f"All inference complete. Results in: {args.output_dir}/")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
