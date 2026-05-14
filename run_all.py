"""
Top-level orchestrator for Experiments 3 and 4.

Two callable functions — import and call from a notebook or script:

    from run_all import run_experiment3, run_experiment4

    run_experiment3(device="cuda:0")
    run_experiment4(device="cuda:0", k=8)

Each function runs all three solver models in sequence and writes results to
disk as soon as each model finishes (JSONL, one record per line).
"""

import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
for _p in [str(ROOT), str(ROOT / "Experiment3"), str(ROOT / "Experiment4")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import get_all_configs

# Experiment 3 internals
from run_experiment3 import (
    load_examples,
    run_filtering,
    run_solving,
    _resolve_filter_wrapper,
    _SOLVER_SHORT_NAMES,
)

# Experiment 4 internals
from run_experiment4 import (
    load_problems,
    run_model as _run_model_exp4,
)

_DEFAULT_SOLVER_ORDER = ["qwen25_math_1.5b", "gemma4_e2b", "phi4_mini"]


# ---------------------------------------------------------------------------
# Experiment 3
# ---------------------------------------------------------------------------

def run_experiment3(
    filter_model: str = "qwen25_7b",
    solver_models: list[str] | None = None,
    enhanced_dir: str | None = None,
    output_dir: str = "Experiment3/results",
    device: str = "cuda:0",
    max_new_tokens: int = 1024,
    limit: int | None = None,
    distractor_seed: int = 42,
    resume: bool = True,
) -> None:
    """
    Run Experiment 3 (Distractor Filtering) for all solver models.

    Stage 1 — filter_model labels each clause RELEVANT/IRRELEVANT once and
               caches results to disk.
    Stage 2 — each solver model runs CoT on the cleaned problem; results are
               written to JSONL as soon as that model finishes.

    Args:
        filter_model:    HF short_name (e.g. 'qwen25_7b') or 'gpt-oss' for
                         Ollama cloud. Requires OLLAMA_API_KEY if 'gpt-oss'.
        solver_models:   List of short_names to run as solvers. Defaults to
                         all three SLMs in a fixed order.
        enhanced_dir:    Path to gsm_enhanced_templates directory.
        output_dir:      Directory where JSONL result files are written.
        device:          HuggingFace device string ('cuda:0', 'cpu', 'auto').
        max_new_tokens:  Token budget for generation.
        limit:           Cap the number of problems (useful for smoke tests).
        distractor_seed: RNG seed for distractor selection from the pool.
        resume:          Skip problems already present in the output file.
    """
    if enhanced_dir is None:
        enhanced_dir = str(ROOT / "Experiment1" / "gsm_enhanced_templates")

    if solver_models is None:
        all_configs = get_all_configs()
        solver_configs = [
            c for name in _DEFAULT_SOLVER_ORDER
            for c in all_configs if c.short_name == name
        ]
    else:
        all_configs = get_all_configs()
        config_by_name = {c.short_name: c for c in all_configs}
        solver_configs = [config_by_name[n] for n in solver_models]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filter_wrapper = _resolve_filter_wrapper(filter_model, device)

    examples = load_examples(enhanced_dir, seed=distractor_seed)
    if limit:
        examples = examples[:limit]

    print(f"\n{'='*60}")
    print(f"Experiment 3 — Distractor Filtering")
    print(f"  Started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Filter model:  {filter_wrapper.short_name}")
    print(f"  Solver models: {[c.short_name for c in solver_configs]}")
    print(f"  Problems:      {len(examples)}")
    print(f"  Output dir:    {output_dir}")
    print(f"{'='*60}")

    # ── Stage 1: filter once ─────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Stage 1 — Filtering with {filter_wrapper.short_name}")
    print(f"{'─'*50}")
    filter_records = run_filtering(
        filter_wrapper=filter_wrapper,
        examples=examples,
        output_dir=output_dir,
        max_new_tokens=max_new_tokens,
    )

    # ── Stage 2: solve — one model at a time, results flushed after each ─────
    for cfg in solver_configs:
        print(f"\n{'─'*50}")
        print(f"  Stage 2 — Solving with {cfg.short_name}  ({cfg.model_id})")
        print(f"{'─'*50}")
        run_solving(
            solver_config=cfg,
            examples=examples,
            filter_records=filter_records,
            filter_model_name=filter_wrapper.short_name,
            output_dir=output_dir,
            device=device,
            max_new_tokens=max_new_tokens,
            resume=resume,
        )
        print(f"  [{cfg.short_name}] results written to {output_dir}/")

    print(f"\n{'='*60}")
    print(f"Experiment 3 complete.  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Experiment 4
# ---------------------------------------------------------------------------

def run_experiment4(
    models: list[str] | None = None,
    data_path: str | None = None,
    output_dir: str = "Experiment4/results",
    device: str = "cuda:0",
    max_new_tokens: int = 1024,
    k: int = 8,
    temperature: float = 0.7,
    n_problems: int | None = None,
    resume: bool = True,
) -> None:
    """
    Run Experiment 4 (PoT + Execution Verification) for all models.

    For each model, samples k Python solutions per problem, executes them in
    isolated subprocesses, and majority-votes the answer. Results are written
    to JSONL as soon as each model finishes.

    Args:
        models:          List of short_names to evaluate. Defaults to all three
                         SLMs in a fixed order.
        data_path:       Path to problems_all.json (depth 1–8 combined dataset).
        output_dir:      Directory where JSONL result files are written.
        device:          HuggingFace device string ('cuda:0', 'cpu', 'auto').
        max_new_tokens:  Token budget for code generation.
        k:               Number of solution samples per problem (default 8).
                         Reduce to 4 if VRAM is tight.
        temperature:     Sampling temperature when k > 1 (default 0.7).
        n_problems:      Cap total problems evaluated (default: all 1600).
        resume:          Skip problems already present in the output file.
    """
    if data_path is None:
        data_path = str(ROOT / "Experiment2" / "data" / "problems_all.json")

    if models is None:
        all_configs = get_all_configs()
        configs = [
            c for name in _DEFAULT_SOLVER_ORDER
            for c in all_configs if c.short_name == name
        ]
    else:
        all_configs = get_all_configs()
        config_by_name = {c.short_name: c for c in all_configs}
        configs = [config_by_name[n] for n in models]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    problems = load_problems(data_path, n_problems)

    print(f"\n{'='*60}")
    print(f"Experiment 4 — PoT + Execution Verification")
    print(f"  Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Models:      {[c.short_name for c in configs]}")
    print(f"  Problems:    {len(problems)}  (k={k}, temp={temperature})")
    print(f"  Output dir:  {output_dir}")
    print(f"{'='*60}")

    for cfg in configs:
        print(f"\n{'─'*50}")
        print(f"  Model: {cfg.short_name}  ({cfg.model_id})")
        print(f"{'─'*50}")
        _run_model_exp4(
            config=cfg,
            problems=problems,
            output_dir=output_dir,
            device=device,
            max_new_tokens=max_new_tokens,
            k=k,
            temperature=temperature,
            resume=resume,
        )
        print(f"  [{cfg.short_name}] results written to {output_dir}/")

    print(f"\n{'='*60}")
    print(f"Experiment 4 complete.  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Run both experiments when executed as a script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run Experiment 3 and/or 4")
    p.add_argument("--exp",     type=str, default="both", choices=("3", "4", "both"))
    p.add_argument("--device",  type=str, default="cuda:0")
    p.add_argument("--k",       type=int, default=8)
    p.add_argument("--filter_model", type=str, default="qwen25_7b")
    p.add_argument("--limit",   type=int, default=None)
    p.add_argument("--no_resume", action="store_true")
    args = p.parse_args()

    if args.exp in ("3", "both"):
        run_experiment3(
            filter_model=args.filter_model,
            device=args.device,
            limit=args.limit,
            resume=not args.no_resume,
        )
    if args.exp in ("4", "both"):
        run_experiment4(
            device=args.device,
            k=args.k,
            n_problems=args.limit,
            resume=not args.no_resume,
        )
