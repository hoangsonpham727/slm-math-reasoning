"""
Experiment 4: Program-of-Thought + Execution Verification

For each problem in the synthetic depth-1–8 dataset, sample k Python solutions,
execute each in an isolated subprocess, then take majority vote over valid answers.

Usage:
    python run_experiment4.py [--models all] [--data_path ../Experiment2/data/problems_all.json]
                              [--output_dir results] [--device cuda:0]
                              [--max_new_tokens 1024] [--k 8] [--temperature 0.7]
                              [--n_problems None] [--no_resume]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from datetime import datetime

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from models import get_all_configs, get_model_wrapper, ModelConfig


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

POT_SYSTEM = (
    "You are a math solver. You solve problems by writing Python code."
)

POT_USER = """Problem:
{problem}

Write Python code to solve this step by step.
Rules:
1. Assign a descriptive variable name to EVERY quantity introduced.
2. Add a brief comment before each line explaining the calculation.
3. The final line must assign the result to: answer = <expression>
4. Use only basic arithmetic (+ - * /). Do not import anything.

Python code:
"""


# ---------------------------------------------------------------------------
# Code extraction & execution
# ---------------------------------------------------------------------------

_RE_FENCE = re.compile(r'```(?:python)?\n(.*?)```', re.DOTALL)


def extract_code_block(text: str) -> str:
    """Strip markdown fences if present; return raw Python code."""
    m = _RE_FENCE.search(text)
    if m:
        return m.group(1).strip()
    # No fences — try to find the first line that looks like Python
    lines = text.strip().splitlines()
    code_lines = []
    in_code = False
    for line in lines:
        if re.match(r'\s*(#.*|[a-zA-Z_]\w*\s*=|\w+\s*[+\-*/])', line):
            in_code = True
        if in_code:
            code_lines.append(line)
    return "\n".join(code_lines).strip() if code_lines else text.strip()


def execute_python(code: str, timeout: int = 10) -> float | None:
    """
    Execute generated Python code in an isolated subprocess.
    Returns the float value of `answer` if execution succeeds, else None.
    Subprocess isolation ensures a crash in generated code won't abort the loop.
    """
    runner = f"import math, fractions\n{code}\nprint(answer)"
    try:
        result = subprocess.run(
            [sys.executable, "-c", runner],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            val = result.stdout.strip().split("\n")[-1]
            return float(val)
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_problems(data_path: str, n_problems: int | None) -> list[dict]:
    with open(data_path) as f:
        data = json.load(f)
    if n_problems:
        data = data[:n_problems]
    return data


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def load_completed_ids(out_path: Path) -> set[str]:
    completed = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    completed.add(json.loads(line)["problem_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


def append_record(record: dict, out_path: Path) -> None:
    with open(out_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Per-model inference loop
# ---------------------------------------------------------------------------

def run_model(
    config: ModelConfig,
    problems: list[dict],
    output_dir: str,
    device: str,
    max_new_tokens: int,
    k: int,
    temperature: float,
    resume: bool,
):
    out_path = Path(output_dir) / f"{config.short_name}_pot_k{k}.jsonl"

    completed_ids = set()
    if resume:
        completed_ids = load_completed_ids(out_path)
        if completed_ids:
            print(f"    [resume] {len(completed_ids)} problems already done.")

    total = len(problems)
    wrapper = get_model_wrapper(config, device)
    model_loaded = False

    correct_count = 0
    done = 0

    for item in problems:
        pid = item["problem_id"]
        if pid in completed_ids:
            done += 1
            continue

        if not model_loaded:
            wrapper.load()
            model_loaded = True

        question     = item["question"]
        gold         = item["ground_truth"]  # float
        depth        = item["depth"]
        prompt       = POT_USER.format(problem=question)

        t0 = time.time()

        # ── Sample k solutions ───────────────────────────────────────────────
        candidates = []
        exec_errors = 0

        for _ in range(k):
            try:
                response = wrapper.generate(
                    POT_SYSTEM, prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(k > 1),   # greedy when k=1, sample otherwise
                )
            except Exception as e:
                exec_errors += 1
                continue

            code = extract_code_block(response)
            prog_answer = execute_python(code)

            if prog_answer is not None:
                candidates.append({"code": code, "prog_answer": prog_answer})
            else:
                exec_errors += 1

        # ── Majority vote ────────────────────────────────────────────────────
        if candidates:
            vote_counts  = Counter(round(c["prog_answer"], 4) for c in candidates)
            predicted_val, vote_won = vote_counts.most_common(1)[0]
            predicted = predicted_val
        else:
            predicted = None
            vote_won  = 0

        # ── Correctness (1 % relative tolerance, matching Exp2 convention) ───
        is_correct = (
            predicted is not None
            and abs(predicted - gold) / (abs(gold) + 1e-9) < 0.01
        )
        elapsed = time.time() - t0

        if is_correct:
            correct_count += 1
        done += 1

        record = {
            "model":            config.short_name,
            "problem_id":       pid,
            "depth":            depth,
            "question":         question,
            "ground_truth":     gold,
            "k_samples":        k,
            "valid_executions": len(candidates),
            "execution_errors": exec_errors,
            "candidates":       candidates,
            "predicted":        predicted,
            "vote_won":         vote_won,
            "is_correct":       is_correct,
            "elapsed_s":        round(elapsed, 3),
        }
        append_record(record, out_path)

        sym = "✓" if is_correct else ("∅" if predicted is None else "✗")
        print(f"    [{config.short_name}|d{depth}] {done}/{total}  "
              f"GT={gold:.2f}  Pred={predicted}  {sym}  "
              f"valid={len(candidates)}/{k}  ({elapsed:.1f}s)")

        if done % 50 == 0:
            acc = correct_count / done
            print(f"    --- Running accuracy: {acc:.4f} ({done}/{total}) ---")

    if model_loaded:
        wrapper.unload()

    n_done = done
    acc = correct_count / n_done if n_done else 0.0
    print(f"\n  [{config.short_name}] Accuracy: {acc:.4f} ({correct_count}/{n_done})")
    print(f"  Results saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Experiment 4: PoT + Execution Verification")
    p.add_argument("--models",         type=str, default="all")
    p.add_argument("--data_path",      type=str,
                   default=str(repo_root / "Experiment2" / "data" / "problems_all.json"))
    p.add_argument("--output_dir",     type=str, default="results")
    p.add_argument("--device",         type=str, default="cuda:0")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--k",              type=int, default=8,
                   help="Number of solution samples per problem (default: 8)")
    p.add_argument("--temperature",    type=float, default=0.7,
                   help="Sampling temperature for k>1 (default: 0.7)")
    p.add_argument("--n_problems",     type=int, default=None,
                   help="Limit total problems evaluated (default: all)")
    p.add_argument("--no_resume",      action="store_true")
    p.add_argument("--smoke_test",     action="store_true",
                   help="Quick sanity check: force n_problems=2 and k=2 per model.")
    return p.parse_args()


def main():
    os.environ.setdefault("HF_HOME", "/mnt/data/hf")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/data/hf/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/data/hf/hub")
    os.environ.setdefault("TMPDIR", "/mnt/data/tmp")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    args = parse_args()
    if args.smoke_test:
        args.n_problems = 2
        args.k = 2
        print("[smoke-test] forcing n_problems=2, k=2.", file=sys.stderr)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_configs = get_all_configs()
    if args.models == "all":
        configs = all_configs
    else:
        wanted = set(args.models.split(","))
        configs = [c for c in all_configs if c.short_name in wanted]
        missing = wanted - {c.short_name for c in configs}
        if missing:
            raise ValueError(f"Unknown model short_name(s): {sorted(missing)}")

    problems = load_problems(args.data_path, args.n_problems)

    smoke_tag = "  [SMOKE TEST]" if args.smoke_test else ""
    print(f"\n{'='*60}")
    print(f"Experiment 4 — PoT + Execution Verification{smoke_tag}")
    print(f"  Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Models:      {[c.short_name for c in configs]}")
    print(f"  Problems:    {len(problems)} (k={args.k}, temp={args.temperature})")
    print(f"  Output dir:  {args.output_dir}")
    print(f"{'='*60}\n")

    for cfg in configs:
        print(f"\n{'─'*50}")
        print(f"  Model: {cfg.short_name}  ({cfg.model_id})")
        print(f"{'─'*50}")
        run_model(
            config=cfg,
            problems=problems,
            output_dir=args.output_dir,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            k=args.k,
            temperature=args.temperature,
            resume=not args.no_resume,
        )

    print(f"\n{'='*60}")
    print(f"Experiment 4 complete.  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
