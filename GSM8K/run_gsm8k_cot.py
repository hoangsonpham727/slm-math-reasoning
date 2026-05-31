"""
GSM8K CoT Inference Runner

Runs Chain-of-Thought (CoT) inference on the filtered GSM8K test set for all
configured SLMs and saves results as JSONL (one record per problem per model).

Reuses:
  - models.py  — model loading / inference
  - dataset_generator.SYSTEM_COT — CoT system prompt

Answer extraction uses a self-contained multi-strategy extractor (no Experiment1
dependency) that handles: \\boxed{}, GSM8K #### format, "Answer:" phrases,
and last-number fallback.

Usage:
    python run_gsm8k_cot.py --models all --device cuda:0
    python run_gsm8k_cot.py --models qwen25_math_1.5b --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent         # GSM8K/
ROOT = HERE.parent                             # repo root

for _p in [str(ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import get_all_configs, get_model_wrapper  # type: ignore

# CoT system prompt (shared with Experiment 2)
sys.path.insert(0, str(ROOT / "Experiment2"))
from dataset_generator import SYSTEM_COT  # type: ignore

# ── Answer extraction (self-contained) ───────────────────────────────────────

_RE_BOXED        = re.compile(r"\\boxed\{([^}]+)\}")
_RE_GSM_HASH     = re.compile(r"####\s*([\-\d,\.]+)")
_RE_ANSWER_PHRASE = [
    re.compile(r"(?is)(?:\*\*\s*)?[Ff]inal\s+[Aa]nswer\s*:\s*\$?\s*([\-\d,\.]+)"),
    re.compile(r"(?is)[Tt]he\s+(?:final\s+)?[Aa]nswer\s+is\s*:?\s*\$?\s*([\-\d,\.]+)"),
    re.compile(r"(?im)^\s*[Aa]nswer\s*:\s*\$?\s*([\-\d,\.]+)"),
]
_RE_LAST_NUMBER  = re.compile(r"(?<!\w)-?\d[\d,]*(?:\.\d+)?")


def _parse_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def extract_model_answer(text: str) -> Optional[float]:
    """
    Multi-strategy answer extraction from model response.
    Priority: \\boxed{} > GSM #### > answer phrases > last number in tail.
    """
    if not text:
        return None

    # 1. \boxed{N}
    m = _RE_BOXED.search(text)
    if m:
        v = _parse_float(m.group(1).strip("$").strip())
        if v is not None:
            return v

    # 2. GSM8K #### format
    matches = _RE_GSM_HASH.findall(text)
    if matches:
        v = _parse_float(matches[-1])
        if v is not None:
            return v

    # 3. Answer-phrase patterns (last match wins)
    best_match = None
    best_pos   = -1
    for pat in _RE_ANSWER_PHRASE:
        for m in pat.finditer(text):
            if m.start() > best_pos:
                best_match = m.group(1)
                best_pos   = m.start()
    if best_match is not None:
        v = _parse_float(best_match)
        if v is not None:
            return v

    # 4. Last number in the tail (last 25 lines, max 2000 chars)
    lines = text.splitlines()
    tail  = "\n".join(lines[-25:]) if len(lines) > 25 else text
    if len(tail) > 2000:
        tail = tail[-2000:]
    nums = _RE_LAST_NUMBER.findall(tail)
    if nums:
        return _parse_float(nums[-1])

    return None


def answers_match(gold: float, pred: Optional[float], eps: float = 1e-5) -> bool:
    if pred is None:
        return False
    tol = max(eps, abs(gold) * eps)
    return abs(gold - pred) <= tol


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_gsm8k(data_path: str) -> list[dict]:
    with open(data_path) as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} GSM8K problems from '{data_path}'")
    return problems


# ── Result I/O ────────────────────────────────────────────────────────────────

def save_record(record: dict, output_dir: str, model_name: str):
    """Append one result record to the model's JSONL file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fpath = Path(output_dir) / f"{model_name}_cot.jsonl"
    with open(fpath, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_completed_ids(output_dir: str, model_name: str) -> set:
    """Return the set of problem_ids already written (for resume support)."""
    fpath = Path(output_dir) / f"{model_name}_cot.jsonl"
    completed = set()
    if fpath.exists():
        with open(fpath) as f:
            for line in f:
                try:
                    completed.add(json.loads(line)["problem_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


# ── Per-model inference loop ──────────────────────────────────────────────────

COT_USER_SUFFIX = (
    "\n\nPlease reason step by step, then give your final answer as: Answer: <number>"
)


def run_model_on_gsm8k(
    config,
    problems: list[dict],
    output_dir: str,
    device: str,
    max_new_tokens: int,
    resume: bool = True,
):
    """Load one model, run all GSM8K problems, unload, save results."""
    completed = load_completed_ids(output_dir, config.short_name) if resume else set()
    if completed:
        print(f"    [resume] {len(completed)} problems already done.")

    wrapper = get_model_wrapper(config, device)
    model_loaded = False
    total = len(problems)

    for i, problem in enumerate(problems):
        pid = problem["problem_id"]
        if pid in completed:
            continue

        if not model_loaded:
            wrapper.load()
            model_loaded = True

        user_prompt = problem["question"] + COT_USER_SUFFIX

        t0 = time.time()
        try:
            response = wrapper.generate(
                system_prompt=SYSTEM_COT,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            response = f"[ERROR: {e}]"
        elapsed = time.time() - t0

        predicted    = extract_model_answer(response)
        is_correct   = answers_match(problem["ground_truth"], predicted)
        is_collapse  = predicted is None

        record = {
            "model":           config.short_name,
            "problem_id":      pid,
            "estimated_steps": problem["estimated_steps"],
            "question":        problem["question"],
            "ground_truth":    problem["ground_truth"],
            "response":        response,
            "predicted":       predicted,
            "is_correct":      is_correct,
            "is_collapse":     is_collapse,
            "elapsed_s":       round(elapsed, 3),
        }
        save_record(record, output_dir, config.short_name)

        sym = "✓" if is_correct else ("∅" if is_collapse else "✗")
        print(f"    [{config.short_name}] {i+1}/{total}  "
              f"GT={problem['ground_truth']:>8.2f}  "
              f"Pred={str(predicted):>8}  {sym}  ({elapsed:.1f}s)")

    if model_loaded:
        wrapper.unload()

    # Print per-step-count accuracy summary
    _print_summary(output_dir, config.short_name)


def _print_summary(output_dir: str, model_name: str):
    fpath = Path(output_dir) / f"{model_name}_cot.jsonl"
    if not fpath.exists():
        return
    records = []
    with open(fpath) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not records:
        return

    total   = len(records)
    correct = sum(r["is_correct"] for r in records)
    print(f"\n  [{model_name}] Overall: {correct}/{total} = {correct/total:.1%}")

    # Group by estimated_steps
    from collections import defaultdict
    by_steps: dict = defaultdict(list)
    for r in records:
        by_steps[r["estimated_steps"]].append(r["is_correct"])

    print(f"  {'Steps':<8}{'N':<8}{'Acc'}")
    for steps in sorted(by_steps):
        grp = by_steps[steps]
        acc = sum(grp) / len(grp)
        print(f"  {steps:<8}{len(grp):<8}{acc:.1%}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GSM8K CoT inference runner")
    p.add_argument("--models",   type=str, default="all",
                   help="Comma-separated model short_names or 'all'")
    p.add_argument("--device",   type=str, default="cuda:0")
    p.add_argument("--data",     type=str,
                   default=str(HERE / "data" / "gsm8k_test.json"),
                   help="Path to gsm8k_test.json")
    p.add_argument("--output_dir", type=str,
                   default=str(HERE / "results"),
                   help="Directory for JSONL output files")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--no_resume",   action="store_true",
                   help="Re-run from scratch instead of resuming")
    return p.parse_args()


def main():

    args = parse_args()

    all_configs = get_all_configs()
    if args.models == "all":
        configs = all_configs
    else:
        wanted  = set(args.models.split(","))
        configs = [c for c in all_configs if c.short_name in wanted]

    problems = load_gsm8k(args.data)

    print(f"\n{'='*65}")
    print(f"GSM8K CoT Inference  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    print(f"  Models  : {[c.short_name for c in configs]}")
    print(f"  Problems: {len(problems)}")
    print(f"  Output  : {args.output_dir}")
    print(f"{'='*65}\n")

    for cfg in configs:
        print(f"\n{'─'*50}")
        print(f"  Model: {cfg.short_name}")
        print(f"{'─'*50}")
        run_model_on_gsm8k(
            config=cfg,
            problems=problems,
            output_dir=args.output_dir,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            resume=not args.no_resume,
        )

    print(f"\n{'='*65}")
    print(f"All done  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    print(f"Results saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
