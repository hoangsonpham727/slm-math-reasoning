"""
GSM8K CISV Inference Runner

Runs the Chunk-wise Incremental Solving with Verification (CISV) pipeline on
the filtered GSM8K test set for all configured SLMs.

Thin wrapper: imports run_exp4 from CISV/chunked_solve.py and init_model from
CISV/llm_wrapper.py with GSM8K-specific field mappings.

GSM8K problems already have:
  - "question"        → problem text
  - "ground_truth"    → numeric gold answer (parsed from #### N)
  - "estimated_steps" → number of arithmetic steps (used as depth proxy)

These map directly to run_exp4's default field names.

Usage:
    python run_gsm8k_cisv.py --models all --device cuda:0
    python run_gsm8k_cisv.py --models phi4_mini --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────

HERE  = Path(__file__).resolve().parent         # GSM8K/
ROOT  = HERE.parent                             # repo root
CISV  = ROOT / "CISV"

for _p in [str(ROOT), str(CISV)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chunked_solve import run_exp4, _MODEL_CHUNK_SIZE   # type: ignore
from llm_wrapper   import init_model, llm as _llm       # type: ignore
import llm_wrapper as _lw                               # type: ignore

# ── Constants ─────────────────────────────────────────────────────────────────

ALL_MODELS      = ["qwen25_math_1.5b", "gemma4_e2b", "phi4_mini"]

# GSM8K field names for run_exp4
GSM8K_FIELDS = dict(
    problem_field = "question",
    answer_field  = "ground_truth",
    depth_field   = "estimated_steps",   # used only for grouping in summary
)

DEFAULT_DATA   = str(HERE / "data" / "gsm8k_test.json")
DEFAULT_OUT    = str(HERE / "results" / "cisv")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GSM8K CISV inference runner")
    p.add_argument("--models",   type=str, default="all",
                   help="Comma-separated model short_names or 'all'")
    p.add_argument("--device",   type=str, default="cuda:0")
    p.add_argument("--data",     type=str, default=DEFAULT_DATA,
                   help="Path to gsm8k_test.json")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUT,
                   help="Directory for JSON output files")
    return p.parse_args()


def main():
    os.environ.setdefault("HF_HOME",               "/mnt/data/hf")
    os.environ.setdefault("TRANSFORMERS_CACHE",     "/mnt/data/hf/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE",  "/mnt/data/hf/hub")
    os.environ.setdefault("TMPDIR",                 "/mnt/data/tmp")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES",   "0")

    args = parse_args()

    if args.models == "all":
        models = ALL_MODELS
    else:
        models = [m.strip() for m in args.models.split(",")]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"GSM8K CISV Inference  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    print(f"  Models  : {models}")
    print(f"  Data    : {args.data}")
    print(f"  Output  : {out_dir}")
    print(f"{'='*65}")

    all_results: dict[str, list[dict]] = {}

    for short_name in models:
        output_path = str(out_dir / f"exp4_chunked_{short_name}.json")

        print(f"\n{'─'*65}")
        print(f"  Model: {short_name}  —  started {datetime.now():%H:%M:%S}")
        print(f"{'─'*65}")

        init_model(short_name, device=args.device)

        chunk_size = _MODEL_CHUNK_SIZE.get(short_name, 2)
        print(f"  chunk_size: {chunk_size}")

        results = run_exp4(
            dataset_path=args.data,
            output_path=output_path,
            llm_fn=_lw.llm,
            chunk_size=chunk_size,
            **GSM8K_FIELDS,
        )

        all_results[short_name] = results
        _lw._wrapper.unload()

        print(f"  [{short_name}] done — results saved to {output_path}")

    # Cross-model summary
    print(f"\n{'='*65}")
    print("FINAL SUMMARY — GSM8K CISV — All models")
    print(f"{'='*65}")
    print(f"\n{'Model':<22}{'Overall':>10}  per-step-count accuracy")
    print("-" * 65)

    for short_name, results in all_results.items():
        if not results:
            continue
        overall = sum(r["correct"] for r in results) / len(results)
        step_groups: dict[int, list] = {}
        for r in results:
            step_groups.setdefault(r["depth"], []).append(r)
        step_acc = "  ".join(
            f"s{s}={sum(r['correct'] for r in g)/len(g):.2f}"
            for s, g in sorted(step_groups.items())
        )
        print(f"  {short_name:<20}{overall:>8.4f}  {step_acc}")

    print(f"\nAll results saved to '{out_dir}/'")


if __name__ == "__main__":
    main()
