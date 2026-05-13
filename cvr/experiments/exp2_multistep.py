"""
Experiment 2 (CVR): Multi-step Degradation

Evaluates three methods across problems at depths 1–8:
  1. SC (Self-Consistency, N=5)
  2. CVR-VerifyOnly (verify + no restart, max_restarts=0)
  3. CVR-Full       (verify + restart)

Dataset: Experiment2/data/problems_depth0X.json (200 problems per depth)

Results saved as JSONL per (model, method) pair, mirroring Experiment2's format.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tqdm import tqdm

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from models import MODEL_CONFIGS, get_model_wrapper
from Experiment2.dataset_generator import MathProblem, parse_answer
from cvr.model_adapter import CVRModelAdapter
from cvr.pipeline import CVRPipeline
from cvr.baselines.baseline_sc import run_sc_on_dataset
from cvr.utils import load_yaml_config, extract_model_final
from cvr.evaluation.eval_multistep import compute_multistep_metrics, compute_depth_ceiling


def load_problems(data_dir: str, depths: list[int], n_per_depth: int = 200) -> list[dict]:
    """Load problems from per-depth JSON files. Returns list of plain dicts."""
    all_problems = []
    for depth in depths:
        fpath = Path(data_dir) / f"problems_depth{depth:02d}.json"
        with open(fpath) as f:
            raw = json.load(f)
        for rec in raw[:n_per_depth]:
            all_problems.append(rec)
    return all_problems


def run_method_on_problems(
    method: str,
    problems: list[dict],
    adapter: CVRModelAdapter,
    config: dict,
    model_name: str,
    limit: Optional[int] = None,
) -> list[dict]:
    """
    Run one method on all problems. Returns list of result records (JSONL-compatible).
    method: 'sc' | 'cvr_verifyonly' | 'cvr_full'
    """
    if limit:
        problems = problems[:limit]

    results = []

    if method == "sc":
        examples = [{"question": p["question"], "ground_truth": p["ground_truth"], **p} for p in problems]
        records = run_sc_on_dataset(
            adapter, examples,
            num_chains=config["generation"]["num_chains"],
            temperature=config["generation"]["temperature"],
        )
        for rec, prob in zip(records, problems):
            rec.setdefault("model", model_name)
            rec.setdefault("regime", "sc")
            rec.setdefault("problem_id", prob.get("problem_id"))
            rec.setdefault("depth", prob.get("depth"))
        return records

    # CVR variants
    cfg = dict(config)
    cfg["restart"] = dict(config["restart"])
    if method == "cvr_verifyonly":
        cfg["restart"]["max_restarts"] = 0
    cfg["verification"] = dict(config["verification"])
    cfg["verification"]["enable_relevance"] = False  # Exp2 has no distractors

    pipeline = CVRPipeline(adapter, cfg)

    for prob in tqdm(problems, desc=f"{model_name}/{method}"):
        t0 = time.perf_counter()
        result = pipeline.solve(prob["question"])
        elapsed = round(time.perf_counter() - t0, 3)

        gt = float(prob["ground_truth"])
        pred_str = result.get("answer")
        try:
            pred_float = float(pred_str) if pred_str else None
        except (ValueError, TypeError):
            pred_float = None

        is_correct = (
            pred_float is not None
            and abs(pred_float - gt) / (abs(gt) + 1e-9) < 0.01
        )

        total_restarts = sum(c.get("total_restarts", 0) for c in result.get("chains", []))
        consistency_failures = sum(c.get("consistency_failures", 0) for c in result.get("chains", []))

        record = {
            "model": model_name,
            "regime": method,
            "problem_id": prob.get("problem_id"),
            "depth": prob.get("depth"),
            "question": prob.get("question"),
            "ground_truth": gt,
            "predicted": pred_float,
            "is_correct": is_correct,
            "is_collapse": pred_float is None,
            "total_restarts": total_restarts,
            "consistency_failures": consistency_failures,
            "relevance_failures": 0,  # disabled for Exp2
            "successful_chains": result.get("successful_chains", 0),
            "confidence": result.get("confidence", 0.0),
            "elapsed_s": elapsed,
        }
        results.append(record)

    return results


def run_experiment2(
    data_dir: str = "Experiment2/data",
    config_path: str = "cvr/config.yaml",
    output_dir: str = "cvr/results",
    device: str = "auto",
    model_filter: Optional[list[str]] = None,
    depths: Optional[list[int]] = None,
    n_per_depth: int = 200,
    limit: Optional[int] = None,
):
    if depths is None:
        depths = list(range(1, 9))

    config = load_yaml_config(config_path)
    problems = load_problems(data_dir, depths, n_per_depth)

    configs = MODEL_CONFIGS
    if model_filter:
        configs = [c for c in configs if c.short_name in model_filter]

    methods = ["sc", "cvr_verifyonly", "cvr_full"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "methods": methods,
        "depths": depths,
        "models": {},
    }

    for cfg in configs:
        print(f"\n{'='*60}\nModel: {cfg.short_name}")
        wrapper = get_model_wrapper(cfg, device=device)
        wrapper.load()
        adapter = CVRModelAdapter(wrapper)

        for method in methods:
            print(f"\n  Method: {method}")
            records = run_method_on_problems(
                method, problems, adapter, config, cfg.short_name, limit
            )
            # Save JSONL
            out_file = output_path / f"{cfg.short_name}_{method}.jsonl"
            with open(out_file, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
            print(f"  Saved {len(records)} records to {out_file}")

            # Quick summary
            metrics = compute_multistep_metrics(records)
            overall = metrics["overall"]
            k_star = compute_depth_ceiling(metrics["by_depth"])
            print(f"  Overall acc: {overall['accuracy']:.3f}  |  k* (ceiling): {k_star}")

            if cfg.short_name not in summary["models"]:
                summary["models"][cfg.short_name] = {}
            summary["models"][cfg.short_name][method] = {
                "overall_accuracy": overall["accuracy"],
                "depth_ceiling_k_star": k_star,
                "avg_restarts": overall.get("avg_restarts_per_problem", 0),
            }

        wrapper.unload()

    summary_file = output_path / "exp2_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")
    return summary
