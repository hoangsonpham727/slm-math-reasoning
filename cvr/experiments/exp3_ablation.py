"""
Experiment 3 (CVR): Ablation Studies

Ablates CVR hyperparameters using the best-performing model on a held-out
subset of depth-2 and depth-4 problems (50 problems per depth = 100 total).

Ablation axes:
  - consistency_votes: 1, 3, 5
  - enable_relevance: True, False
  - max_restarts: 0, 1, 2, 3
  - num_chains: 1, 3, 5, 7
  - one_veto: False (majority), True (any Wrong → fail)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tqdm import tqdm

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from models import MODEL_CONFIGS, get_model_wrapper
from Experiment2.dataset_generator import parse_answer
from cvr.model_adapter import CVRModelAdapter
from cvr.pipeline import CVRPipeline
from cvr.utils import load_yaml_config
from cvr.evaluation.eval_accuracy import compute_accuracy


def load_ablation_problems(data_dir: str, depths: list[int] = None, n_per_depth: int = 50) -> list[dict]:
    if depths is None:
        depths = [2, 4]
    problems = []
    for depth in depths:
        fpath = Path(data_dir) / f"problems_depth{depth:02d}.json"
        with open(fpath) as f:
            raw = json.load(f)
        for rec in raw[:n_per_depth]:
            problems.append(rec)
    return problems


def run_ablation_config(
    problems: list[dict],
    adapter: CVRModelAdapter,
    config: dict,
) -> dict:
    """Run CVR with given config on all problems. Returns accuracy metrics."""
    pipeline = CVRPipeline(adapter, config)
    records = []
    for prob in problems:
        result = pipeline.solve(prob["question"])
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
        records.append({
            "is_correct": is_correct,
            "is_collapse": pred_float is None,
            "depth": prob.get("depth"),
            "total_restarts": sum(c.get("total_restarts", 0) for c in result.get("chains", [])),
        })
    metrics = compute_accuracy(records)
    metrics["collapse_rate"] = round(sum(1 for r in records if r["is_collapse"]) / len(records), 4)
    metrics["avg_restarts"] = round(sum(r["total_restarts"] for r in records) / len(records), 3)
    return metrics


def run_experiment3(
    data_dir: str = "Experiment2/data",
    config_path: str = "cvr/config.yaml",
    output_dir: str = "cvr/results",
    device: str = "auto",
    model_name: str = "qwen25_math_1.5b",
    n_per_depth: int = 50,
    ablation_depths: Optional[list[int]] = None,
):
    base_config = load_yaml_config(config_path)
    problems = load_ablation_problems(data_dir, ablation_depths, n_per_depth)

    model_cfg = next((c for c in MODEL_CONFIGS if c.short_name == model_name), None)
    if model_cfg is None:
        raise ValueError(f"Unknown model: {model_name}. Choose from {[c.short_name for c in MODEL_CONFIGS]}")

    wrapper = get_model_wrapper(model_cfg, device=device)
    wrapper.load()
    adapter = CVRModelAdapter(wrapper)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "n_problems": len(problems),
        "ablations": {},
    }

    def _run(label: str, cfg: dict):
        print(f"\n  Ablation [{label}] ...")
        metrics = run_ablation_config(problems, adapter, cfg)
        results["ablations"][label] = {**metrics, "config_patch": label}
        print(f"    acc={metrics['overall_accuracy']:.3f}  collapse={metrics['collapse_rate']:.3f}  restarts={metrics['avg_restarts']:.2f}")

    import copy

    # 1. Consistency votes
    for k in [1, 3, 5]:
        cfg = copy.deepcopy(base_config)
        cfg["verification"]["consistency_votes"] = k
        cfg["verification"]["enable_relevance"] = False
        _run(f"consistency_votes={k}", cfg)

    # 2. Relevance check on/off
    for rel in [False, True]:
        cfg = copy.deepcopy(base_config)
        cfg["verification"]["enable_relevance"] = rel
        _run(f"enable_relevance={rel}", cfg)

    # 3. Max restarts
    for max_r in [0, 1, 2, 3]:
        cfg = copy.deepcopy(base_config)
        cfg["restart"]["max_restarts"] = max_r
        cfg["verification"]["enable_relevance"] = False
        _run(f"max_restarts={max_r}", cfg)

    # 4. Number of chains
    for n_chains in [1, 3, 5, 7]:
        cfg = copy.deepcopy(base_config)
        cfg["generation"]["num_chains"] = n_chains
        cfg["verification"]["enable_relevance"] = False
        _run(f"num_chains={n_chains}", cfg)

    # 5. One-vote-veto vs majority (requires NodeVerifier one_veto flag)
    # We patch the pipeline's verifier after construction
    for veto in [False, True]:
        cfg = copy.deepcopy(base_config)
        cfg["verification"]["enable_relevance"] = False
        cfg["verification"]["one_veto"] = veto  # read by NodeVerifier if we add support
        label = f"one_veto={veto}"
        print(f"\n  Ablation [{label}] ...")
        # Temporarily patch NodeVerifier to support one_veto
        pipeline = CVRPipeline(adapter, cfg)
        pipeline.verifier.consistency.one_veto = veto
        pipeline.verifier.relevance.one_veto = veto
        records = []
        for prob in problems:
            result = pipeline.solve(prob["question"])
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
            records.append({
                "is_correct": is_correct,
                "is_collapse": pred_float is None,
                "total_restarts": sum(c.get("total_restarts", 0) for c in result.get("chains", [])),
            })
        metrics = compute_accuracy(records)
        metrics["collapse_rate"] = round(sum(1 for r in records if r["is_collapse"]) / len(records), 4)
        metrics["avg_restarts"] = round(sum(r["total_restarts"] for r in records) / len(records), 3)
        results["ablations"][label] = metrics
        print(f"    acc={metrics['overall_accuracy']:.3f}  collapse={metrics['collapse_rate']:.3f}")

    wrapper.unload()

    out_file = output_path / f"exp3_ablation_{model_name}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved to {out_file}")
    return results
