"""
Experiment 1 (CVR): Distractor Robustness

Evaluates four methods on clean vs. distractor-injected GSM problems:
  1. SC (Self-Consistency, N=5)
  2. CVR-NoRel (CVR without relevance check, enable_relevance=False)
  3. CVR-Full  (CVR with relevance check, enable_relevance=True)

Datasets:
  - Original:  Experiment1/gsm_templates/         (100 clean problems)
  - Distractor: Experiment1/gsm_enhanced_templates/ (100 distractor-injected)

Results saved as JSON following Experiment1's nested structure.
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
from cvr.utils import extract_gsm_final, answers_match

# load_original_examples and load_enhanced_distracted_examples are safe to import
# only when called (they use data_preparation.append_distractor, which requires google-genai).
# We do a lazy import inside functions that need them.
from cvr.model_adapter import CVRModelAdapter
from cvr.pipeline import CVRPipeline
from cvr.baselines.baseline_sc import run_sc_on_dataset
from cvr.utils import load_yaml_config
from cvr.evaluation.eval_distractor import compute_distractor_metrics, compute_distractor_metrics_by_type


def _build_verifier_once(config):
    """Build the HF or cloud verifier adapter once; returns None to fall back to the SLM."""
    hf_cfg = config.get("verifier_local_hf", {})
    cloud_cfg = config.get("verifier_cloud", {})
    if hf_cfg.get("enabled", False):
        from cvr.hf_verifier import build_hf_verifier
        v = build_hf_verifier(hf_cfg)
        print(f"  [CVR] Verifier loaded once: {v.model_key}")
        return v
    if cloud_cfg.get("enabled", False):
        from cvr.cloud_verifier import build_cloud_verifier
        v = build_cloud_verifier(cloud_cfg)
        print(f"  [CVR] Verifier loaded once: {v.model_key}")
        return v
    return None


def run_method_on_examples(
    method: str,
    examples: list[dict],
    adapter: CVRModelAdapter,
    config: dict,
    limit: Optional[int] = None,
    verifier_adapter=None,
) -> dict:
    """
    Run one method on a list of examples. Returns accuracy + result records.

    method: 'sc' | 'cvr_norel' | 'cvr_full'
    """
    if limit:
        examples = examples[:limit]

    if method == "sc":
        records = run_sc_on_dataset(
            adapter, examples,
            num_chains=config["generation"]["num_chains"],
            temperature=config["generation"]["temperature"],
            desc=method,
        )
    else:
        # CVR
        enable_rel = (method == "cvr_full")
        cfg = dict(config)
        cfg["verification"] = dict(config["verification"])
        cfg["verification"]["enable_relevance"] = enable_rel

        pipeline = CVRPipeline(adapter, cfg, verifier_adapter=verifier_adapter)
        records = []
        for ex in tqdm(examples, desc=f"{method}"):
            result = pipeline.solve(ex["question"])
            gold = extract_gsm_final(ex.get("answer", ""))
            is_correct = answers_match(gold or "", result.get("answer"))
            # Count relevance failures across all chains
            rel_failures = sum(
                c.get("relevance_failures", 0) for c in result.get("chains", [])
            )
            record = {
                "question": ex.get("question"),
                "question_original": ex.get("question_original"),
                "distractor": ex.get("distractor"),
                "distractor_type": ex.get("distractor_type"),
                "source_file": ex.get("source_file"),
                "id_orig": ex.get("id_orig"),
                "id_shuffled": ex.get("id_shuffled"),
                "ground_truth_extracted": gold,
                "predicted_extracted": result.get("answer"),
                "is_correct": is_correct,
                "is_collapse": result.get("answer") is None,
                "confidence": result.get("confidence", 0.0),
                "successful_chains": result.get("successful_chains", 0),
                "total_chains": result.get("total_chains", 0),
                "relevance_failures": rel_failures,
                "relevance_triggered": rel_failures > 0,
                "elapsed_s": result.get("elapsed_s", 0.0),
            }
            records.append(record)

    n_correct = sum(1 for r in records if r.get("is_correct"))
    n_eval = len(records)
    return {
        "accuracy": round(n_correct / n_eval, 4) if n_eval else 0.0,
        "n_correct": n_correct,
        "n_evaluated": n_eval,
        "results": records,
    }


def run_experiment1(  # noqa: C901
    original_dir: str = "Experiment1/gsm_templates",
    enhanced_dir: str = "Experiment1/gsm_enhanced_templates",
    config_path: str = "cvr/config.yaml",
    output_path: str = "cvr/results/exp1_distractor.json",
    device: str = "auto",
    model_filter: Optional[list[str]] = None,
    limit: Optional[int] = None,
    distractor_seed: int = 42,
):
    config = load_yaml_config(config_path)

    # Lazy import to avoid google-genai at module load time
    import sys as _sys
    _exp1 = str(_repo_root / "Experiment1")
    if _exp1 not in _sys.path:
        _sys.path.insert(0, _exp1)
    from run_inference_eval import load_original_examples, load_enhanced_distracted_examples

    original_examples = load_original_examples(original_dir)
    enhanced_examples = load_enhanced_distracted_examples(enhanced_dir, seed=distractor_seed)

    configs = MODEL_CONFIGS
    if model_filter:
        configs = [c for c in configs if c.short_name in model_filter]

    methods = ["cvr_full"]

    all_results: dict = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_path": config_path,
            "methods": methods,
            "model_filter": model_filter,
            "limit": limit,
            "distractor_seed": distractor_seed,
        },
    }

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Model: {cfg.short_name}")
        wrapper = get_model_wrapper(cfg, device=device)
        wrapper.load()
        adapter = CVRModelAdapter(wrapper)

        verifier_adapter = _build_verifier_once(config)

        model_results: dict = {}
        for method in methods:
            print(f"\n  Method: {method}")
            orig_out = run_method_on_examples(method, original_examples, adapter, config, limit, verifier_adapter=verifier_adapter)
            dist_out = run_method_on_examples(method, enhanced_examples, adapter, config, limit, verifier_adapter=verifier_adapter)
            dist_metrics = compute_distractor_metrics(orig_out["results"], dist_out["results"])
            by_type = compute_distractor_metrics_by_type(dist_out["results"])
            model_results[method] = {
                "original": orig_out,
                "enhanced": dist_out,
                "distractor_metrics": dist_metrics,
                "by_distractor_type": by_type,
            }
            print(f"    Clean acc: {orig_out['accuracy']:.3f}  |  Distractor acc: {dist_out['accuracy']:.3f}  |  Drop: {dist_metrics['accuracy_drop']:.3f}")

        all_results[cfg.short_name] = model_results
        wrapper.unload()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return all_results


def run_exp1_for_adapter(
    adapter,
    verifier_adapter,
    model_short_name: str,
    original_examples: list,
    enhanced_examples: list,
    config: dict,
    limit: Optional[int] = None,
) -> dict:
    """
    Run Exp1 methods against a pre-loaded adapter. No model loading/unloading.
    Returns the per-model results dict (keyed by method).
    """
    methods = ["cvr_full"]
    model_results: dict = {}
    for method in methods:
        print(f"\n  [Exp1] Method: {method}")
        orig_out = run_method_on_examples(method, original_examples, adapter, config, limit, verifier_adapter=verifier_adapter)
        dist_out = run_method_on_examples(method, enhanced_examples, adapter, config, limit, verifier_adapter=verifier_adapter)
        dist_metrics = compute_distractor_metrics(orig_out["results"], dist_out["results"])
        by_type = compute_distractor_metrics_by_type(dist_out["results"])
        model_results[method] = {
            "original": orig_out,
            "enhanced": dist_out,
            "distractor_metrics": dist_metrics,
            "by_distractor_type": by_type,
        }
        print(f"    Clean acc: {orig_out['accuracy']:.3f}  |  Distractor acc: {dist_out['accuracy']:.3f}  |  Drop: {dist_metrics['accuracy_drop']:.3f}")
    return model_results
