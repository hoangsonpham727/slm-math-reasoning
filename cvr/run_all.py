"""
CVR Master Runner

Usage:
    # Sanity test: 5 problems, Qwen only
    python cvr/run_all.py --experiment sanity --model qwen25_math_1.5b --device cuda --limit 5

    # Full Experiment 1 (distractor robustness)
    python cvr/run_all.py --experiment 1 --model all --device auto

    # Full Experiment 2 (multi-step degradation)
    python cvr/run_all.py --experiment 2 --model all --device auto

    # Ablation (uses Qwen by default)
    python cvr/run_all.py --experiment 3 --model qwen25_math_1.5b --device auto

    # Run everything
    python cvr/run_all.py --experiment all --model all --device auto
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Load .env so OLLAMA_API_KEY (and other secrets) are available.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional; key can also be set via export OLLAMA_API_KEY=...

from models import MODEL_CONFIGS


def _model_filter(arg: str) -> list[str] | None:
    if arg == "all":
        return None
    known = {c.short_name for c in MODEL_CONFIGS}
    names = [n.strip() for n in arg.split(",")]
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Valid: {sorted(known)}")
    return names


class _DebugAdapter:
    """
    Wraps CVRModelAdapter for sanity mode: logs every prompt+response pair to
    a list and prints a compact trace to stdout.
    """

    def __init__(self, adapter, log: list, problem_idx: int):
        self._adapter = adapter
        self._log = log
        self._problem_idx = problem_idx
        self._call_idx = 0

    def _record(self, kind: str, system: str, user: str, response: str, **kw):
        entry = {
            "problem": self._problem_idx,
            "call": self._call_idx,
            "kind": kind,
            "system_prompt": system,
            "user_prompt": user,
            "response": response,
            **kw,
        }
        self._log.append(entry)
        # Compact stdout trace
        label = f"[P{self._problem_idx} #{self._call_idx} {kind}]"
        print(f"\n    {label}")
        print(f"      SYS : {system[:120]}")
        print(f"      USER: {user[-300:]}")  # tail of user prompt (contains the step)
        print(f"      OUT : {repr(response)}")
        self._call_idx += 1

    def generate_greedy(self, system_prompt, user_prompt, max_new_tokens=512):
        r = self._adapter.generate_greedy(system_prompt, user_prompt, max_new_tokens)
        self._record("greedy", system_prompt, user_prompt, r, max_new_tokens=max_new_tokens)
        return r

    def generate_sampled(self, system_prompt, user_prompt, max_new_tokens=256, temperature=0.7):
        r = self._adapter.generate_sampled(system_prompt, user_prompt, max_new_tokens, temperature)
        self._record("sampled", system_prompt, user_prompt, r,
                     max_new_tokens=max_new_tokens, temperature=temperature)
        return r

    @property
    def model_key(self):
        return self._adapter.model_key


def run_sanity(model_filter, device, limit=5):
    """Quick smoke-test: 5 depth-2 problems with full prompt/response logging."""
    import json
    from cvr.model_adapter import CVRModelAdapter
    from cvr.pipeline import CVRPipeline
    from cvr.utils import load_yaml_config

    config = load_yaml_config("cvr/config.yaml")
    config["generation"]["num_chains"] = 2

    models = MODEL_CONFIGS if not model_filter else [c for c in MODEL_CONFIGS if c.short_name in model_filter]
    if not models:
        print("No matching models found.")
        return

    problems_path = Path("Experiment2/data/problems_depth02.json")
    with open(problems_path) as f:
        problems = json.load(f)[:limit]

    from models import get_model_wrapper
    for cfg in models:
        print(f"\n{'='*60}\nSanity test: {cfg.short_name}")
        wrapper = get_model_wrapper(cfg, device=device)
        wrapper.load()
        base_adapter = CVRModelAdapter(wrapper)

        debug_log: list[dict] = []
        correct = 0

        for i, prob in enumerate(problems):
            print(f"\n  {'─'*56}")
            print(f"  Problem {i+1}: {prob['question']}")
            print(f"  GT: {prob['ground_truth']}")

            debug_adapter = _DebugAdapter(base_adapter, debug_log, problem_idx=i + 1)
            pipeline = CVRPipeline(debug_adapter, config)
            result = pipeline.solve(prob["question"])

            gt = float(prob["ground_truth"])
            pred = result.get("answer")
            try:
                pred_f = float(pred) if pred else None
            except (ValueError, TypeError):
                pred_f = None
            ok = pred_f is not None and abs(pred_f - gt) / (abs(gt) + 1e-9) < 0.01
            correct += ok

            n_chains = result.get("total_chains", 0)
            ok_chains = result.get("successful_chains", 0)
            restarts = sum(c.get("total_restarts", 0) for c in result.get("chains", []))
            first_chain_steps = len(result["chains"][0]["steps"]) if result.get("chains") else 0

            print(f"\n  ── Result ──")
            print(f"  Pred: {pred}  GT: {gt}  {'✓' if ok else '✗'}")
            print(f"  Chains: {ok_chains}/{n_chains} successful  |  Steps in chain 0: {first_chain_steps}  |  Restarts: {restarts}")

        # Save full debug log
        log_path = Path(f"cvr/sanity_debug_{cfg.short_name}.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(debug_log, f, indent=2)
        print(f"\n  Full prompt/response log saved to {log_path}")
        print(f"\n  Sanity result: {correct}/{len(problems)} correct")
        wrapper.unload()


def main():
    parser = argparse.ArgumentParser(description="CVR Master Runner")
    parser.add_argument("--experiment", choices=["sanity", "1", "2", "3", "all"],
                        default="sanity", help="Which experiment to run")
    parser.add_argument("--model", default="all",
                        help="Model(s) to run: 'all', 'qwen25_math_1.5b', 'gemma4_e2b', 'phi4_mini', or comma-separated")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cuda:0, cpu")
    parser.add_argument("--limit", type=int, default=None, help="Limit problems per split (for quick runs)")
    parser.add_argument("--output-dir", default="cvr/results", help="Output directory for results")
    parser.add_argument("--config", default="cvr/config.yaml", help="Path to CVR config YAML")
    parser.add_argument("--depths", default=None, help="Comma-separated depths for Exp2, e.g. 1,2,3,4")
    parser.add_argument("--ablation-model", default="qwen25_math_1.5b", help="Model to use for ablation (Exp3)")
    args = parser.parse_args()

    try:
        model_filter = _model_filter(args.model)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    depths = None
    if args.depths:
        depths = [int(d) for d in args.depths.split(",")]

    t0 = time.perf_counter()
    exp = args.experiment

    if exp in ("sanity", "all"):
        print("\n" + "="*60)
        print("SANITY TEST")
        run_sanity(model_filter, args.device, limit=args.limit or 5)

    if exp in ("1", "all"):
        print("\n" + "="*60)
        print("EXPERIMENT 1: Distractor Robustness")
        from cvr.experiments.exp1_distractor import run_experiment1
        run_experiment1(
            config_path=args.config,
            output_path=f"{args.output_dir}/exp1_distractor.json",
            device=args.device,
            model_filter=model_filter,
            limit=args.limit,
        )

    if exp in ("2", "all"):
        print("\n" + "="*60)
        print("EXPERIMENT 2: Multi-step Degradation")
        from cvr.experiments.exp2_multistep import run_experiment2
        run_experiment2(
            config_path=args.config,
            output_dir=args.output_dir,
            device=args.device,
            model_filter=model_filter,
            depths=depths,
            limit=args.limit,
        )

    if exp in ("3", "all"):
        print("\n" + "="*60)
        print("EXPERIMENT 3: Ablation Studies")
        from cvr.experiments.exp3_ablation import run_experiment3
        run_experiment3(
            config_path=args.config,
            output_dir=args.output_dir,
            device=args.device,
            model_name=args.ablation_model,
        )

    elapsed = round(time.perf_counter() - t0, 1)
    print(f"\n{'='*60}")
    print(f"Total time: {elapsed}s")


if __name__ == "__main__":
    main()
