"""
inference.py — Run a trained FARN navigator on test problems.

Loads a trained navigator checkpoint and uses it to guide SLM
reasoning on unseen test problems. No reward model is needed
at inference time — the navigator runs its learned policy directly.

Usage:
    python inference.py \
        --model qwen25_math_1.5b \
        --navigator checkpoints/nav_qwen25_math_1.5b_best.pt \
        --dataset gsm8k \
        --max_problems 100
"""

import argparse
import json
import os
import time
import numpy as np
import torch
from typing import Dict, List, Optional

from config import get_config, FARNConfig, MODEL_REGISTRY, ACTION_NAMES
from navigator import Navigator
from environment import FARNEnvironment


def load_test_problems(config: FARNConfig) -> List[Dict]:
    """Load test-split problems from GSM8K.

    Returns:
        List of dicts with 'question' and 'answer' keys.
    """
    from datasets import load_dataset

    print("Loading test set...")
    ds = load_dataset(config.data.dataset_name, "main", split=config.data.test_split)

    problems = []
    for example in ds:
        answer_text = example["answer"]
        if "####" in answer_text:
            final_answer = answer_text.split("####")[-1].strip()
        else:
            final_answer = answer_text.strip()

        problems.append({
            "question": example["question"],
            "answer": final_answer,
        })

    print(f"Loaded {len(problems)} test problems.")
    return problems


def run_inference(
    config: FARNConfig,
    navigator_path: str,
    max_problems: Optional[int] = None,
    output_path: str = "results",
) -> Dict:
    """Run the trained navigator on test problems and collect results.

    Args:
        config: FARN configuration.
        navigator_path: Path to trained navigator checkpoint.
        max_problems: Maximum number of test problems to run.
        output_path: Directory to save results.

    Returns:
        Dict with aggregate metrics and per-problem results.
    """
    # ── Setup ──
    env = FARNEnvironment(config, reward_model=None)  # No reward at inference
    navigator = Navigator(config.navigator, device="cpu")
    navigator.load(navigator_path)

    problems = load_test_problems(config)
    if max_problems:
        problems = problems[:max_problems]

    # ── Run problems ──
    results = {
        "model": config.model.name,
        "navigator_path": navigator_path,
        "num_problems": len(problems),
        "problems": [],
    }

    correct_count = 0
    total_steps = 0
    start_time = time.time()

    for i, problem in enumerate(problems):
        # Reset environment
        state = env.reset(problem["question"], problem["answer"])

        actions_taken = []
        feature_trace = []

        # Run navigator (greedy — no exploration)
        for step in range(config.navigator.max_steps_per_episode):
            action = navigator.select_action(
                state=state,
                episode=0,
                force_greedy=True,           # No exploration at test time
                answer_present=(state[6] > 0.5),
                is_first_step=(step == 0),
            )
            actions_taken.append(ACTION_NAMES[action])

            next_state, _, done, info = env.step(action)

            # Log features for analysis
            feature_trace.append(info["feature_values"])

            state = next_state
            if done:
                break

        # Record result
        is_correct = info.get("correct", False)
        correct_count += int(is_correct)
        total_steps += len(actions_taken)

        problem_result = {
            "index": i,
            "question": problem["question"],
            "ground_truth": problem["answer"],
            "predicted": info.get("final_answer", ""),
            "correct": is_correct,
            "num_steps": len(actions_taken),
            "actions": actions_taken,
            "feature_trace": feature_trace,
        }
        results["problems"].append(problem_result)

        # Progress
        if (i + 1) % 20 == 0:
            running_acc = correct_count / (i + 1)
            elapsed = time.time() - start_time
            print(
                f"  [{i + 1}/{len(problems)}] "
                f"Accuracy: {running_acc:.1%} | "
                f"Avg steps: {total_steps / (i + 1):.1f} | "
                f"Time: {elapsed:.0f}s"
            )

    # ── Aggregate metrics ──
    total_time = time.time() - start_time
    accuracy = correct_count / len(problems)

    results["metrics"] = {
        "accuracy": accuracy,
        "correct": correct_count,
        "total": len(problems),
        "avg_steps": total_steps / len(problems),
        "total_time_seconds": round(total_time, 2),
        "seconds_per_problem": round(total_time / len(problems), 2),
    }

    # ── Action distribution ──
    from collections import Counter
    all_actions = []
    for p in results["problems"]:
        all_actions.extend(p["actions"])
    action_counts = Counter(all_actions)
    results["action_distribution"] = dict(action_counts)

    # ── Save results ──
    os.makedirs(output_path, exist_ok=True)
    result_file = os.path.join(
        output_path,
        f"inference_{config.model.name}.json",
    )
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    # ── Print summary ──
    print(f"\n{'=' * 50}")
    print(f"FARN Inference Results: {config.model.name}")
    print(f"{'=' * 50}")
    print(f"Accuracy: {accuracy:.1%} ({correct_count}/{len(problems)})")
    print(f"Avg steps per problem: {total_steps / len(problems):.1f}")
    print(f"Total time: {total_time:.0f}s")
    print(f"Action distribution: {dict(action_counts)}")
    print(f"Results saved to: {result_file}")

    return results


def run_baseline(
    config: FARNConfig,
    mode: str = "direct",
    max_problems: Optional[int] = None,
    output_path: str = "results",
) -> Dict:
    """Run baseline (no navigator) for comparison.

    Args:
        config: FARN configuration.
        mode: "direct" (no CoT) or "cot" (chain-of-thought).
        max_problems: Maximum problems to evaluate.
        output_path: Directory to save results.

    Returns:
        Dict with aggregate metrics.
    """
    env = FARNEnvironment(config, reward_model=None)
    problems = load_test_problems(config)
    if max_problems:
        problems = problems[:max_problems]

    correct_count = 0
    start_time = time.time()

    for i, problem in enumerate(problems):
        if mode == "direct":
            prompt = (
                f"Solve this math problem. Give only the final numerical answer.\n\n"
                f"Problem: {problem['question']}\n\nAnswer:"
            )
        else:  # cot
            prompt = (
                f"Solve this math problem step by step. Show your work, then give "
                f"the final answer.\n\n"
                f"Problem: {problem['question']}\n\nSolution:"
            )

        response = env.slm.generate(prompt)

        # Extract answer
        import re
        numbers = re.findall(r'[\d.,]+', response)
        if numbers:
            predicted = numbers[-1].replace(",", "")
        else:
            predicted = ""

        try:
            gt = problem["answer"].replace(",", "").replace("$", "").strip()
            if predicted and abs(float(predicted) - float(gt)) < 1e-6:
                correct_count += 1
        except ValueError:
            pass

        if (i + 1) % 50 == 0:
            print(f"  Baseline [{i + 1}/{len(problems)}]: {correct_count / (i + 1):.1%}")

    accuracy = correct_count / len(problems)
    total_time = time.time() - start_time

    results = {
        "model": config.model.name,
        "mode": mode,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": len(problems),
        "time_seconds": round(total_time, 2),
    }

    result_file = os.path.join(output_path, f"baseline_{mode}_{config.model.name}.json")
    os.makedirs(output_path, exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBaseline ({mode}) — {config.model.name}: {accuracy:.1%}")
    return results


# ── CLI Entry Point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run FARN inference or baseline")
    parser.add_argument(
        "--model", type=str, default="qwen25_math_1.5b",
        choices=list(MODEL_REGISTRY.keys()),
    )
    parser.add_argument(
        "--navigator", type=str, default=None,
        help="Path to trained navigator checkpoint. If None, runs baseline.",
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        choices=["direct", "cot"],
        help="Run baseline instead of navigator. 'direct' or 'cot'.",
    )
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    config = get_config(model_name=args.model)
    config.device = args.device

    if args.baseline:
        run_baseline(config, mode=args.baseline,
                     max_problems=args.max_problems, output_path=args.output)
    elif args.navigator:
        run_inference(config, navigator_path=args.navigator,
                      max_problems=args.max_problems, output_path=args.output)
    else:
        print("Specify --navigator <path> or --baseline <direct|cot>")


if __name__ == "__main__":
    main()
