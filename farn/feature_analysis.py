"""
feature_analysis.py — Standalone feature correlation analysis.

PURPOSE
-------
This script validates the 8-feature state representation BEFORE any
navigator training. It answers one question:

    Do the externally computed features actually correlate with the
    failure modes we diagnosed in Phase 1?

Specifically:
  - Noise-awareness features (f1, f2, f3) should degrade predictably
    when distractor sentences are present (Exp 1 setup)
  - Depth-awareness features (f4, f5, f6) should degrade predictably
    as step count increases past the model's ceiling (Exp 2 setup)

This analysis is a standalone publishable result. Even if the navigator
training doesn't work perfectly, showing that the features are
informative of failure is a contribution.

WHAT IT PRODUCES
----------------
  1. Per-feature statistics (mean, std) across distractor vs clean problems
  2. Per-feature statistics across step depth levels
  3. Pearson correlation between each feature and correctness
  4. Scatter plots saved as PNGs
  5. A summary JSON with all statistics

Usage:
    python feature_analysis.py \
        --model qwen25_math_1.5b \
        --dataset gsm8k \
        --n_problems 200 \
        --output_dir analysis/

    # To also run on your Exp1 distractor dataset:
    python feature_analysis.py \
        --model qwen25_math_1.5b \
        --exp1_file data/gsm_symbolic_distractor.json \
        --exp2_file data/step_controlled.json \
        --output_dir analysis/
"""

import argparse
import json
import os
import re
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from feature_extractor import FeatureExtractor
from config import get_config, MODEL_REGISTRY


# ── Synthetic Data Generation ────────────────────────────────────────
# These functions create controlled test cases for analysis without
# needing the SLM — they generate synthetic "SLM outputs" from
# templates so the analysis can run on CPU without any GPU.

def make_clean_problem() -> Tuple[str, List[str]]:
    """Create a clean 3-step GSM8K-style problem with ideal SLM output.

    Returns:
        (problem_text, list_of_step_outputs)
    """
    problem = (
        "Tom has 24 apples. He gives 8 to his friend and then buys "
        "3 more bags each containing 5 apples. How many apples does "
        "Tom have now?"
    )
    steps = [
        "Tom starts with 24 apples. After giving 8 to his friend, he has 24 - 8 = 16 apples.",
        "Tom buys 3 bags with 5 apples each, so he gets 3 * 5 = 15 more apples.",
        "Total apples = 16 + 15 = 31. The answer is #### 31",
    ]
    return problem, steps


def make_distractor_problem() -> Tuple[str, List[str]]:
    """Create the same problem with an irrelevant distractor entity.

    The distractor "7 oranges" is mentioned but should not appear in
    any calculation. A noise-sensitive SLM might try to use it.
    """
    problem = (
        "Tom has 24 apples. He also has 7 oranges which he is saving "
        "for a party. He gives 8 apples to his friend and then buys "
        "3 more bags each containing 5 apples. How many apples does "
        "Tom have now?"
    )
    # Noise-affected SLM tries to incorporate the distractor
    steps_affected = [
        "Tom has 24 apples and 7 oranges. He gives 8 apples away, leaving 24 - 8 = 16. "
        "He also has 7 oranges.",
        "He buys 3 * 5 = 15 apples. Total fruit = 16 + 15 + 7 = 38.",
        "The answer is #### 38",
    ]
    return problem, steps_affected


def make_correct_distractor_steps() -> List[str]:
    """Steps for the distractor problem where SLM correctly ignores distractor."""
    return [
        "Tom has 24 apples. He gives 8 to his friend: 24 - 8 = 16 apples.",
        "Tom buys 3 bags with 5 apples each: 3 * 5 = 15 apples.",
        "Total apples = 16 + 15 = 31. The answer is #### 31",
    ]


def make_deep_chain_problem(num_steps: int) -> Tuple[str, List[str]]:
    """Create a chain-of-operations problem with num_steps steps.

    Each step applies one arithmetic operation to the previous result.
    We simulate SLM outputs with controlled error injection at step >= 4.
    """
    problem = (
        f"Start with 100. "
        + " ".join(
            [f"Step {i}: {'add' if i % 2 == 1 else 'subtract'} {10 + i}."
             for i in range(1, num_steps + 1)]
        )
        + " What is the final result?"
    )

    # Build correct steps
    value = 100
    steps = []
    for i in range(1, num_steps + 1):
        delta = 10 + i
        if i % 2 == 1:
            new_value = value + delta
            expr = f"{value} + {delta} = {new_value}"
        else:
            new_value = value - delta
            expr = f"{value} - {delta} = {new_value}"

        # Inject arithmetic error at step 4+ to simulate depth collapse
        if i >= 4:
            wrong_value = new_value + 3  # Introduce a small error
            expr = f"{value} + {delta} = {wrong_value}"  # Wrong operation sign too
            new_value = wrong_value

        steps.append(f"Step {i}: {expr}. Current value: {new_value}.")
        value = new_value

    return problem, steps


# ── Feature Extraction Loop ──────────────────────────────────────────

def extract_features_for_problem(
    problem_text: str,
    step_outputs: List[str],
) -> List[Dict]:
    """Run the feature extractor over all steps for one problem.

    Args:
        problem_text: The math word problem.
        step_outputs: List of SLM output strings (one per step).

    Returns:
        List of feature dicts (one per step), each containing:
          feature name -> float value
    """
    extractor = FeatureExtractor(problem_text)
    feature_trace = []

    for step_idx, step_output in enumerate(step_outputs):
        state = extractor.extract(step_output, step_idx)
        feature_dict = dict(zip(extractor.feature_names, state.tolist()))
        feature_trace.append(feature_dict)

    return feature_trace


def aggregate_traces(traces: List[List[Dict]]) -> Dict[str, List[float]]:
    """Flatten a list of feature traces into per-feature value lists.

    Args:
        traces: List of traces, each trace is a list of feature dicts.

    Returns:
        Dict mapping feature_name -> flat list of all values across
        all steps and all traces.
    """
    result = defaultdict(list)
    for trace in traces:
        for step_features in trace:
            for fname, fval in step_features.items():
                result[fname].append(fval)
    return dict(result)


# ── Statistical Tests ────────────────────────────────────────────────

def pearson_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson r between two lists of equal length."""
    if len(x) < 2:
        return 0.0
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def cohens_d(group_a: List[float], group_b: List[float]) -> float:
    """Compute Cohen's d effect size between two groups.

    Positive d means group_a has higher mean.
    Values: small ≈ 0.2, medium ≈ 0.5, large ≈ 0.8
    """
    if not group_a or not group_b:
        return 0.0
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    pooled_std = np.sqrt(
        (np.std(group_a, ddof=1) ** 2 + np.std(group_b, ddof=1) ** 2) / 2
    )
    if pooled_std == 0:
        return 0.0
    return float((mean_a - mean_b) / pooled_std)


def compare_feature_groups(
    group_a_traces: List[List[Dict]],
    group_b_traces: List[List[Dict]],
    label_a: str = "clean",
    label_b: str = "distractor",
) -> Dict:
    """Compare feature distributions between two groups of traces.

    Args:
        group_a_traces: Feature traces from condition A (e.g., clean problems).
        group_b_traces: Feature traces from condition B (e.g., with distractors).
        label_a, label_b: Names for the groups.

    Returns:
        Dict with per-feature statistics including mean, std, Cohen's d.
    """
    vals_a = aggregate_traces(group_a_traces)
    vals_b = aggregate_traces(group_b_traces)

    all_features = sorted(set(vals_a) | set(vals_b))
    comparison = {}

    for fname in all_features:
        a = vals_a.get(fname, [])
        b = vals_b.get(fname, [])

        comparison[fname] = {
            f"mean_{label_a}": float(np.mean(a)) if a else None,
            f"std_{label_a}": float(np.std(a)) if a else None,
            f"mean_{label_b}": float(np.mean(b)) if b else None,
            f"std_{label_b}": float(np.std(b)) if b else None,
            "cohens_d": cohens_d(a, b),
            "delta_mean": (float(np.mean(a)) - float(np.mean(b))) if (a and b) else None,
        }

    return comparison


# ── Experiment 1 Analysis: Distractor Sensitivity ───────────────────

def run_exp1_analysis(n_problems: int = 50) -> Dict:
    """Analyse feature behaviour on clean vs. distractor-augmented problems.

    Uses synthetic problems to avoid requiring the SLM.

    Returns:
        Dict with comparison statistics and summary.
    """
    print("\n── Experiment 1: Noise-awareness feature analysis ──")

    clean_traces = []
    distractor_affected_traces = []
    distractor_resilient_traces = []

    for i in range(n_problems):
        # Clean problem
        problem, steps = make_clean_problem()
        trace = extract_features_for_problem(problem, steps)
        clean_traces.append(trace)

        # Distractor problem, noise-affected SLM (uses distractor)
        dist_problem, dist_steps = make_distractor_problem()
        dist_trace = extract_features_for_problem(dist_problem, dist_steps)
        distractor_affected_traces.append(dist_trace)

        # Distractor problem, noise-resilient SLM (ignores distractor)
        resilient_steps = make_correct_distractor_steps()
        resilient_trace = extract_features_for_problem(dist_problem, resilient_steps)
        distractor_resilient_traces.append(resilient_trace)

    # Compare clean vs. noise-affected
    affected_comparison = compare_feature_groups(
        clean_traces, distractor_affected_traces,
        label_a="clean", label_b="noise_affected",
    )

    # Compare clean vs. noise-resilient
    resilient_comparison = compare_feature_groups(
        clean_traces, distractor_resilient_traces,
        label_a="clean", label_b="noise_resilient",
    )

    print(f"\n  {n_problems} clean problems vs {n_problems} distractor-affected problems:")
    print(f"  {'Feature':<30} {'Clean mean':>12} {'Affected mean':>14} {'Cohen d':>8}")
    print(f"  {'-'*64}")

    for fname, stats in affected_comparison.items():
        d = stats["cohens_d"]
        flag = " ◄ LARGE EFFECT" if abs(d) >= 0.8 else (" ◄ medium" if abs(d) >= 0.5 else "")
        print(
            f"  {fname:<30} "
            f"{stats['mean_clean']:>12.3f} "
            f"{stats['mean_noise_affected']:>14.3f} "
            f"{d:>8.3f}{flag}"
        )

    return {
        "affected_vs_clean": affected_comparison,
        "resilient_vs_clean": resilient_comparison,
        "n_problems": n_problems,
    }


# ── Experiment 2 Analysis: Depth Sensitivity ─────────────────────────

def run_exp2_analysis(max_steps: int = 8) -> Dict:
    """Analyse feature behaviour as reasoning chain depth increases.

    Simulates chains of length 1..max_steps with arithmetic error
    injection at step >= 4 (mimicking the empirical depth ceiling).

    Returns:
        Dict with per-step feature means and correlation with correctness.
    """
    print("\n── Experiment 2: Depth-awareness feature analysis ──")

    # For each chain length, collect features at the last step
    step_traces: Dict[int, List[Dict]] = defaultdict(list)
    correctness_by_length: Dict[int, List[float]] = defaultdict(list)

    n_per_length = 20

    for chain_length in range(1, max_steps + 1):
        for _ in range(n_per_length):
            problem, steps = make_deep_chain_problem(chain_length)
            trace = extract_features_for_problem(problem, steps)
            # We care about the LAST step's features (where errors accumulate)
            if trace:
                last_features = trace[-1]
                step_traces[chain_length].append(last_features)
            # Mark as "correct" if chain length <= 3 (no error injection)
            correct = 1.0 if chain_length <= 3 else 0.0
            correctness_by_length[chain_length].append(correct)

    # Per-step feature means
    step_means = {}
    for length, feature_dicts in step_traces.items():
        feature_means = {}
        for fname in next(iter(feature_dicts), {}).keys():
            vals = [fd[fname] for fd in feature_dicts if fname in fd]
            feature_means[fname] = float(np.mean(vals)) if vals else 0.0
        step_means[length] = feature_means

    # Correlation between each feature and correctness
    print(f"\n  Per-step feature means (chain length 1 to {max_steps}):")
    feature_names = list(next(iter(step_means.values()), {}).keys())

    # Header
    col_width = 12
    header = f"  {'Feature':<30}" + "".join(f"  {i:>{col_width}}" for i in range(1, max_steps + 1))
    print(header)
    print(f"  {'-' * (30 + col_width * max_steps + 4)}")

    for fname in feature_names:
        row_vals = [step_means.get(l, {}).get(fname, float('nan'))
                    for l in range(1, max_steps + 1)]
        row_str = f"  {fname:<30}" + "".join(f"  {v:>{col_width}.3f}" for v in row_vals)
        print(row_str)

    # Correlations with correctness
    print(f"\n  Feature-correctness correlations across chain lengths:")
    print(f"  {'Feature':<30} {'Pearson r':>10}  Interpretation")
    print(f"  {'-'*70}")

    # Build flat lists for correlation
    flat_features: Dict[str, List[float]] = defaultdict(list)
    flat_correct: List[float] = []

    for length in sorted(step_traces.keys()):
        correct_val = float(np.mean(correctness_by_length[length]))
        for fd in step_traces[length]:
            for fname, fval in fd.items():
                flat_features[fname].append(fval)
            flat_correct.append(correct_val)

    correlations = {}
    for fname in feature_names:
        r = pearson_correlation(flat_features[fname], flat_correct)
        correlations[fname] = r

        direction = "↑ higher = more correct" if r > 0 else "↓ lower = more correct"
        strength = "strong" if abs(r) >= 0.5 else ("moderate" if abs(r) >= 0.3 else "weak")
        flag = " ◄" if abs(r) >= 0.3 else ""
        print(f"  {fname:<30} {r:>10.3f}  {strength}, {direction}{flag}")

    return {
        "step_means": {str(k): v for k, v in step_means.items()},
        "correlations": correlations,
        "max_steps": max_steps,
    }


# ── GSM8K-based Analysis (with real SLM outputs) ────────────────────

def run_gsm8k_analysis(
    model_name: str,
    n_problems: int = 100,
    output_dir: str = "analysis",
) -> Dict:
    """Run feature analysis on real GSM8K problems with SLM outputs.

    This version requires the SLM to be loaded. It runs direct prompting
    (no navigator), collects the SLM's output, extracts features, and
    checks correctness. The result shows which feature values correlate
    with getting the right answer.

    NOTE: This takes GPU time. For a quick analysis without the SLM,
    use run_exp1_analysis() and run_exp2_analysis() instead.

    Args:
        model_name: SLM identifier.
        n_problems: Number of GSM8K test problems to analyse.
        output_dir: Directory to save results.

    Returns:
        Dict with feature statistics and correlations.
    """
    from datasets import load_dataset
    from environment import SLMWrapper
    from config import get_config

    config = get_config(model_name=model_name)
    slm = SLMWrapper(config)

    print(f"\nRunning GSM8K feature analysis for {model_name}...")
    print(f"Loading {n_problems} test problems...")

    ds = load_dataset("openai/gsm8k", "main", split="test")

    problems_data = []
    for i, ex in enumerate(ds):
        if i >= n_problems:
            break
        gt = ex["answer"].split("####")[-1].strip() if "####" in ex["answer"] else ex["answer"]
        problems_data.append({"question": ex["question"], "answer": gt})

    feature_records = []   # One record per problem
    correct_labels = []    # 1.0 if correct, 0.0 otherwise

    for i, prob in enumerate(problems_data):
        # Prompt SLM with chain-of-thought
        prompt = (
            "Solve this math problem step by step. Show all calculations clearly.\n\n"
            f"Problem: {prob['question']}\n\nSolution:"
        )
        slm_output = slm.generate(prompt)

        # Extract features from the full output (treat as one "step")
        extractor = FeatureExtractor(prob["question"])
        state = extractor.extract(slm_output, step_number=0)
        feature_dict = dict(zip(extractor.feature_names, state.tolist()))

        # Check correctness
        numbers = re.findall(r'[\d,]+\.?\d*', slm_output)
        correct = False
        if numbers:
            try:
                predicted = numbers[-1].replace(",", "")
                gt = prob["answer"].replace(",", "").replace("$", "")
                correct = abs(float(predicted) - float(gt)) < 1e-6
            except ValueError:
                pass

        feature_records.append(feature_dict)
        correct_labels.append(1.0 if correct else 0.0)

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{n_problems}, "
                  f"accuracy so far: {np.mean(correct_labels):.1%}")

    # Compute correlations
    feature_names = list(feature_records[0].keys())
    correlations = {}
    for fname in feature_names:
        vals = [r[fname] for r in feature_records]
        r = pearson_correlation(vals, correct_labels)
        correlations[fname] = r

    print(f"\n  Feature-correctness correlations on GSM8K ({model_name}):")
    print(f"  {'Feature':<30} {'Pearson r':>10}  Effect")
    print(f"  {'-'*55}")
    for fname, r in sorted(correlations.items(), key=lambda x: -abs(x[1])):
        flag = " ◄ INFORMATIVE" if abs(r) >= 0.2 else ""
        print(f"  {fname:<30} {r:>10.3f}{flag}")

    results = {
        "model": model_name,
        "n_problems": n_problems,
        "overall_accuracy": float(np.mean(correct_labels)),
        "feature_correlations_with_correct": correlations,
        "feature_means_correct": {
            fname: float(np.mean([feature_records[i][fname]
                                  for i, c in enumerate(correct_labels) if c > 0.5]))
            for fname in feature_names
        },
        "feature_means_incorrect": {
            fname: float(np.mean([feature_records[i][fname]
                                  for i, c in enumerate(correct_labels) if c < 0.5]))
            for fname in feature_names
        },
    }

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"feature_analysis_{model_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    return results


# ── Print Full Summary ───────────────────────────────────────────────

def print_analysis_summary(exp1: Dict, exp2: Dict) -> None:
    """Print a tidy summary of both experiments' findings."""
    print("\n")
    print("=" * 70)
    print("  FEATURE ANALYSIS SUMMARY")
    print("=" * 70)

    print("\n  NOISE-AWARENESS FEATURES (should degrade with distractors):")
    noise_features = ["f1_relevance_ratio", "f2_unused_entities", "f3_entity_op_alignment"]
    for fname in noise_features:
        stats = exp1["affected_vs_clean"].get(fname, {})
        d = stats.get("cohens_d", 0)
        verdict = "✓ WORKS" if abs(d) >= 0.5 else "△ weak"
        print(f"    {fname:<34} Cohen d = {d:+.3f}  {verdict}")

    print("\n  DEPTH-AWARENESS FEATURES (should degrade with chain length):")
    depth_features = ["f4_step_progress", "f5_arith_verify", "f6_bounds_plausible"]
    for fname in depth_features:
        r = exp2["correlations"].get(fname, 0)
        verdict = "✓ WORKS" if abs(r) >= 0.3 else "△ weak"
        print(f"    {fname:<34} Pearson r = {r:+.3f}  {verdict}")

    print("\n  PROGRESS FEATURES (structural, expected to correlate weakly):")
    progress_features = ["f7_answer_present", "f8_step_count"]
    for fname in progress_features:
        r = exp2["correlations"].get(fname, 0)
        print(f"    {fname:<34} Pearson r = {r:+.3f}")

    print()


# ── CLI Entry Point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Standalone feature correlation analysis for FARN"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="If provided, also runs GSM8K analysis with real SLM outputs (requires GPU)",
    )
    parser.add_argument(
        "--n_problems", type=int, default=50,
        help="Number of synthetic problems for Exp1/Exp2 analysis",
    )
    parser.add_argument(
        "--n_gsm8k", type=int, default=100,
        help="Number of GSM8K problems for real-SLM analysis",
    )
    parser.add_argument(
        "--max_steps", type=int, default=8,
        help="Maximum chain length for Exp2 depth analysis",
    )
    parser.add_argument(
        "--output_dir", type=str, default="analysis",
        help="Directory to save analysis results",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Run synthetic analyses (always, no GPU needed) ──
    exp1_results = run_exp1_analysis(n_problems=args.n_problems)
    exp2_results = run_exp2_analysis(max_steps=args.max_steps)

    print_analysis_summary(exp1_results, exp2_results)

    # Save
    with open(os.path.join(args.output_dir, "exp1_noise_analysis.json"), "w") as f:
        json.dump(exp1_results, f, indent=2)
    with open(os.path.join(args.output_dir, "exp2_depth_analysis.json"), "w") as f:
        json.dump(exp2_results, f, indent=2)
    print(f"Synthetic analysis results saved to: {args.output_dir}/")

    # ── Optional: real SLM analysis ──
    if args.model:
        gsm8k_results = run_gsm8k_analysis(
            model_name=args.model,
            n_problems=args.n_gsm8k,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
