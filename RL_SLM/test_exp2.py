"""
Test RL agent on Experiment 2 held-out set (reasoning depth).

Loads the trained DQN policy for each model and runs greedy inference (ε=0)
on all 1,600 EXP2 problems (depths 1–8, idx 0–199 per depth).
Reports per-model and per-depth accuracy and saves full results.

Usage:
    python test_exp2.py \\
        --models qwen25_math_1.5b gemma4_e2b phi4_mini \\
        --checkpoint_dir ./checkpoints \\
        --output ./results/test_exp2_results.json
"""
import argparse
import gc
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_rl_dir  = Path(__file__).resolve().parent
_env_dir = _rl_dir / "environment"
for _p in [str(_rl_dir), str(_env_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from environment.Dueling_DQN_net import Dueling_DQN
from environment.RL_env import RLEnv


def parse_args():
    p = argparse.ArgumentParser(description="Test RL agent on Experiment 2")
    p.add_argument("--models",         nargs="+",
                   default=["qwen25_math_1.5b", "gemma4_e2b", "phi4_mini"])
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--output",         default="./results/test_exp2_results.json")
    p.add_argument("--data_dir",       default=None)
    p.add_argument("--eval_model",     default="gpt-oss:120b")
    p.add_argument("--eval_base_url",  default=None)
    p.add_argument("--eval_api_key",   default=None)
    p.add_argument("--max_depth",      type=int, default=5)
    p.add_argument("--max_width",      type=int, default=5)
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


def test_one_model(model_name: str, args) -> dict:
    """Run greedy DQN policy on all EXP2 problems. Returns result dict."""
    ckpt_path = os.path.join(args.checkpoint_dir, model_name, "policy_final.pt")
    if not os.path.exists(ckpt_path):
        print(f"[test_exp2] WARNING: checkpoint not found at {ckpt_path}, skipping {model_name}")
        return {"model": model_name, "accuracy": None, "error": "checkpoint not found", "records": []}

    eval_config = {
        "model_name": args.eval_model,
        "base_url":   args.eval_base_url,
        "api_key":    args.eval_api_key,
    }

    env = RLEnv(
        dataset        = "EXP2",
        is_test        = True,
        LLM_name       = model_name,
        problem_indexs = None,
        max_depth      = args.max_depth,
        max_width      = args.max_width,
        random_problems= False,
        random_seed    = args.seed,
        eval_config    = eval_config,
        data_dir       = args.data_dir,
    )

    device = torch.device(args.device)
    policy = Dueling_DQN(input_size=7, output_size=5).to(device)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.eval()

    correct = 0
    records = []
    depth_correct  = defaultdict(int)
    depth_total    = defaultdict(int)

    pbar = tqdm(range(env.total_problems), desc=f"EXP2 {model_name}", unit="prob")
    for i in pbar:
        state, finished = env.reset()
        if finished:
            break

        record = env.ds[i % env.total_problems]
        depth  = record.get("depth", 0)

        done         = False
        final_reward = 0.0
        actions_taken = []

        while not done:
            with torch.no_grad():
                s_t    = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = policy(s_t).argmax(dim=1).item()
            actions_taken.append(action)
            state, final_reward, done = env.step(action)

        is_correct = bool(final_reward)
        correct += int(is_correct)
        depth_correct[depth] += int(is_correct)
        depth_total[depth]   += 1
        pbar.set_postfix(correct=correct, acc=f"{correct/max(len(records)+1,1):.2f}", depth=depth)

        records.append({
            "problem_id":   record.get("problem_id", f"prob_{i}"),
            "depth":        depth,
            "problem":      env.problem,
            "ground_truth": float(env.ans),
            "is_correct":   is_correct,
            "actions":      actions_taken,
        })

    pbar.close()
    accuracy = correct / max(len(records), 1)
    per_depth = {
        d: {"correct": depth_correct[d], "total": depth_total[d],
            "accuracy": depth_correct[d] / max(depth_total[d], 1)}
        for d in sorted(depth_total)
    }

    print(f"[test_exp2] {model_name}: overall {correct}/{len(records)} = {accuracy:.3f}")
    for d, stats in per_depth.items():
        print(f"            depth {d}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.3f}")

    del env, policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model":     model_name,
        "dataset":   "EXP2",
        "accuracy":  accuracy,
        "n_correct": correct,
        "n_total":   len(records),
        "per_depth": per_depth,
        "records":   records,
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    all_results = []
    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"[test_exp2] Model: {model_name}")
        print(f"{'='*60}")
        result = test_one_model(model_name, args)
        all_results.append(result)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[test_exp2] Results saved → {args.output}")

    print("\n── Summary ──────────────────────────────")
    for r in all_results:
        acc = f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "N/A"
        print(f"  {r['model']:30s}  EXP2 overall accuracy: {acc}")


if __name__ == "__main__":
    main()
