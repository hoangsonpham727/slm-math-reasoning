"""
train.py — Navigator training loop.

Trains the Dueling DQN navigator by running episodes in the FARN
environment. Each episode:
  1. Sample a hard problem (one the SLM gets wrong with direct prompting)
  2. Navigator selects actions, SLM generates reasoning steps
  3. Feature extractor computes state, reward model scores the step
  4. Transitions are stored in replay buffer
  5. Navigator weights are updated via Double DQN

Usage:
    python train.py --model qwen25_math_1.5b --reward prm --episodes 500
    python train.py --model phi4_mini --reward ccqa --episodes 300
"""

import argparse
import json
import os
import random
import time
import numpy as np
import torch
from typing import Dict, List, Tuple

from config import get_config, FARNConfig, MODEL_REGISTRY
from navigator import Navigator
from replay_buffer import ReplayBuffer
from environment import FARNEnvironment, RewardModel


def load_gsm8k_problems(config: FARNConfig) -> List[Dict]:
    """Load GSM8K training problems.

    Each problem is a dict with keys 'question' and 'answer'.
    The answer field in GSM8K contains the reasoning chain followed
    by "#### <number>", so we extract the final number.

    Args:
        config: FARN configuration.

    Returns:
        List of problem dicts with 'question' and 'answer' keys.
    """
    from datasets import load_dataset

    print("Loading GSM8K dataset...")
    ds = load_dataset(config.data.dataset_name, "main", split=config.data.train_split)

    problems = []
    for example in ds:
        # Extract the final answer from "#### <number>"
        answer_text = example["answer"]
        # The answer is after the last "####"
        if "####" in answer_text:
            final_answer = answer_text.split("####")[-1].strip()
        else:
            final_answer = answer_text.strip()

        problems.append({
            "question": example["question"],
            "answer": final_answer,
        })

    print(f"Loaded {len(problems)} problems from GSM8K train split.")
    return problems


def filter_hard_problems(
    problems: List[Dict],
    env: FARNEnvironment,
    max_problems: int = 500,
) -> List[Dict]:
    """Filter to problems the SLM gets wrong with direct prompting.

    This follows RLoT's strategy: only train the navigator on problems
    where the SLM needs help. Easy problems don't provide useful
    training signal.

    Args:
        problems: All training problems.
        env: Environment with loaded SLM.
        max_problems: Maximum number of hard problems to keep.

    Returns:
        Filtered list of hard problems.
    """
    print("Filtering for hard problems (SLM gets wrong with direct prompting)...")
    hard = []

    # Shuffle to get a diverse sample
    shuffled = problems.copy()
    random.shuffle(shuffled)

    for i, prob in enumerate(shuffled):
        if len(hard) >= max_problems:
            break

        if i % 50 == 0:
            print(f"  Screened {i} problems, found {len(hard)} hard ones...")

        # Test with direct prompting (no navigator)
        prompt = f"Solve this math problem. Give only the final numerical answer.\n\nProblem: {prob['question']}\n\nAnswer:"
        response = env.slm.generate(prompt)

        # Check if the SLM got it right
        try:
            # Extract number from response
            import re
            numbers = re.findall(r'[\d.,]+', response)
            if numbers:
                slm_answer = numbers[-1].replace(",", "")
                gt_answer = prob["answer"].replace(",", "").replace("$", "")
                if abs(float(slm_answer) - float(gt_answer)) < 1e-6:
                    continue  # SLM got it right — skip
        except (ValueError, IndexError):
            pass

        hard.append(prob)

    print(f"Found {len(hard)} hard problems out of {min(len(shuffled), max_problems * 3)} screened.")
    return hard


def train(config: FARNConfig) -> None:
    """Main training loop.

    Args:
        config: Full FARN configuration.
    """
    # ── Setup ──
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Initialise components
    reward_model = RewardModel(config)
    env = FARNEnvironment(config, reward_model=reward_model)
    navigator = Navigator(config.navigator, device="cpu")  # Navigator always on CPU
    buffer = ReplayBuffer(config.navigator.replay_buffer_size)

    # Load and filter problems
    all_problems = load_gsm8k_problems(config)
    hard_problems = filter_hard_problems(
        all_problems, env, config.data.max_train_problems
    )

    if not hard_problems:
        print("ERROR: No hard problems found. The SLM may be too strong for "
              "this dataset, or the answer extraction is failing.")
        return

    # ── Training log ──
    log = {
        "model": config.model.name,
        "reward_type": config.reward.reward_type,
        "num_hard_problems": len(hard_problems),
        "episodes": [],
    }

    # ── Training loop ──
    print(f"\nStarting training: {config.navigator.num_episodes} episodes")
    print(f"  Model: {config.model.name}")
    print(f"  Reward: {config.reward.reward_type}")
    print(f"  Hard problems: {len(hard_problems)}")
    print(f"  Navigator params: {navigator.online_net.count_parameters()}")
    print()

    best_avg_reward = -float("inf")

    for episode in range(config.navigator.num_episodes):
        episode_start = time.time()

        # Sample a random hard problem
        problem = random.choice(hard_problems)
        state = env.reset(problem["question"], problem["answer"])

        episode_reward = 0.0
        episode_steps = 0
        actions_taken = []

        # ── Episode loop ──
        for step in range(config.navigator.max_steps_per_episode):
            # Navigator selects action
            action = navigator.select_action(
                state=state,
                episode=episode,
                answer_present=(state[6] > 0.5),   # f7
                is_first_step=(step == 0),
            )
            actions_taken.append(action)

            # Environment executes action
            next_state, reward, done, info = env.step(action)

            # Store transition
            buffer.push(state, action, reward, next_state, done)

            episode_reward += reward
            episode_steps += 1
            state = next_state

            # ── DQN update ──
            if len(buffer) >= config.navigator.batch_size:
                batch = buffer.sample(config.navigator.batch_size)
                loss = navigator.update(*batch)

            if done:
                break

        # ── Post-episode bookkeeping ──

        # Update target network periodically
        if episode % config.navigator.target_update_freq == 0:
            navigator.sync_target()

        # Logging
        episode_time = time.time() - episode_start
        epsilon = navigator._get_epsilon(episode)

        episode_log = {
            "episode": episode,
            "reward": episode_reward,
            "steps": episode_steps,
            "epsilon": epsilon,
            "correct": info.get("correct", False),
            "actions": [config_action_name(a) for a in actions_taken],
            "time_seconds": round(episode_time, 2),
        }
        log["episodes"].append(episode_log)

        # Print progress
        if episode % 10 == 0:
            recent = log["episodes"][-10:]
            avg_reward = np.mean([e["reward"] for e in recent])
            avg_correct = np.mean([e["correct"] for e in recent])

            print(
                f"Episode {episode:4d} | "
                f"Reward: {episode_reward:.3f} | "
                f"Avg(10): {avg_reward:.3f} | "
                f"Correct: {avg_correct:.1%} | "
                f"Steps: {episode_steps} | "
                f"ε: {epsilon:.3f} | "
                f"Buffer: {len(buffer)} | "
                f"Time: {episode_time:.1f}s"
            )

        # Save checkpoint
        if (episode + 1) % config.navigator.save_every == 0:
            ckpt_path = os.path.join(
                config.navigator.checkpoint_dir,
                f"nav_{config.model.name}_ep{episode + 1}.pt",
            )
            navigator.save(ckpt_path)

            # Check if this is the best model
            recent = log["episodes"][-50:]
            avg_reward = np.mean([e["reward"] for e in recent])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_path = os.path.join(
                    config.navigator.checkpoint_dir,
                    f"nav_{config.model.name}_best.pt",
                )
                navigator.save(best_path)

    # ── Save final model and log ──
    final_path = os.path.join(
        config.navigator.checkpoint_dir,
        f"nav_{config.model.name}_final.pt",
    )
    navigator.save(final_path)

    log_path = os.path.join(
        config.navigator.checkpoint_dir,
        f"training_log_{config.model.name}.json",
    )
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining log saved to {log_path}")

    print("\nTraining complete!")


def config_action_name(action_idx: int) -> str:
    """Convert action index to name (safe version)."""
    from config import ACTION_NAMES
    if 0 <= action_idx < len(ACTION_NAMES):
        return ACTION_NAMES[action_idx]
    return f"unknown_{action_idx}"


# ── CLI Entry Point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train the FARN navigator")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen25_math_1.5b",
        choices=list(MODEL_REGISTRY.keys()),
        help="Which SLM to train the navigator for",
    )
    parser.add_argument(
        "--reward",
        type=str,
        default="prm",
        choices=["prm", "ccqa"],
        help="Reward signal: 'prm' (Math-Shepherd) or 'ccqa' (cycle-consistency)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for SLM inference",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    config = get_config(model_name=args.model, reward_type=args.reward)
    config.navigator.num_episodes = args.episodes
    config.device = args.device
    config.seed = args.seed

    train(config)


if __name__ == "__main__":
    main()
