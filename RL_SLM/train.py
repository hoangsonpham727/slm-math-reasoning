"""
RL-of-Thoughts training entry point for SLM Math Reasoning.

Wires together:
  - RLEnv (dataset + ENV reasoning loop with Ollama evaluator for state)
  - Dueling DQN (policy + target networks)
  - ExperienceReplay buffer
  - Reward model selected via --reward_type {prm,ccqa,programmatic}

The reward model swap is handled entirely via PRM.get_reward_model();
the DQN training loop itself is not reward-model-aware.

Usage:
    python train.py \\
        --dataset GSM8K \\
        --model qwen25_math_1.5b \\
        --reward_type programmatic \\
        --eval_model gpt-oss:120b \\
        --n_episodes 500
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
_rl_dir  = Path(__file__).resolve().parent
_env_dir = _rl_dir / "environment"
for _p in [str(_rl_dir), str(_env_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from DQN_buffer import ExperienceReplay, Experience
from environment.Dueling_DQN_net import Dueling_DQN
from environment.RL_env import RLEnv
from PRM import get_reward_model


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="RLoT SLM Training")

    # Environment
    p.add_argument("--dataset",       default="GSM8K",
                   choices=["GSM8K", "MATH", "GPQA", "MMLU-STEM", "StrategyQA", "EXP1", "EXP2"])
    p.add_argument("--model",         default="qwen25_math_1.5b",
                   help="Model short_name or HF model_id (see models.py)")
    p.add_argument("--data_dir",      default=None,
                   help="Repo root for EXP1/EXP2 data; auto-detected if omitted")
    p.add_argument("--max_depth",     type=int, default=5)
    p.add_argument("--max_width",     type=int, default=5)

    # Ollama evaluator (state scoring)
    p.add_argument("--eval_model",    default="gpt-oss:120b")
    p.add_argument("--eval_base_url", default=None,
                   help="Ollama base URL; reads OLLAMA_URL env var if omitted")
    p.add_argument("--eval_api_key",  default=None,
                   help="Ollama API key; reads OLLAMA_API_KEY env var if omitted")

    # Reward model
    p.add_argument("--reward_type",   default="programmatic",
                   choices=["prm", "ccqa", "programmatic"],
                   help="Reward model: original PRM, CCQA, or arithmetic verifier")
    p.add_argument("--prm_name",      default=None,
                   help="PRM checkpoint name (required when --reward_type prm)")

    # DQN hyper-parameters
    p.add_argument("--n_episodes",    type=int,   default=500)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--buffer_cap",    type=int,   default=10_000)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--eps_start",     type=float, default=1.0)
    p.add_argument("--eps_end",       type=float, default=0.05)
    p.add_argument("--eps_decay",     type=int,   default=200)
    p.add_argument("--target_update", type=int,   default=10,
                   help="Copy policy → target every N episodes")
    p.add_argument("--rm_weight",     type=float, default=0.1,
                   help="Weight of reward-model signal added to env reward")

    # Infra
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--save_dir",      default="./checkpoints")
    p.add_argument("--save_every",    type=int,   default=50,
                   help="Save a checkpoint every N episodes")

    return p.parse_args()


# ── Training loop ─────────────────────────────────────────────────────────────

def main():
    os.environ["HF_HOME"] = "/mnt/data/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/mnt/data/hf/transformers"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/data/hf/hub"
    os.environ["TMPDIR"] = "/mnt/data/tmp"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # use first GPU only
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Evaluator config passed into ENV (Ollama GPT-OSS for state scoring)
    eval_config = {
        "model_name": args.eval_model,
        "base_url":   args.eval_base_url,   # None → falls back to env var
        "api_key":    args.eval_api_key,    # None → falls back to env var
    }

    print(f"[train] Dataset:     {args.dataset}")
    print(f"[train] Reasoning:   {args.model}")
    print(f"[train] Evaluator:   {args.eval_model}")
    print(f"[train] Reward type: {args.reward_type}")

    env = RLEnv(
        dataset        = args.dataset,
        is_test        = False,
        LLM_name       = args.model,
        problem_indexs = None,
        max_depth      = args.max_depth,
        max_width      = args.max_width,
        random_problems= True,
        random_seed    = args.seed,
        eval_config    = eval_config,
        data_dir       = args.data_dir,
    )

    # Reward model — selected entirely via flag, no loop changes needed
    rm_kwargs = {}
    if args.reward_type == "prm":
        if args.prm_name is None:
            raise ValueError("--prm_name is required when --reward_type prm")
        rm_kwargs = {"PRM_name": args.prm_name, "device": args.device}
    elif args.reward_type == "ccqa":
        rm_kwargs = {"device": args.device}

    reward_model = get_reward_model(args.reward_type, **rm_kwargs)

    # Networks
    device = torch.device(args.device)
    policy = Dueling_DQN(input_size=7, output_size=5).to(device)
    target = Dueling_DQN(input_size=7, output_size=5).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    buffer    = ExperienceReplay(capacity=args.buffer_cap, random_seed=args.seed)

    steps_done = 0

    for episode in range(args.n_episodes):
        state, finished = env.reset()
        if finished:
            print("[train] Dataset exhausted — training complete.")
            break

        episode_reward = 0.0
        done = False

        while not done:
            # ε-greedy action
            epsilon = args.eps_end + (args.eps_start - args.eps_end) * \
                      np.exp(-steps_done / max(args.eps_decay, 1))
            if np.random.rand() < epsilon:
                action = np.random.randint(5)
            else:
                with torch.no_grad():
                    s_t    = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    action = policy(s_t).argmax(dim=1).item()

            next_state, env_reward, done = env.step(action)

            # Augment env reward with reward-model signal
            step_texts = env.core.thought_each_step
            prm_input  = reward_model.covert_to_input(env.problem, step_texts)
            scores, _  = reward_model.get_step_scores(prm_input)
            rm_reward  = float(scores[-1]) if scores else 0.0

            reward         = float(env_reward) + args.rm_weight * rm_reward
            episode_reward += reward

            buffer.append(Experience(
                state      = np.array(state,      dtype=np.float32),
                action     = action,
                reward     = reward,
                done       = done,
                next_state = np.array(next_state, dtype=np.float32),
            ))
            state      = next_state
            steps_done += 1

            # DQN update
            if len(buffer) >= args.batch_size:
                states, actions, rewards, dones, next_states = buffer.sample(args.batch_size)
                s  = torch.tensor(states).to(device)
                a  = torch.tensor(actions).to(device)
                r  = torch.tensor(rewards).to(device)
                d  = torch.tensor(dones,   dtype=torch.float32).to(device)
                ns = torch.tensor(next_states).to(device)

                q_vals   = policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q   = target(ns).max(1)[0]
                    target_q = r + args.gamma * next_q * (1.0 - d)

                loss = nn.functional.mse_loss(q_vals, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Sync target network
        if episode % args.target_update == 0:
            target.load_state_dict(policy.state_dict())

        # Logging + checkpoint
        if episode % args.save_every == 0:
            ckpt = os.path.join(args.save_dir, f"policy_ep{episode:04d}.pt")
            torch.save(policy.state_dict(), ckpt)
            print(f"[ep {episode:4d}]  reward={episode_reward:.3f}  "
                  f"eps={epsilon:.3f}  buf={len(buffer)}  saved → {ckpt}")

    final = os.path.join(args.save_dir, "policy_final.pt")
    torch.save(policy.state_dict(), final)
    print(f"[train] Final checkpoint saved → {final}")


if __name__ == "__main__":
    main()
