"""
config.py — Central configuration for the FARN framework.

All hyperparameters, model identifiers, action definitions, and feature
extraction settings are defined here so that every other module imports
from a single source of truth.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Model Configurations ─────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Configuration for a single SLM."""
    name: str                    # Short identifier
    hf_model_id: str             # HuggingFace model path
    size_billions: float         # Approximate parameter count
    use_system_prompt: bool      # Whether the model supports system prompts
    answer_format: str           # Expected answer format: "####" or "\\boxed{}"
    max_new_tokens: int = 512    # Max tokens per reasoning step
    load_in_4bit: bool = False   # Whether to quantise to 4-bit


# Registry of all SLMs in the project.
# Each model's depth_ceiling and noise_sensitivity are empirical values
# from Phase 1 experiments — used for feature normalisation and analysis.
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "qwen25_math_1.5b": ModelConfig(
        name="qwen25_math_1.5b",
        hf_model_id="Qwen/Qwen2.5-Math-1.5B-Instruct",
        size_billions=1.5,
        use_system_prompt=True,
        answer_format="\\boxed{}",
    ),
    "phi4_mini": ModelConfig(
        name="phi4_mini",
        hf_model_id="microsoft/Phi-4-mini-instruct",
        size_billions=3.8,
        use_system_prompt=True,
        answer_format="####",
    ),
    "gemma4_e2b": ModelConfig(
        name="gemma4_e2b",
        hf_model_id="google/gemma-4-E2B-IT",
        size_billions=2.0,
        use_system_prompt=True,
        answer_format="####",
    ),
    "deepseek_math_7b": ModelConfig(
        name="deepseek_math_7b",
        hf_model_id="deepseek-ai/deepseek-math-7b-instruct",
        size_billions=7.0,
        use_system_prompt=False,        # DeepSeek-Math has no system prompt
        answer_format="\\boxed{}",
        load_in_4bit=True,              # Quantise to fit alongside PRM
    ),
}


# ── Action Space ─────────────────────────────────────────────────────

# The five logic blocks from RLoT, kept unchanged for Direction 1.
# Each action is an integer index used by the DQN.
ACTION_NAMES = [
    "reason",       # 0: Reason one step
    "decompose",    # 1: Break into subtasks
    "debate",       # 2: Generate multiple plans, select best
    "refine",       # 3: Review and revise current step
    "terminate",    # 4: Produce final answer
]

NUM_ACTIONS = len(ACTION_NAMES)


# ── Feature Extractor Settings ───────────────────────────────────────

@dataclass
class FeatureConfig:
    """Settings for the 8-feature external state vector."""

    num_features: int = 8       # Dimensionality of state vector

    # Bounds-plausibility settings (f6)
    bounds_multiplier: float = 100.0   # Result must be within 100x of max input
    allow_negative: bool = False       # Whether negative intermediates are plausible

    # Normalisation ranges for continuous features
    max_step_count: int = 8            # Maximum reasoning steps (matches Exp 2)
    max_entities: int = 20             # Cap for entity count normalisation


# ── Navigator (DQN) Hyperparameters ──────────────────────────────────

@dataclass
class NavigatorConfig:
    """Hyperparameters for the Dueling DQN navigator."""

    # Architecture
    state_dim: int = 8              # Must match FeatureConfig.num_features
    num_actions: int = NUM_ACTIONS  # 5 actions
    hidden_1: int = 64              # First hidden layer
    hidden_2: int = 32              # Second hidden layer

    # Training
    learning_rate: float = 1e-3
    gamma: float = 0.99             # Discount factor
    epsilon_start: float = 1.0      # Initial exploration rate
    epsilon_end: float = 0.05       # Final exploration rate
    epsilon_decay: int = 500        # Episodes over which epsilon decays
    batch_size: int = 32
    replay_buffer_size: int = 10000
    target_update_freq: int = 10    # Update target network every N episodes
    num_episodes: int = 500         # Total training episodes
    max_steps_per_episode: int = 8  # Max actions before forced termination

    # Checkpointing
    save_every: int = 50            # Save navigator weights every N episodes
    checkpoint_dir: str = "checkpoints"


# ── Reward Settings ──────────────────────────────────────────────────

@dataclass
class RewardConfig:
    """Configuration for the reward signal."""

    # Which reward to use: "prm" (Math-Shepherd) or "ccqa" (cycle-consistency)
    reward_type: str = "prm"

    # Math-Shepherd PRM settings
    prm_model_id: str = "peiyi9979/math-shepherd-mistral-7b-prm"
    prm_load_in_4bit: bool = True

    # CCQA settings
    ccqa_model_id: str = "google/flan-t5-base"
    ccqa_similarity_threshold: float = 0.7  # Below this, reasoning is suspect


# ── Dataset Settings ─────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Dataset paths and splits."""

    dataset_name: str = "openai/gsm8k"
    train_split: str = "train"
    test_split: str = "test"
    # Only train on problems the SLM gets wrong (hard problems)
    filter_hard: bool = True
    # Number of hard problems to use for training
    max_train_problems: int = 500


# ── Master Config ────────────────────────────────────────────────────

@dataclass
class FARNConfig:
    """Top-level configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=lambda: MODEL_REGISTRY["qwen25_math_1.5b"])
    features: FeatureConfig = field(default_factory=FeatureConfig)
    navigator: NavigatorConfig = field(default_factory=NavigatorConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    data: DataConfig = field(default_factory=DataConfig)
    device: str = "cuda"   # "cuda" or "cpu"
    seed: int = 42


def get_config(model_name: str = "qwen25_math_1.5b", reward_type: str = "prm") -> FARNConfig:
    """Factory function to create a config for a specific model and reward."""
    cfg = FARNConfig()
    cfg.model = MODEL_REGISTRY[model_name]
    cfg.reward.reward_type = reward_type
    return cfg
