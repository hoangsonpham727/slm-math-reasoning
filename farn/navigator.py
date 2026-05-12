"""
navigator.py — Dueling DQN navigator model.

This is a tiny MLP (~3.5K parameters) that maps the 8-dimensional
external state vector to Q-values over the 5 logic-block actions.

Architecture (Dueling DQN, following Wang et al., 2016):

    State (8) ──► Hidden1 (64, ReLU) ──► Hidden2 (32, ReLU) ──┬──► Value stream  (32→1)
                                                                │
                                                                └──► Advantage stream (32→5)

    Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))

The dueling architecture separates state-value estimation from
action-advantage estimation, which helps the navigator learn which
states are valuable independently of which action is taken —
important when some states (e.g., "arithmetic error detected")
strongly constrain the best action.

Training uses Double DQN (Van Hasselt et al., 2016) with a target
network for stability.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple

from config import NavigatorConfig, ACTION_NAMES


class DuelingDQN(nn.Module):
    """Dueling DQN network for the navigator.

    The forward pass computes Q-values for all actions given a state.
    The dueling decomposition helps the network learn state values
    and action advantages separately.
    """

    def __init__(self, config: NavigatorConfig):
        super().__init__()
        self.config = config

        # Shared feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_1),
            nn.ReLU(),
            nn.Linear(config.hidden_1, config.hidden_2),
            nn.ReLU(),
        )

        # Value stream: estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(config.hidden_2, 1),
        )

        # Advantage stream: estimates A(s, a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.hidden_2, config.num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions.

        Args:
            state: Tensor of shape (batch, state_dim) or (state_dim,).

        Returns:
            Q-values tensor of shape (batch, num_actions) or (num_actions,).
        """
        features = self.feature_layer(state)

        value = self.value_stream(features)           # (batch, 1)
        advantage = self.advantage_stream(features)   # (batch, num_actions)

        # Dueling combination: Q = V + (A - mean(A))
        # Subtracting mean(A) ensures identifiability
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q_values

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Navigator:
    """High-level navigator that wraps the DQN for action selection and training.

    Handles:
    - Epsilon-greedy action selection
    - Double DQN weight updates
    - Target network synchronisation
    - Model saving/loading
    - Action masking (enforce transition rules)
    """

    def __init__(self, config: NavigatorConfig, device: str = "cpu"):
        """Initialise the navigator with online and target networks.

        Args:
            config: Navigator hyperparameters.
            device: "cpu" or "cuda". Navigator itself always runs on CPU
                    (it's tiny), but we keep the option for consistency.
        """
        self.config = config
        self.device = torch.device(device)

        # Online network (updated every step)
        self.online_net = DuelingDQN(config).to(self.device)

        # Target network (updated periodically for stability)
        self.target_net = DuelingDQN(config).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target network is never trained directly

        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=config.learning_rate,
        )
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stability

        # Exploration tracking
        self.steps_done = 0

        print(f"Navigator initialised: {self.online_net.count_parameters()} parameters")

    def select_action(
        self,
        state: np.ndarray,
        episode: int = 0,
        force_greedy: bool = False,
        answer_present: bool = False,
        is_first_step: bool = False,
    ) -> int:
        """Select an action using epsilon-greedy policy with transition rules.

        Transition rules (from RLoT):
          1. If answer is already present, only "terminate" is allowed.
          2. If this is the first step, "refine" is converted to "reason".

        Args:
            state: numpy array of shape (8,).
            episode: Current episode number (for epsilon decay).
            force_greedy: If True, skip exploration (for evaluation).
            answer_present: Whether the SLM has already produced an answer.
            is_first_step: Whether this is the first action in the episode.

        Returns:
            Action index (0-4).
        """
        # ── Transition rule 1: answer present → force terminate ──
        if answer_present:
            return ACTION_NAMES.index("terminate")

        # ── Epsilon-greedy exploration ──
        epsilon = self._get_epsilon(episode)

        if not force_greedy and random.random() < epsilon:
            action = random.randint(0, self.config.num_actions - 1)
        else:
            # Greedy: pick action with highest Q-value
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.online_net(state_tensor)
                action = q_values.argmax(dim=1).item()

        # ── Transition rule 2: no refine on first step ──
        if is_first_step and action == ACTION_NAMES.index("refine"):
            action = ACTION_NAMES.index("reason")

        self.steps_done += 1
        return action

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        """Perform a Double DQN update on a batch of transitions.

        Double DQN: use online network to SELECT the best next action,
        but use target network to EVALUATE its Q-value. This reduces
        overestimation bias.

        Args:
            states:      (batch, state_dim)
            actions:     (batch,) — integer action indices
            rewards:     (batch,) — single-step rewards
            next_states: (batch, state_dim)
            dones:       (batch,) — 1.0 if episode ended, 0.0 otherwise

        Returns:
            Loss value (float) for logging.
        """
        states = states.to(self.device)
        actions = actions.to(self.device).long()
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values: Q(s, a) from online network
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target:
        # 1. Online network selects the best next action
        with torch.no_grad():
            next_action_online = self.online_net(next_states).argmax(dim=1)
            # 2. Target network evaluates that action's Q-value
            next_q_target = self.target_net(next_states).gather(
                1, next_action_online.unsqueeze(1)
            ).squeeze(1)
            # 3. Bellman target: r + γ * Q_target(s', argmax_a Q_online(s', a))
            target_q = rewards + self.config.gamma * next_q_target * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def sync_target(self) -> None:
        """Copy online network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _get_epsilon(self, episode: int) -> float:
        """Compute current exploration rate with exponential decay."""
        return self.config.epsilon_end + (
            self.config.epsilon_start - self.config.epsilon_end
        ) * np.exp(-episode / self.config.epsilon_decay)

    def save(self, path: str) -> None:
        """Save navigator weights and optimizer state."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
        }, path)
        print(f"Navigator saved to {path}")

    def load(self, path: str) -> None:
        """Load navigator weights and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.steps_done = checkpoint["steps_done"]
        print(f"Navigator loaded from {path}")
