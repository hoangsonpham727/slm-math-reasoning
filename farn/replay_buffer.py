"""
replay_buffer.py — Experience replay buffer for DQN training.

Stores (state, action, reward, next_state, done) transitions and
samples random mini-batches for training. This breaks the temporal
correlation between consecutive samples, which is critical for
stable DQN learning.

The buffer is a simple circular buffer with fixed maximum size.
When full, the oldest transitions are overwritten.
"""

import random
import numpy as np
import torch
from collections import deque
from typing import Tuple, NamedTuple


class Transition(NamedTuple):
    """A single experience transition."""
    state: np.ndarray       # shape (state_dim,)
    action: int             # integer action index
    reward: float           # single-step reward
    next_state: np.ndarray  # shape (state_dim,)
    done: bool              # whether episode ended


class ReplayBuffer:
    """Fixed-size circular replay buffer.

    Usage:
        buffer = ReplayBuffer(capacity=10000)
        buffer.push(state, action, reward, next_state, done)
        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
    """

    def __init__(self, capacity: int = 10000):
        """Initialise the buffer.

        Args:
            capacity: Maximum number of transitions to store.
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition.

        Args:
            state: Current state vector.
            action: Action taken.
            reward: Reward received.
            next_state: State after taking the action.
            done: Whether the episode terminated.
        """
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a random batch and return as tensors.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones),
            each as a torch.Tensor.
        """
        transitions = random.sample(self.buffer, batch_size)

        states = torch.FloatTensor(np.array([t.state for t in transitions]))
        actions = torch.LongTensor([t.action for t in transitions])
        rewards = torch.FloatTensor([t.reward for t in transitions])
        next_states = torch.FloatTensor(np.array([t.next_state for t in transitions]))
        dones = torch.FloatTensor([float(t.done) for t in transitions])

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer({len(self)}/{self.buffer.maxlen})"
