"""
StepwiseGenerator — generates one reasoning step at a time.

At each call, the generator receives the verified prior steps and produces
only the next step. If the model generates multiple steps, everything after
the next step header is discarded.
"""

from __future__ import annotations

from cvr.model_adapter import CVRModelAdapter
from cvr.prompts import (
    STEPWISE_GENERATION_SYSTEM,
    STEPWISE_GENERATION_USER,
    format_prior_steps_block,
)
from cvr.utils import parse_step_text


class StepwiseGenerator:
    """
    Generates one reasoning step conditioned on verified prior steps.

    Args:
        adapter: CVRModelAdapter wrapping a loaded model
        temperature: sampling temperature for generation (default: 0.7)
        max_new_tokens: token budget per step (default: 256)
    """

    def __init__(
        self,
        adapter: CVRModelAdapter,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ):
        self.adapter = adapter
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def generate_step(
        self,
        question: str,
        verified_steps: list[dict],
        step_number: int,
    ) -> dict:
        """
        Generate step `step_number` given the verified prior steps.

        Returns:
            dict with 'index' (int) and 'text' (str) keys
        """
        prior_block = format_prior_steps_block(verified_steps)
        user_prompt = STEPWISE_GENERATION_USER.format(
            question=question,
            prior_steps_block=prior_block,
            step_number=step_number,
        )

        raw_output = self.adapter.generate_sampled(
            system_prompt=STEPWISE_GENERATION_SYSTEM,
            user_prompt=user_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        step_text = parse_step_text(raw_output, step_number)

        return {
            "index": step_number,
            "text": step_text,
        }
