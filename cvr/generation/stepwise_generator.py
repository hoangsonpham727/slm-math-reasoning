"""
SolutionGenerator — generates a complete solution then parses it into steps.

Small math-trained SLMs (Qwen2.5-Math, Phi-4, Gemma4) are trained to produce
full solutions, not single steps. Asking for "one step only" causes the model to
ignore the instruction. The correct approach is to let the model generate its full
chain-of-thought and then parse the output into individual Step N: blocks for
per-step verification.

On restart, the generator receives verified steps as context and generates the
remaining steps (from the failed step number onwards).
"""

from __future__ import annotations

from cvr.model_adapter import CVRModelAdapter
from cvr.prompts import (
    STEPWISE_GENERATION_SYSTEM,
    STEPWISE_GENERATION_USER,
    format_prior_steps_block,
)
from cvr.utils import parse_all_steps


class StepwiseGenerator:
    """
    Generates a complete solution and returns a parsed list of step dicts.

    Args:
        adapter: CVRModelAdapter wrapping a loaded model
        temperature: sampling temperature for generation (default: 0.7)
        max_new_tokens: token budget for the full solution (default: 1024)
    """

    def __init__(
        self,
        adapter: CVRModelAdapter,
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
    ):
        self.adapter = adapter
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def generate_all_steps(self, question: str) -> list[dict]:
        """
        Generate a full solution to the problem and parse it into step dicts.

        Returns:
            list of {"index": int, "text": str} dicts
        """
        user_prompt = STEPWISE_GENERATION_USER.format(question=question)

        raw_output = self.adapter.generate_sampled(
            system_prompt=STEPWISE_GENERATION_SYSTEM,
            user_prompt=user_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        return parse_all_steps(raw_output)
