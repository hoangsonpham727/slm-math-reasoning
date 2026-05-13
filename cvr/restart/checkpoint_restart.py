"""
CheckpointRestarter — re-generates a failed step from the last verified checkpoint.

When verification fails at step k, we truncate the chain at step k-1 (the last
verified state) and re-prompt the model. The corrective hint in the system prompt
depends on the failure type:
  - consistency failure → arithmetic-care hint
  - relevance failure   → "ignore irrelevant info" hint

Temperature is increased on each retry attempt to encourage a different
computation path. Budget: max_restarts per step.
"""

from __future__ import annotations

from cvr.model_adapter import CVRModelAdapter
from cvr.prompts import (
    RESTART_CONSISTENCY_SYSTEM,
    RESTART_CONSISTENCY_USER,
    RESTART_RELEVANCE_SYSTEM,
    RESTART_RELEVANCE_USER,
    format_prior_steps_block,
)
from cvr.utils import parse_step_text


class CheckpointRestarter:
    """
    Re-generates a step from the last verified checkpoint.

    Args:
        adapter: CVRModelAdapter wrapping a loaded model
        max_restarts: maximum retry attempts per step (default: 2)
        base_temperature: starting temperature for restart generation
        temperature_increment: added per attempt to diversify outputs
    """

    def __init__(
        self,
        adapter: CVRModelAdapter,
        max_restarts: int = 2,
        base_temperature: float = 0.7,
        temperature_increment: float = 0.15,
    ):
        self.adapter = adapter
        self.max_restarts = max_restarts
        self.base_temperature = base_temperature
        self.temperature_increment = temperature_increment

    def restart(
        self,
        question: str,
        verified_steps: list[dict],
        step_number: int,
        failure_type: str,
        attempt: int,
        max_new_tokens: int = 256,
    ) -> dict:
        """
        Re-generate step `step_number` from the verified checkpoint.

        Args:
            question: original problem text
            verified_steps: list of verified step dicts (the checkpoint)
            step_number: 1-based index of the step to re-generate
            failure_type: 'consistency' or 'relevance'
            attempt: 0-indexed retry count (scales temperature)
            max_new_tokens: generation budget per step

        Returns:
            dict with 'index' and 'text' keys
        """
        if failure_type == "relevance":
            system_prompt = RESTART_RELEVANCE_SYSTEM
            user_template = RESTART_RELEVANCE_USER
        else:
            system_prompt = RESTART_CONSISTENCY_SYSTEM
            user_template = RESTART_CONSISTENCY_USER

        prior_block = format_prior_steps_block(verified_steps)
        user_prompt = user_template.format(
            question=question,
            prior_steps_block=prior_block,
            step_number=step_number,
        )

        temperature = self.base_temperature + attempt * self.temperature_increment

        raw_output = self.adapter.generate_sampled(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        step_text = parse_step_text(raw_output, step_number)

        return {
            "index": step_number,
            "text": step_text,
        }
