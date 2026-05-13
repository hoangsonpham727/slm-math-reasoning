"""
CheckpointRestarter — re-generates from the last verified checkpoint.

When verification fails at step k, the restarter prompts the model to continue
from step k given the verified prior steps (the checkpoint). The model generates
step k onward as a full continuation, which is then parsed into individual steps
for sequential re-verification.

Temperature is increased on each retry attempt to encourage a different
computation path.
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
from cvr.utils import parse_all_steps


class CheckpointRestarter:
    """
    Re-generates from the last verified checkpoint.

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

    def restart_and_continue(
        self,
        question: str,
        verified_steps: list[dict],
        step_number: int,
        failure_type: str,
        attempt: int,
        max_new_tokens: int = 1024,
    ) -> list[dict]:
        """
        Re-generate step `step_number` and all subsequent steps from the checkpoint.

        The model receives verified_steps as context and generates a continuation
        starting from step_number. The raw output is parsed into a step list so
        the pipeline can verify each step sequentially.

        Args:
            question: original problem text
            verified_steps: verified steps before the failed one (the checkpoint)
            step_number: 1-based index of the step that failed
            failure_type: 'consistency' or 'relevance'
            attempt: 0-indexed retry count (scales temperature)
            max_new_tokens: generation budget for the full continuation

        Returns:
            list of {"index": int, "text": str} dicts starting at step_number
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
            step_number_plus1=step_number + 1,
        )

        temperature = self.base_temperature + attempt * self.temperature_increment

        raw_output = self.adapter.generate_sampled(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        steps = parse_all_steps(raw_output)

        # Re-index steps so they start at step_number if the model didn't label them.
        # If parse_all_steps returned a single unlabeled chunk, assign step_number.
        if len(steps) == 1 and steps[0]["index"] == 1 and step_number > 1:
            steps[0]["index"] = step_number

        # Filter to steps >= step_number in case the model repeated prior steps.
        steps = [s for s in steps if s["index"] >= step_number]

        return steps
