"""
NCV@k-Binary consistency verifier.

Checks whether a reasoning step's calculation is correct given the problem
statement and all previously verified steps.

Design (from NCV paper arXiv:2510.02816, Section 2.3–2.4):
- Binary mode: constrain output to 4 tokens max
- k-fold independent sampling with majority vote
- Each check sees: Problem + verified prior steps + current step
- Sequential: only verified steps appear in context (unverified ones are excluded)
- Conservative: unparseable output counts as "Wrong"
"""

from __future__ import annotations

from cvr.model_adapter import CVRModelAdapter
from cvr.prompts import CONSISTENCY_CHECK_SYSTEM, CONSISTENCY_CHECK_USER
from cvr.utils import format_steps_as_text, parse_binary_vote


class ConsistencyChecker:
    """
    NCV@k binary consistency check.

    Args:
        adapter: CVRModelAdapter wrapping a loaded model
        num_votes: number of independent binary judgments (NCV default: 3)
        max_tokens: output token budget for each judgment (NCV default: 4)
        temperature: sampling temperature for verification (NCV default: ~0.3)
        one_veto: if True, ANY "Wrong" vote fails the step (stricter than majority)
    """

    def __init__(
        self,
        adapter: CVRModelAdapter,
        num_votes: int = 3,
        max_tokens: int = 4,
        temperature: float = 0.3,
        one_veto: bool = False,
    ):
        self.adapter = adapter
        self.num_votes = num_votes
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.one_veto = one_veto

    def check(
        self,
        question: str,
        prior_steps: list[dict],
        current_step: dict,
    ) -> dict:
        """
        Run k independent binary checks on `current_step`.

        Returns:
            passed      bool   — True if majority (or all, if one_veto) say Correct
            votes       list   — individual bool results per vote
            vote_count  tuple  — (num_correct, num_wrong)
            raw_outputs list   — raw model responses per vote
        """
        prior_text = format_steps_as_text(prior_steps) if prior_steps else "(none yet)"
        user_prompt = CONSISTENCY_CHECK_USER.format(
            question=question,
            prior_steps=prior_text,
            step_number=current_step["index"],
            step_text=current_step["text"],
        )

        votes: list[bool] = []
        raw_outputs: list[str] = []

        for _ in range(self.num_votes):
            raw = self.adapter.generate_sampled(
                system_prompt=CONSISTENCY_CHECK_SYSTEM,
                user_prompt=user_prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            raw_outputs.append(raw)
            votes.append(parse_binary_vote(raw))

        num_correct = sum(votes)
        num_wrong = self.num_votes - num_correct

        if self.one_veto:
            passed = num_wrong == 0
        else:
            passed = num_correct > self.num_votes // 2

        return {
            "passed": passed,
            "votes": votes,
            "vote_count": (num_correct, num_wrong),
            "raw_outputs": raw_outputs,
        }
