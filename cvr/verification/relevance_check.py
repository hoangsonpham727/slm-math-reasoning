"""
Distractor-aware relevance verifier.

Checks whether a reasoning step uses ONLY information that is necessary
and relevant to solving the problem. This catches distractor incorporation:
when an SLM uses an irrelevant number or fact from an injected distractor
sentence, this check flags it.

Uses the same NCV@k-Binary mechanism as ConsistencyChecker — 3-vote majority,
4-token max output — but with the relevance prompts.
"""

from __future__ import annotations

from cvr.model_adapter import CVRModelAdapter
from cvr.prompts import RELEVANCE_CHECK_SYSTEM, RELEVANCE_CHECK_USER
from cvr.utils import parse_binary_vote


class RelevanceChecker:
    """
    Binary relevance check for distractor detection.

    Args:
        adapter: CVRModelAdapter wrapping a loaded model
        num_votes: number of independent binary judgments (default: 3)
        max_tokens: output token budget (default: 4)
        temperature: sampling temperature (default: 0.3)
        one_veto: if True, ANY "No" vote fails the step
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

    def check(self, question: str, current_step: dict) -> dict:
        """
        Run k independent relevance checks on `current_step`.

        Returns:
            passed      bool   — True if majority (or all, if one_veto) say Yes
            votes       list   — individual bool results per vote
            vote_count  tuple  — (num_yes, num_no)
            raw_outputs list   — raw model responses per vote
        """
        user_prompt = RELEVANCE_CHECK_USER.format(
            question=question,
            step_text=current_step["text"],
        )

        votes: list[bool] = []
        raw_outputs: list[str] = []

        for _ in range(self.num_votes):
            raw = self.adapter.generate_sampled(
                system_prompt=RELEVANCE_CHECK_SYSTEM,
                user_prompt=user_prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            raw_outputs.append(raw)
            votes.append(parse_binary_vote(raw))

        num_yes = sum(votes)
        num_no = self.num_votes - num_yes

        if self.one_veto:
            passed = num_no == 0
        else:
            passed = num_yes > self.num_votes // 2

        return {
            "passed": passed,
            "votes": votes,
            "vote_count": (num_yes, num_no),
            "raw_outputs": raw_outputs,
        }
