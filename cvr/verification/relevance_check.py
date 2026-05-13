"""
Distractor-aware relevance verifier.

Checks whether a reasoning step uses ONLY information that is necessary
and relevant to solving the problem. This catches distractor incorporation:
when an SLM uses an irrelevant number or fact from an injected distractor
sentence, this check flags it.

Uses the same NCV@k-Binary mechanism as ConsistencyChecker — same retry/abstain
logic for rate-limited cloud verifiers.

NOTE: Only enable this for Experiment 1 (distractor robustness). For Experiment 2
(multi-step degradation), there are no distractors — enable_relevance should be
False in config.yaml to save API calls.
"""

from __future__ import annotations

import time

from cvr.model_adapter import CVRModelAdapter
from cvr.prompts import RELEVANCE_CHECK_SYSTEM, RELEVANCE_CHECK_USER
from cvr.utils import parse_binary_vote


class RelevanceChecker:
    """
    Binary relevance check for distractor detection.

    Args:
        adapter: CVRModelAdapter wrapping a loaded model
        num_votes: number of independent binary judgments (default: 1 for cloud)
        max_tokens: output token budget
        temperature: sampling temperature (default: 0.3)
        one_veto: if True, ANY "No" vote fails the step
        call_delay_s: seconds to sleep between consecutive API calls
        retry_delay_s: seconds to wait before retrying an empty response
    """

    def __init__(
        self,
        adapter: CVRModelAdapter,
        num_votes: int = 1,
        max_tokens: int = 64,
        temperature: float = 0.3,
        one_veto: bool = False,
        call_delay_s: float = 0.0,
        retry_delay_s: float = 2.0,
    ):
        self.adapter = adapter
        self.num_votes = num_votes
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.one_veto = one_veto
        self.call_delay_s = call_delay_s
        self.retry_delay_s = retry_delay_s

    def _call_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        raw = self.adapter.generate_sampled(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if not raw.strip() and self.retry_delay_s > 0:
            time.sleep(self.retry_delay_s)
            raw = self.adapter.generate_sampled(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        return raw

    def check(self, question: str, current_step: dict) -> dict:
        """
        Run k independent relevance checks on `current_step`.

        Returns:
            passed      bool   — True if majority (or all, if one_veto) say Yes,
                                 or True if all votes abstained
            votes       list   — individual bool results per non-empty vote
            vote_count  tuple  — (num_yes, num_no)
            raw_outputs list   — raw model responses per vote
            abstained   int    — number of votes that were empty/skipped
        """
        user_prompt = RELEVANCE_CHECK_USER.format(
            question=question,
            step_text=current_step["text"],
        )

        votes: list[bool] = []
        raw_outputs: list[str] = []
        abstained = 0

        for i in range(self.num_votes):
            if i > 0 and self.call_delay_s > 0:
                time.sleep(self.call_delay_s)

            raw = self._call_with_retry(RELEVANCE_CHECK_SYSTEM, user_prompt)
            raw_outputs.append(raw)

            if not raw.strip():
                abstained += 1
            else:
                votes.append(parse_binary_vote(raw))

        num_yes = sum(votes)
        num_no = len(votes) - num_yes

        if not votes:
            passed = True  # all abstained → can't verify → pass
        elif self.one_veto:
            passed = num_no == 0
        else:
            passed = num_yes > len(votes) // 2

        return {
            "passed": passed,
            "votes": votes,
            "vote_count": (num_yes, num_no),
            "raw_outputs": raw_outputs,
            "abstained": abstained,
        }
