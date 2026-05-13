"""
NCV@k-Binary consistency verifier.

Checks whether a reasoning step's calculation is correct given the problem
statement and all previously verified steps.

Design (from NCV paper arXiv:2510.02816, Section 2.3–2.4):
- Binary mode: constrain output to 64 tokens max
- k-fold independent sampling with majority vote (default k=1 for cloud models)
- Each check sees: Problem + verified prior steps + current step
- Sequential: only verified steps appear in context (unverified ones are excluded)

Empty-response handling:
  Cloud verifiers (rate-limited) may return empty strings. An empty response is
  treated as "abstain" — it is excluded from the vote count. If ALL votes abstain,
  the step is passed (rate limiting is an infra issue, not a correctness signal).
  A single retry with backoff is attempted before classifying as abstain.
"""

from __future__ import annotations

import time

from cvr.model_adapter import CVRModelAdapter
from cvr.prompts import CONSISTENCY_CHECK_SYSTEM, CONSISTENCY_CHECK_USER
from cvr.utils import format_steps_as_text, parse_binary_vote


class ConsistencyChecker:
    """
    NCV@k binary consistency check.

    Args:
        adapter: CVRModelAdapter wrapping a loaded model
        num_votes: number of independent binary judgments (default: 1 for cloud)
        max_tokens: output token budget for each judgment
        temperature: sampling temperature for verification
        one_veto: if True, ANY "Wrong" vote fails the step (stricter than majority)
        call_delay_s: seconds to sleep between consecutive API calls (rate limiting)
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
        """Make one verification call, retrying once if the response is empty."""
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

    def check(
        self,
        question: str,
        prior_steps: list[dict],
        current_step: dict,
    ) -> dict:
        """
        Run k independent binary checks on `current_step`.

        Returns:
            passed      bool   — True if majority (or all, if one_veto) say Correct,
                                 or True if all responses were empty (abstain)
            votes       list   — individual bool results per non-empty vote
            vote_count  tuple  — (num_correct, num_wrong) counting non-empty votes only
            raw_outputs list   — raw model responses per vote
            abstained   int    — number of votes that were empty/skipped
        """
        prior_text = format_steps_as_text(prior_steps) if prior_steps else "(none yet)"
        user_prompt = CONSISTENCY_CHECK_USER.format(
            question=question,
            prior_steps=prior_text,
            step_text=current_step["text"],
        )

        votes: list[bool] = []
        raw_outputs: list[str] = []
        abstained = 0

        for i in range(self.num_votes):
            if i > 0 and self.call_delay_s > 0:
                time.sleep(self.call_delay_s)

            raw = self._call_with_retry(CONSISTENCY_CHECK_SYSTEM, user_prompt)
            raw_outputs.append(raw)

            if not raw.strip():
                abstained += 1  # treat as abstain, not Wrong
            else:
                votes.append(parse_binary_vote(raw))

        num_correct = sum(votes)
        num_wrong = len(votes) - num_correct

        # All abstained → can't verify → pass (rate limiting ≠ wrong step)
        if not votes:
            passed = True
        elif self.one_veto:
            passed = num_wrong == 0
        else:
            passed = num_correct > len(votes) // 2

        return {
            "passed": passed,
            "votes": votes,
            "vote_count": (num_correct, num_wrong),
            "raw_outputs": raw_outputs,
            "abstained": abstained,
        }
