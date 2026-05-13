"""
NodeVerifier — orchestrates consistency and relevance checks per step.

A step passes ONLY if both checks pass. The failure_type field tells
the restarter which corrective hint to use.
"""

from __future__ import annotations

from cvr.model_adapter import CVRModelAdapter
from cvr.verification.consistency_check import ConsistencyChecker
from cvr.verification.relevance_check import RelevanceChecker


class NodeVerifier:
    """
    Runs consistency + (optionally) relevance checks for each reasoning step.

    Args:
        adapter: CVRModelAdapter wrapping a loaded model
        consistency_votes: k for consistency check (default: 1 for cloud)
        relevance_votes: k for relevance check (default: 1 for cloud)
        verification_temperature: sampling temperature for all checks
        max_output_tokens: token budget per check call
        enable_relevance: if False, only consistency is checked (default for Exp2)
        one_veto: if True, use one-vote-veto instead of majority
        call_delay_s: seconds between consecutive API calls (rate limiting guard)
        retry_delay_s: seconds to wait before retrying an empty response
    """

    def __init__(
        self,
        adapter: CVRModelAdapter,
        consistency_votes: int = 1,
        relevance_votes: int = 1,
        verification_temperature: float = 0.3,
        max_output_tokens: int = 64,
        enable_relevance: bool = False,
        one_veto: bool = False,
        call_delay_s: float = 1.0,
        retry_delay_s: float = 2.0,
    ):
        self.consistency = ConsistencyChecker(
            adapter,
            num_votes=consistency_votes,
            max_tokens=max_output_tokens,
            temperature=verification_temperature,
            one_veto=one_veto,
            call_delay_s=call_delay_s,
            retry_delay_s=retry_delay_s,
        )
        self.relevance = RelevanceChecker(
            adapter,
            num_votes=relevance_votes,
            max_tokens=max_output_tokens,
            temperature=verification_temperature,
            one_veto=one_veto,
            call_delay_s=call_delay_s,
            retry_delay_s=retry_delay_s,
        )
        self.enable_relevance = enable_relevance

    def verify_step(
        self,
        question: str,
        prior_verified_steps: list[dict],
        current_step: dict,
    ) -> dict:
        """
        Verify one reasoning step.

        Returns:
            passed              bool   — True if step should be accepted
            failure_type        str|None  — None | 'consistency' | 'relevance'
            consistency_result  dict   — from ConsistencyChecker.check()
            relevance_result    dict|None — from RelevanceChecker.check(), or None if skipped
        """
        cons_result = self.consistency.check(question, prior_verified_steps, current_step)
        if not cons_result["passed"]:
            return {
                "passed": False,
                "failure_type": "consistency",
                "consistency_result": cons_result,
                "relevance_result": None,
            }

        rel_result = None
        if self.enable_relevance:
            rel_result = self.relevance.check(question, current_step)
            if not rel_result["passed"]:
                return {
                    "passed": False,
                    "failure_type": "relevance",
                    "consistency_result": cons_result,
                    "relevance_result": rel_result,
                }

        return {
            "passed": True,
            "failure_type": None,
            "consistency_result": cons_result,
            "relevance_result": rel_result,
        }
