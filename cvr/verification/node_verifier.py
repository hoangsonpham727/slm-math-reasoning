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
        consistency_votes: k for consistency check (default: 3)
        relevance_votes: k for relevance check (default: 3)
        verification_temperature: sampling temperature for all checks
        enable_relevance: if False, only consistency is checked (ablation mode)
        one_veto: if True, use one-vote-veto instead of majority
    """

    def __init__(
        self,
        adapter: CVRModelAdapter,
        consistency_votes: int = 3,
        relevance_votes: int = 3,
        verification_temperature: float = 0.3,
        enable_relevance: bool = True,
        one_veto: bool = False,
    ):
        self.consistency = ConsistencyChecker(
            adapter,
            num_votes=consistency_votes,
            temperature=verification_temperature,
            one_veto=one_veto,
        )
        self.relevance = RelevanceChecker(
            adapter,
            num_votes=relevance_votes,
            temperature=verification_temperature,
            one_veto=one_veto,
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
