"""
Final answer selection via majority vote across verified reasoning chains.

The answer appearing most often across successful chains wins. If no chain
succeeded, returns None with confidence 0.
"""

from __future__ import annotations

from collections import Counter


def majority_vote(chains: list[dict]) -> dict:
    """
    Select the final answer by majority vote over successful chains.

    Args:
        chains: list of chain result dicts, each expected to have:
                  success (bool), answer (str|None)

    Returns:
        answer            str|None  — winning answer, or None if no chains succeeded
        vote_count        int       — votes for the winning answer
        total_chains      int       — total chains attempted
        successful_chains int       — chains that produced a parseable answer
        confidence        float     — vote_count / successful_chains
    """
    successful = [c for c in chains if c.get("success") and c.get("answer") is not None]

    if not successful:
        return {
            "answer": None,
            "vote_count": 0,
            "total_chains": len(chains),
            "successful_chains": 0,
            "confidence": 0.0,
        }

    answers = [c["answer"] for c in successful]
    counter = Counter(answers)
    best_answer, best_count = counter.most_common(1)[0]

    return {
        "answer": best_answer,
        "vote_count": best_count,
        "total_chains": len(chains),
        "successful_chains": len(successful),
        "confidence": best_count / len(successful),
    }
