"""
Token cost estimation for CVR vs. baselines.

Token counts are estimated from prompt + response string lengths since the
CVR pipeline doesn't have a direct hook into the tokenizer. The estimate uses
a word-count × 1.3 proxy (conservative; actual BPE tokens are typically 1.1–1.5
per word for math text).

For exact counts, a caller can pass a tokenizer to count_tokens_exact().
"""

from __future__ import annotations

import re


def _approx_tokens(text: str) -> int:
    """Estimate token count from a string (word count × 1.3 proxy)."""
    words = len(text.split())
    return max(1, round(words * 1.3))


def estimate_tokens_per_chain(chain: dict) -> dict:
    """
    Estimate token cost of a single CVR chain.

    Counts:
        generation_tokens    — tokens in all step prompts + step responses
        verification_tokens  — tokens in all verification prompts + 4-token responses
        total_tokens         — sum of above
    """
    gen_tokens = 0
    ver_tokens = 0

    for step in chain.get("steps", []):
        # Generation: approximate from step text length
        gen_tokens += _approx_tokens(step.get("text", ""))

        vresult = step.get("verification") or {}
        for check_key in ("consistency_result", "relevance_result"):
            check = vresult.get(check_key)
            if check is None:
                continue
            for raw in check.get("raw_outputs", []):
                ver_tokens += _approx_tokens(raw) + 4  # response ≤ 4 tokens

    return {
        "generation_tokens": gen_tokens,
        "verification_tokens": ver_tokens,
        "total_tokens": gen_tokens + ver_tokens,
    }


def estimate_tokens_for_result(result: dict) -> dict:
    """
    Aggregate token estimates across all chains in a CVR result dict.

    Returns per-chain average and total.
    """
    chains = result.get("chains", [])
    if not chains:
        return {"avg_tokens_per_chain": 0, "total_tokens": 0, "n_chains": 0}

    chain_costs = [estimate_tokens_per_chain(c) for c in chains]
    total = sum(c["total_tokens"] for c in chain_costs)
    avg = total / len(chain_costs)

    return {
        "avg_tokens_per_chain": round(avg),
        "total_tokens": total,
        "n_chains": len(chains),
        "avg_generation_tokens": round(sum(c["generation_tokens"] for c in chain_costs) / len(chain_costs)),
        "avg_verification_tokens": round(sum(c["verification_tokens"] for c in chain_costs) / len(chain_costs)),
    }


def compute_avg_tokens(results: list[dict]) -> dict:
    """Average token cost across a list of problem results."""
    if not results:
        return {}
    per_result = [estimate_tokens_for_result(r) for r in results]
    keys = ["avg_tokens_per_chain", "total_tokens", "avg_generation_tokens", "avg_verification_tokens"]
    return {
        k: round(sum(r.get(k, 0) for r in per_result) / len(per_result))
        for k in keys
    }


def count_tokens_exact(text: str, tokenizer) -> int:
    """Exact token count using a HuggingFace tokenizer (optional, when available)."""
    return len(tokenizer.encode(text, add_special_tokens=False))
