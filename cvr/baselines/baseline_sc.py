"""
Self-Consistency (SC) baseline.

Generates N independent CoT solutions (using the existing SYSTEM_COT prompt
from Experiment 2) and selects the final answer by majority vote.

This is the only baseline not already in the project. Direct and CoT single-pass
baselines are in Experiment2/run_experiment2.py and are not duplicated here.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from collections import Counter

from tqdm import tqdm

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from Experiment2.dataset_generator import SYSTEM_COT  # reuse existing prompt
from cvr.model_adapter import CVRModelAdapter
from cvr.utils import extract_model_final, answers_match, extract_gsm_final


class SelfConsistencyBaseline:
    """
    Self-Consistency: generate N chains, majority vote on final answers.

    Args:
        adapter: CVRModelAdapter wrapping a loaded model
        num_chains: number of independent samples (default: 5)
        temperature: sampling temperature (default: 0.7)
        max_new_tokens: token budget per chain (default: 512)
    """

    def __init__(
        self,
        adapter: CVRModelAdapter,
        num_chains: int = 5,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ):
        self.adapter = adapter
        self.num_chains = num_chains
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def solve(self, question: str) -> dict:
        """
        Run SC on one problem. Returns the majority-vote answer and metadata.

        Returns:
            answer            str|None
            confidence        float     — fraction of chains agreeing on winning answer
            vote_count        int
            total_chains      int
            successful_chains int       — chains that produced a parseable answer
            elapsed_s         float
        """
        t0 = time.perf_counter()

        answers: list[str | None] = []
        responses: list[str] = []

        for _ in range(self.num_chains):
            response = self.adapter.generate_sampled(
                system_prompt=SYSTEM_COT,
                user_prompt=question + "\nPlease reason step by step, then give your final answer as: Answer: <number>",
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            responses.append(response)
            answers.append(extract_model_final(response))

        valid = [(i, a) for i, a in enumerate(answers) if a is not None]
        elapsed = round(time.perf_counter() - t0, 3)

        if not valid:
            return {
                "answer": None,
                "confidence": 0.0,
                "vote_count": 0,
                "total_chains": self.num_chains,
                "successful_chains": 0,
                "responses": responses,
                "elapsed_s": elapsed,
            }

        answer_list = [a for _, a in valid]
        counter = Counter(answer_list)
        best_answer, best_count = counter.most_common(1)[0]

        return {
            "answer": best_answer,
            "confidence": round(best_count / len(valid), 4),
            "vote_count": best_count,
            "total_chains": self.num_chains,
            "successful_chains": len(valid),
            "responses": responses,
            "elapsed_s": elapsed,
        }


def run_sc_on_dataset(
    adapter: CVRModelAdapter,
    examples: list[dict],
    num_chains: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    limit: int | None = None,
    desc: str = "sc",
) -> list[dict]:
    """
    Run SC on a list of example dicts.

    Each example must have 'question' and 'answer' (GSM-style with ####) or
    'ground_truth' (float, for Exp2-style problems).

    Returns list of result records compatible with eval_accuracy.compute_accuracy().
    """
    sc = SelfConsistencyBaseline(adapter, num_chains, temperature, max_new_tokens)
    if limit is not None:
        examples = examples[:limit]

    results = []
    for ex in tqdm(examples, desc=desc):
        result = sc.solve(ex["question"])

        # Determine ground truth
        if "ground_truth" in ex:
            # Experiment 2 style (numeric)
            gt = ex["ground_truth"]
            pred_str = result["answer"]
            try:
                pred_float = float(pred_str) if pred_str else None
            except (ValueError, TypeError):
                pred_float = None
            is_correct = (
                pred_float is not None
                and abs(pred_float - gt) / (abs(gt) + 1e-9) < 0.01
            )
        else:
            # Experiment 1 style (GSM #### label)
            gold = extract_gsm_final(ex.get("answer", ""))
            is_correct = answers_match(gold or "", result["answer"])

        record = {
            "question": ex.get("question"),
            "ground_truth": ex.get("ground_truth", ex.get("answer")),
            "answer": result["answer"],
            "confidence": result["confidence"],
            "is_correct": is_correct,
            "is_collapse": result["answer"] is None,
            "total_chains": result["total_chains"],
            "successful_chains": result["successful_chains"],
            "elapsed_s": result["elapsed_s"],
        }
        # Carry through Exp1 metadata if present
        for key in ("source_file", "id_orig", "id_shuffled", "distractor", "distractor_type", "depth"):
            if key in ex:
                record[key] = ex[key]

        results.append(record)

    return results
