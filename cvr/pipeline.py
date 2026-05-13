"""
CVRPipeline — end-to-end Checkpoint Verification and Restart pipeline.

For a single math problem:
  1. Run N independent reasoning chains
  2. Each chain: stepwise generate → NCV verify → restart from checkpoint on failure
  3. Majority vote over chains' final answers

The pipeline is stateless between problems and can be reused across a dataset.
"""

from __future__ import annotations

import time

import torch

from cvr.model_adapter import CVRModelAdapter
from cvr.generation.stepwise_generator import StepwiseGenerator
from cvr.verification.node_verifier import NodeVerifier
from cvr.restart.checkpoint_restart import CheckpointRestarter
from cvr.selection.majority_vote import majority_vote
from cvr.utils import is_final_step, extract_model_final


class CVRPipeline:
    """
    Full CVR pipeline for a single model.

    Args:
        adapter: CVRModelAdapter wrapping a loaded BaseModelWrapper
        config: dict loaded from cvr/config.yaml (via load_yaml_config)
    """

    def __init__(self, adapter: CVRModelAdapter, config: dict):
        self.adapter = adapter
        self.config = config

        gen_cfg = config["generation"]
        ver_cfg = config["verification"]
        rst_cfg = config["restart"]

        self.num_chains = gen_cfg["num_chains"]
        self.max_steps = gen_cfg["max_steps"]

        self.generator = StepwiseGenerator(
            adapter,
            temperature=gen_cfg["temperature"],
            max_new_tokens=gen_cfg["max_new_tokens_per_step"],
        )
        self.verifier = NodeVerifier(
            adapter,
            consistency_votes=ver_cfg["consistency_votes"],
            relevance_votes=ver_cfg["relevance_votes"],
            verification_temperature=ver_cfg["verification_temperature"],
            max_output_tokens=ver_cfg["max_output_tokens"],
            enable_relevance=ver_cfg["enable_relevance"],
        )
        self.restarter = CheckpointRestarter(
            adapter,
            max_restarts=rst_cfg["max_restarts"],
            base_temperature=gen_cfg["temperature"],
            temperature_increment=rst_cfg["temperature_increment"],
        )

    def solve(self, question: str) -> dict:
        """
        Run the full CVR pipeline on one math problem.

        Returns:
            answer              str|None  — final answer after majority vote
            confidence          float     — fraction of successful chains that agree
            vote_count          int
            total_chains        int
            successful_chains   int
            chains              list      — per-chain result dicts
            elapsed_s           float
        """
        t0 = time.perf_counter()

        chains = []
        for chain_idx in range(self.num_chains):
            with torch.no_grad():
                chain = self._solve_single_chain(question, chain_idx)
            chains.append(chain)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        vote_result = majority_vote(chains)

        elapsed = round(time.perf_counter() - t0, 3)
        return {**vote_result, "chains": chains, "elapsed_s": elapsed}

    def _solve_single_chain(self, question: str, chain_idx: int) -> dict:
        """Generate and verify one reasoning chain. Returns a chain result dict."""
        verified_steps: list[dict] = []
        total_restarts = 0
        consistency_failures = 0
        relevance_failures = 0

        for step_idx in range(self.max_steps):
            step_number = step_idx + 1

            # Generate next step
            current_step = self.generator.generate_step(
                question, verified_steps, step_number
            )

            final = is_final_step(current_step["text"])

            # Verify the step
            vresult = self.verifier.verify_step(question, verified_steps, current_step)

            if vresult["passed"]:
                current_step["verification"] = vresult
                verified_steps.append(current_step)
                if final:
                    break
            else:
                # Track failure type before restart loop
                if vresult["failure_type"] == "relevance":
                    relevance_failures += 1
                else:
                    consistency_failures += 1

                # Restart loop
                restarted = False
                for attempt in range(self.restarter.max_restarts):
                    total_restarts += 1
                    new_step = self.restarter.restart(
                        question,
                        verified_steps,
                        step_number,
                        vresult["failure_type"],
                        attempt,
                        max_new_tokens=self.config["generation"]["max_new_tokens_per_step"],
                    )
                    new_vresult = self.verifier.verify_step(
                        question, verified_steps, new_step
                    )
                    if new_vresult["passed"]:
                        new_step["verification"] = new_vresult
                        verified_steps.append(new_step)
                        restarted = True
                        if is_final_step(new_step["text"]):
                            final = True
                        break

                if not restarted:
                    # All restarts exhausted — chain fails
                    return {
                        "success": False,
                        "steps": verified_steps,
                        "answer": None,
                        "chain_idx": chain_idx,
                        "total_restarts": total_restarts,
                        "consistency_failures": consistency_failures,
                        "relevance_failures": relevance_failures,
                    }

                if final:
                    break

        # Extract final answer from the full verified chain
        full_solution = "\n".join(s["text"] for s in verified_steps)
        answer = extract_model_final(full_solution)

        return {
            "success": True,
            "steps": verified_steps,
            "answer": answer,
            "chain_idx": chain_idx,
            "total_restarts": total_restarts,
            "consistency_failures": consistency_failures,
            "relevance_failures": relevance_failures,
        }
