"""
CVRPipeline — end-to-end Checkpoint Verification and Restart pipeline.

For a single math problem:
  1. Run N independent reasoning chains
  2. Each chain:
     a. Generate complete solution (let the model do what it's trained for)
     b. Parse output into individual Step N: blocks
     c. Verify each step sequentially with NCV binary checks
     d. On failure: restart from the last verified checkpoint, generating a
        continuation (step k onward), parse + re-verify step k
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
        verifier_adapter: optional pre-built adapter for the verifier (used in
            debug/sanity mode to intercept cloud verifier calls)
    """

    def __init__(self, adapter: CVRModelAdapter, config: dict, verifier_adapter=None):
        self.adapter = adapter
        self.config = config

        gen_cfg = config["generation"]
        ver_cfg = config["verification"]
        rst_cfg = config["restart"]

        self.num_chains = gen_cfg["num_chains"]
        self.max_steps = gen_cfg["max_steps"]

        # Use higher token budget: model generates the full solution, not one step.
        full_solution_tokens = gen_cfg.get(
            "max_new_tokens_full_solution",
            gen_cfg["max_new_tokens_per_step"] * self.max_steps,
        )

        self.generator = StepwiseGenerator(
            adapter,
            temperature=gen_cfg["temperature"],
            max_new_tokens=full_solution_tokens,
        )

        # Verifier priority:
        #   1. caller-supplied (sanity/debug mode wraps it in _DebugAdapter)
        #   2. local HuggingFace model (verifier_local_hf.enabled)
        #   3. cloud Ollama model  (verifier_cloud.enabled)
        #   4. same local SLM as generator (fallback)
        if verifier_adapter is not None:
            _verifier_adapter = verifier_adapter
        else:
            hf_cfg = config.get("verifier_local_hf", {})
            cloud_cfg = config.get("verifier_cloud", {})
            if hf_cfg.get("enabled", False):
                from cvr.hf_verifier import build_hf_verifier
                _verifier_adapter = build_hf_verifier(hf_cfg)
                print(f"  [CVR] Using local HF verifier: {_verifier_adapter.model_key}")
            elif cloud_cfg.get("enabled", False):
                from cvr.cloud_verifier import build_cloud_verifier
                _verifier_adapter = build_cloud_verifier(cloud_cfg)
                print(f"  [CVR] Using cloud verifier: {_verifier_adapter.model_key}")
            else:
                _verifier_adapter = adapter

        self.verifier = NodeVerifier(
            _verifier_adapter,
            consistency_votes=ver_cfg["consistency_votes"],
            relevance_votes=ver_cfg["relevance_votes"],
            verification_temperature=ver_cfg["verification_temperature"],
            max_output_tokens=ver_cfg["max_output_tokens"],
            enable_relevance=ver_cfg["enable_relevance"],
            call_delay_s=ver_cfg.get("call_delay_s", 1.0),
            retry_delay_s=ver_cfg.get("retry_delay_s", 2.0),
        )

        # Restart uses same token budget as full solution (continuation = remaining steps).
        restart_tokens = gen_cfg.get(
            "max_new_tokens_full_solution",
            gen_cfg["max_new_tokens_per_step"] * self.max_steps,
        )
        self.restarter = CheckpointRestarter(
            adapter,
            max_restarts=rst_cfg["max_restarts"],
            base_temperature=gen_cfg["temperature"],
            temperature_increment=rst_cfg["temperature_increment"],
        )
        self._restart_tokens = restart_tokens

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
        """
        Generate and verify one reasoning chain.

        Strategy:
          - Generate the full solution once (model outputs its natural format).
          - Parse into Step N: blocks.
          - Verify each block; on failure restart from the last verified checkpoint,
            generating a continuation that replaces the failed step onward.
        """
        verified_steps: list[dict] = []
        total_restarts = 0
        consistency_failures = 0
        relevance_failures = 0

        # Generate complete solution upfront; parse into step list.
        pending_steps = self.generator.generate_all_steps(question)

        step_pos = 0
        while step_pos < len(pending_steps) and len(verified_steps) < self.max_steps:
            current_step = pending_steps[step_pos]

            # Verify the step.
            vresult = self.verifier.verify_step(question, verified_steps, current_step)

            if vresult["passed"]:
                current_step["verification"] = vresult
                verified_steps.append(current_step)
                if is_final_step(current_step["text"]):
                    break
                step_pos += 1

            else:
                # Track failure type.
                if vresult["failure_type"] == "relevance":
                    relevance_failures += 1
                else:
                    consistency_failures += 1

                step_number = current_step["index"]
                restarted = False

                for attempt in range(self.restarter.max_restarts):
                    total_restarts += 1
                    new_steps = self.restarter.restart_and_continue(
                        question,
                        verified_steps,
                        step_number,
                        vresult["failure_type"],
                        attempt,
                        max_new_tokens=self._restart_tokens,
                    )
                    if not new_steps:
                        continue

                    # Verify only the first new step; if it passes, accept the
                    # continuation and replace the pending queue.
                    first_new = new_steps[0]
                    new_vresult = self.verifier.verify_step(
                        question, verified_steps, first_new
                    )
                    if new_vresult["passed"]:
                        first_new["verification"] = new_vresult
                        verified_steps.append(first_new)
                        # Replace remaining queue with the model-generated continuation.
                        pending_steps = new_steps[1:]
                        step_pos = 0
                        restarted = True
                        if is_final_step(first_new["text"]):
                            pending_steps = []
                        break

                if not restarted:
                    return {
                        "success": False,
                        "steps": verified_steps,
                        "answer": None,
                        "chain_idx": chain_idx,
                        "total_restarts": total_restarts,
                        "consistency_failures": consistency_failures,
                        "relevance_failures": relevance_failures,
                    }

                if not pending_steps:
                    break

        full_solution = "\n".join(
            f"Step {s['index']}: {s['text']}" for s in verified_steps
        )
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
