"""
feature_extractor.py — Compute the 8-feature external state vector.

This module replaces RLoT's self-evaluation state with externally
verifiable features organised into three groups:

  Group A — Noise-awareness (targeting Exp 1 distractor sensitivity):
    f1: relevance_ratio    — fraction of problem entities referenced
    f2: unused_count       — number of entities not yet used (normalised)
    f3: entity_op_align    — alignment between values and operations

  Group B — Depth-awareness (targeting Exp 2 depth collapse):
    f4: step_progress      — current step / estimated total steps
    f5: arith_verify       — does the SLM's arithmetic check out?
    f6: bounds_plausible   — is the latest result numerically plausible?

  Group C — Progress (structural):
    f7: answer_present     — has the SLM produced an answer marker?
    f8: step_count_norm    — normalised step count

Every feature is normalised to [0, 1] so the navigator MLP sees a
uniform input scale.

Key design principle: NO feature depends on the SLM evaluating itself.
Every feature is computed by external code from the SLM's raw text output.
"""

import re
import numpy as np
from typing import List, Optional, Tuple

from entity_registry import EntityRegistry
from config import FeatureConfig


class FeatureExtractor:
    """Computes the 8-dimensional state vector from SLM output.

    Usage:
        extractor = FeatureExtractor(problem_text, config)
        # After each SLM reasoning step:
        state = extractor.extract(slm_output_so_far, step_number)
        # state is a numpy array of shape (8,), all values in [0, 1]
    """

    # Regex for arithmetic operations in SLM output.
    # Matches expressions like "5 + 3 = 8", "12 * 4 = 48", "100 - 25 = 75"
    # Also handles implicit operations written as "5 + 3" without "= result"
    ARITH_EXPR_PATTERN = re.compile(
        r"""
        ([\d.,]+)            # First operand
        \s*                  # Optional whitespace
        ([+\-*/×÷])          # Operator
        \s*                  # Optional whitespace
        ([\d.,]+)            # Second operand
        (?:                  # Optional "= result" group
            \s*=\s*
            ([\d.,]+)        # Stated result
        )?
        """,
        re.VERBOSE,
    )

    # Patterns that indicate the SLM has produced a final answer.
    ANSWER_PATTERNS = [
        re.compile(r"####\s*[\d.,]+"),                # GSM8K format
        re.compile(r"\\boxed\{[^}]+\}"),              # LaTeX boxed format
        re.compile(r"(?:the\s+)?answer\s+is\s+[\d.,]+", re.IGNORECASE),
        re.compile(r"(?:final\s+)?answer\s*[:=]\s*[\d.,]+", re.IGNORECASE),
        re.compile(r"therefore.*?=\s*[\d.,]+", re.IGNORECASE),
    ]

    # Pattern to extract any number from text (used for bounds checking)
    ANY_NUMBER = re.compile(r"(?<!\w)([\d,]+\.?\d*)(?!\w)")

    def __init__(self, problem_text: str, config: FeatureConfig = None):
        """Initialise the feature extractor for a single problem.

        Args:
            problem_text: The full math word problem.
            config: Feature extraction settings. Uses defaults if None.
        """
        self.config = config or FeatureConfig()
        self.problem_text = problem_text

        # Build entity registry from the problem text
        self.registry = EntityRegistry(problem_text)

        # Estimate total steps from problem complexity.
        # Heuristic: number of entities roughly equals number of operations + 1,
        # so estimated steps ≈ number of entities.
        self.estimated_total_steps = max(self.registry.total_entities(), 2)

        # Track cumulative SLM output across steps for reference updates
        self.cumulative_output = ""
        self.step_outputs: List[str] = []

    def extract(self, latest_step_output: str, step_number: int) -> np.ndarray:
        """Compute the full 8-feature state vector after a reasoning step.

        Args:
            latest_step_output: The SLM's output from the most recent step.
            step_number: Current step index (0-based).

        Returns:
            numpy array of shape (8,), all values in [0, 1].
        """
        # Accumulate outputs
        self.step_outputs.append(latest_step_output)
        self.cumulative_output += "\n" + latest_step_output

        # Update entity references with the new output
        self.registry.update_references(latest_step_output)

        # ── Group A: Noise-awareness features ────────────────────────

        # f1: Relevance ratio — what fraction of problem entities
        #     has the SLM referenced in its reasoning so far?
        f1 = self.registry.relevance_ratio()

        # f2: Unused entity count — how many entities are still unused?
        #     Normalised by total entities (or max_entities cap).
        total = self.registry.total_entities()
        f2_raw = self.registry.unused_count()
        f2 = f2_raw / max(total, 1)  # Normalise to [0, 1]; 0 = all used

        # f3: Entity-operation alignment — are the number of arithmetic
        #     operations consistent with the number of values referenced?
        f3 = self._entity_operation_alignment()

        # ── Group B: Depth-awareness features ────────────────────────

        # f4: Step progress ratio — how far through the expected chain?
        f4 = min((step_number + 1) / self.estimated_total_steps, 1.0)

        # f5: Arithmetic verification — does the latest computation check out?
        f5 = self._verify_arithmetic(latest_step_output)

        # f6: Bounds plausibility — is the latest numerical result reasonable?
        f6 = self._check_bounds(latest_step_output)

        # ── Group C: Progress features ───────────────────────────────

        # f7: Answer present — has the SLM produced an answer marker?
        f7 = self._check_answer_present()

        # f8: Step count — normalised by maximum allowed steps.
        f8 = min((step_number + 1) / self.config.max_step_count, 1.0)

        state = np.array([f1, f2, f3, f4, f5, f6, f7, f8], dtype=np.float32)
        return state

    def _entity_operation_alignment(self) -> float:
        """f3: Measure alignment between referenced values and operations.

        For well-formed arithmetic:  operations ≈ values - 1
        (each operation combines two values into one result).

        Returns:
            Float in [0, 1]. 1.0 = perfect alignment, 0.0 = severe mismatch.
        """
        # Count distinct numerical values in cumulative output
        value_matches = self.ANY_NUMBER.findall(self.cumulative_output)
        num_values = len(set(value_matches))

        # Count arithmetic operators in cumulative output
        operators = re.findall(r'[+\-*/×÷]', self.cumulative_output)
        # Filter out negative signs (heuristic: minus after operator or at start)
        # This is imperfect but good enough for a state feature
        num_ops = len(operators)

        if num_values == 0 and num_ops == 0:
            return 0.5  # No arithmetic yet — neutral

        expected_ops = max(num_values - 1, 0)
        denominator = max(num_ops, num_values, 1)
        alignment = 1.0 - abs(num_ops - expected_ops) / denominator

        return max(0.0, min(1.0, alignment))

    def _verify_arithmetic(self, step_output: str) -> float:
        """f5: Verify arithmetic expressions in the latest step.

        Extracts expressions like "X op Y = Z" and checks whether
        X op Y actually equals Z using Python eval.

        Returns:
            1.0 if all extracted expressions verify correctly.
            0.0 if any expression is wrong.
            0.5 if no arithmetic expression could be parsed.
        """
        matches = list(self.ARITH_EXPR_PATTERN.finditer(step_output))

        if not matches:
            return 0.5  # No arithmetic found — uncertain

        all_correct = True
        any_checked = False

        for match in matches:
            operand1_str = match.group(1).replace(",", "")
            operator = match.group(2)
            operand2_str = match.group(3).replace(",", "")
            stated_result_str = match.group(4)

            if stated_result_str is None:
                continue  # Expression without "= result" — can't verify

            any_checked = True

            try:
                op1 = float(operand1_str)
                op2 = float(operand2_str)
                stated = float(stated_result_str.replace(",", ""))

                # Normalise operator symbols
                if operator in ("×", "*"):
                    computed = op1 * op2
                elif operator in ("÷", "/"):
                    if op2 == 0:
                        all_correct = False
                        continue
                    computed = op1 / op2
                elif operator == "+":
                    computed = op1 + op2
                elif operator == "-":
                    computed = op1 - op2
                else:
                    continue

                # Allow small floating-point tolerance
                if abs(computed - stated) > max(abs(stated) * 1e-6, 1e-6):
                    all_correct = False

            except ValueError:
                continue  # Unparseable — skip

        if not any_checked:
            return 0.5  # Had expressions but none with "= result"

        return 1.0 if all_correct else 0.0

    def _check_bounds(self, step_output: str) -> float:
        """f6: Check if the latest numerical result is plausible.

        Plausibility means:
          - Non-negative (for typical GSM8K problems about counts/money)
          - Within bounds_multiplier × max(input values)

        Returns:
            1.0 if plausible, 0.0 if implausible, 0.5 if no number found.
        """
        # Extract the last number in the step output as the "result"
        numbers = self.ANY_NUMBER.findall(step_output)
        if not numbers:
            return 0.5  # No number to check

        try:
            latest_result = float(numbers[-1].replace(",", ""))
        except ValueError:
            return 0.5

        # Check non-negativity (configurable)
        if not self.config.allow_negative and latest_result < 0:
            return 0.0

        # Check magnitude bounds
        entity_values = self.registry.entity_values()
        if not entity_values:
            return 0.5  # No reference values to compare against

        max_input = max(abs(v) for v in entity_values)
        upper_bound = max_input * self.config.bounds_multiplier

        if abs(latest_result) > upper_bound:
            return 0.0

        return 1.0

    def _check_answer_present(self) -> float:
        """f7: Check if the SLM has produced a final-answer marker.

        Returns:
            1.0 if an answer marker is found, 0.0 otherwise.
        """
        for pattern in self.ANSWER_PATTERNS:
            if pattern.search(self.cumulative_output):
                return 1.0
        return 0.0

    def get_initial_state(self) -> np.ndarray:
        """Return the state vector before any reasoning has occurred.

        This is the "blank" state that the navigator sees at episode start.
        """
        total = self.registry.total_entities()
        return np.array([
            0.0,    # f1: no entities referenced yet
            1.0,    # f2: all entities unused (normalised to 1.0)
            0.5,    # f3: no arithmetic yet — neutral
            0.0,    # f4: step progress = 0
            0.5,    # f5: no arithmetic to verify — uncertain
            1.0,    # f6: no result yet — assume plausible
            0.0,    # f7: no answer present
            0.0,    # f8: step count = 0
        ], dtype=np.float32)

    def reset(self) -> None:
        """Reset for a new episode with the same problem."""
        self.registry.reset_references()
        self.cumulative_output = ""
        self.step_outputs.clear()

    @property
    def feature_names(self) -> List[str]:
        """Human-readable names for each feature dimension."""
        return [
            "f1_relevance_ratio",
            "f2_unused_entities",
            "f3_entity_op_alignment",
            "f4_step_progress",
            "f5_arith_verify",
            "f6_bounds_plausible",
            "f7_answer_present",
            "f8_step_count",
        ]
