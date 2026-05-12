"""
entity_registry.py — Extract and track mathematical entities from problem text.

This module parses a math word problem to identify all named numerical
quantities (e.g., "5 apples", "$3.50 per hour", "12 students") and builds
a registry that the feature extractor queries at each reasoning step.

The registry answers two questions:
  1. What entities exist in the problem? (for computing relevance ratio f1)
  2. Which entities has the SLM referenced so far? (for unused-count f2)

Design choice: pure regex, no neural model. This is deliberate —
the entity registry must be fast (runs every step), deterministic,
and not subject to the same failure modes as the SLM itself.
"""

import re
from dataclasses import dataclass, field
from typing import List, Set, Tuple


@dataclass
class MathEntity:
    """A single numerical entity extracted from the problem text.

    Attributes:
        value:      The numerical value as a float (e.g., 3.5)
        raw_text:   The original text span (e.g., "$3.50")
        context:    Surrounding noun phrase (e.g., "per hour")
        position:   Character offset in the original problem
    """
    value: float
    raw_text: str
    context: str
    position: int


class EntityRegistry:
    """Maintains the set of mathematical entities for one problem.

    Usage:
        registry = EntityRegistry(problem_text)
        # After each SLM reasoning step:
        registry.update_references(slm_output_step_n)
        ratio = registry.relevance_ratio()
        unused = registry.unused_count()
    """

    # Pattern matches integers, decimals, fractions, currency, and percentages.
    # Examples matched: "42", "3.14", "$5.99", "1/2", "75%", "1,000"
    NUMBER_PATTERN = re.compile(
        r"""
        (?<!\w)                       # Not preceded by a word character
        (?:
            \$\s*                      # Optional dollar sign
        )?
        (?:
            \d{1,3}(?:,\d{3})*        # Integer with comma separators: 1,000
            (?:\.\d+)?                 # Optional decimal part
            |
            \d+\.\d+                   # Plain decimal: 3.14
            |
            \d+/\d+                    # Fraction: 1/2
            |
            \d+                        # Plain integer: 42
        )
        (?:\s*%)?                      # Optional percent sign
        (?!\w)                         # Not followed by a word character
        """,
        re.VERBOSE,
    )

    # Words that surround a number and give it semantic context.
    # We capture up to 4 words on each side of the number.
    CONTEXT_WINDOW = 4

    def __init__(self, problem_text: str):
        """Parse the problem text and build the entity registry.

        Args:
            problem_text: The full math word problem string.
        """
        self.problem_text = problem_text
        self.entities: List[MathEntity] = []
        self.referenced_indices: Set[int] = set()

        self._extract_entities()

    def _extract_entities(self) -> None:
        """Find all numerical entities in the problem text."""
        words = self.problem_text.split()

        for match in self.NUMBER_PATTERN.finditer(self.problem_text):
            raw = match.group().strip()
            value = self._parse_value(raw)
            if value is None:
                continue

            # Extract context: the noun phrase surrounding this number.
            pos = match.start()
            context = self._get_context(pos, raw)

            self.entities.append(MathEntity(
                value=value,
                raw_text=raw,
                context=context,
                position=pos,
            ))

    def _parse_value(self, raw: str) -> float | None:
        """Convert a raw number string to a float.

        Handles: "$5.99" -> 5.99, "1,000" -> 1000, "75%" -> 75,
                 "1/2" -> 0.5, "42" -> 42.0
        """
        try:
            # Strip currency and percent symbols
            cleaned = raw.replace("$", "").replace("%", "").replace(",", "").strip()

            # Handle fractions
            if "/" in cleaned:
                parts = cleaned.split("/")
                return float(parts[0]) / float(parts[1])

            return float(cleaned)
        except (ValueError, ZeroDivisionError):
            return None

    def _get_context(self, position: int, raw: str) -> str:
        """Extract the noun-phrase context around a number.

        Takes up to CONTEXT_WINDOW words before and after the number
        in the original problem text.
        """
        # Find word boundaries around the number
        before = self.problem_text[:position].split()
        after = self.problem_text[position + len(raw):].split()

        ctx_before = before[-self.CONTEXT_WINDOW:] if before else []
        ctx_after = after[:self.CONTEXT_WINDOW] if after else []

        return " ".join(ctx_before + [raw] + ctx_after)

    def update_references(self, slm_output: str) -> None:
        """Check which entities the SLM has referenced in its latest output.

        We consider an entity "referenced" if its numerical value appears
        in the SLM's output. We match by value (not raw text) to handle
        cases where the SLM reformats numbers (e.g., "1,000" -> "1000").

        Args:
            slm_output: The SLM's output from the latest reasoning step.
        """
        # Extract all numbers from the SLM's output
        output_numbers = set()
        for match in self.NUMBER_PATTERN.finditer(slm_output):
            val = self._parse_value(match.group().strip())
            if val is not None:
                output_numbers.add(val)

        # Check each entity against the output numbers
        for i, entity in enumerate(self.entities):
            if entity.value in output_numbers:
                self.referenced_indices.add(i)
            # Also check for close floating-point matches
            elif any(abs(entity.value - n) < 1e-6 for n in output_numbers):
                self.referenced_indices.add(i)

    def relevance_ratio(self) -> float:
        """f1: Fraction of problem entities referenced by the SLM so far.

        Returns:
            Float in [0, 1]. Higher = SLM is using more of the problem data.
            Returns 1.0 if problem has no entities (edge case).
        """
        if not self.entities:
            return 1.0
        return len(self.referenced_indices) / len(self.entities)

    def unused_count(self) -> int:
        """f2: Number of problem entities NOT yet referenced.

        Returns:
            Non-negative integer. Lower = SLM is more complete.
        """
        return len(self.entities) - len(self.referenced_indices)

    def entity_values(self) -> List[float]:
        """Return all numerical values for bounds-checking (used by f6)."""
        return [e.value for e in self.entities]

    def total_entities(self) -> int:
        """Total number of entities in the problem."""
        return len(self.entities)

    def reset_references(self) -> None:
        """Clear all reference tracking (for re-use across episodes)."""
        self.referenced_indices.clear()

    def __repr__(self) -> str:
        refs = len(self.referenced_indices)
        total = len(self.entities)
        return f"EntityRegistry({refs}/{total} referenced)"
