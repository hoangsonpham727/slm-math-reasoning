"""
Shared utilities for Experiment 4 — Chunked Incremental Solve.
"""

import json
import re
import os
from pathlib import Path


# ── Arithmetic extraction ────────────────────────────────────────────────────

def _normalize_latex(text: str) -> str:
    """
    Convert LaTeX math notation to plain arithmetic so the regex can match.

    Processing order:
      1. \\frac{a}{b}  →  decimal
      2. \\text{...}, \\mathrm{...}, \\mathbf{...}  →  discard unit words
      3. $ and \\$  →  removed
      4. \\times → ×,  \\div → ÷,  \\cdot → ×
      5. Commas in numbers: 1,000 → 1000
    """
    # 1. \frac{a}{b} → decimal  (all fractions, including those before '=')
    def _frac(m):
        try:
            num, den = float(m.group(1)), float(m.group(2))
            return str(num / den) if den != 0 else "0"
        except ValueError:
            return m.group(0)
    text = re.sub(r'\\frac\{([\d.]+)\}\{([\d.]+)\}', _frac, text)

    # 3. Strip LaTeX wrappers: \text{...}, \mathrm{...}, \mathbf{...}
    #    Keep numeric content, discard unit words like "dollars", "hours"
    def _unwrap_text(m):
        inner = m.group(1).strip()
        # If inner is purely numeric, keep it
        if re.match(r'^[\d.]+$', inner):
            return inner
        # Otherwise discard (it's a unit label like " dollars")
        return ' '
    text = re.sub(r'\\(?:text|mathrm|mathbf)\{([^}]*)\}', _unwrap_text, text)

    # 4. Strip dollar signs (both $ and \$)
    text = text.replace(r'\$', '')
    text = text.replace('$', '')

    # 5. Operator conversions
    text = text.replace(r'\times', '×').replace(r'\div', '÷')
    text = text.replace(r'\cdot', '×')

    # 6. Remove commas from numbers (1,000 → 1000)
    text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
    text = re.sub(r'(\d),(\d{3})', r'\1\2', text)  # handle millions

    return text


def _is_numeric_key(k: str) -> bool:
    """Check if a string is a numeric value (used to filter context keys)."""
    try:
        float(k)
        return True
    except ValueError:
        return False


def _substitute_symbolic_operands(text: str, context: dict) -> str:
    """
    Replace known variable names with their numeric values so the arithmetic
    regex can match symbolic expressions like 'current_total - 4 = 28'.

    Only substitutes non-numeric, non-dunder context keys that map to numbers.
    Sorts by length descending to avoid partial matches.
    """
    subs = {
        k: str(round(v, 10) if isinstance(v, float) else v)
        for k, v in context.items()
        if not k.startswith("__") and not _is_numeric_key(k)
           and isinstance(v, (int, float))
    }
    for name in sorted(subs.keys(), key=len, reverse=True):
        text = re.sub(r'\b' + re.escape(name) + r'\b', subs[name], text)
    return text


def extract_arithmetic_expressions(text: str, context: dict | None = None) -> list[dict]:
    """
    Extract all 'a op b = c' patterns from natural CoT text.
    Normalises LaTeX operators (\\times, \\div, \\frac) before matching.
    If context is provided, substitutes symbolic variable names with their
    numeric values before regex matching.
    Returns list of dicts: left, op, right, claimed, full_match.
    """
    text = _normalize_latex(text)
    if context:
        text = _substitute_symbolic_operands(text, context)
    pattern = (
        r'([\d]+(?:\.[\d]+)?)\s*'
        r'([+\-*/×÷])\s*'
        r'([\d]+(?:\.[\d]+)?)\s*=\s*'
        r'([\d]+(?:\.[\d]+)?)'
    )
    results = []
    for m in re.finditer(pattern, text):
        op = m.group(2).replace('×', '*').replace('÷', '/')
        results.append({
            "left":        float(m.group(1)),
            "op":          op,
            "right":       float(m.group(3)),
            "claimed":     float(m.group(4)),
            "full_match":  m.group(0),
        })
    return results


def compute_expression(left: float, op: str, right: float) -> float | None:
    """Evaluate a single arithmetic expression. Returns None on error."""
    try:
        if op == '+': return left + right
        if op == '-': return left - right
        if op == '*': return left * right
        if op == '/': return left / right if right != 0 else None
    except Exception:
        return None


def resolve_operand(value: float, context: dict) -> float:
    """
    If value matches a stored context key, return the corrected value.
    Otherwise return as-is. Handles downstream propagation automatically.
    """
    return context.get(str(round(value, 10)), value)


def _is_double_count(expr: dict, prev: dict, tolerance: float = 1e-4) -> bool:
    """
    Detect the double-counting pattern where a model re-applies an operand
    that was already consumed in the previous expression.

    Classic Phi4 failure:
        6 * 5 = 30     ← correct
        30 * 5 = 150   ← re-uses 5 that was already multiplied in step above

    Signature:
      - current left  ≈ previous claimed result  (chains on the result)
      - current right ≈ previous right OR left   (reuses an already-used operand)
      - same operator
    """
    if prev is None:
        return False
    left_chains   = abs(expr["left"]  - prev["claimed"]) <= tolerance
    right_reused  = (abs(expr["right"] - prev["right"]) <= tolerance or
                     abs(expr["right"] - prev["left"])  <= tolerance)
    same_op       = expr["op"] == prev["op"]
    return left_chains and right_reused and same_op


def verify_and_correct_expressions(
    expressions: list[dict],
    context: dict,
    tolerance: float = 1e-4,
) -> list[dict]:
    """
    Verify each arithmetic expression the SLM produced (read-only audit).

    Does NOT mutate context or resolve operands against previous corrections,
    because aggressive propagation causes collisions when the same numeric
    value appears in unrelated semantic roles (e.g. "135" as both total
    earnings and a wrong claimed result).

    Each expression is verified using its literal operands.  The caller
    decides what value to carry forward (prefer ANSWER: tag, then last
    non-double-counted corrected_value).

    Adds status="double_count" when the model re-applies an already-consumed
    operand (e.g. Phi4 double-multiplying "hours" in a rate × time problem).
    """
    log = []
    prev = None
    for expr in expressions:
        left = expr["left"]
        right = expr["right"]
        computed = compute_expression(left, expr["op"], right)

        if computed is None:
            entry = {
                **expr,
                "resolved_left":  left,
                "resolved_right": right,
                "computed":       None,
                "status":         "error",
                "corrected_value": None,
            }
            log.append(entry)
            prev = expr
            continue

        is_correct = abs(computed - expr["claimed"]) <= tolerance
        corrected_value = computed if not is_correct else expr["claimed"]

        # Detect double-count AFTER verifying arithmetic (it may be arithmetically
        # correct but semantically wrong — the hallmark of this failure mode).
        if _is_double_count(expr, prev, tolerance):
            status = "double_count"
            # The correct carry-forward value is the PREVIOUS expression's result,
            # not this one. Store it so the caller can use it directly.
            corrected_value = prev["claimed"]
        else:
            status = "correct" if is_correct else "corrected"

        entry = {
            **expr,
            "resolved_left":  left,
            "resolved_right": right,
            "computed":       round(computed,         10),
            "status":         status,
            "corrected_value": round(corrected_value, 10),
        }
        log.append(entry)
        prev = expr
    return log


def extract_last_number(text: str) -> float | None:
    """
    Fallback: return the last plain number in the text.
    Prefers \\boxed{N}. Strips LaTeX fractions first so denominators
    like the '2' in \\frac{3}{2}M are never mistaken for the answer.
    """
    # Prefer an explicit boxed numeric answer
    boxed = re.search(r'\\boxed\{([-\d.]+)\}', text)
    if boxed:
        try:
            return float(boxed.group(1))
        except ValueError:
            pass
    # Remove LaTeX fractions (\frac{a}{b}) before scanning for numbers
    cleaned = re.sub(r'\\frac\{[^}]*\}\{[^}]*\}', '', text)
    numbers = re.findall(r'[-]?\d+(?:\.\d+)?', cleaned)
    try:
        return float(numbers[-1]) if numbers else None
    except ValueError:
        return None


# ── Sentence splitting and chunking ─────────────────────────────────────────

def split_into_clauses(problem_text: str) -> list[str]:
    """Split problem text into individual sentences."""
    text = problem_text.replace('\n', '. ')
    parts = re.split(r'(?<=[.?!])\s+', text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]


def chunk_clauses(clauses: list[str], chunk_size: int = 2) -> list[list[str]]:
    """Group clauses into chunks of at most chunk_size."""
    return [clauses[i:i + chunk_size] for i in range(0, len(clauses), chunk_size)]


# ── Answer comparison ────────────────────────────────────────────────────────

def answers_match(predicted: str | None, gold: str | None, eps: float = 0.01) -> bool:
    """Numeric-tolerant comparison. Returns False on parse failure."""
    if predicted is None or gold is None:
        return False
    try:
        return abs(float(predicted) - float(gold)) <= eps
    except (ValueError, TypeError):
        return str(predicted).strip() == str(gold).strip()


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_dataset(path: str) -> list[dict]:
    """Load a JSON or JSONL dataset file."""
    path = Path(path)
    with open(path) as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def save_results(results: list[dict], output_path: str) -> None:
    """Save results list to a JSON file, creating parent dirs as needed."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def checkpoint_results(results: list[dict], output_path: str) -> None:
    """
    Write a numbered checkpoint file alongside the main output path.
    File is named <stem>_ckpt_<N><suffix> where N = len(results).
    """
    p = Path(output_path)
    ckpt_path = p.parent / f"{p.stem}_ckpt_{len(results)}{p.suffix}"
    with open(ckpt_path, "w") as f:
        json.dump(results, f, indent=2)
