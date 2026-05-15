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
      \\frac{a}{b}  →  decimal string  (e.g. \\frac{1}{4} → 0.25)
      \\times       →  ×
      \\div         →  ÷
    """
    def _frac(m):
        try:
            num, den = float(m.group(1)), float(m.group(2))
            return str(num / den) if den != 0 else "0"
        except ValueError:
            return m.group(0)

    text = re.sub(r'\\frac\{([\d.]+)\}\{([\d.]+)\}', _frac, text)
    text = text.replace(r'\times', '×').replace(r'\div', '÷')
    return text


def extract_arithmetic_expressions(text: str) -> list[dict]:
    """
    Extract all 'a op b = c' patterns from natural CoT text.
    Normalises LaTeX operators (\\times, \\div, \\frac) before matching.
    Returns list of dicts: left, op, right, claimed, full_match.
    """
    text = _normalize_latex(text)
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


def verify_and_correct_expressions(
    expressions: list[dict],
    context: dict,
    tolerance: float = 1e-4,
) -> list[dict]:
    """
    Verify each expression and correct arithmetic errors.
    For each expression:
      1. Resolve operands against context (replaces wrong SLM values
         with previously corrected values).
      2. Compute the true result.
      3. If claimed != computed, CORRECT it and propagate forward.
    Context is MUTATED in-place so subsequent expressions and
    subsequent chunks can reference corrected values.
    """
    log = []
    for expr in expressions:
        left  = resolve_operand(expr["left"],  context)
        right = resolve_operand(expr["right"], context)
        computed = compute_expression(left, expr["op"], right)

        if computed is None:
            log.append({
                **expr,
                "resolved_left":  left,
                "resolved_right": right,
                "computed":       None,
                "status":         "error",
                "corrected_value": None,
            })
            continue

        is_correct = abs(computed - expr["claimed"]) <= tolerance
        corrected_value = computed if not is_correct else expr["claimed"]

        # Map the SLM's (possibly wrong) claimed value → corrected value
        # so downstream expressions that cite the wrong number get fixed.
        context[str(round(expr["claimed"],     10))] = corrected_value
        context[str(round(corrected_value,     10))] = corrected_value
        context["__last__"] = corrected_value

        log.append({
            **expr,
            "resolved_left":  left,
            "resolved_right": right,
            "computed":       round(computed,         10),
            "status":         "correct" if is_correct else "corrected",
            "corrected_value": round(corrected_value, 10),
        })
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
