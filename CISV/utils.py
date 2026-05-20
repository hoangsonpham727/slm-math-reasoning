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

    # 7. Strip plain-text unit words immediately after a digit
    #    "9 hours × 15 dollars = 135" → "9 × 15 = 135"
    text = re.sub(
        r'(?<=[0-9])\s+(?:hours?|minutes?|seconds?|days?|weeks?|months?|years?|'
        r'dollars?|cents?|euros?|rupees?|'
        r'apples?|books?|miles?|cookies?|items?|pieces?|balls?|flowers?|coins?|'
        r'marbles?|pencils?|stars?|tickets?|gallons?|points?|pounds?)\b',
        '',
        text,
        flags=re.IGNORECASE,
    )

    return text


def _expand_paren_fractions(text: str) -> str:
    """Replace (A/B) parenthesised fractions with their decimal value.
    Turns '5 * (1/5) = 1' into '5 * 0.2 = 1' so the binary regex can match.
    """
    def _eval(m):
        try:
            num = float(m.group(1))
            den = float(m.group(2))
            return str(num / den) if den != 0 else m.group(0)
        except ValueError:
            return m.group(0)
    return re.sub(r'\(\s*([\d.]+)\s*/\s*([\d.]+)\s*\)', _eval, text)


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
    Extract arithmetic expressions from natural CoT text.

    Supports:
      • Binary:  a op b = c
      • Chained: a op b op c = d  (decomposed into two binary steps)

    Processing order:
      1. _normalize_latex  (strip $, LaTeX ops, unit words, commas)
      2. _substitute_symbolic_operands  (replace context variable names)
      3. Pass 1 — chained 3-operand regex; each match → two binary dicts
      4. Pass 2 — binary regex; positions covered by Pass 1 are skipped

    Results are returned sorted by their position in the text so that
    downstream chain-tracing and double-count detection see them in order.
    """
    text = _normalize_latex(text)
    text = _expand_paren_fractions(text)
    if context:
        text = _substitute_symbolic_operands(text, context)

    raw: list[tuple[int, dict]] = []   # (start_pos, expr_dict)
    covered: list[tuple[int, int]] = []  # spans claimed by chained matches

    def _norm_op(op: str) -> str:
        return op.replace('×', '*').replace('÷', '/')

    # ── Pass 1: 3-operand chained expressions  a op1 b op2 c = d ─────────────
    chain_pat = (
        r'(?<![\d/])([\d]+(?:\.[\d]+)?)\s*'  # a
        r'([+\-*/×÷])\s*'                     # op1
        r'([\d]+(?:\.[\d]+)?)\s*'             # b
        r'([+\-*/×÷])\s*'                     # op2
        r'([\d]+(?:\.[\d]+)?)\s*=\s*'         # c =
        r'([\d]+(?:\.[\d]+)?)'                # d (claimed)
    )
    for m in re.finditer(chain_pat, text):
        a   = float(m.group(1))
        op1 = _norm_op(m.group(2))
        b   = float(m.group(3))
        op2 = _norm_op(m.group(4))
        c   = float(m.group(5))
        d   = float(m.group(6))

        intermediate = compute_expression(a, op1, b)
        if intermediate is None:
            continue

        # Step 1: a op1 b = intermediate  (synthetic — no "= X" in text)
        raw.append((m.start(), {
            "left":       a,
            "op":         op1,
            "right":      b,
            "claimed":    round(intermediate, 10),
            "full_match": f"{m.group(1)} {m.group(2)} {m.group(3)}",
        }))
        # Step 2: intermediate op2 c = d
        raw.append((m.start() + 1, {
            "left":       round(intermediate, 10),
            "op":         op2,
            "right":      c,
            "claimed":    d,
            "full_match": m.group(0),
        }))
        covered.append(m.span())

    # ── Pass 2: binary expressions, skip positions covered by chained ─────────
    binary_pat = (
        r'(?<![\d/])([\d]+(?:\.[\d]+)?)\s*'
        r'([+\-*/×÷])\s*'
        r'([\d]+(?:\.[\d]+)?)\s*=\s*'
        r'([\d]+(?:\.[\d]+)?)'
    )
    for m in re.finditer(binary_pat, text):
        ms, me = m.span()
        if any(cs <= ms < ce or cs < me <= ce for cs, ce in covered):
            continue
        op = _norm_op(m.group(2))
        raw.append((ms, {
            "left":       float(m.group(1)),
            "op":         op,
            "right":      float(m.group(3)),
            "claimed":    float(m.group(4)),
            "full_match": m.group(0),
        }))

    # Sort by text position to preserve the order expressions appear in
    raw.sort(key=lambda x: x[0])
    return [d for _, d in raw]


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

    Guard: if the previous expression was already arithmetically wrong
    (status="corrected"), the current expression is likely a re-statement of
    the same computation with the corrected answer — not a genuine double-count.
    Example: model writes a template line "X - Y = X" then the actual
    computation "X - Y = Z (correct)". Without this guard the second line
    would be wrongly flagged as a double-count of the first.
    """
    if prev is None:
        return False
    # Don't fire if prev was itself arithmetically wrong — current likely fixes it.
    if prev.get("status") == "corrected":
        return False
    # Only flag multiplicative double-counts. Additive/subtractive repetition
    # (same bonus applied twice, same fixed cost on two items) is legitimate.
    if expr["op"] not in ('*', '/'):
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
            prev = entry   # use enriched entry so status is available
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
        prev = entry   # use enriched entry so status is available next iteration
    return log


def extract_last_number(text: str) -> float | None:
    """
    Fallback: return the last plain number in the text.
    Prefers \\boxed{N}, then evaluates plain-text fractions N/M,
    then falls back to the last bare number.
    Strips LaTeX fractions first so denominators are not mistaken for answers.
    """
    # 1. Prefer an explicit boxed numeric answer
    boxed = re.search(r'\\boxed\{([-\d.]+)\}', text)
    if boxed:
        try:
            return float(boxed.group(1))
        except ValueError:
            pass

    # 2. Remove LaTeX fractions (\frac{a}{b}) before scanning
    cleaned = re.sub(r'\\frac\{[^}]*\}\{[^}]*\}', '', text)

    # 3. Look for plain-text fractions N/M — find the LAST one and evaluate it
    #    Handles "ANSWER: 200/3" → 66.667, "remaining = 280/3" → 93.333
    frac_matches = list(re.finditer(
        r'([-]?\d+(?:\.\d+)?)\s*/\s*([-]?\d+(?:\.\d+)?)', cleaned
    ))
    if frac_matches:
        last = frac_matches[-1]
        try:
            den = float(last.group(2))
            if den != 0:
                return float(last.group(1)) / den
        except ValueError:
            pass

    # 4. Fall back to last bare number
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
