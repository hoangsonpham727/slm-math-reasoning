"""
CVR utilities.

Re-exports existing answer extraction functions from the project so CVR
code has a single import point. Adds step-parsing helpers specific to CVR.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Ensure repo root is on the path for cross-package imports.
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Lazy import for parse_answer (Experiment2 has no problematic transitive deps).
def _import_parse_answer():
    from Experiment2.dataset_generator import parse_answer as _pa
    return _pa


def parse_answer(response: str):
    """Float answer extraction for Exp2-style problems. Source: Experiment2/dataset_generator.py."""
    return _import_parse_answer()(response)


# ── Answer extraction functions ──────────────────────────────────────────────
# These are copied from Experiment1/run_inference_eval.py (pure regex, no deps).
# Experiment1/run_inference_eval.py imports data_preparation which transitively
# requires google-genai; we cannot safely do a module-level import of that file.
# Source: Experiment1/run_inference_eval.py:38–206

_RE_BOXED_START = re.compile(r"\\boxed\{")
_RE_CHAT_TURN = re.compile(r"(?i)\n(?:User|Assistant|System):\s*")
_RE_ANSWER_PHRASES = (
    re.compile(r"(?is)(?:\*\*\s*)?Final answer\s*:\s*([^\n]+)"),
    re.compile(r"(?is)The\s+final\s+answer\s+is\s*:?\s*([^\n]+)"),
    re.compile(r"(?is)The\s+answer\s+is\s*:?\s*([^\n]+)"),
    re.compile(r"(?im)^\s*Answer:\s*([^\n]+)"),
)
_TAIL_MAX_LINES = 25
_TAIL_MAX_CHARS = 2000
_RE_NUMERIC_TOKEN = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _primary_solution_text(text: str) -> str:
    m = _RE_CHAT_TURN.search(text)
    return text[: m.start()] if m else text


def _extraction_tail(text: str) -> str:
    lines = text.splitlines()
    tail = "\n".join(lines[-_TAIL_MAX_LINES:]) if len(lines) > _TAIL_MAX_LINES else text
    return tail[-_TAIL_MAX_CHARS:] if len(tail) > _TAIL_MAX_CHARS else tail


def _extract_answer_phrase_last(text: str):
    best = None
    for pat in _RE_ANSWER_PHRASES:
        for m in pat.finditer(text):
            if best is None or m.start() > best.start():
                best = m
    if best is None:
        return None
    s = best.group(1).strip()
    return s or None


def _extract_numeric_candidate(text: str):
    if not text:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"(?i)^(?:is|=)\s*", "", cleaned)
    cleaned = cleaned.replace("$", "").replace("%", "").replace(",", "")
    cleaned = re.sub(r"\\+", "", cleaned).strip()
    if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", cleaned):
        return cleaned
    matches = _RE_NUMERIC_TOKEN.findall(cleaned)
    return matches[-1].replace(",", "") if matches else None


def _extract_boxed_last(text: str):
    best = None
    for m in _RE_BOXED_START.finditer(text):
        pos = m.end()
        depth = 1
        j = pos
        while j < len(text) and depth:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            j += 1
        if depth == 0:
            best = text[pos: j - 1]
    return best


def extract_gsm_final(answer: str):
    """Extract gold label after the last #### line in a GSM-style answer string."""
    if not answer or not answer.strip():
        return None
    matches = list(re.finditer(r"####\s*(.+)", answer, flags=re.MULTILINE))
    if not matches:
        return None
    gsm = matches[-1].group(1).strip()
    return _extract_numeric_candidate(gsm) or gsm


def extract_model_final(text: str):
    """Multi-stage answer extraction: ####, phrases, \\boxed{}, last number in tail."""
    if not text:
        return None
    text = _primary_solution_text(text)
    if not text.strip():
        return None
    m = list(re.finditer(r"####\s*(.+)", text, flags=re.MULTILINE))
    if m:
        gsm = m[-1].group(1).strip()
        return _extract_numeric_candidate(gsm) or gsm
    phrase = _extract_answer_phrase_last(text)
    if phrase:
        parsed = _extract_numeric_candidate(phrase)
        return parsed if parsed is not None else phrase
    boxed = _extract_boxed_last(text)
    if boxed is not None:
        parsed = _extract_numeric_candidate(boxed)
        if parsed is not None:
            return parsed
        s = boxed.strip()
        if s:
            return s
    tail = _extraction_tail(text)
    candidates = _RE_NUMERIC_TOKEN.findall(tail)
    return candidates[-1].replace(",", "") if candidates else None


def _normalize_answer(s: str) -> str:
    t = s.strip().replace(",", "").replace("$", "").replace("%", "")
    t = re.sub(r"\\+", "", t).strip()
    tm = t.replace(",", "")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", tm):
        return tm
    num = re.search(r"-?\d+(?:\.\d+)?", tm)
    return num.group(0) if num else t


def answers_match(gold: str, pred, eps: float = 1e-5) -> bool:
    """Numeric-tolerant answer comparison. Source: Experiment1/run_inference_eval.py."""
    if pred is None:
        return False
    g = _normalize_answer(gold)
    p = _normalize_answer(pred)
    if not g or not p:
        return g == p
    try:
        gf, pf = float(g), float(p)
        return abs(gf - pf) <= eps * max(1.0, abs(gf), abs(pf))
    except ValueError:
        return g.casefold() == p.casefold()

__all__ = [
    "extract_model_final",
    "extract_gsm_final",
    "answers_match",
    "parse_answer",
    "parse_step_text",
    "format_steps_as_text",
    "parse_binary_vote",
    "is_final_step",
]

# ---------------------------------------------------------------------------
# Step parsing
# ---------------------------------------------------------------------------

_STEP_HEADER = re.compile(r"Step\s*(\d+)\s*:", re.IGNORECASE)


def parse_step_text(raw_output: str, expected_step_num: int) -> str:
    """
    Extract the content of step `expected_step_num` from raw model output.

    SLMs sometimes generate multiple steps at once. This function truncates
    the output at the start of the *next* step header (Step N+1:) so only
    the current step is returned.
    """
    raw_output = raw_output.strip()
    # Find the start of Step (expected_step_num + 1) and cut there.
    next_step = expected_step_num + 1
    pattern = re.compile(rf"Step\s*{next_step}\s*:", re.IGNORECASE)
    m = pattern.search(raw_output)
    if m:
        raw_output = raw_output[: m.start()].strip()
    # Also strip the "Step N:" prefix if the model included it.
    current_pattern = re.compile(rf"^\s*Step\s*{expected_step_num}\s*:\s*", re.IGNORECASE)
    raw_output = current_pattern.sub("", raw_output, count=1).strip()
    return raw_output


def format_steps_as_text(steps: list[dict]) -> str:
    """Format a list of verified step dicts into readable text for prompts."""
    lines = [f"Step {s['index']}: {s['text']}" for s in steps]
    return "\n".join(lines)


def parse_binary_vote(raw_token: str) -> bool:
    """
    Parse the first token of a NCV binary response.

    Returns True for 'Correct'/'Yes'/'Right'/'True'.
    Returns False for 'Wrong'/'No'/'Incorrect'/'False' or anything unparseable
    (conservative: unknown → fail).
    """
    tok = raw_token.strip().split()[0].lower() if raw_token.strip() else ""
    tok = re.sub(r"[^a-z]", "", tok)  # strip punctuation
    return tok in {"correct", "yes", "right", "true"}


def is_final_step(step_text: str) -> bool:
    """
    Heuristic: detect whether a step contains the final answer.
    Triggers on common answer-signalling phrases.
    """
    text_lower = step_text.lower()
    patterns = [
        r"####",
        r"the\s+(?:final\s+)?answer\s+is",
        r"therefore[,\s]",
        r"in\s+total",
        r"altogether",
        r"answer\s*:",
        r"\\boxed\{",
    ]
    return any(re.search(p, text_lower) for p in patterns)


def load_yaml_config(path: str) -> dict:
    """Load a YAML config file. Requires PyYAML."""
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)
