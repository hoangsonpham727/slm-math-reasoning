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
    "parse_all_steps",
    "format_steps_as_text",
    "parse_binary_vote",
    "is_final_step",
]

# ---------------------------------------------------------------------------
# Step parsing
# ---------------------------------------------------------------------------

# Matches any "Step N:" header, with optional surrounding markdown bold markers.
_STEP_HEADER = re.compile(r"\*{0,2}\s*Step\s*(\d+)\s*[:.]\*{0,2}", re.IGNORECASE)
# Matches numbered-list intro lines like "1. Do something" or "2) Do something".
_NUMBERED_LIST_LINE = re.compile(r"^\s*\d+[.)]\s+\S")
# Matches markdown bold/header markers at line start.
_MD_HEADER = re.compile(r"^\s*\*{1,2}|^\s*#{1,3}\s")


def parse_step_text(raw_output: str, expected_step_num: int) -> str:
    """
    Extract the content of step `expected_step_num` from raw model output.

    Handles three common failure modes for small models:
    1. Multiple steps generated at once — truncated at the next step boundary.
    2. Markdown bold step headers like **Step 2:** — handled by _STEP_HEADER regex.
    3. Numbered-list preamble ("1. Calculate X\n2. Calculate Y\n...") before the
       actual arithmetic — stripped by discarding lines that look like list items
       without numeric content.
    """
    raw_output = raw_output.strip()

    # Truncate at the next step boundary (handles plain and bold markdown headers).
    next_step = expected_step_num + 1
    boundary = re.compile(rf"\*{{0,2}}\s*Step\s*{next_step}\s*[:.]\*{{0,2}}", re.IGNORECASE)
    m = boundary.search(raw_output)
    if m:
        raw_output = raw_output[: m.start()].strip()

    # If the current step header appears somewhere mid-text (model wrote a preamble
    # before the actual step content), take only what comes after the header.
    current_header = re.compile(
        rf"\*{{0,2}}\s*Step\s*{expected_step_num}\s*[:.][^\n]*\*{{0,2}}",
        re.IGNORECASE,
    )
    m2 = current_header.search(raw_output)
    if m2:
        raw_output = raw_output[m2.end():].strip()

    # Drop pure numbered-list lines that are a plan/outline with no arithmetic.
    # Keep a line if the body (after "N. ") contains any digit.
    filtered_lines = []
    for line in raw_output.splitlines():
        stripped = line.strip()
        if not stripped:
            filtered_lines.append(line)
            continue
        if _NUMBERED_LIST_LINE.match(stripped):
            body = re.sub(r"^\d+[.)]\s*", "", stripped)
            if not re.search(r"\d", body):
                continue  # pure label like "1. Calculate the fee" — drop it
        filtered_lines.append(line)

    return "\n".join(filtered_lines).strip()


def format_steps_as_text(steps: list[dict]) -> str:
    """Format a list of verified step dicts into readable text for prompts."""
    lines = [f"Step {s['index']}: {s['text']}" for s in steps]
    return "\n".join(lines)


_VOTE_CORRECT_RE = re.compile(r"\b(correct|yes|right|true)\b", re.IGNORECASE)
_VOTE_WRONG_RE = re.compile(r"\b(wrong|no|incorrect|false)\b", re.IGNORECASE)


def parse_binary_vote(raw_output: str) -> bool:
    """
    Parse a NCV binary verification response.

    Small models (1.5B–4B) often output preamble before the verdict word, so we
    search the full response with word-boundary regex and take the LAST match
    from each verdict group, then compare positions.

    Using word boundaries avoids 'correct' matching inside 'incorrect', or
    'no' matching inside 'known'.

    Returns True  for last verdict in {correct, yes, right, true}.
    Returns False for last verdict in {wrong, no, incorrect, false} or unparseable
    (conservative: unknown → fail).
    """
    if not raw_output or not raw_output.strip():
        return False

    correct_matches = list(_VOTE_CORRECT_RE.finditer(raw_output))
    wrong_matches = list(_VOTE_WRONG_RE.finditer(raw_output))

    if not correct_matches and not wrong_matches:
        return False  # unparseable → conservative fail

    last_correct = correct_matches[-1].start() if correct_matches else -1
    last_wrong = wrong_matches[-1].start() if wrong_matches else -1
    return last_correct > last_wrong


def parse_all_steps(raw_output: str) -> list[dict]:
    """
    Parse a complete model solution into a list of step dicts.

    Finds all 'Step N:' headers (including markdown bold variants) and splits
    the output at those boundaries. Returns [{"index": N, "text": "..."}, ...].
    If no step headers are found, returns the entire output as step 1.
    """
    raw_output = raw_output.strip()
    matches = list(_STEP_HEADER.finditer(raw_output))
    if not matches:
        return [{"index": 1, "text": raw_output}]

    steps = []
    for i, match in enumerate(matches):
        step_num = int(match.group(1))
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_output)
        text = raw_output[content_start:content_end].strip()
        steps.append({"index": step_num, "text": text})
    return steps


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
