"""
Experiment 3: Distractor Filtering

Two-stage pipeline on the enhanced GSM8K dataset:
  Stage 1 — Filter: ask the model to label each clause RELEVANT or IRRELEVANT.
  Stage 2 — Solve: run CoT solve on the relevant-only problem text.

Usage:
    python run_experiment3.py [--models all] [--enhanced_dir ../Experiment1/gsm_enhanced_templates]
                              [--output_dir results] [--device cuda:0]
                              [--max_new_tokens 1024] [--limit None] [--no_resume]
                              [--distractor_seed 42]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from models import get_all_configs, get_model_wrapper, ModelConfig


# ---------------------------------------------------------------------------
# Inlined helpers from Experiment1 (avoids importing data_preparation.py
# which requires google.genai / ollama that are not needed here)
# ---------------------------------------------------------------------------

def _append_distractor(question: str, distractor, position: str = "mid") -> str:
    q = question.strip()
    if isinstance(distractor, dict):
        distractor = str(distractor.get("text", ""))
    d = str(distractor).strip().rstrip(".")
    if not d:
        return q
    if position == "end":
        return f"{q[:-1]}, {d}?" if q.endswith("?") else f"{q} {d}."
    sentences = re.split(r'(?<=[.!])\s+', q)
    if len(sentences) <= 1:
        return f"{q[:-1]}, {d}?" if q.endswith("?") else f"{q} {d}."
    sentences.insert(len(sentences) - 1, d + ".")
    return " ".join(sentences)


def load_enhanced_distracted_examples(enhanced_dir: str, seed: int = 42) -> list[dict]:
    import glob as _glob
    import random as _random
    rng = _random.Random(seed)
    paths = sorted(_glob.glob(os.path.join(enhanced_dir, "*.json")))
    out = []
    for filepath in paths:
        with open(filepath, encoding="utf-8") as f:
            record = json.load(f)
        pool = record.get("dynamic_distractor_pool") or []
        if not pool:
            continue
        original_q = record.get("question", "")
        a = record.get("answer", "")
        if not original_q or not a:
            continue
        chosen = rng.choice(pool)
        distractor_text = chosen.get("text", "") if isinstance(chosen, dict) else str(chosen)
        distractor_type = chosen.get("type", "UNKNOWN") if isinstance(chosen, dict) else "UNKNOWN"
        distracted = _append_distractor(original_q, distractor_text)
        out.append({
            "question": distracted,
            "answer": a,
            "question_original": original_q,
            "distractor": distractor_text,
            "distractor_type": distractor_type,
            "id_orig": record.get("id_orig"),
            "source_file": os.path.basename(filepath),
        })
    return out


_RE_BOXED_START = re.compile(r"\\boxed\{")
_RE_CHAT_TURN   = re.compile(r"(?i)\n(?:User|Assistant|System):\s*")
_RE_ANSWER_PHRASES = (
    re.compile(r"(?is)(?:\*\*\s*)?Final answer\s*:\s*([^\n]+)"),
    re.compile(r"(?is)The\s+final\s+answer\s+is\s*:?\s*([^\n]+)"),
    re.compile(r"(?is)The\s+answer\s+is\s*:?\s*([^\n]+)"),
    re.compile(r"(?im)^\s*Answer:\s*([^\n]+)"),
)
_RE_NUMERIC_TOKEN = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _primary_solution_text(text: str) -> str:
    m = _RE_CHAT_TURN.search(text)
    return text[: m.start()] if m else text


def _extract_boxed_last(text: str):
    best = None
    for m in _RE_BOXED_START.finditer(text):
        pos, depth, j = m.end(), 1, m.end()
        while j < len(text) and depth:
            depth += (text[j] == "{") - (text[j] == "}")
            j += 1
        if depth == 0:
            best = text[pos: j - 1]
    return best


def _extract_numeric_candidate(text: str):
    if not text:
        return None
    cleaned = re.sub(r"(?i)^(?:is|=)\s*", "", text.strip())
    cleaned = cleaned.replace("$", "").replace("%", "").replace(",", "")
    cleaned = re.sub(r"\\+", "", cleaned).strip()
    if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", cleaned):
        return cleaned
    matches = _RE_NUMERIC_TOKEN.findall(cleaned)
    return matches[-1].replace(",", "") if matches else None


def extract_gsm_final(answer: str):
    if not answer or not answer.strip():
        return None
    matches = list(re.finditer(r"####\s*(.+)", answer, flags=re.MULTILINE))
    if not matches:
        return None
    return _extract_numeric_candidate(matches[-1].group(1).strip()) or matches[-1].group(1).strip()


def extract_model_final(text: str):
    if not text:
        return None
    text = _primary_solution_text(text)
    if not text.strip():
        return None
    m = list(re.finditer(r"####\s*(.+)", text, flags=re.MULTILINE))
    if m:
        return _extract_numeric_candidate(m[-1].group(1).strip()) or m[-1].group(1).strip()
    best = None
    for pat in _RE_ANSWER_PHRASES:
        for hit in pat.finditer(text):
            if best is None or hit.start() > best.start():
                best = hit
    if best:
        parsed = _extract_numeric_candidate(best.group(1).strip())
        if parsed is not None:
            return parsed
        return best.group(1).strip() or None
    boxed = _extract_boxed_last(text)
    if boxed is not None:
        parsed = _extract_numeric_candidate(boxed)
        if parsed is not None:
            return parsed
        return boxed.strip() or None
    tail_lines = text.splitlines()[-25:]
    tail = "\n".join(tail_lines)[-2000:]
    candidates = _RE_NUMERIC_TOKEN.findall(tail)
    return candidates[-1].replace(",", "") if candidates else None


def _normalize_answer(s: str) -> str:
    t = s.strip().replace(",", "").replace("$", "").replace("%", "")
    t = re.sub(r"\\+", "", t).strip()
    if re.fullmatch(r"-?\d+(?:\.\d+)?", t):
        return t
    num = re.search(r"-?\d+(?:\.\d+)?", t)
    return num.group(0) if num else t


def answers_match(gold: str, pred, eps: float = 1e-5) -> bool:
    if pred is None:
        return False
    g = _normalize_answer(gold)
    p = _normalize_answer(str(pred))
    if not g or not p:
        return g == p
    try:
        return abs(float(g) - float(p)) <= eps * max(1.0, abs(float(g)), abs(float(p)))
    except ValueError:
        return g.casefold() == p.casefold()


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

FILTER_SYSTEM = (
    "You are a math problem analyzer. "
    "Your job is to identify which parts of a problem are needed to answer "
    "the question and which are irrelevant distractors."
)

FILTER_USER = """Problem:
{problem}

Instructions:
1. Break the problem into individual clauses (one per line).
2. For each clause, label it RELEVANT or IRRELEVANT.
   - RELEVANT: the clause contains a quantity or fact directly needed to answer the question.
   - IRRELEVANT: the clause does not affect the answer (it may sound related but is not needed).

Output format (one clause per line, nothing else):
Clause 1: <text> | RELEVANT
Clause 2: <text> | IRRELEVANT
"""

SOLVE_SYSTEM = (
    "You are a careful math assistant. Reason step by step. "
    "Your final line MUST be exactly: #### <numeric_answer>. "
    "Use only a number in <numeric_answer> (no words, currency symbols, units, or extra text). "
    "If the answer is an integer, write an integer; otherwise write a decimal."
)

SOLVE_USER = """{clean_problem}

Solve step by step."""


# ---------------------------------------------------------------------------
# Filter response parser
# ---------------------------------------------------------------------------

_RE_CLAUSE = re.compile(
    r'[Cc]lause\s*\d+\s*:\s*(.+?)\s*\|\s*(RELEVANT|IRRELEVANT)',
    re.IGNORECASE,
)


def parse_filter_response(response: str) -> tuple[list[str], list[str]]:
    """Return (relevant_clauses, irrelevant_clauses). Both may be empty on parse failure."""
    relevant = []
    irrelevant = []
    for line in response.strip().splitlines():
        m = _RE_CLAUSE.match(line.strip())
        if m:
            text, label = m.group(1).strip(), m.group(2).upper()
            if label == "RELEVANT":
                relevant.append(text)
            else:
                irrelevant.append(text)
    return relevant, irrelevant


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_examples(enhanced_dir: str, seed: int = 42) -> list[dict]:
    """Load enhanced GSM8K problems with a single distractor injected per problem."""
    examples = load_enhanced_distracted_examples(enhanced_dir, seed=seed)
    # Assign a stable problem_id for resume support
    for i, ex in enumerate(examples):
        ex.setdefault("problem_id", ex.get("source_file", f"prob_{i:04d}"))
    return examples


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def load_completed_ids(out_path: Path) -> set[str]:
    completed = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    completed.add(json.loads(line)["problem_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


def append_record(record: dict, out_path: Path) -> None:
    with open(out_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Per-model inference loop
# ---------------------------------------------------------------------------

def run_model(
    config: ModelConfig,
    examples: list[dict],
    output_dir: str,
    device: str,
    max_new_tokens: int,
    limit: int | None,
    resume: bool,
):
    out_path = Path(output_dir) / f"{config.short_name}_distractor_filter.jsonl"

    completed_ids = set()
    if resume:
        completed_ids = load_completed_ids(out_path)
        if completed_ids:
            print(f"    [resume] {len(completed_ids)} problems already done.")

    todo = examples[:limit] if limit else examples
    total = len(todo)

    wrapper = get_model_wrapper(config, device)
    model_loaded = False

    correct_count = 0
    done = 0

    for ex in todo:
        pid = ex["problem_id"]
        if pid in completed_ids:
            done += 1
            continue

        if not model_loaded:
            wrapper.load()
            model_loaded = True

        distracted_q = ex["question"]           # already has distractor appended
        gold_raw = extract_gsm_final(ex["answer"]) or extract_model_final(ex["answer"])

        t0 = time.time()

        # ── Stage 1: filter ──────────────────────────────────────────────────
        filter_prompt = FILTER_USER.format(problem=distracted_q)
        try:
            filter_response = wrapper.generate(
                FILTER_SYSTEM, filter_prompt, max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            filter_response = f"[ERROR: {e}]"

        relevant, irrelevant = parse_filter_response(filter_response)
        if not relevant:
            clean_problem = distracted_q
            filter_failed = True
        else:
            clean_problem = " ".join(relevant)
            filter_failed = False

        # ── Stage 2: solve ───────────────────────────────────────────────────
        solve_prompt = SOLVE_USER.format(clean_problem=clean_problem)
        try:
            solve_response = wrapper.generate(
                SOLVE_SYSTEM, solve_prompt, max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            solve_response = f"[ERROR: {e}]"

        predicted = extract_model_final(solve_response)
        is_correct = answers_match(gold_raw or "", predicted)
        elapsed = time.time() - t0

        if is_correct:
            correct_count += 1
        done += 1

        record = {
            "model":               config.short_name,
            "problem_id":          pid,
            "distracted_question": distracted_q,
            "question_original":   ex.get("question_original", ""),
            "distractor":          ex.get("distractor", ""),
            "distractor_type":     ex.get("distractor_type", "UNKNOWN"),
            "ground_truth":        gold_raw,
            "filter_response":     filter_response,
            "relevant_clauses":    relevant,
            "irrelevant_clauses":  irrelevant,
            "filter_failed":       filter_failed,
            "clean_problem":       clean_problem,
            "solve_response":      solve_response,
            "predicted":           predicted,
            "is_correct":          is_correct,
            "elapsed_s":           round(elapsed, 3),
        }
        append_record(record, out_path)

        sym = "✓" if is_correct else ("∅" if predicted is None else "✗")
        ff  = " [filter_fail]" if filter_failed else ""
        print(f"    [{config.short_name}] {done}/{total}  "
              f"GT={gold_raw}  Pred={predicted}  {sym}{ff}  ({elapsed:.1f}s)")

        if done % 50 == 0:
            acc = correct_count / done
            print(f"    --- Running accuracy: {acc:.4f} ({done}/{total}) ---")

    if model_loaded:
        wrapper.unload()

    n_done = done
    acc = correct_count / n_done if n_done else 0.0
    print(f"\n  [{config.short_name}] Accuracy: {acc:.4f} ({correct_count}/{n_done})")
    print(f"  Results saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Experiment 3: Distractor Filtering")
    p.add_argument("--models",         type=str, default="all")
    p.add_argument("--enhanced_dir",   type=str,
                   default=str(repo_root / "Experiment1" / "gsm_enhanced_templates"))
    p.add_argument("--output_dir",     type=str, default="results")
    p.add_argument("--device",         type=str, default="cuda:0")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--limit",          type=int, default=None)
    p.add_argument("--distractor_seed",type=int, default=42)
    p.add_argument("--no_resume",      action="store_true")
    return p.parse_args()


def main():
    os.environ.setdefault("HF_HOME", "/mnt/data/hf")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/data/hf/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/data/hf/hub")
    os.environ.setdefault("TMPDIR", "/mnt/data/tmp")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_configs = get_all_configs()
    if args.models == "all":
        configs = all_configs
    else:
        wanted = set(args.models.split(","))
        configs = [c for c in all_configs if c.short_name in wanted]
        missing = wanted - {c.short_name for c in configs}
        if missing:
            raise ValueError(f"Unknown model short_name(s): {sorted(missing)}")

    examples = load_examples(args.enhanced_dir, seed=args.distractor_seed)
    print(f"\n{'='*60}")
    print(f"Experiment 3 — Distractor Filtering")
    print(f"  Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Models:      {[c.short_name for c in configs]}")
    print(f"  Problems:    {len(examples)} (limit={args.limit})")
    print(f"  Output dir:  {args.output_dir}")
    print(f"{'='*60}\n")

    for cfg in configs:
        print(f"\n{'─'*50}")
        print(f"  Model: {cfg.short_name}  ({cfg.model_id})")
        print(f"{'─'*50}")
        run_model(
            config=cfg,
            examples=examples,
            output_dir=args.output_dir,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            resume=not args.no_resume,
        )

    print(f"\n{'='*60}")
    print(f"Experiment 3 complete.  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
