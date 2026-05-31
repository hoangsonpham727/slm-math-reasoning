"""
Experiment 3: Distractor Filtering

Two-stage pipeline on the enhanced GSM8K dataset:
  Stage 1 — Filter: label each clause RELEVANT or IRRELEVANT.
             Runs once and caches results; reused across all solver models.
  Stage 2 — Solve (each SLM): CoT solve on the relevant-only problem text.

Filter model options (--filter_model):
  qwen25_7b   — Qwen/Qwen2.5-7B-Instruct loaded locally via HuggingFace (default)
  gpt-oss     — gpt-oss:120b via Ollama cloud (requires OLLAMA_API_KEY env var)

Usage:
    python run_experiment3.py [--solver_models all]
                              [--filter_model qwen25_7b|gpt-oss]
                              [--enhanced_dir ../Experiment1/gsm_enhanced_templates]
                              [--output_dir results] [--device cuda:0]
                              [--max_new_tokens 1024] [--limit None] [--no_resume]
                              [--distractor_seed 42] [--smoke_test]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime
import ollama
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from models import get_all_configs, get_model_wrapper, ModelConfig


# ---------------------------------------------------------------------------
# Ollama cloud wrapper — same .load() / .generate() / .unload() interface
# as the HuggingFace wrappers in models.py
# ---------------------------------------------------------------------------

_OLLAMA_MODEL = "gpt-oss:120b"
_OLLAMA_HOST  = "https://ollama.com"


class OllamaFilterWrapper:
    """Thin wrapper around the Ollama cloud API for the filtering stage."""

    def __init__(self):
        self.short_name = "gpt-oss"

    def load(self):
        api_key = os.getenv("OLLAMA_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OLLAMA_API_KEY environment variable is not set. "
                "Export it before running with --filter_model gpt-oss."
            )
        print(f"  [gpt-oss] Using Ollama cloud ({_OLLAMA_HOST}, model={_OLLAMA_MODEL}).")

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 1024, **_) -> str:
        from ollama import Client
        client = Client(
            host=_OLLAMA_HOST,
            headers={"Authorization": "Bearer " + os.getenv("OLLAMA_API_KEY", "")},
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        response = ""
        for part in client.chat(_OLLAMA_MODEL, messages=messages, stream=True):
            response += part["message"]["content"]
        return response

    def unload(self):
        print(f"  [gpt-oss] Done (no GPU resources to free).")


# ---------------------------------------------------------------------------
# Inlined helpers from Experiment1 (avoids importing data_preparation.py
# which requires google.genai that is not needed here)
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

FILTER_USER = """Instructions:
1. Break the problem into individual clauses (one per line).
2. For each clause, label it RELEVANT or IRRELEVANT.
   - RELEVANT: the clause contains a quantity or fact directly needed to compute the answer.
   - IRRELEVANT: the clause does not affect the answer (it may sound related but is not needed).

Output format — one clause per line, nothing else:
Clause N: <text> | RELEVANT
Clause N: <text> | IRRELEVANT

--- Example 1 ---
Problem:
John has 10 hectares of a pineapple field. There are 100 pineapples per hectare. John can harvest his pineapples every 3 months. Some agronomists suggest adding a 5% loss factor due to pests each harvest. How many pineapples can John harvest within a year?

Clause 1: John has 10 hectares of a pineapple field. | RELEVANT
Clause 2: There are 100 pineapples per hectare. | RELEVANT
Clause 3: John can harvest his pineapples every 3 months. | RELEVANT
Clause 4: Some agronomists suggest adding a 5% loss factor due to pests each harvest. | IRRELEVANT

--- Example 2 ---
Problem:
A tower is made out of 4 blue blocks, twice as many yellow blocks, and an unknown number of red blocks. Each blue block measures about 3 inches in side length. If there are 32 blocks in the tower in total, how many red blocks are there?

Clause 1: A tower is made out of 4 blue blocks, twice as many yellow blocks, and an unknown number of red blocks. | RELEVANT
Clause 2: Each blue block measures about 3 inches in side length. | IRRELEVANT
Clause 3: If there are 32 blocks in the tower in total, how many red blocks are there? | RELEVANT

--- Example 3 ---
Problem:
It takes Carmen 10 minutes to finish a crossword puzzle and 5 minutes to finish a sudoku puzzle. Over the weekend she solved 3 crossword puzzles and 8 sudoku puzzles. Carmen started the weekend with a cup of coffee at 7 am, which she drank before any puzzles. How much time did she spend playing these games?

Clause 1: It takes Carmen 10 minutes to finish a crossword puzzle and 5 minutes to finish a sudoku puzzle. | RELEVANT
Clause 2: Over the weekend she solved 3 crossword puzzles and 8 sudoku puzzles. | RELEVANT
Clause 3: Carmen started the weekend with a cup of coffee at 7 am, which she drank before any puzzles. | IRRELEVANT

--- Now label this problem ---
Problem:
{problem}
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
# Stage 1: filtering (Qwen2.5-7B runs once, results cached to disk)
# ---------------------------------------------------------------------------

FILTER_CACHE_FILENAME = "filter_cache_{filter_model}.jsonl"


def run_filtering(
    filter_wrapper,           # HF BaseModelWrapper or OllamaFilterWrapper
    examples: list[dict],
    output_dir: str,
    max_new_tokens: int,
) -> list[dict]:
    """
    Run Stage 1 with filter_wrapper over all examples.
    Results are cached to disk so solver models can reuse them without
    reloading the filter model.
    Returns the full list of filtered records (one per example).
    """
    cache_path = Path(output_dir) / FILTER_CACHE_FILENAME.format(
        filter_model=filter_wrapper.short_name
    )

    # Load existing cache
    cached: dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    cached[rec["problem_id"]] = rec
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"  [filter cache] {len(cached)} problems already filtered.")

    need_filtering = [ex for ex in examples if ex["problem_id"] not in cached]

    if need_filtering:
        print(f"  [filter] Running {filter_wrapper.short_name} on "
              f"{len(need_filtering)} problems ...")
        filter_wrapper.load()

        for i, ex in enumerate(need_filtering):
            pid = ex["problem_id"]
            distracted_q = ex["question"]
            filter_prompt = FILTER_USER.format(problem=distracted_q)
            try:
                filter_response = filter_wrapper.generate(
                    FILTER_SYSTEM, filter_prompt, max_new_tokens=max_new_tokens,
                )
            except Exception as e:
                filter_response = f"[ERROR: {e}]"

            relevant, irrelevant = parse_filter_response(filter_response)
            filter_failed = not relevant
            clean_problem = distracted_q if filter_failed else " ".join(relevant)

            rec = {
                "problem_id":       pid,
                "filter_model":     filter_wrapper.short_name,
                "filter_response":  filter_response,
                "relevant_clauses": relevant,
                "irrelevant_clauses": irrelevant,
                "filter_failed":    filter_failed,
                "clean_problem":    clean_problem,
            }
            cached[pid] = rec
            with open(cache_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

            ff = " [fail]" if filter_failed else ""
            print(f"    [filter {i+1}/{len(need_filtering)}] {pid}{ff}")

        filter_wrapper.unload()
        print(f"  [filter] Done. Cache saved to {cache_path}")
    else:
        print(f"  [filter] All problems already in cache — skipping filter model load.")

    # Return in original example order
    return [cached[ex["problem_id"]] for ex in examples]


# ---------------------------------------------------------------------------
# Stage 2: solving (one SLM at a time)
# ---------------------------------------------------------------------------

def run_solving(
    solver_config: ModelConfig,
    examples: list[dict],
    filter_records: list[dict],
    filter_model_name: str,
    output_dir: str,
    device: str,
    max_new_tokens: int,
    resume: bool,
):
    """Run Stage 2 for one solver model using pre-computed filter_records."""
    out_path = Path(output_dir) / f"{solver_config.short_name}_distractor_filter.jsonl"

    completed_ids = set()
    if resume:
        completed_ids = load_completed_ids(out_path)
        if completed_ids:
            print(f"    [resume] {len(completed_ids)} problems already done.")

    total = len(examples)
    wrapper = get_model_wrapper(solver_config, device)
    model_loaded = False
    correct_count = 0
    done = 0

    for ex, fr in zip(examples, filter_records):
        pid = ex["problem_id"]
        if pid in completed_ids:
            done += 1
            continue

        if not model_loaded:
            wrapper.load()
            model_loaded = True

        gold_raw = extract_gsm_final(ex["answer"]) or extract_model_final(ex["answer"])
        clean_problem = fr["clean_problem"]
        filter_failed = fr["filter_failed"]

        t0 = time.time()
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
            "solver_model":        solver_config.short_name,
            "filter_model":        filter_model_name,
            "problem_id":          pid,
            "distracted_question": ex["question"],
            "question_original":   ex.get("question_original", ""),
            "distractor":          ex.get("distractor", ""),
            "distractor_type":     ex.get("distractor_type", "UNKNOWN"),
            "ground_truth":        gold_raw,
            "filter_response":     fr["filter_response"],
            "relevant_clauses":    fr["relevant_clauses"],
            "irrelevant_clauses":  fr["irrelevant_clauses"],
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
        print(f"    [{solver_config.short_name}] {done}/{total}  "
              f"GT={gold_raw}  Pred={predicted}  {sym}{ff}  ({elapsed:.1f}s)")

        if done % 50 == 0:
            acc = correct_count / done
            print(f"    --- Running accuracy: {acc:.4f} ({done}/{total}) ---")

    if model_loaded:
        wrapper.unload()

    n_done = done
    acc = correct_count / n_done if n_done else 0.0
    print(f"\n  [{solver_config.short_name}] Accuracy: {acc:.4f} ({correct_count}/{n_done})")
    print(f"  Results saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# The three original SLMs used as solvers (filter model excluded)
_SOLVER_SHORT_NAMES = {"qwen25_math_1.5b", "gemma4_e2b", "phi4_mini"}


_OLLAMA_FILTER_NAME = "gpt-oss"


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 3: Distractor Filtering")
    p.add_argument("--solver_models",  type=str, default="all",
                   help="Comma-separated solver short_names or 'all' "
                        f"(choices: {', '.join(sorted(_SOLVER_SHORT_NAMES))})")
    p.add_argument("--filter_model",   type=str, default="qwen25_7b",
                   help="Model used for Stage 1 clause labeling. "
                        "Use a HuggingFace short_name (e.g. qwen25_7b) "
                        f"or '{_OLLAMA_FILTER_NAME}' for Ollama cloud (requires OLLAMA_API_KEY). "
                        "(default: qwen25_7b)")
    p.add_argument("--enhanced_dir",   type=str,
                   default=str(repo_root / "Experiment1" / "gsm_enhanced_templates"))
    p.add_argument("--output_dir",     type=str, default="results")
    p.add_argument("--device",         type=str, default="cuda:0")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--limit",          type=int, default=None)
    p.add_argument("--distractor_seed",type=int, default=42)
    p.add_argument("--no_resume",      action="store_true")
    p.add_argument("--smoke_test",     action="store_true",
                   help="Quick sanity check: force limit=2.")
    return p.parse_args()


def _resolve_filter_wrapper(filter_model_name: str, device: str):
    """Return a wrapper object (HF or Ollama) for the requested filter model."""
    if filter_model_name == _OLLAMA_FILTER_NAME:
        return OllamaFilterWrapper()
    all_configs = get_all_configs()
    config_by_name = {c.short_name: c for c in all_configs}
    if filter_model_name not in config_by_name:
        raise ValueError(
            f"Unknown filter_model: '{filter_model_name}'. "
            f"Use '{_OLLAMA_FILTER_NAME}' for Ollama cloud, or one of: "
            f"{sorted(config_by_name)}"
        )
    return get_model_wrapper(config_by_name[filter_model_name], device)


def main():
    os.environ.setdefault("HF_HOME", "/mnt/data/hf")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/data/hf/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/data/hf/hub")
    os.environ.setdefault("TMPDIR", "/mnt/data/tmp")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    args = parse_args()
    if args.smoke_test:
        args.limit = 2
        print("[smoke-test] forcing limit=2.", file=sys.stderr)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_configs = get_all_configs()
    config_by_name = {c.short_name: c for c in all_configs}

    # Resolve filter wrapper (HF model or Ollama cloud)
    filter_wrapper = _resolve_filter_wrapper(args.filter_model, args.device)

    # Resolve solver model configs
    if args.solver_models == "all":
        solver_configs = [c for c in all_configs if c.short_name in _SOLVER_SHORT_NAMES]
    else:
        wanted = set(args.solver_models.split(","))
        solver_configs = [config_by_name[n] for n in wanted if n in config_by_name]
        missing = wanted - set(config_by_name)
        if missing:
            raise ValueError(f"Unknown solver model short_name(s): {sorted(missing)}")

    examples = load_examples(args.enhanced_dir, seed=args.distractor_seed)
    if args.limit:
        examples = examples[:args.limit]

    smoke_tag = "  [SMOKE TEST]" if args.smoke_test else ""
    print(f"\n{'='*60}")
    print(f"Experiment 3 — Distractor Filtering{smoke_tag}")
    print(f"  Started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Filter model:  {filter_wrapper.short_name}")
    print(f"  Solver models: {[c.short_name for c in solver_configs]}")
    print(f"  Problems:      {len(examples)}")
    print(f"  Output dir:    {args.output_dir}")
    print(f"{'='*60}\n")

    # ── Stage 1: filter once, cache to disk ──────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Stage 1 — Filtering distractors")
    print(f"{'─'*50}")
    filter_records = run_filtering(
        filter_wrapper=filter_wrapper,
        examples=examples,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
    )

    # ── Stage 2: solve with each SLM ─────────────────────────────────────────
    for cfg in solver_configs:
        print(f"\n{'─'*50}")
        print(f"  Stage 2 — Solving with {cfg.short_name}  ({cfg.model_id})")
        print(f"{'─'*50}")
        run_solving(
            solver_config=cfg,
            examples=examples,
            filter_records=filter_records,
            filter_model_name=filter_wrapper.short_name,
            output_dir=args.output_dir,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            resume=not args.no_resume,
        )

    print(f"\n{'='*60}")
    print(f"Experiment 3 complete.  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
