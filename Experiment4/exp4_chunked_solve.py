"""
Experiment 4 — Chunked Incremental Solve (Dataset B, synthetic depth 1-8).

Pipeline per problem:
  problem → split into clauses → group into 2-clause chunks
  for each chunk:
    1. Prompt SLM with verified context + chunk clauses
    2. Extract 'a op b = c' patterns via regex
    3. Verify / correct each expression; propagate errors forward
    4. Fallback to last-number extraction if no patterns found
  final_answer = last verified value in chain (not SLM's stated answer)
"""

import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent          # Experiment4/
for _p in [str(ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import (
    extract_arithmetic_expressions,
    verify_and_correct_expressions,
    extract_last_number,
    split_into_clauses,
    chunk_clauses,
    answers_match,
    save_results,
    load_dataset,
    checkpoint_results,
)


def extract_answer_tag(text: str) -> float | None:
    """Extract the number from an explicit ANSWER: <number> line."""
    m = re.search(r'ANSWER\s*:\s*([-\d,]+(?:\.\d+)?)', text)
    if m:
        try:
            return float(m.group(1).replace(',', ''))
        except ValueError:
            return None
    return None

# ── Prompt templates ─────────────────────────────────────────────────────────

CHUNK_SYSTEM = (
    "You are a math solver. "
    "Solve the given problem using only the known values provided. "
    "Show your work step by step."
)

CHUNK_USER = """\
Known values:
{context_text}

New information (follow these steps IN ORDER):
{chunk_text}

Using ONLY the known values and the new information above, \
work through each numbered step IN THE ORDER GIVEN. \
Show each calculation as: number operation number = result.
After all steps, compute the single final value that remains after \
applying every operation in sequence. Write your final answer as:
ANSWER: <number>
"""


# ── Context helpers ──────────────────────────────────────────────────────────

def _is_numeric_key(key: str) -> bool:
    try:
        float(key)
        return True
    except ValueError:
        return False


def format_context(context: dict) -> str:
    # Only expose current_total — named variables like diana_hour persist
    # across chunks and cause the model to operate on stale values instead
    # of the running total.
    if "current_total" not in context:
        return "(none yet — this is the start of the problem)"
    return (
        f"current_total = {context['current_total']}"
        "  ← use ONLY this as your starting amount"
    )


_GOAL_PATTERN = re.compile(
    r'^\s*(?:how\s+(?:much|many)|what\s+is|find\s+(?:the|how))',
    re.IGNORECASE,
)

def _is_goal_clause(clause: str) -> bool:
    """Return True for pure question sentences that contain no computation."""
    return bool(_GOAL_PATTERN.match(clause)) and '=' not in clause


def _snapshot(context: dict) -> dict:
    return {
        k: v for k, v in context.items()
        if not k.startswith("__") and not _is_numeric_key(k)
    }


# ── Variable name inference ──────────────────────────────────────────────────

def infer_variable_name(clause_text: str, chunk_idx: int) -> str:
    """Heuristic: proper noun + quantity noun. Fallback: step_N_result."""
    name = re.search(r'\b([A-Z][a-z]+)\b', clause_text)
    unit = re.search(
        r'(dollars?|apples?|books?|hours?|miles?|cookies?|points?|'
        r'items?|pounds?|gallons?|minutes?|pieces?|cents?|tickets?|'
        r'balls?|flowers?|pencils?|stars?|coins?|marbles?)',
        clause_text, re.IGNORECASE,
    )
    if name and unit:
        return f"{name.group(1).lower()}_{unit.group(1).lower()}"
    elif name:
        return f"{name.group(1).lower()}_value"
    return f"step_{chunk_idx + 1}_result"


def update_context_with_name(
    context: dict, clause_text: str, chunk_idx: int, value: float
) -> None:
    var_name = infer_variable_name(clause_text, chunk_idx)
    context[var_name] = value
    # current_total is the canonical running amount shown prominently to the model
    context["current_total"] = value
    context[str(round(value, 10))] = value
    context["__last__"] = value


# ── Single chunk solver ──────────────────────────────────────────────────────

def solve_chunk(
    chunk_clauses_list: list[str],
    context: dict,
    llm_fn,
    max_retries: int = 3,
) -> dict:
    """
    Solve one 2-clause chunk against verified context.
    Returns a detailed log dict for analysis.
    """
    # Number each clause so the model follows the stated order
    chunk_text = "\n".join(
        f"{i+1}. {c}" for i, c in enumerate(chunk_clauses_list)
    )
    context_before = _snapshot(context)

    prompt = CHUNK_USER.format(
        context_text=format_context(context),
        chunk_text=chunk_text,
    )

    temperatures = [0.0, 0.3, 0.6]
    slm_response = ""
    for attempt in range(max_retries):
        temp = temperatures[attempt] if attempt < len(temperatures) else 0.6
        slm_response = llm_fn(prompt, system=CHUNK_SYSTEM, temperature=temp)
        expressions = extract_arithmetic_expressions(slm_response)

        answer_tag = extract_answer_tag(slm_response)

        if expressions or answer_tag is not None:
            verification_log = []
            if expressions:
                verification_log = verify_and_correct_expressions(
                    expressions, context
                )
            return {
                "chunk_text":         chunk_text,
                "context_before":     context_before,
                "slm_response":       slm_response,
                "expressions_found":  [e["full_match"] for e in expressions],
                "verification_log":   verification_log,
                "answer_tag":         answer_tag,
                "context_after":      _snapshot(context),
                "extraction_method":  "arithmetic",
                "num_retries":        attempt,
            }

    # Fallback: last number in response
    last_num = extract_last_number(slm_response)
    if last_num is not None:
        context["__last__"] = last_num
        context[str(round(last_num, 10))] = last_num

    return {
        "chunk_text":         chunk_text,
        "context_before":     context_before,
        "slm_response":       slm_response,
        "expressions_found":  [],
        "verification_log":   [],
        "answer_tag":         None,
        "context_after":      _snapshot(context),
        "extraction_method":  "last_number" if last_num is not None else "failed",
        "num_retries":        max_retries,
    }


# ── Main experiment loop ─────────────────────────────────────────────────────

def run_exp4(
    dataset_path: str,
    output_path: str,
    problem_field: str,
    answer_field: str,
    depth_field: str,
    steps_field: str | None = None,
    chunk_size: int = 2,
    max_retries: int = 3,
    llm_fn=None,
) -> list[dict]:
    """
    Run Experiment 4 — Chunked Incremental Solve.

    Args:
        dataset_path:  Path to Dataset B JSON/JSONL file.
        output_path:   Where to write final results JSON.
        problem_field: Field name for the question text.
        answer_field:  Field name for the ground truth answer.
        depth_field:   Field name for problem depth (int).
        steps_field:   Optional field name for pre-split steps list.
                       If present and contains text steps, sentence splitting
                       is skipped.
        chunk_size:    Number of clauses per chunk (fixed at 2).
        max_retries:   Retries per chunk when no expressions are extracted.
        llm_fn:        Callable with signature llm(prompt, system, temperature).
                       Defaults to experiments.llm_wrapper.llm.
    """
    if llm_fn is None:
        from llm_wrapper import llm
        llm_fn = llm

    dataset = load_dataset(dataset_path)
    results = []

    for i, item in enumerate(dataset):
        problem = item[problem_field]
        gold    = str(item[answer_field])
        depth   = item[depth_field]

        # Clause extraction: prefer stored metadata, else sentence-split
        if steps_field and steps_field in item:
            clauses = item[steps_field]
            if clauses and isinstance(clauses[0], dict):
                clauses = [s.get("text", str(s)) for s in clauses]
        else:
            clauses = split_into_clauses(problem)

        # Drop goal-only question sentences — they contain no computation,
        # waste retries, and poison context via garbage fallback extraction.
        clauses = [
            c for c in clauses
            if not _is_goal_clause(c)
        ]

        chunks = chunk_clauses(clauses, chunk_size)

        # Incremental solve
        context    = {}
        chunk_logs = []

        for ci, chunk in enumerate(chunks):
            chunk_result = solve_chunk(chunk, context, llm_fn, max_retries)

            # Decide the chunk's final value:
            # 1. Prefer explicit ANSWER: tag (model's combined final value)
            # 2. Fall back to last verified expression
            # 3. Fall back to last_number extraction
            chunk_value = None

            answer_tag = chunk_result.get("answer_tag")
            if answer_tag is not None:
                chunk_value = answer_tag

            if chunk_value is None and chunk_result["verification_log"]:
                last_v = chunk_result["verification_log"][-1]
                if last_v["corrected_value"] is not None:
                    chunk_value = last_v["corrected_value"]

            if chunk_value is None and chunk_result["extraction_method"] == "last_number":
                chunk_value = context.get("__last__")

            if chunk_value is not None:
                update_context_with_name(
                    context, " ".join(chunk), ci, chunk_value
                )

            # Refresh context_after now that update_context_with_name has run
            chunk_result["context_after"] = _snapshot(context)
            chunk_logs.append(chunk_result)

        # Final answer from verified chain, not SLM's stated answer
        chain_answer = context.get("__last__")
        predicted    = str(round(chain_answer, 10)) if chain_answer is not None else None
        correct      = answers_match(predicted, gold)

        total_expr      = sum(len(cl["verification_log"]) for cl in chunk_logs)
        corrected_count = sum(
            1 for cl in chunk_logs
            for v in cl["verification_log"] if v["status"] == "corrected"
        )
        extract_fails = sum(
            1 for cl in chunk_logs if cl["extraction_method"] == "failed"
        )
        unverified = sum(
            1 for cl in chunk_logs if cl["extraction_method"] == "last_number"
        )

        results.append({
            "id":                        i,
            "problem":                   problem,
            "gold":                      gold,
            "depth":                     depth,
            "num_clauses":               len(clauses),
            "num_chunks":                len(chunks),
            "chunk_size":                chunk_size,
            "predicted":                 predicted,
            "correct":                   correct,
            "total_expressions_checked": total_expr,
            "corrections_applied":       corrected_count,
            "extraction_failures":       extract_fails,
            "unverified_fallbacks":      unverified,
            "chunk_logs":                chunk_logs,
        })

        if (i + 1) % 50 == 0:
            acc  = sum(r["correct"] for r in results) / len(results)
            corr = sum(r["corrections_applied"] for r in results)
            fails = sum(r["extraction_failures"] for r in results)
            print(
                f"[Exp4] {i+1}/{len(dataset)} | "
                f"Acc: {acc:.3f} | Corrections: {corr} | "
                f"Extract fails: {fails}"
            )
            checkpoint_results(results, output_path)

    # Summary by depth
    print(f"\n{'='*65}")
    print(f"EXPERIMENT 4 — Chunked Incremental Solve (chunk_size={chunk_size})")
    print(f"{'='*65}")
    print(
        f"\n{'Depth':<8}{'N':<8}{'Acc':<10}"
        f"{'Corr/prob':<12}{'Unverif':<12}{'ExtFail'}"
    )
    print("-" * 65)

    depth_groups: dict[int, list] = {}
    for r in results:
        depth_groups.setdefault(r["depth"], []).append(r)

    for d in sorted(depth_groups.keys()):
        g   = depth_groups[d]
        acc = sum(r["correct"] for r in g) / len(g)
        corr = sum(r["corrections_applied"] for r in g) / len(g)
        unv  = sum(r["unverified_fallbacks"] for r in g) / len(g)
        ef   = sum(r["extraction_failures"] for r in g) / len(g)
        print(
            f"{d:<8}{len(g):<8}{acc:<10.4f}"
            f"{corr:<12.2f}{unv:<12.2f}{ef:.2f}"
        )

    overall = sum(r["correct"] for r in results) / len(results)
    print(f"\nOverall accuracy: {overall:.4f}")
    print(
        f"Total corrections applied: "
        f"{sum(r['corrections_applied'] for r in results)}"
    )

    save_results(results, output_path)
    return results


# ── Multi-model runner ───────────────────────────────────────────────────────

_ALL_MODELS = ["qwen25_math_1.5b", "gemma4_e2b", "phi4_mini"]

_FIELD_DEFAULTS = dict(
    problem_field = "question",
    answer_field  = "ground_truth",
    depth_field   = "depth",
    steps_field   = None,
    chunk_size    = 2,
)


def run_all_models(
    models: list[str] | None = None,
    device: str = "cuda:0",
    data_path: str | None = None,
    output_dir: str | None = None,
) -> dict[str, list[dict]]:
    """
    Run Experiment 4 for every model in sequence.
    Each model is loaded, evaluated over the full dataset, then unloaded
    before the next model is loaded — keeps peak VRAM at one model at a time.
    Checkpoints are written every 50 problems per model.

    Args:
        models:     List of short_names to run. Defaults to all three SLMs.
        device:     HuggingFace device string ('cuda:0', 'cpu', 'auto').
        data_path:  Path to problems_all.json. Defaults to Experiment2/data/.
        output_dir: Directory for result files. Defaults to <root>/results/.

    Returns:
        Dict mapping short_name → results list.
    """
    from datetime import datetime
    from llm_wrapper import init_model, llm as _llm_fn
    import llm_wrapper as _lw

    if models is None:
        models = _ALL_MODELS

    dataset_path = data_path or str(
        ROOT / "Experiment2" / "data" / "problems_all.json"
    )
    out_dir = Path(output_dir) if output_dir else ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {}

    print(f"\n{'='*65}")
    print(f"EXPERIMENT 4 — All models  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    print(f"  Models  : {models}")
    print(f"  Dataset : {dataset_path}")
    print(f"  Output  : {out_dir}")
    print(f"{'='*65}")

    for short_name in models:
        output_path = str(out_dir / f"exp4_chunked_{short_name}.json")

        print(f"\n{'─'*65}")
        print(f"  Model: {short_name}  —  started {datetime.now():%H:%M:%S}")
        print(f"{'─'*65}")

        init_model(short_name, device=device)

        results = run_exp4(
            dataset_path=dataset_path,
            output_path=output_path,
            llm_fn=_lw.llm,
            **_FIELD_DEFAULTS,
        )

        all_results[short_name] = results
        _lw._wrapper.unload()

        print(f"  [{short_name}] done — results saved to {output_path}")

    # ── Cross-model summary ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("FINAL SUMMARY — All models")
    print(f"{'='*65}")
    print(f"\n{'Model':<22}{'Overall':>10}  per-depth accuracy")
    print("-" * 65)

    for short_name, results in all_results.items():
        overall = sum(r["correct"] for r in results) / len(results)
        depth_groups: dict[int, list] = {}
        for r in results:
            depth_groups.setdefault(r["depth"], []).append(r)
        depth_acc = "  ".join(
            f"d{d}={sum(r['correct'] for r in g)/len(g):.2f}"
            for d, g in sorted(depth_groups.items())
        )
        print(f"  {short_name:<20}{overall:>8.4f}  {depth_acc}")

    return all_results


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment 4 — Chunked Incremental Solve"
    )
    parser.add_argument(
        "--model", default="qwen25_math_1.5b",
        help="Single model short_name (ignored when --all is set)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all three models in sequence",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="HuggingFace device string (default: cuda:0)",
    )
    parser.add_argument(
        "--data", default=None,
        help="Path to Dataset B JSON (default: Experiment2/data/problems_all.json)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path / dir (single model: file path; --all: directory)",
    )
    args = parser.parse_args()

    if args.all:
        run_all_models(
            device=args.device,
            data_path=args.data,
            output_dir=args.output,
        )
    else:
        dataset_path = args.data or str(
            ROOT / "Experiment2" / "data" / "problems_all.json"
        )
        output_path = args.output or str(
            ROOT / "results" / f"exp4_chunked_{args.model}.json"
        )

        from llm_wrapper import init_model, llm
        init_model(args.model, device=args.device)

        run_exp4(
            dataset_path=dataset_path,
            output_path=output_path,
            llm_fn=llm,
            **_FIELD_DEFAULTS,
        )
