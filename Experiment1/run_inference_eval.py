"""
Experiment 1: Irrelevant Distractor Injection

Run GSM inference for all models in models.py on original vs enhanced (distracted) questions.
Computes accuracy on both type of questions; saves full generations to JSON.

Usage:
    python run_inference_eval.py [--mode original enhanced both] [--original-dir ./gsm_templates] 
                                 [--enhanced-dir ./gsm_enhanced_templates] [--distractor-seed 42] 
                                 [--output ./inference_results.json] [--device auto] 
                                 [--max-new-tokens 512] [--limit None] [--models None (all models) | qwen25_math_1.5b | gemma4_e2b | phi4_mini] 
                                 [--smoke-test None]
"""

from __future__ import annotations
from pathlib import Path
import argparse
import glob
import json
import os
import random
import re
import sys
from tqdm import tqdm
from datetime import datetime, timezone
from typing import Any, Optional

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from models import MODEL_CONFIGS, ModelConfig, get_model_wrapper
from data_preparation import append_distractor

# ---------------------------------------------------------------------------
# Answer extraction (truncate leakage, GSM ####, phrases, \\boxed{}, tail)
# ---------------------------------------------------------------------------

_RE_BOXED_START = re.compile(r"\\boxed\{")
_RE_CHAT_TURN = re.compile(r"(?i)\n(?:User|Assistant|System):\s*")
# Last occurrence wins: phrases that typically introduce the final answer line.
_RE_ANSWER_PHRASES: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?is)(?:\*\*\s*)?Final answer\s*:\s*([^\n]+)"),
    re.compile(r"(?is)The\s+final\s+answer\s+is\s*:?\s*([^\n]+)"),
    re.compile(r"(?is)The\s+answer\s+is\s*:?\s*([^\n]+)"),
    re.compile(r"(?im)^\s*Answer:\s*([^\n]+)"),
)
_TAIL_MAX_LINES = 25
_TAIL_MAX_CHARS = 2000
_RE_NUMERIC_TOKEN = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _primary_solution_text(text: str) -> str:
    """Keep only text before a second-role chat turn (model leakage / multi-turn)."""
    if not text:
        return text
    m = _RE_CHAT_TURN.search(text)
    if m:
        return text[: m.start()]
    return text


def _extraction_tail(text: str) -> str:
    """Last portion of the solution for 'last number' fallback (bounded)."""
    lines = text.splitlines()
    if len(lines) > _TAIL_MAX_LINES:
        tail = "\n".join(lines[-_TAIL_MAX_LINES :])
    else:
        tail = text
    if len(tail) > _TAIL_MAX_CHARS:
        tail = tail[-_TAIL_MAX_CHARS :]
    return tail


def _extract_answer_phrase_last(text: str) -> Optional[str]:
    """Content after the last answer-introducing phrase in the text."""
    best: Optional[re.Match[str]] = None
    for pat in _RE_ANSWER_PHRASES:
        for m in pat.finditer(text):
            if best is None or m.start() > best.start():
                best = m
    if best is None:
        return None
    s = best.group(1).strip()
    return s or None


def _extract_numeric_candidate(text: str) -> Optional[str]:
    """Extract a numeric answer candidate from mixed-format text."""
    if not text:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None

    # Remove common wrappers while keeping numeric payload.
    cleaned = re.sub(r"(?i)^(?:is|=)\s*", "", cleaned)
    cleaned = cleaned.replace("$", "").replace("%", "")
    cleaned = cleaned.replace(",", "")
    cleaned = re.sub(r"\\+", "", cleaned).strip()

    # Prefer strict full-match first, then fallback to last numeric token.
    if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", cleaned):
        return cleaned
    matches = _RE_NUMERIC_TOKEN.findall(cleaned)
    if matches:
        return matches[-1].replace(",", "")
    return None


def extract_gsm_final(answer: str) -> Optional[str]:
    """Final answer after the last #### line in a GSM-style solution string."""
    if not answer or not answer.strip():
        return None
    matches = list(re.finditer(r"####\s*(.+)", answer, flags=re.MULTILINE))
    if not matches:
        return None
    gsm = matches[-1].group(1).strip()
    return _extract_numeric_candidate(gsm) or gsm


def _extract_boxed_last(text: str) -> Optional[str]:
    """Content of the last \\boxed{...} using brace balancing."""
    best: Optional[str] = None
    for m in _RE_BOXED_START.finditer(text):
        pos = m.end()  # first char inside outer braces
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
            best = text[pos : j - 1]
    return best


def extract_model_final(text: str) -> Optional[str]:
    """
    After stripping chat leakage: GSM #### line; answer phrases; \\boxed{...};
    then last numeric token in a bounded tail of the text.
    """
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
        if parsed is not None:
            return parsed
        return phrase

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
    if candidates:
        return candidates[-1].replace(",", "")
    return None


def _normalize_answer(s: str) -> str:
    t = s.strip()
    t = t.replace(",", "").replace("$", "")
    t = t.replace("%", "")
    t = re.sub(r"\\+", "", t)
    t = t.strip()
    tm = t.replace(",", "")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", tm):
        return tm
    num = re.search(r"-?\d+(?:\.\d+)?", tm)
    if num:
        return num.group(0)
    return t


def answers_match(gold: str, pred: Optional[str], eps: float = 1e-5) -> bool:
    if pred is None:
        return False
    g = _normalize_answer(gold)
    p = _normalize_answer(pred)
    if not g or not p:
        return g == p
    try:
        gf = float(g)
        pf = float(p)
        return abs(gf - pf) <= eps * max(1.0, abs(gf), abs(pf))
    except ValueError:
        return g.casefold() == p.casefold()


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def load_original_examples(template_dir: str) -> list[dict[str, Any]]:
    pattern = os.path.join(template_dir, "*.json")
    paths = sorted(glob.glob(pattern))
    out: list[dict[str, Any]] = []
    for filepath in paths:
        with open(filepath, encoding="utf-8") as f:
            record = json.load(f)
        q = record.get("question", "")
        a = record.get("answer", "")
        if not q or not a:
            print(
                f"Warning: skipping {os.path.basename(filepath)}: missing question or answer.",
                file=sys.stderr,
            )
            continue
        row = {
            "question": q,
            "answer": a,
            "id_orig": record.get("id_orig"),
            "id_shuffled": record.get("id_shuffled"),
            "source_file": os.path.basename(filepath),
        }
        out.append(row)
    return out


def load_enhanced_distracted_examples(
    enhanced_dir: str, seed: int = 42
) -> list[dict[str, Any]]:
    pattern = os.path.join(enhanced_dir, "*.json")
    paths = sorted(glob.glob(pattern))
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for filepath in paths:
        with open(filepath, encoding="utf-8") as f:
            record = json.load(f)
        pool = record.get("dynamic_distractor_pool") or []
        if not pool:
            print(
                f"Warning: skipping {os.path.basename(filepath)}: "
                "missing or empty dynamic_distractor_pool.",
                file=sys.stderr,
            )
            continue
        original_q = record.get("question", "")
        a = record.get("answer", "")
        if not original_q or not a:
            print(
                f"Warning: skipping {os.path.basename(filepath)}: missing question or answer.",
                file=sys.stderr,
            )
            continue
        chosen = rng.choice(pool)
        distractor_text = chosen.get("text", "") if isinstance(chosen, dict) else str(chosen)
        distractor_type = chosen.get("type", "UNKNOWN") if isinstance(chosen, dict) else "UNKNOWN"
        distracted = append_distractor(original_q, distractor_text)
        row = {
            "question": distracted,
            "answer": a,
            "question_original": original_q,
            "distractor": distractor_text,
            "distractor_type": distractor_type,
            "id_orig": record.get("id_orig"),
            "id_shuffled": record.get("id_shuffled"),
            "source_file": os.path.basename(filepath),
        }
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful math assistant. Reason step by step. "
    "Your final line MUST be exactly: #### <numeric_answer>. "
    "Use only a number in <numeric_answer> (no words, currency symbols, units, or extra text). "
    "If the answer is an integer, write an integer; otherwise write a decimal."
)


def _select_configs(
    all_configs: list[ModelConfig], short_names: Optional[list[str]]
) -> list[ModelConfig]:
    if not short_names:
        return list(all_configs)
    name_set = set(short_names)
    chosen = [c for c in all_configs if c.short_name in name_set]
    missing = name_set - {c.short_name for c in chosen}
    if missing:
        raise ValueError(f"Unknown model short_name(s): {sorted(missing)}")
    return chosen


def run_all_models_on_examples(
    examples: list[dict[str, Any]],
    configs: list[ModelConfig],
    system_prompt: str,
    device: str,
    max_new_tokens: int,
    limit: Optional[int],
) -> dict[str, Any]:
    """
    Returns dict keyed by short_name: accuracy, n_evaluated, n_skipped_no_gold, results.
    """
    if limit is not None:
        examples = examples[:limit]

    results_by_model: dict[str, Any] = {}

    for cfg in configs:
        wrapper = get_model_wrapper(cfg, device=device)
        wrapper.load()
        rows_out: list[dict[str, Any]] = []
        correct = 0
        n_eval = 0
        n_no_gold = 0

        for ex in tqdm(examples):
            gold_raw = extract_gsm_final(ex["answer"])
            if gold_raw is None:
                # Fallback for rare non-GSM-formatted gold labels.
                gold_raw = extract_model_final(ex["answer"])
            if gold_raw is None:
                n_no_gold += 1
                rows_out.append(
                    {
                        "source_file": ex.get("source_file"),
                        "id_orig": ex.get("id_orig"),
                        "id_shuffled": ex.get("id_shuffled"),
                        "question": ex.get("question"),
                        "question_original": ex.get("question_original"),
                        "distractor": ex.get("distractor"),
                        "ground_truth_extracted": None,
                        "predicted_extracted": None,
                        "correct": False,
                        "full_model_output": "",
                        "error": "failed to extract gold answer",
                    }
                )
                continue

            user_prompt = ex["question"]
            try:
                full_text = wrapper.generate(
                    system_prompt, user_prompt, max_new_tokens=max_new_tokens
                )
            except Exception as e:
                pred = None
                full_text = ""
                err = str(e)
                rows_out.append(
                    {
                        "source_file": ex.get("source_file"),
                        "id_orig": ex.get("id_orig"),
                        "id_shuffled": ex.get("id_shuffled"),
                        "question": ex.get("question"),
                        "question_original": ex.get("question_original"),
                        "distractor": ex.get("distractor"),
                        "ground_truth_extracted": gold_raw,
                        "predicted_extracted": pred,
                        "correct": False,
                        "full_model_output": full_text,
                        "error": err,
                    }
                )
                n_eval += 1
                continue

            pred = extract_model_final(full_text)
            ok = answers_match(gold_raw, pred)
            if ok:
                correct += 1
            n_eval += 1

            rows_out.append(
                {
                    "source_file": ex.get("source_file"),
                    "id_orig": ex.get("id_orig"),
                    "id_shuffled": ex.get("id_shuffled"),
                    "question": ex.get("question"),
                    "question_original": ex.get("question_original"),
                    "distractor": ex.get("distractor"),
                    "ground_truth_extracted": gold_raw,
                    "predicted_extracted": pred,
                    "correct": ok,
                    "full_model_output": full_text,
                }
            )

        acc = correct / n_eval if n_eval else 0.0
        results_by_model[cfg.short_name] = {
            "model_id": cfg.model_id,
            "family": cfg.family,
            "accuracy": acc,
            "n_correct": correct,
            "n_evaluated": n_eval,
            "n_skipped_no_gold": n_no_gold,
            "results": rows_out,
        }
        wrapper.unload()

    return results_by_model


def _config_to_jsonable(cfg: ModelConfig) -> dict[str, Any]:
    return {
        "model_id": cfg.model_id,
        "short_name": cfg.short_name,
        "family": cfg.family,
        "dtype": cfg.dtype,
    }


def main() -> None:
    os.environ["HF_HOME"] = "/mnt/data/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/mnt/data/hf/transformers"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/data/hf/hub"
    os.environ["TMPDIR"] = "/mnt/data/tmp"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # use first GPU only
    
    parser = argparse.ArgumentParser(
        description="GSM inference evaluation: original vs enhanced (distracted) templates."
    )
    parser.add_argument(
        "--mode",
        choices=("original", "enhanced", "both"),
        default="both",
        help="Which dataset(s) to run (default: both).",
    )
    parser.add_argument(
        "--original-dir",
        default="./gsm_templates",
        help="Directory of original template JSON files (default: ./gsm_templates).",
    )
    parser.add_argument(
        "--enhanced-dir",
        default="./gsm_enhanced_templates",
        help="Directory of enhanced JSON with dynamic_distractor_pool (default: ./gsm_enhanced_templates).",
    )
    parser.add_argument(
        "--distractor-seed",
        type=int,
        default=42,
        help="RNG seed for distractor choice; matches export-distracted stability (default: 42).",
    )
    parser.add_argument(
        "--output",
        default="./inference_results.json",
        help="Output JSON path (default: ./inference_results.json).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Device placement for HF models. "
            "Use 'auto' (default) or pin to one GPU with 'cuda:0' (or '0')."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Generation budget (default: 512).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of problems to evaluate per run (default: all).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model short_name values (default: all in MODEL_CONFIGS).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a quick sanity check: force mode=both and limit=1 so every selected "
            "model is loaded and evaluated on 1 question from original and enhanced sets."
        ),
    )

    args = parser.parse_args()
    if args.smoke_test:
        args.mode = "both"
        args.limit = 1
        print(
            "[smoke-test] forcing mode=both and limit=1 "
            "(one question from original and one from enhanced).",
            file=sys.stderr,
        )

    configs = _select_configs(MODEL_CONFIGS, args.models)
    out_path = os.path.abspath(args.output)

    payload: dict[str, Any] = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "original_dir": os.path.abspath(args.original_dir),
            "enhanced_dir": os.path.abspath(args.enhanced_dir),
            "distractor_seed": args.distractor_seed,
            "device": args.device,
            "max_new_tokens": args.max_new_tokens,
            "limit": args.limit,
            "models_requested": args.models,
            "model_configs": [_config_to_jsonable(c) for c in configs],
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
        },
    }

    def save_checkpoint() -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    if args.mode in ("original", "both"):
        orig_ex = load_original_examples(args.original_dir)
        print(
            f"[original] Loaded {len(orig_ex)} examples from {args.original_dir}",
            file=sys.stderr,
        )
        payload["original"] = run_all_models_on_examples(
            orig_ex,
            configs,
            DEFAULT_SYSTEM_PROMPT,
            args.device,
            args.max_new_tokens,
            args.limit,
        )
        for sn, block in payload["original"].items():
            print(
                f"  [original] {sn}: accuracy={block['accuracy']:.4f} "
                f"({block['n_correct']}/{block['n_evaluated']})",
                file=sys.stderr,
            )
        save_checkpoint()
        print(f"[original] Wrote checkpoint to {out_path}", file=sys.stderr)

    if args.mode in ("enhanced", "both"):
        enh_ex = load_enhanced_distracted_examples(
            args.enhanced_dir, seed=args.distractor_seed
        )
        print(
            f"[enhanced] Loaded {len(enh_ex)} examples from {args.enhanced_dir} "
            f"(seed={args.distractor_seed})",
            file=sys.stderr,
        )
        payload["enhanced"] = run_all_models_on_examples(
            enh_ex,
            configs,
            DEFAULT_SYSTEM_PROMPT,
            args.device,
            args.max_new_tokens,
            args.limit,
        )
        for sn, block in payload["enhanced"].items():
            print(
                f"  [enhanced] {sn}: accuracy={block['accuracy']:.4f} "
                f"({block['n_correct']}/{block['n_evaluated']})",
                file=sys.stderr,
            )
        save_checkpoint()
        print(f"[enhanced] Wrote checkpoint to {out_path}", file=sys.stderr)

    save_checkpoint()

    print(f"Wrote results to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
