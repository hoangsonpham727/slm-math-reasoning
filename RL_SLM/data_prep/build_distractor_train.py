"""
Build distractor-augmented training data from Experiment 1's unused distractors.

Each of the 100 enhanced templates has a dynamic_distractor_pool of 5 distractors.
Experiment 1's evaluation used exactly one per problem (selected with random.Random(42)
in sorted filename order — matching export_distracted_jsonl in data_preparation.py).
This script takes the remaining 4 unused distractors per problem and injects each
into the question via append_distractor(), producing 400 training records.

Output: RL_SLM/data/train_distractor/problems_distractor_train.json
"""

import glob
import json
import os
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def append_distractor(question: str, distractor, position: str = "mid") -> str:
    """Inline copy of Experiment1/data_preparation.append_distractor."""
    q = question.strip()
    if isinstance(distractor, dict):
        distractor = str(distractor.get("text", ""))
    d = str(distractor).strip().rstrip(".")
    if not d:
        return q
    if position == "end":
        if q.endswith("?"):
            return f"{q[:-1]}, {d}?"
        return f"{q} {d}."
    sentences = re.split(r'(?<=[.!])\s+', q)
    if len(sentences) <= 1:
        if q.endswith("?"):
            return f"{q[:-1]}, {d}?"
        return f"{q} {d}."
    mid_point = len(sentences) - 1
    sentences.insert(mid_point, d + ".")
    return " ".join(sentences)


def extract_gsm_answer(answer_str: str) -> float:
    """Extract numeric answer from GSM-format string (e.g. '...#### 10')."""
    if "####" in answer_str:
        raw = answer_str.split("####")[-1]
    else:
        raw = answer_str
    cleaned = re.sub(r"[^0-9.\-]", "", raw.strip())
    return float(cleaned) if cleaned else 0.0


def build_distractor_training_set(
    enhanced_dir: str,
    output_path: str,
    eval_seed: int = 42,
) -> int:
    """
    For each template, replay the evaluation RNG to identify the used distractor,
    then inject each of the remaining unused distractors into the question.

    Returns the number of records written.
    """
    files = sorted(glob.glob(os.path.join(enhanced_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {enhanced_dir}")

    # Replay the exact same RNG used during Experiment 1 evaluation
    rng = random.Random(eval_seed)

    records = []
    for filepath in files:
        with open(filepath, encoding="utf-8") as f:
            template = json.load(f)

        pool = template.get("dynamic_distractor_pool") or []
        if not pool:
            print(f"Warning: skipping {os.path.basename(filepath)} — empty pool")
            rng.choice(pool) if pool else None
            continue

        # Identify which distractor was consumed at evaluation time
        used = rng.choice(pool)
        used_text = used["text"] if isinstance(used, dict) else str(used)

        original_q = template.get("question", "")
        ground_truth = extract_gsm_answer(template.get("answer", ""))
        source_file = os.path.basename(filepath)
        id_orig = template.get("id_orig")
        base_idx = template.get("id_shuffled", 0)

        unused = [d for d in pool if (d.get("text") if isinstance(d, dict) else d) != used_text]

        for i, distractor in enumerate(unused):
            d_text = distractor["text"] if isinstance(distractor, dict) else str(distractor)
            d_type = distractor.get("type", "UNKNOWN") if isinstance(distractor, dict) else "UNKNOWN"
            augmented_q = append_distractor(original_q, d_text)

            records.append({
                "problem_id": f"train_dist_{base_idx:04d}_d{i}",
                "question": augmented_q,
                "ground_truth": ground_truth,
                "source_file": source_file,
                "id_orig": id_orig,
                "distractor_type": d_type,
                "distractor_text": d_text,
                "question_original": original_q,
            })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} records to {output_path}")
    return len(records)


if __name__ == "__main__":
    enhanced_dir = str(REPO_ROOT / "Experiment1" / "gsm_enhanced_templates")
    output_path  = str(REPO_ROOT / "RL_SLM" / "data" / "train_distractor" / "problems_distractor_train.json")
    build_distractor_training_set(enhanced_dir, output_path)
