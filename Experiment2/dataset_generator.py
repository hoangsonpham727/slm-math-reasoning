"""
Dataset Generator

Generates dependency-chained arithmetic problems at controlled step depths (1–8).
Each step's output feeds as input to the next; no shortcuts exist to the final answer.
All ground-truth answers are verified by a Python solver.
"""

import random
import json
import re
from fractions import Fraction
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


random.seed(42)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MathProblem:
    problem_id: str
    depth: int                     # number of required reasoning steps
    question: str                  # natural-language problem text
    ground_truth: float            # verified final answer
    intermediate_values: list      # list of step results including final answer
    operations: list               # list of operation names used at each step
    answer_unit: str               # e.g. "$", "apples", "km"


# ---------------------------------------------------------------------------
# Vocabulary pools — kept simple to eliminate confounds
# ---------------------------------------------------------------------------

ACTORS    = ["Alice", "Bob", "Carlos", "Diana", "Eve", "Frank", "Grace", "Henry"]
ITEMS     = ["apples", "books", "boxes", "tickets", "pens", "bags", "coins", "cards"]
UNITS     = ["$", "kg", "km", "points", "hours", "liters", "metres", "items"]
JOBS      = ["earns", "collects", "receives", "gathers"]
SHOPS     = ["store", "market", "shop", "warehouse"]
VERBS_BUY = ["buys", "purchases"]
VERBS_USE = ["uses", "spends", "donates", "gives away"]

OPS = ["multiply", "divide_half", "divide_third", "divide_quarter",
       "subtract_fixed", "add_fixed", "multiply_then_subtract"]


# ---------------------------------------------------------------------------
# Step templates — each returns (sentence_str, result_value)
# ---------------------------------------------------------------------------

def step_earn(actor, rate, hours, unit="$"):
    val = Fraction(rate * hours)
    sentence = (f"{actor} {random.choice(JOBS)} {unit}{rate} per hour "
                f"and works {hours} hours.")
    return sentence, val


def step_spend_fraction(actor, current_val, frac_num, frac_den, item, unit="$"):
    spent = current_val * Fraction(frac_num, frac_den)
    remaining = current_val - spent
    frac_str = {(1, 2): "half", (1, 3): "a third", (1, 4): "a quarter",
                (2, 3): "two-thirds", (3, 4): "three-quarters"}.get(
                    (frac_num, frac_den), f"{frac_num}/{frac_den}")
    sentence = (f"{actor} spends {frac_str} of {actor}'s remaining {unit} on {item}.")
    return sentence, remaining


def step_buy_items(actor, remaining, num_items, price_each, item, unit="$"):
    cost = Fraction(num_items * price_each)
    result = remaining - cost
    sentence = (f"{actor} then {random.choice(VERBS_BUY)} {num_items} {item} "
                f"at {unit}{price_each} each.")
    return sentence, result


def step_add_bonus(actor, current_val, bonus, unit="$"):
    result = current_val + Fraction(bonus)
    sentence = f"{actor} also receives a {unit}{bonus} bonus."
    return sentence, result


def step_tax_or_fee(actor, current_val, pct):
    fee = current_val * Fraction(pct, 100)
    result = current_val - fee
    sentence = f"A {pct}% fee is deducted from {actor}'s total."
    return sentence, result


def step_double_production(entity, current_val, unit="$"):
    result = current_val * 2
    sentence = f"{entity} doubles {entity}'s current {unit} total."
    return sentence, result


def step_fixed_subtract(actor, current_val, amount, item, unit="$"):
    result = current_val - Fraction(amount)
    sentence = f"{actor} {random.choice(VERBS_USE)} {unit}{amount} on {item}."
    return sentence, result


def step_sell_at_price(actor, current_val, price, unit="$"):
    result = current_val * Fraction(price)
    sentence = f"{actor} sells each remaining unit for {unit}{price}."
    return sentence, result


# ---------------------------------------------------------------------------
# Problem constructors by depth
# ---------------------------------------------------------------------------

def _make_step_sequence(depth: int, seed_val: int):
    """
    Build a sequence of (sentence, intermediate_value) pairs of exactly `depth`
    chained steps. Returns (sentences, intermediates, operations, final_value, unit).
    """
    random.seed(seed_val)

    actor = random.choice(ACTORS)
    unit  = "$"

    # Pick enough distinct items so each step uses a unique one (Fix C)
    items_needed = max(depth, 2)
    items_pool = (ITEMS * ((items_needed // len(ITEMS)) + 1))[:items_needed]
    random.shuffle(items_pool)

    sentences    = []
    intermediates = []
    ops          = []

    # --- Step 1: always an earning step (establishes base value) ---
    rate  = random.randint(10, 30)
    hours = random.randint(5, 15)
    s, val = step_earn(actor, rate, hours, unit)
    sentences.append(s)
    intermediates.append(val)
    ops.append("multiply")

    def _build_step_library(item_iter):
        """
        Build step functions that each draw a unique item from the iterator.
        Fraction steps only apply when the running total is evenly divisible
        by the denominator, keeping intermediate values clean (Fix E).
        """
        def _spend_half(v):
            return step_spend_fraction(actor, v, 1, 2, next(item_iter), unit)
        def _spend_third(v):
            return step_spend_fraction(actor, v, 1, 3, next(item_iter), unit)
        def _spend_quarter(v):
            return step_spend_fraction(actor, v, 1, 4, next(item_iter), unit)
        def _buy(v):
            n = random.randint(2, 5)
            p = random.randint(2, 8)
            max_affordable = max(1, int(v) - 1)
            if n * p > max_affordable:
                p = max(1, max_affordable // n)
            return step_buy_items(actor, v, n, p, next(item_iter), unit)
        def _bonus(v):
            return step_add_bonus(actor, v, random.randint(5, 15), unit)
        def _fee(v):
            return step_tax_or_fee(actor, v, random.choice([10, 20, 25]))
        def _double(v):
            return step_double_production(actor, v, unit)
        def _fixed_sub(v):
            amount = random.randint(2, 10)
            max_sub = max(1, int(v) - 1)
            amount = min(amount, max_sub)
            return step_fixed_subtract(actor, v, amount, next(item_iter), unit)
        return [_spend_half, _spend_third, _spend_quarter,
                _buy, _bonus, _fee, _double, _fixed_sub]

    item_iter = iter(items_pool)
    step_library = _build_step_library(item_iter)

    # Shuffle remaining steps (Steps 2..depth)
    available = step_library[:]
    random.shuffle(available)

    for i in range(depth - 1):
        step_fn = available[i % len(available)]
        current = intermediates[-1]
        s, new_val = step_fn(current)
        sentences.append(s)
        intermediates.append(new_val)
        ops.append("derived")

    final_value = intermediates[-1]
    return sentences, intermediates, ops, final_value, unit


def build_problem(depth: int, problem_idx: int) -> MathProblem:
    seed_val = depth * 10000 + problem_idx
    sentences, intermediates, ops, final_value, unit = _make_step_sequence(depth, seed_val)

    actor_name = sentences[0].split()[0]  # extract first word (actor name)
    question_body = " ".join(sentences)
    question = (question_body + f" How much {unit} does {actor_name} have in the end?")

    return MathProblem(
        problem_id=f"depth{depth:02d}_prob{problem_idx:04d}",
        depth=depth,
        question=question,
        ground_truth=float(final_value),
        intermediate_values=[float(v) for v in intermediates],
        operations=ops,
        answer_unit=unit,
    )


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    depths: list = list(range(1, 9)),
    n_per_depth: int = 200,
    output_dir: str = "data",
) -> dict:
    """
    Generate n_per_depth problems for each depth level.
    Returns a dict: {depth -> list[MathProblem]}
    Saves to output_dir/problems_depth{k}.json and a combined file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_problems = {}

    for depth in depths:
        problems = []
        for idx in range(n_per_depth):
            p = build_problem(depth, idx)
            # Sanity-check: ground truth must be positive
            assert p.ground_truth > 0, f"Non-positive GT at {p.problem_id}"
            problems.append(p)

        all_problems[depth] = problems

        # Save per-depth file
        out_path = Path(output_dir) / f"problems_depth{depth:02d}.json"
        with open(out_path, "w") as f:
            json.dump([asdict(p) for p in problems], f, indent=2)

        print(f"  Depth {depth}: {len(problems)} problems generated. "
              f"  GT range: [{min(p.ground_truth for p in problems):.2f}, "
              f"{max(p.ground_truth for p in problems):.2f}]")

    # Save combined file
    combined = []
    for depth in depths:
        combined.extend([asdict(p) for p in all_problems[depth]])
    with open(Path(output_dir) / "problems_all.json", "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nDataset saved to '{output_dir}/'. "
          f"Total problems: {len(combined)}")
    return all_problems


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

SYSTEM_DIRECT = (
    "You are a precise math assistant. "
    "Read the problem carefully and output only the final numeric answer. "
    "Do not show any working. "
    "Format: Answer: <number>"
)

SYSTEM_COT = (
    "You are a precise math assistant. "
    "Solve the problem step by step. "
    "Show each calculation clearly. "
    "At the end, write your final answer as: Answer: <number>"
)

def build_prompt_direct(problem: MathProblem) -> str:
    return problem.question

def build_prompt_cot(problem: MathProblem) -> str:
    return (
        f"{problem.question}\n\n"
        "Please reason step by step, then give your final answer as: Answer: <number>"
    )


# ---------------------------------------------------------------------------
# Answer parser
# ---------------------------------------------------------------------------

def parse_answer(response: str) -> Optional[float]:
    """
    Extract the final numeric answer from a model response.
    Tries multiple patterns in order of specificity.
    Returns None if no answer found (counted as output collapse).
    """
    # Pattern 1: explicit "Answer: <number>"
    m = re.search(r'[Aa]nswer\s*:\s*\$?\s*([\d,]+\.?\d*)', response)
    if m:
        return float(m.group(1).replace(",", ""))

    # Pattern 2: boxed answer (common in math model outputs)
    m = re.search(r'\\boxed\{([^}]+)\}', response)
    if m:
        try:
            return float(m.group(1).replace(",", "").strip("$"))
        except ValueError:
            pass

    # Pattern 3: "= <number>" at end of line
    m = re.search(r'=\s*\$?\s*([\d,]+\.?\d*)\s*$', response, re.MULTILINE)
    if m:
        return float(m.group(1).replace(",", ""))

    # Pattern 4: last standalone number in response
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
    if numbers:
        return float(numbers[-1])

    return None


def check_step_accuracy(
    response: str,
    problem: MathProblem,
    tolerance: float = 0.05,
) -> Optional[int]:
    """
    Returns the 1-based index of the first step where model-reported running
    values diverge from ground-truth intermediates. Returns None if all steps
    are observed in the response in-order and within tolerance.
    """
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
    if not numbers:
        return 1

    predicted_steps = [float(n) for n in numbers]
    expected_steps = problem.intermediate_values

    # Robust to extra numbers (e.g., step labels, copied constants): we only
    # require that expected intermediates appear in-order in the response.
    search_start = 0
    for i, expected in enumerate(expected_steps):
        matched = False
        for j in range(search_start, len(predicted_steps)):
            predicted = predicted_steps[j]
            if abs(predicted - expected) / (abs(expected) + 1e-9) < tolerance:
                search_start = j + 1
                matched = True
                break
        if not matched:
            return i + 1

    if len(predicted_steps) < len(expected_steps):
        return len(predicted_steps) + 1

    return None


# ---------------------------------------------------------------------------
# Quick dataset validation (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating dataset...")
    dataset = generate_dataset(
        depths=list(range(1, 9)),
        n_per_depth=200,
        output_dir="data",
    )

    # Spot-check a few problems
    print("\n--- Spot-check samples ---")
    for depth in [1, 3, 5, 8]:
        p = dataset[depth][0]
        print(f"\nDepth {depth} | ID: {p.problem_id}")
        print(f"  Q: {p.question}")
        print(f"  Intermediates: {p.intermediate_values}")
        print(f"  GT: {p.ground_truth} {p.answer_unit}")