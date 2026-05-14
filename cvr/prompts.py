"""
All CVR prompt templates in one place.

Each template is split into system_prompt + user_prompt to match the
wrapper.generate(system_prompt, user_prompt) interface.

Template variables use Python str.format() — {question}, {prior_steps}, etc.
"""

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

STEPWISE_GENERATION_SYSTEM = (
    "You are a precise math assistant. "
    "Solve math problems step by step. "
    "Label every step exactly as 'Step 1:', 'Step 2:', etc. "
    "Show the arithmetic calculation on each step explicitly."
)

STEPWISE_GENERATION_USER = """\
Problem: {question}

Solve completely. Label each step as 'Step 1:', 'Step 2:', etc. Show the arithmetic on every step.\
"""

# Used when there are no prior steps yet.
NO_PRIOR_STEPS = ""

# ---------------------------------------------------------------------------
# Verification — Consistency (NCV Binary)
# ---------------------------------------------------------------------------

CONSISTENCY_CHECK_SYSTEM = (
    "You are a math verifier. "
    "Your response MUST start with exactly one word: 'Correct' or 'Wrong'. "
    "Write that word first, then optionally a short explanation."
)

CONSISTENCY_CHECK_USER = """\
Problem: {question}

Verified steps so far:
{prior_steps}

Step to check:
{step_text}

Check ONLY the arithmetic calculations shown. Do not evaluate the approach, units, or whether this step is necessary. Is the arithmetic correct? Start your response with 'Correct' or 'Wrong'.\
"""

# ---------------------------------------------------------------------------
# Verification — Relevance (distractor check)
# ---------------------------------------------------------------------------

RELEVANCE_CHECK_SYSTEM = (
    "You are a math verifier checking whether a solution step uses distractor information. "
    "Math problems sometimes contain extra numbers or facts that are NOT needed to reach the answer. "
    "Your response MUST start with exactly one word: 'Yes' or 'No'. "
    "Write that word first, then optionally a short explanation."
)

RELEVANCE_CHECK_USER = """\
Problem: {question}

Step to check:
{step_text}

Some problems contain distractor numbers or facts that are not needed to solve the problem. \
Does this step use ONLY numbers and facts that are required to reach the answer? \
Answer 'No' if the step relies on any extra or irrelevant information from the problem. \
Start your response with 'Yes' or 'No'.\
"""

# ---------------------------------------------------------------------------
# Restart prompts
# ---------------------------------------------------------------------------

RESTART_CONSISTENCY_SYSTEM = (
    "You are a precise math assistant. A previous attempt at a step contained "
    "an arithmetic error. Double-check every calculation before writing. "
    "Label each step exactly as 'Step N:' and show all arithmetic explicitly."
)

RESTART_CONSISTENCY_USER = """\
Problem: {question}

{prior_steps_block}The previous attempt at Step {step_number} had an arithmetic error. Continue carefully from Step {step_number}, double-checking every calculation.
Label each step as 'Step {step_number}:', 'Step {step_number_plus1}:', etc.\
"""

RESTART_RELEVANCE_SYSTEM = (
    "You are a precise math assistant. A previous attempt at a step used "
    "information that is NOT needed to solve the problem. "
    "Use ONLY the numbers and facts that are directly required. "
    "Ignore any extra or irrelevant information in the problem statement. "
    "Label each step exactly as 'Step N:' and show all arithmetic explicitly."
)

RESTART_RELEVANCE_USER = """\
Problem: {question}

{prior_steps_block}The previous attempt at Step {step_number} used irrelevant information. Continue from Step {step_number}, using only information directly needed.
Label each step as 'Step {step_number}:', 'Step {step_number_plus1}:', etc.\
"""

# ---------------------------------------------------------------------------
# Final answer extraction (from full verified chain)
# ---------------------------------------------------------------------------

FINAL_ANSWER_SYSTEM = (
    "You are a math assistant. Given a completed multi-step solution, "
    "state the final numeric answer. "
    "Your response MUST be exactly: #### <number> "
    "(just a number, no words, units, or extra text)."
)

FINAL_ANSWER_USER = """\
Problem: {question}

Solution steps:
{all_steps}

What is the final numeric answer? Write: #### <number>\
"""


def format_prior_steps_block(steps: list[dict]) -> str:
    """Format verified steps into the '{prior_steps_block}' slot in generation prompts."""
    if not steps:
        return ""
    lines = [f"Step {s['index']}: {s['text']}" for s in steps]
    return "\n".join(lines) + "\n\n"
