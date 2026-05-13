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
    "You are a precise math assistant. Solve problems one step at a time. "
    "Write ONLY the next single reasoning step — do not skip ahead or write multiple steps. "
    "Show your arithmetic explicitly. End the step when you have one new result."
)

STEPWISE_GENERATION_USER = """\
Problem: {question}

{prior_steps_block}Write ONLY Step {step_number} now:\
"""

# Used when there are no prior steps yet.
NO_PRIOR_STEPS = ""

# ---------------------------------------------------------------------------
# Verification — Consistency (NCV Binary)
# ---------------------------------------------------------------------------

CONSISTENCY_CHECK_SYSTEM = (
    "You are a math verifier. You will be given a math problem, verified prior steps, "
    "and a new step to check. Answer with exactly ONE word: Correct or Wrong. "
    "Do not explain. Do not add punctuation."
)

CONSISTENCY_CHECK_USER = """\
Problem: {question}

Verified steps so far:
{prior_steps}

Now check this step:
Step {step_number}: {step_text}

Is this step's calculation correct given the problem and prior steps? Answer: Correct or Wrong\
"""

# ---------------------------------------------------------------------------
# Verification — Relevance (distractor check)
# ---------------------------------------------------------------------------

RELEVANCE_CHECK_SYSTEM = (
    "You are a math verifier. Check whether a reasoning step uses ONLY information "
    "that is necessary and relevant to solving the problem. "
    "Answer with exactly ONE word: Yes or No. Do not explain. Do not add punctuation."
)

RELEVANCE_CHECK_USER = """\
Problem: {question}

Step {step_number}: {step_text}

Does this step use only information that is necessary and directly relevant to solving the problem? Answer: Yes or No\
"""

# ---------------------------------------------------------------------------
# Restart prompts
# ---------------------------------------------------------------------------

RESTART_CONSISTENCY_SYSTEM = (
    "You are a precise math assistant. A previous attempt at the next step contained "
    "an arithmetic error. Solve problems one step at a time. "
    "Double-check every calculation before writing your answer. "
    "Write ONLY the next single step."
)

RESTART_CONSISTENCY_USER = """\
Problem: {question}

{prior_steps_block}The previous attempt at Step {step_number} had an arithmetic error. Try again carefully.
Write ONLY Step {step_number}:\
"""

RESTART_RELEVANCE_SYSTEM = (
    "You are a precise math assistant. A previous attempt at the next step used "
    "information that is NOT needed to solve the problem. "
    "IMPORTANT: Use ONLY the numbers and facts that are directly required. "
    "Ignore any extra or irrelevant information in the problem statement. "
    "Write ONLY the next single step."
)

RESTART_RELEVANCE_USER = """\
Problem: {question}

{prior_steps_block}The previous attempt at Step {step_number} used irrelevant information. Try again, ignoring unnecessary details.
Write ONLY Step {step_number}:\
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
