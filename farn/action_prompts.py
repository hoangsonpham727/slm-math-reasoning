"""
action_prompts.py — Prompt templates for the five logic blocks.

Each logic block corresponds to an action the navigator can select.
When the navigator picks action i, the corresponding prompt template
is formatted with the current problem and reasoning context, then
sent to the SLM to generate the next reasoning step.

These prompts are adapted from RLoT's block designs. The key
difference in FARN is that we do NOT include self-evaluation prompts
(RLoT asks the SLM to score itself after each step — we replace
that with external feature extraction).
"""

from typing import List


def format_reasoning_context(steps: List[str]) -> str:
    """Format the accumulated reasoning steps into a context string.

    Args:
        steps: List of SLM outputs from previous reasoning steps.

    Returns:
        Formatted string showing the reasoning history.
    """
    if not steps:
        return "No previous reasoning steps."

    lines = []
    for i, step in enumerate(steps, 1):
        lines.append(f"Step {i}: {step.strip()}")
    return "\n".join(lines)


# ── Prompt Templates ─────────────────────────────────────────────────

# Each function takes the problem text and previous steps, and returns
# the full prompt string to send to the SLM.


def prompt_reason(problem: str, previous_steps: List[str]) -> str:
    """Action 0: Reason one step.

    Asks the SLM to perform exactly one reasoning step. This is the
    basic building block — a single forward step in the chain.
    """
    context = format_reasoning_context(previous_steps)

    return f"""Problem: {problem}

Previous reasoning:
{context}

Perform exactly one reasoning step to make progress toward the answer. Show your work clearly, including any calculations. Do not jump to the final answer unless this step directly produces it.

Step {len(previous_steps) + 1}:"""


def prompt_decompose(problem: str, previous_steps: List[str]) -> str:
    """Action 1: Decompose into subtasks.

    Asks the SLM to break the current problem state into smaller
    subtasks, solve each, and summarise. This helps with complex
    multi-part problems.
    """
    context = format_reasoning_context(previous_steps)

    return f"""Problem: {problem}

Previous reasoning:
{context}

Break the remaining work into smaller subtasks. For each subtask:
1. State what needs to be calculated
2. Perform the calculation
3. State the result

After completing all subtasks, briefly summarise the results.

Subtasks:"""


def prompt_debate(problem: str, previous_steps: List[str]) -> str:
    """Action 2: Debate multiple approaches.

    Asks the SLM to generate 2-3 different plans or approaches,
    compare them, select the most promising one, and take one step
    using that approach. This introduces diversity and helps avoid
    getting stuck on a wrong path.
    """
    context = format_reasoning_context(previous_steps)

    return f"""Problem: {problem}

Previous reasoning:
{context}

Consider 2-3 different approaches to continue solving this problem.
For each approach:
- Briefly describe the strategy
- Identify potential issues

Then select the most promising approach and take one reasoning step using it.

Approaches:"""


def prompt_refine(problem: str, previous_steps: List[str]) -> str:
    """Action 3: Refine the current reasoning.

    Asks the SLM to review and correct the most recent reasoning step.
    This is the primary error-correction mechanism within the standard
    action space.

    Note: This action is automatically converted to "reason" if it
    appears as the first action (nothing to refine yet).
    """
    context = format_reasoning_context(previous_steps)

    return f"""Problem: {problem}

Previous reasoning:
{context}

Review the most recent reasoning step carefully. Check for:
- Arithmetic errors
- Incorrect use of given values
- Logical mistakes
- Missing information

If you find errors, correct them. If the step is correct, confirm it and clarify any ambiguities.

Review:"""


def prompt_terminate(problem: str, previous_steps: List[str]) -> str:
    """Action 4: Terminate and give final answer.

    Asks the SLM to synthesise all previous reasoning steps and
    produce the final numerical answer in the expected format.
    """
    context = format_reasoning_context(previous_steps)

    return f"""Problem: {problem}

Previous reasoning:
{context}

Based on all the reasoning steps above, provide the final numerical answer to the problem. Show the answer clearly.

The answer is:"""


# ── Dispatcher ───────────────────────────────────────────────────────

# Maps action index to prompt function for clean dispatch.
ACTION_PROMPT_FNS = [
    prompt_reason,      # 0
    prompt_decompose,   # 1
    prompt_debate,      # 2
    prompt_refine,      # 3
    prompt_terminate,   # 4
]


def get_action_prompt(
    action: int,
    problem: str,
    previous_steps: List[str],
) -> str:
    """Get the prompt for a given action.

    Args:
        action: Action index (0-4).
        problem: The original problem text.
        previous_steps: List of SLM outputs from previous steps.

    Returns:
        The formatted prompt string.

    Raises:
        IndexError: If action is out of range.
    """
    return ACTION_PROMPT_FNS[action](problem, previous_steps)
