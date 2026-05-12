"""
Intermediate-step programmatic verifier (Reward Model B).

Extracts arithmetic expressions of the form  LHS = RHS  from a reasoning
step (where LHS contains only digits, whitespace, and +−*/().),  evaluates
LHS with Python, and returns the fraction of expressions that are correct.

Returns 0.5 when no verifiable arithmetic expression is found, so it
degrades gracefully on non-arithmetic steps rather than assigning 0.

Reliable on GSM8K and the Exp 2 step-controlled chains where every
intermediate value appears explicitly as  "... = <number>".

Security note: eval() operates only on strings matched by a whitelist
regex that permits digits, spaces, and the five arithmetic operators.
No imports, attribute access, or arbitrary code can reach eval().

PRM-compatible adapter methods (covert_to_input / get_step_scores) allow
train.py to call this reward exactly like the original PRM.
"""
import re

# Whitelist: LHS = digits / whitespace / ( ) + - * /  only.
# RHS = optional leading minus, digits, optional decimal.
_EXPR_RE = re.compile(
    r'([\d\s\+\-\*\/\(\)\.]+)\s*=\s*(-?[\d]+(?:\.[\d]+)?)'
)


class ProgrammaticReward:

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(self, step_text: str) -> float:
        """
        Score one reasoning step.  Returns float in [0, 1], or 0.5 if
        no arithmetic expression can be verified.
        """
        matches = _EXPR_RE.findall(step_text)
        if not matches:
            return 0.5

        correct = 0
        for lhs, rhs in matches:
            try:
                # compile() with mode="eval" raises SyntaxError on statements,
                # giving an extra layer of safety on top of the regex whitelist.
                computed = eval(compile(lhs.strip(), "<string>", "eval"))  # nosec
                if abs(computed - float(rhs)) < 1e-6:
                    correct += 1
            except Exception:
                pass

        return correct / len(matches)

    # ------------------------------------------------------------------
    # PRM-compatible adapter interface
    # ------------------------------------------------------------------

    def covert_to_input(self, problem: str, thoughts: list):
        """Match PRM.covert_to_input signature; returns (problem, thoughts)."""
        return (problem, thoughts)

    def get_step_scores(self, prm_input):
        """
        Score each thought step individually (no cross-step context needed).
        Returns (scores: list[float], n_steps: int).
        """
        _, thoughts = prm_input
        scores = [self.compute(t) for t in thoughts]
        return scores, len(thoughts)
