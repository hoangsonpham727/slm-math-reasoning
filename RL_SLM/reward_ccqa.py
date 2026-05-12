"""
CCQA-style cycle-consistency reward (Reward Model A).

After each reasoning step the accumulated chain is fed to a frozen
Flan-T5-base (250 MB) with the prompt:
  "Generate the question for this solution: <steps>"
The model's output is compared to the original problem text via Jaccard
word-overlap, which becomes the step reward (float in [0, 1]).

The class-level singleton (_model / _tokenizer) means the 250 MB checkpoint
is loaded at most once per process even if multiple ENV instances are created.

PRM-compatible adapter methods (covert_to_input / get_step_scores) allow
train.py to call this reward exactly like the original PRM.
"""
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class CCQAReward:
    _model     = None
    _tokenizer = None

    def __init__(self, device: str = "cpu", model_name: str = "google/flan-t5-base"):
        if CCQAReward._model is None:
            CCQAReward._tokenizer = T5Tokenizer.from_pretrained(model_name)
            CCQAReward._model = (
                T5ForConditionalGeneration
                .from_pretrained(model_name)
                .to(device)
            )
            CCQAReward._model.eval()
        self.device = device

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(self, problem: str, steps: list) -> float:
        """
        Concatenate all steps into a solution string, ask Flan-T5 to
        regenerate the original question, and return Jaccard overlap
        with the original problem text.
        """
        solution = "\n".join(steps)
        prompt   = f"Generate the question for this solution: {solution}"

        inputs = CCQAReward._tokenizer(
            prompt,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 512,
        ).to(self.device)

        with torch.no_grad():
            output_ids = CCQAReward._model.generate(**inputs, max_new_tokens=128)

        regenerated = CCQAReward._tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )
        return _jaccard(problem, regenerated)

    # ------------------------------------------------------------------
    # PRM-compatible adapter interface
    # ------------------------------------------------------------------

    def covert_to_input(self, problem: str, thoughts: list):
        """Match PRM.covert_to_input signature; returns (problem, thoughts)."""
        return (problem, thoughts)

    def get_step_scores(self, prm_input):
        """
        Score each prefix of the thought chain incrementally.
        Returns (scores: list[float], n_steps: int).
        """
        problem, thoughts = prm_input
        scores = [self.compute(problem, thoughts[: i + 1]) for i in range(len(thoughts))]
        return scores, len(thoughts)


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)
