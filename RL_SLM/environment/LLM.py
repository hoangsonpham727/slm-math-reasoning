"""
LLM_api — thin wrapper that connects the RL environment to the HuggingFace
model wrappers defined in models.py (repo root).

Accepts a model short_name (e.g. "qwen25_math_1.5b") or a full HF model_id
(e.g. "Qwen/Qwen2.5-Math-1.5B-Instruct").  The model is loaded once on
construction and kept resident for the lifetime of the ENV object.
"""
import sys
from pathlib import Path

# Make the repo root importable so models.py is reachable from this subdirectory.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from models import MODEL_CONFIGS, get_model_wrapper


class LLM_api:
    """
    Interface expected by ENV:
      get_text(prompt: str) -> str
      reset_token()
      print_usage()
    """

    def __init__(self, model: str, device: str = "auto"):
        cfg = None
        for c in MODEL_CONFIGS:
            if c.short_name == model or c.model_id == model:
                cfg = c
                break
        if cfg is None:
            raise ValueError(
                f"Unknown model '{model}'. "
                f"Valid short_names: {[c.short_name for c in MODEL_CONFIGS]}"
            )
        self._wrapper   = get_model_wrapper(cfg, device=device)
        self._wrapper.load()
        self._n_calls   = 0

    def get_text(self, prompt: str, max_new_tokens: int = 1024) -> str:
        self._n_calls += 1
        # Pass the full prompt as the user turn; system prompt left empty so
        # the reasoning model is not given any meta-instructions here.
        return self._wrapper.generate(
            system_prompt  = "",
            user_prompt    = prompt,
            max_new_tokens = max_new_tokens,
        )

    def reset_token(self):
        self._n_calls = 0

    def print_usage(self):
        print(f"  [LLM_api] total generate calls: {self._n_calls}")
