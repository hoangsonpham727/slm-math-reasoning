"""
Thin wrapper around models.py for the Experiment 4 chunked incremental solver.

Usage:
    from llm_wrapper import llm, init_model

    init_model("qwen25_math_1.5b", device="cuda:0")  # call once at startup
    response = llm("What is 2 + 2?", system="You are a math solver.")
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import get_all_configs, get_model_wrapper

_wrapper = None


def init_model(short_name: str = "qwen25_math_1.5b", device: str = "auto") -> None:
    """Load the specified model. Must be called before llm()."""
    global _wrapper
    configs = {c.short_name: c for c in get_all_configs()}
    if short_name not in configs:
        raise ValueError(
            f"Unknown model '{short_name}'. "
            f"Available: {list(configs.keys())}"
        )
    if _wrapper is not None:
        _wrapper.unload()
    _wrapper = get_model_wrapper(configs[short_name], device=device)
    _wrapper.load()


def llm(
    prompt: str,
    system: str = "You are a math solver. Solve problems step by step.",
    temperature: float = 0.0,
    max_new_tokens: int = 512,
) -> str:
    """Call the loaded SLM. Returns response as a plain string."""
    if _wrapper is None:
        raise RuntimeError(
            "Model not loaded. Call init_model() before llm()."
        )
    return _wrapper.generate(
        system_prompt=system,
        user_prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0.0),
    )


if __name__ == "__main__":
    init_model("qwen25_math_1.5b")
    response = llm("What is 2 + 2?")
    assert isinstance(response, str) and len(response) > 0, "Empty response"
    print("Wrapper OK")
    print("Response:", response[:200])
