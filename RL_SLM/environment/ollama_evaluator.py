"""
Ollama-based step evaluator.

Replaces SLM self-evaluation in ENV.thought_2_state with an external
GPT-OSS model served through Ollama Cloud, so the reasoning model and
the evaluator are always different models.

Config is read from environment variables matching data_preparation.py:
  OLLAMA_API_KEY  — Bearer token
  OLLAMA_URL      — base URL (default: https://ollama.com)
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_MODEL    = "gpt-oss:120b"
_DEFAULT_BASE_URL = "https://ollama.com"


class OllamaEvaluator:
    """
    Drop-in replacement for LLM_api that routes evaluation prompts to an
    Ollama-hosted model.  Exposes get_text(prompt) -> str only.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model_name = model_name
        self.api_key    = api_key    or os.getenv("OLLAMA_API_KEY", "")
        self.base_url   = (base_url or os.getenv("OLLAMA_URL", _DEFAULT_BASE_URL)).rstrip("/")

    def get_text(self, prompt: str) -> str:
        """POST prompt to Ollama /api/chat and return the assistant reply."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model":    self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream":   False,
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
