"""
CloudVerifierAdapter — routes binary verification calls to gpt-oss:120b via Ollama cloud.

Uses the same ollama.Client pattern from Experiment1/data_preparation.py:
    Client(host="https://ollama.com", headers={"Authorization": "Bearer <OLLAMA_API_KEY>"})

Only verification calls go to the cloud. Generation stays on the local SLM.
The adapter exposes generate_sampled / generate_greedy so it drops in as a
direct replacement for CVRModelAdapter in NodeVerifier.
"""

from __future__ import annotations

import os


class CloudVerifierAdapter:
    """
    Wraps the Ollama cloud client with the same interface as CVRModelAdapter
    so it can be passed to NodeVerifier without any other changes.
    """

    def __init__(
        self,
        model: str = "gpt-oss:120b",
        host: str = "https://ollama.com",
        api_key: str | None = None,
    ):
        try:
            from ollama import Client
        except ImportError as e:
            raise ImportError(
                "ollama package is required. Install with: pip install ollama"
            ) from e

        key = api_key or os.environ.get("OLLAMA_API_KEY", "")
        self._model = model
        self._client = Client(
            host=host,
            headers={"Authorization": f"Bearer {key}"},
        )

    @property
    def model_key(self) -> str:
        return self._model

    def generate_sampled(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.3,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        response = ""
        for part in self._client.chat(
            self._model,
            messages=messages,
            stream=True,
            options={"num_predict": max_new_tokens, "temperature": temperature},
        ):
            response += part["message"]["content"]
        return response

    def generate_greedy(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 64,
    ) -> str:
        return self.generate_sampled(
            system_prompt, user_prompt, max_new_tokens, temperature=0.0
        )


def build_cloud_verifier(cfg: dict) -> CloudVerifierAdapter:
    """Build from the verifier_cloud section of config.yaml."""
    return CloudVerifierAdapter(
        model=cfg.get("model", "gpt-oss:120b"),
        host=cfg.get("host", "https://ollama.com"),
        api_key=cfg.get("api_key"),
    )
