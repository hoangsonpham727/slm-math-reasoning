"""
LocalHFVerifierAdapter — runs a local HuggingFace model as the CVR verifier.

Provides the same generate_sampled / generate_greedy interface as
CVRModelAdapter and CloudVerifierAdapter so it drops in as a verifier
without any pipeline changes.

Recommended model: meta-llama/Meta-Llama-3.1-8B-Instruct
  - Strong instruction following for binary Correct/Wrong responses
  - No rate limits (fully local)
  - Runs in ~16 GB (bf16) or ~5 GB (4-bit) VRAM

Usage in config.yaml:
    verifier_local_hf:
      enabled: true
      model_id: "meta-llama/Meta-Llama-3.1-8B-Instruct"
      device: "cuda"
      quantization: "none"   # "none" | "4bit" | "8bit"
"""

from __future__ import annotations

import torch


def pull_model(model_id: str) -> None:
    """
    Download a model from HuggingFace Hub into the local cache.
    Respects HF_TOKEN env var for gated models (e.g. Llama 3.1).
    """
    from huggingface_hub import snapshot_download
    import os

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"Downloading {model_id} from HuggingFace Hub...")
    path = snapshot_download(model_id, token=token)
    print(f"Saved to: {path}")


class LocalHFVerifierAdapter:
    """
    Wraps a locally loaded HuggingFace model as a CVR verifier adapter.

    Args:
        model_id: HuggingFace model ID, e.g. "meta-llama/Meta-Llama-3.1-8B-Instruct"
        device: "auto", "cuda", "cuda:0", "cpu"
        quantization: "none" (bf16), "4bit", or "8bit"
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        device: str = "auto",
        quantization: str = "none",
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import os

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self._model_id = model_id

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        load_kwargs: dict = {"device_map": device, "torch_dtype": torch.bfloat16}
        if quantization == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            load_kwargs.pop("torch_dtype", None)
        elif quantization == "8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs.pop("torch_dtype", None)

        print(f"  [HF verifier] Loading {model_id} ({quantization}) ...")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, token=token, **load_kwargs
        )
        self._model.eval()
        print(f"  [HF verifier] Ready on {next(self._model.parameters()).device}")

    @property
    def model_key(self) -> str:
        return self._model_id.split("/")[-1]

    def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(next(self._model.parameters()).device)

        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self._model.generate(input_ids, **gen_kwargs)

        new_tokens = output_ids[0][input_ids.shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_sampled(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.3,
    ) -> str:
        return self._generate(system_prompt, user_prompt, max_new_tokens, True, temperature)

    def generate_greedy(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 64,
    ) -> str:
        return self._generate(system_prompt, user_prompt, max_new_tokens, False, 0.0)


def build_hf_verifier(cfg: dict) -> LocalHFVerifierAdapter:
    """Build from the verifier_local_hf section of config.yaml."""
    return LocalHFVerifierAdapter(
        model_id=cfg.get("model_id", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        device=cfg.get("device", "auto"),
        quantization=cfg.get("quantization", "none"),
    )
