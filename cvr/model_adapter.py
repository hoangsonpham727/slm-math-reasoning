"""
Thin adapter around BaseModelWrapper that adds sampled generation.

The existing wrappers all use greedy decoding (do_sample=False). CVR needs
sampling for NCV-style binary verification (temperature=0.3, multiple votes)
and for diverse chain generation (temperature=0.7). Rather than modifying
models.py, this adapter accesses the underlying model and
tokenizer_or_processor directly to issue sampled calls.

Each model family has slightly different chat-template and input handling —
this adapter mirrors the exact logic from each wrapper's generate() method,
but with do_sample=True and a configurable temperature.
"""

from __future__ import annotations

import torch
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import BaseModelWrapper


class CVRModelAdapter:
    """
    Wraps a loaded BaseModelWrapper and exposes:
      - generate_greedy(system, user, max_new_tokens) — delegates to wrapper
      - generate_sampled(system, user, max_new_tokens, temperature) — sampled
    """

    def __init__(self, wrapper: "BaseModelWrapper"):
        self._wrapper = wrapper
        self._family = wrapper.config.family
        self._short_name = wrapper.config.short_name

    @property
    def model_key(self) -> str:
        return self._short_name

    def generate_greedy(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        return self._wrapper.generate(system_prompt, user_prompt, max_new_tokens)

    def generate_sampled(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        family = self._family
        if family == "qwen_math":
            return self._sample_qwen(system_prompt, user_prompt, max_new_tokens, temperature)
        elif family == "gemma4":
            return self._sample_gemma4(system_prompt, user_prompt, max_new_tokens, temperature)
        elif family == "phi4":
            return self._sample_phi4(system_prompt, user_prompt, max_new_tokens, temperature)
        else:
            raise ValueError(f"Unknown model family: {family}")

    # ── per-family sampled generation ─────────────────────────────────────────

    def _sample_qwen(self, system_prompt, user_prompt, max_new_tokens, temperature):
        wrapper = self._wrapper
        tokenizer = wrapper.tokenizer_or_processor
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(wrapper.model.device)
        with torch.no_grad():
            output_ids = wrapper.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _sample_gemma4(self, system_prompt, user_prompt, max_new_tokens, temperature):
        wrapper = self._wrapper
        processor = wrapper.tokenizer_or_processor
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(wrapper.model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output_ids = wrapper.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
            )
        new_tokens = output_ids[0][input_len:]
        return processor.decode(new_tokens, skip_special_tokens=True)

    def _sample_phi4(self, system_prompt, user_prompt, max_new_tokens, temperature):
        wrapper = self._wrapper
        tokenizer = wrapper.tokenizer_or_processor
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            raw_inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(wrapper.model.device)
        except Exception:
            fallback_text = (
                f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
            )
            raw_inputs = tokenizer(fallback_text, return_tensors="pt")

        if isinstance(raw_inputs, Mapping) or hasattr(raw_inputs, "input_ids"):
            if not isinstance(raw_inputs, Mapping):
                raw_inputs = dict(raw_inputs)
            model_inputs = {k: v.to(wrapper.model.device) for k, v in raw_inputs.items()}
            input_len = model_inputs["input_ids"].shape[1]
            with torch.no_grad():
                output_ids = wrapper.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id,
                )
        else:
            inputs_tensor = raw_inputs.to(wrapper.model.device)
            input_len = inputs_tensor.shape[1]
            with torch.no_grad():
                output_ids = wrapper.model.generate(
                    inputs_tensor,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id,
                )

        new_tokens = output_ids[0][input_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
