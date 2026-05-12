"""
Handles loading and inference for three models with their specific quirks:
  - Qwen/Qwen2.5-Math-1.5B-Instruct       (standard chat template + \boxed{} output)
  - google/gemma-4-E2B-IT                  (Gemma4ForConditionalGeneration + AutoProcessor)
  - microsoft/Phi-4-mini-instruct          (standard chat; may need eager attn on V100)
"""

import torch
import gc
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

# ── Lazy imports so the file can be imported even before transformers is ready ──
_transformers = None

def _get_transformers():
    global _transformers
    if _transformers is None:
        import transformers
        _transformers = transformers
    return _transformers


# ---------------------------------------------------------------------------
# Model configuration registry
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    model_id: str
    short_name: str
    family: str           # "qwen_math" | "gemma4" | "phi4"
    dtype: str            # "bfloat16" | "float16" | "auto"
    trust_remote_code: bool = False
    use_flash_attn: bool  = True   # set False to force sdpa/eager


MODEL_CONFIGS = [
    ModelConfig(
        model_id     = "Qwen/Qwen2.5-Math-1.5B-Instruct",
        short_name   = "qwen25_math_1.5b",
        family       = "qwen_math",
        dtype        = "bfloat16",
        trust_remote_code = False,
    ),
    ModelConfig(
        model_id     = "google/gemma-4-E2B-IT",
        short_name   = "gemma4_e2b",
        family       = "gemma4",
        dtype        = "bfloat16",
        trust_remote_code = False,
    ),
    ModelConfig(
        model_id     = "microsoft/Phi-4-mini-instruct",
        short_name   = "phi4_mini",
        family       = "phi4",
        dtype        = "bfloat16",
        trust_remote_code = False,
        use_flash_attn    = False,
    ),
]


# ---------------------------------------------------------------------------
# Model wrapper classes
# ---------------------------------------------------------------------------

class BaseModelWrapper:
    """Shared interface for all model wrappers."""

    def __init__(self, config: ModelConfig, device: str = "auto"):
        self.config = config
        self.device = device
        self.model  = None
        self.tokenizer_or_processor = None

    def load(self):
        raise NotImplementedError

    def _resolve_device_map(self):
        """
        Convert CLI `device` values into a HF-compatible device_map.
        - "auto"/"balanced"/... are passed through.
        - "cuda:0" or "0" pins the whole model to one GPU.
        - "cpu" pins to CPU.
        """
        d = str(self.device).strip()
        if d in {"auto", "balanced", "balanced_low_0", "sequential"}:
            return d
        if d.isdigit():
            return {"": f"cuda:{d}"}
        if d.startswith("cuda:"):
            return {"": d}
        if d in {"cpu", "mps"}:
            return {"": d}
        # Fallback: preserve existing behavior for unknown values.
        return d

    @staticmethod
    def _normalize_model_inputs(inputs, device):
        """
        Normalize tokenizer/processor outputs into either:
        - dict[str, Tensor] for kwargs passing to generate, or
        - Tensor for positional passing.
        Returns (model_inputs, input_len).
        """
        if isinstance(inputs, Mapping) or hasattr(inputs, "input_ids"):
            if not isinstance(inputs, Mapping):
                inputs = dict(inputs)
            model_inputs = {k: v.to(device) for k, v in inputs.items()}
            input_len = model_inputs["input_ids"].shape[1]
            return model_inputs, input_len
        model_inputs = inputs.to(device)
        input_len = model_inputs.shape[1]
        return model_inputs, input_len

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        raise NotImplementedError

    def unload(self):
        """Free GPU memory between model runs."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer_or_processor is not None:
            del self.tokenizer_or_processor
            self.tokenizer_or_processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [{self.config.short_name}] unloaded.")


# ── Qwen 2.5 Math ──────────────────────────────────────────────────────────

class QwenMathWrapper(BaseModelWrapper):

    def load(self):
        tr = _get_transformers()
        dtype = getattr(torch, self.config.dtype)
        print(f"  Loading {self.config.model_id} ...")
        self.tokenizer_or_processor = tr.AutoTokenizer.from_pretrained(
            self.config.model_id
        )
        self.model = tr.AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            dtype=dtype,
            device_map=self._resolve_device_map(),
            attn_implementation="sdpa",
        )
        self.model.eval()
        print(f"  [{self.config.short_name}] loaded on {self.device}.")

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 512) -> str:
        tokenizer = self.tokenizer_or_processor
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Gemma 4 E2B ────────────────────────────────────────────────────────────

class Gemma4Wrapper(BaseModelWrapper):
    """
    Gemma 4 E2B IT uses Gemma4ForConditionalGeneration and AutoProcessor.
    Requires transformers >= 4.53.0.
    Thinking mode: disabled by default (no <|think|> in system prompt).
    """

    def load(self):
        tr = _get_transformers()
        torch_dtype = getattr(torch, self.config.dtype)

        print(f"  Loading {self.config.model_id} ...")
        self.tokenizer_or_processor = tr.AutoProcessor.from_pretrained(
            self.config.model_id
        )
        self.model = tr.AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            dtype=torch_dtype,
            device_map=self._resolve_device_map(),
        )
        self.model.eval()
        print(f"  [{self.config.short_name}] loaded on {self.device}.")

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 512) -> str:
        processor = self.tokenizer_or_processor
        # Standard system/user/assistant roles 
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user",   "content": [{"type": "text", "text": user_prompt}]},
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        new_tokens = output_ids[0][input_len:]
        return processor.decode(new_tokens, skip_special_tokens=True)


# ── Phi-4-mini-instruct ────────────────────────────────────────────────────

class Phi4MiniWrapper(BaseModelWrapper):
    """
    Phi-4-mini-instruct (3.8B). Uses standard chat format.
    On V100 or older GPUs, set use_flash_attn=False in the config.
    """

    def load(self):
        tr = _get_transformers()
        torch_dtype = getattr(torch, self.config.dtype)

        print(f"  Loading {self.config.model_id} ...")
        self.tokenizer_or_processor = tr.AutoTokenizer.from_pretrained(
            self.config.model_id
        )
        if self.config.use_flash_attn:
            attn_candidates = ["flash_attention_2", "sdpa", "eager"]
        else:
            attn_candidates = ["sdpa", "eager"]

        last_error: Optional[Exception] = None
        for attn_impl in attn_candidates:
            try:
                print(
                    f"Trying attn_implementation={attn_impl} "
                    f"for {self.config.short_name}"
                )
                self.model = tr.AutoModelForCausalLM.from_pretrained(
                    self.config.model_id,
                    dtype=torch_dtype,
                    device_map=self._resolve_device_map(),
                    attn_implementation=attn_impl,
                )
                print(
                    f"Loaded with attn_implementation={attn_impl} "
                    f"for {self.config.short_name}"
                )
                break
            except Exception as e:
                last_error = e
                print(
                    f"    failed attn_implementation={attn_impl} "
                    f"for {self.config.short_name}: {e}"
                )
        else:
            raise RuntimeError(
                f"Unable to load {self.config.model_id} with attention backends "
                f"{attn_candidates}. Last error: {last_error}"
            ) from last_error

        self.model.eval()
        print(f"  [{self.config.short_name}] loaded on {self.device}.")

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 512) -> str:
        tokenizer = self.tokenizer_or_processor
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        try:
            # Preferred path: let tokenizer produce model-ready tensors directly.
            raw_inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
        except Exception as e:
            # Some tokenizer/template/runtime combinations (seen with Phi-4) can
            # fail inside apply_chat_template with errors like
            # "the first argument must be callable". Fall back to plain prompting.
            print(
                f"    apply_chat_template failed for {self.config.short_name}: {e}. "
                "Falling back to raw prompt formatting."
            )
            fallback_text = (
                f"System: {system_prompt}\n"
                f"User: {user_prompt}\n"
                "Assistant:"
            )
            raw_inputs = tokenizer(fallback_text, return_tensors="pt")

        inputs, _ = self._normalize_model_inputs(raw_inputs, self.model.device)

        with torch.no_grad():
            if isinstance(inputs, Mapping):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                output_ids = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
        if isinstance(inputs, Mapping):
            input_len = inputs["input_ids"].shape[1]
        else:
            input_len = inputs.shape[1]
        new_tokens = output_ids[0][input_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_WRAPPER_MAP = {
    "qwen_math":       QwenMathWrapper,
    "gemma4":          Gemma4Wrapper,
    "phi4":            Phi4MiniWrapper,
}


def get_model_wrapper(config: ModelConfig, device: str = "auto") -> BaseModelWrapper:
    cls = _WRAPPER_MAP.get(config.family)
    if cls is None:
        raise ValueError(f"Unknown model family: {config.family}")
    return cls(config, device)


def get_all_configs() -> list:
    return MODEL_CONFIGS