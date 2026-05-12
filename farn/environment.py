"""
environment.py — MDP environment for navigator training and inference.

This module wraps together:
  - The SLM (generates reasoning steps)
  - The FeatureExtractor (computes external state)
  - The reward model (PRM or CCQA, scores intermediate results)

It exposes a gym-like interface:
  state = env.reset(problem)
  next_state, reward, done, info = env.step(action)

The environment is where the key architectural difference from RLoT
manifests: after each SLM generation step, instead of asking the SLM
to self-evaluate, we run the FeatureExtractor to compute the state
from external signals.
"""

import re
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import FARNConfig, ACTION_NAMES
from feature_extractor import FeatureExtractor
from action_prompts import get_action_prompt


class SLMWrapper:
    """Wraps a HuggingFace causal LM for single-step generation.

    Handles tokenisation, chat template formatting, and generation
    for each SLM in the project. Abstracts away model-specific
    differences (system prompt support, answer format, etc.).
    """

    def __init__(self, config: FARNConfig):
        """Load the SLM onto the GPU.

        Args:
            config: Full FARN configuration.
        """
        self.config = config
        self.model_config = config.model
        self.device = config.device

        print(f"Loading SLM: {self.model_config.hf_model_id}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.hf_model_id,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model — optionally quantised to 4-bit
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }

        if self.model_config.load_in_4bit:
            # Requires bitsandbytes
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            load_kwargs["device_map"] = self.device

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.hf_model_id,
            **load_kwargs,
        )
        self.model.eval()

        print(f"SLM loaded: {self.model_config.name} "
              f"({self.model_config.size_billions}B params)")

    def generate(self, prompt: str) -> str:
        """Generate a single reasoning step from the SLM.

        Args:
            prompt: The formatted prompt including problem and context.

        Returns:
            The SLM's generated text (decoded, stripped of prompt).
        """
        # Build messages in chat format
        messages = []
        if self.model_config.use_system_prompt:
            messages.append({
                "role": "system",
                "content": "You are a helpful math assistant. Solve problems step by step, showing all calculations clearly.",
            })
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        try:
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback for models without chat templates
            input_text = prompt

        # Tokenise and generate
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.model_config.max_new_tokens,
                do_sample=False,        # Greedy for reproducibility
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the new tokens (strip the prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()


class RewardModel:
    """Provides step-level reward signals for navigator training.

    Supports two reward sources:
      - PRM (Process Reward Model): Math-Shepherd scores each step
      - CCQA (Cycle-Consistency): Flan-T5 regenerates question from
        reasoning, similarity to original = reward

    At inference time, the reward model is NOT used (navigator runs
    the trained policy directly). This class is only for training.
    """

    def __init__(self, config: FARNConfig):
        """Load the reward model.

        Args:
            config: Full FARN configuration.
        """
        self.config = config
        self.reward_type = config.reward.reward_type

        if self.reward_type == "prm":
            self._load_prm()
        elif self.reward_type == "ccqa":
            self._load_ccqa()
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    def _load_prm(self) -> None:
        """Load Math-Shepherd PRM for process reward scoring."""
        print(f"Loading PRM: {self.config.reward.prm_model_id}...")

        # Math-Shepherd is a 7B model — load in 4-bit to fit alongside SLM
        from transformers import BitsAndBytesConfig

        self.prm_tokenizer = AutoTokenizer.from_pretrained(
            self.config.reward.prm_model_id,
            trust_remote_code=True,
        )
        self.prm_model = AutoModelForCausalLM.from_pretrained(
            self.config.reward.prm_model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            trust_remote_code=True,
        )
        self.prm_model.eval()
        print("PRM loaded.")

    def _load_ccqa(self) -> None:
        """Load Flan-T5 for cycle-consistency reward."""
        from transformers import AutoModelForSeq2SeqLM

        print(f"Loading CCQA model: {self.config.reward.ccqa_model_id}...")

        self.ccqa_tokenizer = AutoTokenizer.from_pretrained(
            self.config.reward.ccqa_model_id,
        )
        self.ccqa_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.reward.ccqa_model_id,
        ).to(self.config.device)
        self.ccqa_model.eval()
        print("CCQA model loaded.")

    def score(
        self,
        problem: str,
        reasoning_steps: List[str],
    ) -> float:
        """Compute the reward for the current reasoning state.

        Args:
            problem: Original problem text.
            reasoning_steps: All reasoning steps so far.

        Returns:
            Reward score (float). Higher is better.
        """
        if self.reward_type == "prm":
            return self._score_prm(problem, reasoning_steps)
        else:
            return self._score_ccqa(problem, reasoning_steps)

    def _score_prm(self, problem: str, steps: List[str]) -> float:
        """Score using Math-Shepherd PRM.

        Math-Shepherd expects input formatted as:
            problem + " Step 1: ... Step 2: ... "
        and outputs a score indicating correctness of the process.
        """
        # Format the reasoning chain for the PRM
        chain = problem
        for i, step in enumerate(steps, 1):
            chain += f" Step {i}: {step.strip()} ки\n"

        inputs = self.prm_tokenizer(
            chain,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.prm_model.device)

        with torch.no_grad():
            outputs = self.prm_model(**inputs)
            # Use the logit for the "+" token as the reward signal
            # (Math-Shepherd convention: higher logit for "+" = better step)
            logits = outputs.logits[:, -1, :]

            # Get token IDs for "+" and "-"
            plus_id = self.prm_tokenizer.encode("+", add_special_tokens=False)
            minus_id = self.prm_tokenizer.encode("-", add_special_tokens=False)

            if plus_id and minus_id:
                plus_logit = logits[0, plus_id[0]].item()
                minus_logit = logits[0, minus_id[0]].item()
                # Sigmoid to normalise to [0, 1]
                score = torch.sigmoid(
                    torch.tensor(plus_logit - minus_logit)
                ).item()
            else:
                score = 0.5  # Fallback

        return score

    def _score_ccqa(self, problem: str, steps: List[str]) -> float:
        """Score using CCQA cycle-consistency.

        1. Concatenate all reasoning steps into a "solution"
        2. Ask Flan-T5 to generate a question from that solution
        3. Compute text similarity between generated question and original
        4. Similarity score = reward

        If reasoning is corrupted (by distractor or error), the
        regenerated question will diverge from the original.
        """
        # Build the "solution" from reasoning steps
        solution = " ".join(s.strip() for s in steps)

        # Prompt Flan-T5 to regenerate the question
        prompt = f"Generate a math question that would produce this solution: {solution}"

        inputs = self.ccqa_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.ccqa_model.device)

        with torch.no_grad():
            outputs = self.ccqa_model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        regenerated_q = self.ccqa_tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )

        # Compute similarity (simple word-overlap / Jaccard)
        similarity = self._jaccard_similarity(problem, regenerated_q)

        return similarity

    @staticmethod
    def _jaccard_similarity(text_a: str, text_b: str) -> float:
        """Compute Jaccard similarity between two texts.

        Uses word-level tokens (lowercased, stripped of punctuation).
        Returns float in [0, 1].
        """
        def tokenize(text: str) -> set:
            return set(re.findall(r'\w+', text.lower()))

        tokens_a = tokenize(text_a)
        tokens_b = tokenize(text_b)

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b

        return len(intersection) / len(union)


class FARNEnvironment:
    """MDP environment for the FARN navigator.

    Wraps the SLM, feature extractor, and reward model into a
    gym-like interface that the training loop interacts with.

    Lifecycle of one episode:
        1. env.reset(problem_text, ground_truth_answer)
        2. Loop:
            a. action = navigator.select_action(state)
            b. next_state, reward, done, info = env.step(action)
            c. If done: break
        3. Check if SLM's final answer matches ground truth
    """

    def __init__(self, config: FARNConfig, reward_model: Optional[RewardModel] = None):
        """Initialise the environment.

        Args:
            config: Full FARN configuration.
            reward_model: Reward model for training. Can be None for
                          inference (when navigator is already trained).
        """
        self.config = config
        self.slm = SLMWrapper(config)
        self.reward_model = reward_model

        # These are set per-episode in reset()
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.problem_text: str = ""
        self.ground_truth: str = ""
        self.reasoning_steps: List[str] = []
        self.current_step: int = 0
        self.done: bool = False

    def reset(self, problem_text: str, ground_truth: str = "") -> np.ndarray:
        """Start a new episode with a problem.

        Args:
            problem_text: The math word problem to solve.
            ground_truth: The correct answer (for evaluation).

        Returns:
            Initial state vector of shape (8,).
        """
        self.problem_text = problem_text
        self.ground_truth = ground_truth
        self.reasoning_steps = []
        self.current_step = 0
        self.done = False

        # Create a fresh feature extractor for this problem
        self.feature_extractor = FeatureExtractor(
            problem_text, self.config.features
        )

        return self.feature_extractor.get_initial_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one action in the environment.

        1. Convert action to prompt
        2. Run SLM to generate reasoning step
        3. Extract features from the output
        4. Compute reward
        5. Check termination conditions

        Args:
            action: Action index (0-4).

        Returns:
            Tuple of (next_state, reward, done, info).
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")

        action_name = ACTION_NAMES[action]

        # ── Step 1: Build prompt and generate ──
        prompt = get_action_prompt(
            action=action,
            problem=self.problem_text,
            previous_steps=self.reasoning_steps,
        )

        slm_output = self.slm.generate(prompt)
        self.reasoning_steps.append(slm_output)
        self.current_step += 1

        # ── Step 2: Extract external features ──
        state = self.feature_extractor.extract(slm_output, self.current_step - 1)

        # ── Step 3: Check termination ──
        terminated = (
            action_name == "terminate"
            or self.current_step >= self.config.navigator.max_steps_per_episode
            or state[6] > 0.5  # f7: answer_present
        )
        self.done = terminated

        # ── Step 4: Compute reward ──
        if self.reward_model is not None:
            reward = self.reward_model.score(
                self.problem_text, self.reasoning_steps
            )
        else:
            reward = 0.0  # No reward during pure inference

        # ── Step 5: Build info dict ──
        info = {
            "action_name": action_name,
            "slm_output": slm_output,
            "step_number": self.current_step,
            "feature_values": {
                name: float(val)
                for name, val in zip(
                    self.feature_extractor.feature_names, state
                )
            },
        }

        # If episode is done, extract the final answer for evaluation
        if terminated:
            info["final_answer"] = self._extract_answer()
            info["correct"] = self._check_answer(info["final_answer"])

        return state, reward, terminated, info

    def _extract_answer(self) -> str:
        """Extract the final numerical answer from all reasoning steps.

        Tries multiple answer formats (####, \\boxed{}, "the answer is").
        Returns the extracted answer string, or empty string if none found.
        """
        full_output = "\n".join(self.reasoning_steps)

        # Try GSM8K format: #### 42
        match = re.search(r'####\s*([\d.,]+)', full_output)
        if match:
            return match.group(1).replace(",", "")

        # Try LaTeX boxed format: \boxed{42}
        match = re.search(r'\\boxed\{([^}]+)\}', full_output)
        if match:
            return match.group(1).strip()

        # Try "the answer is 42" format
        match = re.search(
            r'(?:the\s+)?(?:final\s+)?answer\s+is\s+([\d.,]+)',
            full_output,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).replace(",", "")

        # Fallback: last number in the output
        numbers = re.findall(r'([\d.,]+)', full_output)
        if numbers:
            return numbers[-1].replace(",", "")

        return ""

    def _check_answer(self, extracted: str) -> bool:
        """Check if the extracted answer matches the ground truth.

        Args:
            extracted: The answer extracted from SLM output.

        Returns:
            True if the answers match numerically.
        """
        if not extracted or not self.ground_truth:
            return False

        try:
            # Clean both answers
            gt_clean = self.ground_truth.replace(",", "").replace("$", "").strip()
            ex_clean = extracted.replace(",", "").replace("$", "").strip()

            gt_val = float(gt_clean)
            ex_val = float(ex_clean)

            return abs(gt_val - ex_val) < 1e-6
        except ValueError:
            return extracted.strip() == self.ground_truth.strip()

    def get_reasoning_trace(self) -> str:
        """Return the full reasoning trace for logging/debugging."""
        lines = [f"Problem: {self.problem_text}", ""]
        for i, step in enumerate(self.reasoning_steps, 1):
            lines.append(f"--- Step {i} ---")
            lines.append(step)
            lines.append("")
        return "\n".join(lines)
