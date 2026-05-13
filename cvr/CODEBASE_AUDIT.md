# CVR Codebase Audit

**Project root:** `/Users/macbook/Desktop/SIT 330/2.3/SLM-Math-Reasoning/`

## Three SLMs (`models.py`)

| Key | Model ID | Family |
|---|---|---|
| `qwen25_math_1.5b` | `Qwen/Qwen2.5-Math-1.5B-Instruct` | `qwen_math` |
| `gemma4_e2b` | `google/gemma-4-E2B-IT` | `gemma4` |
| `phi4_mini` | `microsoft/Phi-4-mini-instruct` | `phi4` |

## Model Interface

- Factory: `get_model_wrapper(config: ModelConfig, device="auto") -> BaseModelWrapper`
- All wrappers: `load()`, `generate(system_prompt, user_prompt, max_new_tokens=512)`, `unload()`
- Internal attributes: `wrapper.model`, `wrapper.tokenizer_or_processor`, `wrapper.config.family`
- Default: greedy decoding (`do_sample=False`). CVR adapter adds sampling via `model_adapter.py`.
- Gemma4 uses `AutoProcessor` (not `AutoTokenizer`); its `apply_chat_template` takes content dicts.

## Distractor Dataset (Experiment 1)

- **Original:** `Experiment1/gsm_templates/` — 100 JSON files
  - Schema: `{question, answer, id_orig, id_shuffled, question_annotated, answer_annotated}`
- **Distractor:** `Experiment1/gsm_enhanced_templates/` — 100 JSON files
  - Extra field: `dynamic_distractor_pool: [{type, text}, ...]` (5 distractors/problem)
  - Distractor types: TYPE_A (scope confusion), TYPE_B (wrong formula), TYPE_C (unit trap), TYPE_D (temporal bait)
- **Load functions** (reuse from `Experiment1/run_inference_eval.py`):
  - `load_original_examples(template_dir)` → `list[dict]`
  - `load_enhanced_distracted_examples(enhanced_dir, seed=42)` → `list[dict]`
  - Output dict keys: `question, answer, question_original, distractor, distractor_type, id_orig, id_shuffled, source_file`

## Multi-step Dataset (Experiment 2)

- **Location:** `Experiment2/data/problems_all.json` (1,600 problems) or per-depth `problems_depth0X.json`
- **Schema:** `{problem_id, depth, question, ground_truth, intermediate_values, operations, answer_unit}`
- **Load:** `json.load(open(path))` directly; convert records to `MathProblem` dataclass if needed
- **Step count:** depth field (1–8); `intermediate_values` list has one entry per step

## Existing Utilities (reuse — do NOT reimplement)

| Function | Location | Purpose |
|---|---|---|
| `extract_model_final(text)` | `Experiment1/run_inference_eval.py:140` | Multi-stage answer extraction |
| `extract_gsm_final(answer)` | `Experiment1/run_inference_eval.py:110` | Gold label extraction |
| `answers_match(gold, pred, eps)` | `Experiment1/run_inference_eval.py:194` | Numeric comparison |
| `parse_answer(response)` | `Experiment2/dataset_generator.py` | Float extraction for Exp 2 |
| `check_step_accuracy(response, problem)` | `Experiment2/dataset_generator.py` | Step-level error detection |
| `SYSTEM_COT`, `SYSTEM_DIRECT` | `Experiment2/dataset_generator.py` | Baseline system prompts |
| `build_prompt_cot(problem)` | `Experiment2/dataset_generator.py` | CoT user prompt builder |

## Existing Baselines

- **Direct** — Experiment 2 `direct` regime (`SYSTEM_DIRECT`)
- **CoT** — Experiment 2 `cot` regime (`SYSTEM_COT` + step-by-step prompt)
- **Self-Consistency** — NOT implemented; `cvr/baselines/baseline_sc.py` adds it

## Result Formats

- **Experiment 1 style:** nested JSON `{meta, original: {model: {accuracy, n_correct, n_evaluated, results: [...]}}, enhanced: {...}}`
- **Experiment 2 style:** JSONL per `{model}_{regime}.jsonl`, one record per problem with fields: `model, regime, problem_id, depth, question, ground_truth, response, predicted, is_correct, is_collapse, first_error_step, elapsed_s`

## Inference Mechanism

- Local Hugging Face Transformers (no vLLM, no API)
- Models loaded one at a time, unloaded between runs
- Greedy decoding via `model.generate(do_sample=False)` in wrappers
- Chat template applied via `tokenizer.apply_chat_template()` (Qwen, Phi) or `processor.apply_chat_template()` (Gemma)
