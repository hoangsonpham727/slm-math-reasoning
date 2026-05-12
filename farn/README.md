# FARN — Failure-Aware Reasoning Navigator

An inference-time framework that enhances mathematical reasoning in Small Language Models (SLMs)
by replacing self-evaluation-based state representations with externally computed, failure-mode-aware
features. Inspired by RLoT (Hao et al., 2025) but with a fundamentally different state design.

## Architecture

```
Problem Text ──► Entity Registry (regex extraction)
                        │
                        ▼
              ┌─────────────────────┐
              │   SLM generates     │◄── Action prompt from Navigator
              │   one reasoning     │
              │   step              │
              └────────┬────────────┘
                       │
                       ▼
              ┌─────────────────────┐
              │  Feature Extractor  │  ◄── Pure Python, no model
              │  f1: relevance ratio│
              │  f2: unused count   │
              │  f3: entity-op align│
              │  f4: step progress  │
              │  f5: arith verify   │
              │  f6: bounds check   │
              │  f7: answer present │
              │  f8: step count     │
              └────────┬────────────┘
                       │
                       ▼
              ┌─────────────────────┐
              │  Navigator (DQN)    │  ◄── ~3.5K params, CPU-trainable
              │  8-dim state → action│
              └────────┬────────────┘
                       │
                       ▼
              Action: {Reason, Decompose, Debate, Refine, Terminate}
```

## File Structure

```
farn/
├── README.md
├── config.py              # All hyperparameters and model configs
├── feature_extractor.py   # 8-feature external state computation
├── entity_registry.py     # Problem text parsing and entity extraction
├── navigator.py           # Dueling DQN navigator model
├── action_prompts.py      # Prompt templates for each logic block
├── replay_buffer.py       # Experience replay for DQN training
├── environment.py         # MDP environment wrapping SLM + features
├── train.py               # Navigator training loop
├── inference.py           # Run trained navigator on test problems
├── evaluate.py            # Evaluation and metrics
└── feature_analysis.py    # Standalone feature correlation analysis
```

## Usage

### 1. Feature Analysis (no training needed)
```bash
python feature_analysis.py --model qwen25_math_1.5b --dataset gsm8k
```

### 2. Train Navigator
```bash
python train.py --model qwen25_math_1.5b --reward prm --episodes 500
```

### 3. Evaluate
```bash
python inference.py --model qwen25_math_1.5b --navigator checkpoints/nav_qwen.pt --dataset gsm8k
```
