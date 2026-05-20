# SLM Math Reasoning

This repository explores and evaluates the mathematical reasoning capabilities of various Small Language Models (SLMs). The experiments examine techniques to improve or benchmark the performance of these models on reasoning tasks.

## Models Evaluated

The code supports inference and evaluation for the following small language models:
- **Qwen2.5-Math-1.5B-Instruct** (`Qwen/Qwen2.5-Math-1.5B-Instruct`)
- **Gemma 4 E2B** (`google/gemma-4-E2B-IT`)
- **Phi-4-mini** (`microsoft/Phi-4-mini-instruct`)
- **Qwen2.5-7B-Instruct** (`Qwen/Qwen2.5-7B-Instruct`)

## Project Structure

- **`models.py`**: Handles loading, configuration, and inference for the different SLMs with their specific formatting and generation quirks.
- **`Experiment1/`**: Contains scripts for the first set of experiments, including data preparation, inference, and plotting accuracy drop.
- **`Experiment2/`**: Contains scripts for the second set of experiments, including dataset generation and analysis.
- **`2-stage/`**: Implements and evaluates a 2-stage pipeline approach.
- **`CISV/`**: Implements Chunk-wise Incremental Solving with Verification (CISV) and includes comprehensive analysis and results plotting scripts.

## Setup

1. Clone the repository.
2. Install the required dependencies (such as `transformers`, `torch`, `matplotlib`, `pandas`, `numpy`).
3. Scripts can be run individually or via the command line within their respective folders.

For example, to analyze CISV results:
```bash
python CISV/analyse_results.py
```
