"""
Experiment 1 plotting utility: accuracy and accuracy drop.

Reads `inference_results.json` from `run_inference_eval.py` and generates:
  1) Grouped bar plot for original vs distracted accuracy by model.
  2) Line plot for accuracy drop (original - distracted) by model.

Usage:
    python plot_accuracy_drop.py \
        --input ./inference_results.json \
        --output-dir ./figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_payload(input_path: Path) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "original" not in payload or "enhanced" not in payload:
        raise ValueError(
            "Input JSON must contain both 'original' and 'enhanced' result blocks."
        )
    return payload


def _collect_metrics(payload: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    original = payload["original"]
    enhanced = payload["enhanced"]

    models = sorted(set(original.keys()) & set(enhanced.keys()))
    if not models:
        raise ValueError("No overlapping model keys found between original and enhanced.")

    acc_orig = np.array([float(original[m]["accuracy"]) for m in models], dtype=float)
    acc_enh = np.array([float(enhanced[m]["accuracy"]) for m in models], dtype=float)
    return models, acc_orig, acc_enh


def _plot_grouped_bar(
    models: list[str],
    acc_orig: np.ndarray,
    acc_enh: np.ndarray,
    out_path: Path,
) -> None:
    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_orig = ax.bar(x - width / 2, acc_orig, width=width, label="Original", color="#4C72B0")
    bars_enh = ax.bar(x + width / 2, acc_enh, width=width, label="Distracted", color="#DD8452")

    # Annotate each bar with its value.
    for bar in list(bars_orig) + list(bars_enh):
        h = bar.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_drop_line(
    models: list[str],
    acc_orig: np.ndarray,
    acc_enh: np.ndarray,
    out_path: Path,
) -> None:
    drop = acc_orig - acc_enh
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, drop, marker="o", linewidth=2, color="#C44E52")

    for i, d in enumerate(drop):
        ax.annotate(f"{d:.3f}", (x[i], d), textcoords="offset points", xytext=(0, 8), ha="center")

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy Drop")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
    ymax = max(0.02, float(drop.max()) * 1.2)
    ymin = min(-0.02, float(drop.min()) * 1.2)
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Experiment 1 bar/line plots for accuracy drop."
    )
    parser.add_argument(
        "--input",
        default="./inference_results.json",
        help="Path to Experiment 1 inference results JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="./figures",
        help="Directory to save output figures.",
    )
    parser.add_argument(
        "--bar-name",
        default="fig_exp1_accuracy_bar.png",
        help="Output filename for grouped bar plot.",
    )
    parser.add_argument(
        "--line-name",
        default="fig_exp1_accuracy_drop_line.png",
        help="Output filename for accuracy-drop line plot.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_payload(input_path)
    models, acc_orig, acc_enh = _collect_metrics(payload)

    bar_out = output_dir / args.bar_name
    line_out = output_dir / args.line_name

    _plot_grouped_bar(models, acc_orig, acc_enh, bar_out)
    _plot_drop_line(models, acc_orig, acc_enh, line_out)

    print(f"Saved grouped bar plot: {bar_out}")
    print(f"Saved drop line plot:   {line_out}")


if __name__ == "__main__":
    main()
