"""
GSM8K Comparative Analysis: CoT vs CISV

Loads results for all models from GSM8K/results/ and generates:
  - Fig 1: Bar chart — overall accuracy per model per method (CoT, CISV)
           with Wilson 95% CIs.
  - Fig 2: Accuracy by estimated step count (line plot, stratified by
           estimated_steps) — shows where CISV helps vs CoT.
  - Fig 3: Summary comparison table printed to stdout.

Usage:
    python analysis.py [--cot_dir results] [--cisv_dir results/cisv]
                       [--output_dir figures]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches


# ── Style constants ───────────────────────────────────────────────────────────

MODEL_LABELS = {
    "qwen25_math_1.5b": "Qwen2.5-Math-1.5B",
    "gemma4_e2b":        "Gemma 4 E2B",
    "phi4_mini":         "Phi-4-mini",
}
MODEL_COLORS = {
    "qwen25_math_1.5b": "#534AB7",
    "gemma4_e2b":        "#D85A30",
    "phi4_mini":         "#BA7517",
}
METHOD_STYLE = {
    "cot":     {"linestyle": "--", "marker": "o", "linewidth": 2.0},
    "chunked": {"linestyle": "-",  "marker": "D", "linewidth": 2.4},
}
METHOD_LABELS = {
    "cot":     "CoT",
    "chunked": "CISV (Ours)",
}
METHOD_COLORS = {
    "cot":     "#70AD47",
    "chunked": "#ED7D31",
}


# ── Wilson CI ────────────────────────────────────────────────────────────────

def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_cot_results(cot_dir: str) -> pd.DataFrame:
    """Load CoT JSONL files ({model}_cot.jsonl) into a DataFrame."""
    records = []
    for fpath in sorted(Path(cot_dir).glob("*_cot.jsonl")):
        model = fpath.stem.replace("_cot", "")
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    records.append({
                        "model":           model,
                        "method":          "cot",
                        "problem_id":      r["problem_id"],
                        "estimated_steps": int(r.get("estimated_steps", 0)),
                        "correct":         bool(r["is_correct"]),
                    })
                except (json.JSONDecodeError, KeyError):
                    continue
    if not records:
        raise FileNotFoundError(f"No *_cot.jsonl files found in '{cot_dir}'")
    df = pd.DataFrame(records)
    print(f"  CoT results : {len(df):,} records  ({df['model'].nunique()} models)")
    return df


def load_cisv_results(cisv_dir: str) -> pd.DataFrame:
    """Load CISV JSON files (exp4_chunked_{model}.json) into a DataFrame."""
    records = []
    for fpath in sorted(Path(cisv_dir).glob("exp4_chunked_*.json")):
        model = fpath.stem.replace("exp4_chunked_", "")
        if "_ckpt_" in model:
            model = model[:model.index("_ckpt_")]
        with open(fpath) as f:
            results = json.load(f)
        for r in results:
            records.append({
                "model":           model,
                "method":          "chunked",
                "problem_id":      r.get("id", ""),
                "estimated_steps": int(r.get("depth", r.get("estimated_steps", 0))),
                "correct":         bool(r["correct"]),
            })
    if not records:
        raise FileNotFoundError(f"No exp4_chunked_*.json files found in '{cisv_dir}'")
    df = pd.DataFrame(records)
    print(f"  CISV results: {len(df):,} records  ({df['model'].nunique()} models)")
    return df


# ── Accuracy computation ──────────────────────────────────────────────────────

def compute_overall_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, method): overall accuracy + Wilson CI."""
    rows = []
    for (model, method), grp in df.groupby(["model", "method"]):
        n = len(grp)
        k = grp["correct"].sum()
        acc = k / n if n > 0 else 0.0
        lo, hi = _wilson_ci(k, n)
        rows.append({"model": model, "method": method,
                     "n": n, "k": k, "acc": acc, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows).sort_values(["model", "method"])


def compute_acc_by_steps(df: pd.DataFrame, bin_size: int = 2) -> pd.DataFrame:
    """Per (model, method, step_bin): accuracy + Wilson CI."""
    df = df.copy()
    df["step_bin"] = ((df["estimated_steps"] - 1) // bin_size) * bin_size + 1
    rows = []
    for (model, method, step_bin), grp in df.groupby(["model", "method", "step_bin"]):
        n = len(grp)
        k = grp["correct"].sum()
        acc = k / n if n > 0 else 0.0
        lo, hi = _wilson_ci(k, n)
        rows.append({"model": model, "method": method, "step_bin": step_bin,
                     "n": n, "k": k, "acc": acc, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows).sort_values(["model", "method", "step_bin"])


# ── Figures ───────────────────────────────────────────────────────────────────

def _save(fig, output_dir: str, name: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = Path(output_dir) / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {Path(output_dir) / name}.{{pdf,png}}")


def fig1_overall_accuracy(overall_df: pd.DataFrame, output_dir: str):
    """Figure 1: Grouped bar chart — overall accuracy per model per method."""
    models  = sorted(overall_df["model"].unique(),
                     key=lambda m: MODEL_LABELS.get(m, m))
    methods = ["cot", "chunked"]
    x       = np.arange(len(models))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(models)), 5))

    for i, method in enumerate(methods):
        accs  = []
        lo_s  = []
        hi_s  = []
        for model in models:
            row = overall_df[(overall_df["model"] == model) &
                             (overall_df["method"] == method)]
            if row.empty:
                accs.append(0.0); lo_s.append(0.0); hi_s.append(0.0)
            else:
                accs.append(float(row["acc"].values[0]))
                lo_s.append(float(row["ci_lo"].values[0]))
                hi_s.append(float(row["ci_hi"].values[0]))

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, accs, width,
                      label=METHOD_LABELS[method],
                      color=METHOD_COLORS[method],
                      edgecolor="white", linewidth=0.6)
        # Error bars from Wilson CI
        yerr_lo = [a - l for a, l in zip(accs, lo_s)]
        yerr_hi = [h - a for a, h in zip(accs, hi_s)]
        ax.errorbar(x + offset,
                    accs,
                    yerr=[yerr_lo, yerr_hi],
                    fmt="none", color="black", capsize=4, linewidth=1.2)
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=10)
    ax.set_ylabel("Accuracy (GSM8K test, ≥3 steps)", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9)
    ax.set_title("GSM8K: CoT vs CISV Overall Accuracy", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, output_dir, "fig1_gsm8k_overall_accuracy")


def fig2_accuracy_by_steps(steps_df: pd.DataFrame, output_dir: str):
    """Figure 2: Accuracy by estimated step count, one subplot per model."""
    models     = sorted(steps_df["model"].unique(),
                        key=lambda m: MODEL_LABELS.get(m, m))
    step_bins  = sorted(steps_df["step_bin"].unique())
    ncols      = min(3, len(models))
    nrows      = (len(models) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             sharey=True, sharex=True)
    axes_flat = np.array(axes).flatten() if len(models) > 1 else [axes]

    for ax_idx, model in enumerate(models):
        ax   = axes_flat[ax_idx]
        m_df = steps_df[steps_df["model"] == model]

        for method in ["cot", "chunked"]:
            r_df = m_df[m_df["method"] == method].sort_values("step_bin")
            if r_df.empty:
                continue
            style = METHOD_STYLE[method]
            color = METHOD_COLORS[method]
            label_str = f"{METHOD_LABELS[method]} (n={r_df['n'].sum()})"
            ax.plot(r_df["step_bin"], r_df["acc"],
                    color=color, label=label_str, **style)
            ax.fill_between(r_df["step_bin"], r_df["ci_lo"], r_df["ci_hi"],
                            alpha=0.15, color=color)

        ax.set_title(MODEL_LABELS.get(model, model), fontsize=11, fontweight="bold")
        ax.set_xlabel("Estimated arithmetic steps", fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=9)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks(step_bins)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[len(models):]:
        ax.set_visible(False)

    fig.suptitle("GSM8K: Accuracy by Problem Complexity (Estimated Steps)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "fig2_gsm8k_accuracy_by_steps")


# ── Summary printing ──────────────────────────────────────────────────────────

def print_summary_table(overall_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("GSM8K SUMMARY: Overall accuracy per model × method")
    print("=" * 70)
    pivot = overall_df.pivot_table(index="model", columns="method", values="acc")
    pivot.index = [MODEL_LABELS.get(m, m) for m in pivot.index]
    # Reorder columns
    for col in ["cot", "chunked"]:
        if col not in pivot.columns:
            pivot[col] = float("nan")
    pivot = pivot[["cot", "chunked"]]
    pivot.columns = [METHOD_LABELS[c] for c in pivot.columns]
    print(pivot.round(4).to_string())

    print("\n" + "=" * 70)
    print("GSM8K SUMMARY: CISV Δ vs CoT per model")
    print("=" * 70)
    for model in sorted(overall_df["model"].unique()):
        cot_row = overall_df[(overall_df["model"] == model) &
                             (overall_df["method"] == "cot")]
        chk_row = overall_df[(overall_df["model"] == model) &
                             (overall_df["method"] == "chunked")]
        if cot_row.empty or chk_row.empty:
            continue
        delta = chk_row["acc"].values[0] - cot_row["acc"].values[0]
        n_cot = int(cot_row["n"].values[0])
        n_chk = int(chk_row["n"].values[0])
        print(f"  {MODEL_LABELS.get(model, model):<28} "
              f"Δ = {delta:+.3f}  "
              f"(CoT n={n_cot}, CISV n={n_chk})")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="GSM8K comparative analysis: CoT vs CISV"
    )
    p.add_argument("--cot_dir",    default=str(here / "results"),
                   help="Directory with *_cot.jsonl files")
    p.add_argument("--cisv_dir",   default=str(here / "results" / "cisv"),
                   help="Directory with exp4_chunked_*.json files")
    p.add_argument("--output_dir", default=str(here / "figures"),
                   help="Where to save figures")
    p.add_argument("--step_bin_size", type=int, default=2,
                   help="Bin width for step-count stratification (default: 2)")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df_cot  = load_cot_results(args.cot_dir)
    df_cisv = load_cisv_results(args.cisv_dir)
    df_all  = pd.concat([df_cot, df_cisv], ignore_index=True)

    print("\nComputing metrics...")
    overall_df = compute_overall_accuracy(df_all)
    steps_df   = compute_acc_by_steps(df_all, bin_size=args.step_bin_size)

    print_summary_table(overall_df)
    overall_df.to_csv(Path(args.output_dir) / "gsm8k_overall_accuracy.csv", index=False)
    steps_df.to_csv(Path(args.output_dir) / "gsm8k_accuracy_by_steps.csv", index=False)

    print("\nGenerating figures...")
    fig1_overall_accuracy(overall_df, args.output_dir)
    fig2_accuracy_by_steps(steps_df, args.output_dir)

    print(f"\nAll figures and CSVs saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
