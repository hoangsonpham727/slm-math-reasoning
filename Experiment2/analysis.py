"""
Experiment 2: Reasoning Depth — Analysis & Plotting

Computes per-model metrics across depth levels and generates publication-ready plots:
  - Fig 1: Acc@k curves per model (direct vs CoT)
  - Fig 2: LLM–SLM accuracy gap (dΔ/dk)
  - Fig 3: Step-level accuracy heatmap
  - Fig 4: Output collapse rate vs depth

Usage:
    python analysis.py [--results_dir results] [--output_dir figures]
                       [--regime direct,cot]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Metric constants
# ---------------------------------------------------------------------------

DEPTH_CEILING_DROP = 0.30   # k* defined as first k with >30% relative accuracy drop
TOLERANCE = 0.01            # 1% tolerance for exact-match


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: str) -> pd.DataFrame:
    """Load all JSONL result files into a single DataFrame."""
    records = []
    for fpath in sorted(Path(results_dir).glob("*.jsonl")):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    if not records:
        raise FileNotFoundError(
            f"No result JSONL files found in '{results_dir}'. "
            "Run run_inference.py first."
        )
    df = pd.DataFrame(records)
    print(f"Loaded {len(df):,} result records from {results_dir}/")
    return df


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_acc_at_k(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      model, regime, depth, n_total, n_correct, n_collapse, acc, collapse_rate,
      step_acc
    """
    rows = []
    for (model, regime, depth), grp in df.groupby(["model", "regime", "depth"]):
        n_total   = len(grp)
        n_correct = grp["is_correct"].sum()
        n_collapse= grp["is_collapse"].sum()
        acc       = n_correct / n_total if n_total > 0 else 0.0
        col_rate  = n_collapse / n_total if n_total > 0 else 0.0
        # Step-level accuracy: fraction of steps solved before first failure.
        # If no failure is detected, count all depth steps as correct.
        if n_total > 0:
            first_error = grp.get("first_error_step", pd.Series([np.nan] * n_total))
            solved_steps = np.where(first_error.isna(), depth, first_error - 1)
            solved_steps = np.clip(solved_steps, 0, depth)
            step_acc = float(np.mean(solved_steps / depth))
        else:
            step_acc = 0.0
        rows.append({
            "model": model, "regime": regime, "depth": depth,
            "n_total": n_total, "n_correct": n_correct,
            "n_collapse": n_collapse, "acc": acc, "collapse_rate": col_rate,
            "step_acc": step_acc,
        })
    return pd.DataFrame(rows).sort_values(["model", "regime", "depth"])


def compute_depth_ceiling(acc_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (model, regime):
      - baseline_acc = acc at depth=1
      - k* = first depth where acc < baseline_acc * (1 - DEPTH_CEILING_DROP)
    """
    rows = []
    for (model, regime), grp in acc_df.groupby(["model", "regime"]):
        grp_sorted = grp.sort_values("depth")
        depth1 = grp_sorted[grp_sorted["depth"] == 1]
        if depth1.empty:
            continue
        baseline = depth1["acc"].values[0]
        threshold = baseline * (1 - DEPTH_CEILING_DROP)
        k_star = None
        for _, row in grp_sorted.iterrows():
            if row["depth"] > 1 and row["acc"] < threshold:
                k_star = int(row["depth"])
                break
        rows.append({
            "model": model, "regime": regime,
            "baseline_acc": round(baseline, 4),
            "threshold": round(threshold, 4),
            "k_star": k_star,
        })
    return pd.DataFrame(rows)


def compute_gap(acc_df: pd.DataFrame, reference_model: str = None) -> pd.DataFrame:
    """
    Compute the accuracy gap vs. the highest-performing model at each depth.
    If reference_model is given, use that specific model as the reference.
    """
    rows = []
    for regime, reg_grp in acc_df.groupby("regime"):
        # Reference: model with highest mean accuracy
        if reference_model:
            ref_model = reference_model
        else:
            mean_acc = reg_grp.groupby("model")["acc"].mean()
            ref_model = mean_acc.idxmax()

        ref_data = reg_grp[reg_grp["model"] == ref_model][["depth", "acc"]]
        ref_data = ref_data.set_index("depth")["acc"]

        for model, m_grp in reg_grp.groupby("model"):
            if model == ref_model:
                continue
            for _, row in m_grp.iterrows():
                ref_acc = ref_data.get(row["depth"], float("nan"))
                rows.append({
                    "model": model, "regime": regime, "depth": row["depth"],
                    "acc": row["acc"], "ref_model": ref_model,
                    "ref_acc": ref_acc, "gap": ref_acc - row["acc"],
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Colour palette: one colour per model, consistent across all plots
MODEL_COLORS = {
    "qwen25_math_1.5b": "#534AB7",   # purple
    "gemma4_e2b":       "#D85A30",   # coral
    "phi4_mini":        "#BA7517",   # amber
}
MODEL_LABELS = {
    "qwen25_math_1.5b": "Qwen2.5-Math-1.5B",
    "gemma4_e2b":       "Gemma 4 E2B",
    "phi4_mini":        "Phi-4-mini",
}
REGIME_STYLE = {"direct": "--", "cot": "-"}
REGIME_LABEL = {"direct": "Direct", "cot": "CoT"}
REGIME_COLORS = {"direct": "#1f77b4", "cot": "#d62728"}  # blue vs red


def fig_acc_curves(acc_df: pd.DataFrame, output_dir: str):
    """Figure 1: Acc@k curves for each model, direct vs CoT."""
    depths = sorted(acc_df["depth"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True, sharex=True)
    axes_flat = axes.flatten()

    models = sorted(acc_df["model"].unique())
    for ax_idx, model in enumerate(models):
        ax = axes_flat[ax_idx]
        m_df = acc_df[acc_df["model"] == model]
        label = MODEL_LABELS.get(model, model)

        for regime in ["direct", "cot"]:
            r_df = m_df[m_df["regime"] == regime].sort_values("depth")
            if r_df.empty:
                continue
            regime_color = REGIME_COLORS[regime]
            ax.plot(
                r_df["depth"], r_df["acc"],
                linestyle=REGIME_STYLE[regime],
                marker="o" if regime == "cot" else "s",
                markersize=5,
                color=regime_color, linewidth=2.2,
                label=f"{REGIME_LABEL[regime]}",
            )
            # Shade 95% Wilson CI
            for _, row in r_df.iterrows():
                n, k = int(row["n_total"]), int(row["n_correct"])
                if n > 0:
                    ci = _wilson_ci(k, n)
                    ax.fill_between(
                        [row["depth"] - 0.05, row["depth"] + 0.05],
                        [ci[0]] * 2, [ci[1]] * 2,
                        alpha=0.18, color=regime_color,
                    )

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Reasoning depth (steps)", fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=9)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks(depths)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes_flat[len(models):]:
        ax.set_visible(False)

    plt.tight_layout()
    out = Path(output_dir) / "fig1_acc_curves.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def fig_gap_slope(acc_df: pd.DataFrame, output_dir: str):
    """Figure 2: Accuracy gap (best model − each model) growing with depth."""
    gap_df = compute_gap(acc_df)
    if gap_df.empty:
        print("  [skip] Only one model; gap plot skipped.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    depths = sorted(gap_df["depth"].unique())

    for ax, regime in zip(axes, ["direct", "cot"]):
        r_df = gap_df[gap_df["regime"] == regime]
        for model, m_df in r_df.groupby("model"):
            m_df = m_df.sort_values("depth")
            color = MODEL_COLORS.get(model, "black")
            label = MODEL_LABELS.get(model, model)
            ax.plot(m_df["depth"], m_df["gap"],
                    marker="o", color=color, linewidth=2, label=label)

        ax.set_title(f"Regime: {REGIME_LABEL[regime]}", fontsize=11)
        ax.set_xlabel("Reasoning depth (steps)", fontsize=9)
        ax.set_ylabel("Accuracy gap vs. best model", fontsize=9)
        ax.set_xticks(depths)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(output_dir) / "fig2_gap_slope.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def fig_step_accuracy_heatmap(acc_df: pd.DataFrame, output_dir: str):
    """Figure 3: Step-level accuracy heatmap — rows = models, cols = depths."""
    depths = sorted(acc_df["depth"].unique())
    models = sorted(acc_df["model"].unique())

    regimes = acc_df["regime"].unique()
    fig, axes = plt.subplots(1, len(regimes), figsize=(14, 4))
    if len(regimes) == 1:
        axes = [axes]

    for ax, regime in zip(axes, sorted(regimes)):
        r_df = acc_df[acc_df["regime"] == regime]
        matrix = np.zeros((len(models), len(depths)))
        for i, model in enumerate(models):
            for j, depth in enumerate(depths):
                cell = r_df[(r_df["model"] == model) & (r_df["depth"] == depth)]
                if not cell.empty:
                    matrix[i, j] = cell["step_acc"].values[0]

        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels([f"d{d}" for d in depths], fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=9)
        ax.set_title(f"Step-Level Accuracy — {REGIME_LABEL[regime]}", fontsize=10)

        # Annotate cells
        for i in range(len(models)):
            for j in range(len(depths)):
                val = matrix[i, j]
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=7, color="black" if val < 0.6 else "white")

        plt.colorbar(im, ax=ax, fraction=0.046, label="Step-level accuracy")


    plt.tight_layout()
    out = Path(output_dir) / "fig3_step_accuracy_heatmap.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def fig_collapse_rate(acc_df: pd.DataFrame, output_dir: str):
    """Figure 4: Output collapse rate (non-parseable) vs depth."""
    depths = sorted(acc_df["depth"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, regime in zip(axes, ["direct", "cot"]):
        r_df = acc_df[acc_df["regime"] == regime]
        for model, m_df in r_df.groupby("model"):
            m_df = m_df.sort_values("depth")
            color = MODEL_COLORS.get(model, "black")
            label = MODEL_LABELS.get(model, model)
            ax.plot(m_df["depth"], m_df["collapse_rate"],
                    marker="s", color=color, linewidth=2,
                    linestyle="--", label=label)

        ax.set_title(f"Output Collapse — {REGIME_LABEL[regime]}", fontsize=10)
        ax.set_xlabel("Depth", fontsize=9)
        ax.set_ylabel("Collapse rate", fontsize=9)
        ax.set_xticks(depths)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(output_dir) / "fig4_collapse_rate.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_tables(acc_df: pd.DataFrame, ceiling_df: pd.DataFrame):
    print("\n" + "="*70)
    print("SUMMARY: Accuracy@k (CoT regime)")
    print("="*70)
    cot_acc = acc_df[acc_df["regime"] == "cot"].pivot_table(
        index="model", columns="depth", values="acc"
    )
    cot_acc.index = [MODEL_LABELS.get(m, m) for m in cot_acc.index]
    print(cot_acc.round(3).to_string())

    print("\n" + "="*70)
    print("SUMMARY: Depth Ceiling k*")
    print("="*70)
    ceiling_display = ceiling_df.copy()
    ceiling_display["model"] = ceiling_display["model"].map(
        lambda m: MODEL_LABELS.get(m, m)
    )
    print(ceiling_display.to_string(index=False))

    print("\n" + "="*70)
    print("SUMMARY: Mean Step-Level Accuracy across depths (CoT regime)")
    print("="*70)
    cot_step = acc_df[acc_df["regime"] == "cot"].groupby("model")["step_acc"].mean()
    cot_step.index = [MODEL_LABELS.get(m, m) for m in cot_step.index]
    print(cot_step.round(3).to_string())


def save_summary_csv(acc_df, ceiling_df, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    acc_df.to_csv(Path(output_dir) / "metrics_accuracy.csv", index=False)
    ceiling_df.to_csv(Path(output_dir) / "metrics_ceiling.csv", index=False)
    print(f"\n  CSVs saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir",  type=str, default="figures")
    parser.add_argument("--regime",      type=str, default="direct,cot")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = load_results(args.results_dir)

    print("Computing metrics...")
    acc_df     = compute_acc_at_k(df)
    ceiling_df = compute_depth_ceiling(acc_df)

    print_summary_tables(acc_df, ceiling_df)
    save_summary_csv(acc_df, ceiling_df, args.output_dir)

    print("\nGenerating figures...")
    fig_acc_curves(acc_df, args.output_dir)
    fig_gap_slope(acc_df, args.output_dir)
    fig_step_accuracy_heatmap(acc_df, args.output_dir)
    fig_collapse_rate(acc_df, args.output_dir)

    print(f"\nAll figures and CSVs saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()