"""
Comparative Analysis: Direct vs CoT vs Chunk-wise Incremental Solving with Verification

Generates publication-ready plots comparing all three methods per model:
  - Fig 1: Accuracy@depth curves (3 lines per model subplot)
  - Fig 2: Method comparison bar chart (mean accuracy per method × model)
  - Fig 3: Exp4 correction / fallback statistics heatmap
  - Fig 4: Accuracy delta (Chunked − CoT) per depth

Usage:
    python analyse_results.py \
        [--exp2_dir  ../Experiment2/results] \
        [--exp4_dir  ../results] \
        [--output_dir figures]
"""

import argparse
import json
import typing
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches


# ── Styling constants ────────────────────────────────────────────────────────

MODEL_LABELS = {
    "qwen25_math_1.5b": "Qwen2.5-Math-1.5B",
    "gemma4_e2b":        "Gemma 4 E2B",
    "phi4_mini":         "Phi-4-mini",
}
MODEL_COLORS = {
    "qwen25_math_1.5b": "#534AB7",   # purple
    "gemma4_e2b":        "#D85A30",   # coral
    "phi4_mini":         "#BA7517",   # amber
}
METHOD_STYLE = {
    "direct":  {"linestyle": ":",  "marker": "s", "linewidth": 1.8},
    "cot":     {"linestyle": "--", "marker": "o", "linewidth": 2.0},
    "chunked": {"linestyle": "-",  "marker": "D", "linewidth": 2.4},
}
METHOD_LABELS = {
    "direct":  "Direct",
    "cot":     "CoT",
    "chunked": "Chunked (Ours)",
}
METHOD_COLORS = {
    "direct":  "#5B9BD5",   # steel blue
    "cot":     "#70AD47",   # green
    "chunked": "#ED7D31",   # orange
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_exp2(results_dir: str) -> pd.DataFrame:
    """Load Experiment 2 JSONL baselines (direct + CoT) into a DataFrame."""
    records = []
    for fpath in sorted(Path(results_dir).glob("*.jsonl")):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    records.append({
                        "model":   r["model"],
                        "method":  r["regime"],      # "direct" or "cot"
                        "depth":   int(r["depth"]),
                        "correct": bool(r["is_correct"]),
                    })
                except (json.JSONDecodeError, KeyError):
                    continue
    if not records:
        raise FileNotFoundError(f"No JSONL files found in '{results_dir}'")
    df = pd.DataFrame(records)
    print(f"  Loaded {len(df):,} baseline records (direct + CoT)")
    return df


def load_exp4(results_dir: str) -> pd.DataFrame:
    """Load Experiment 4 JSON result files (one per model) into a DataFrame."""
    records = []
    for fpath in sorted(Path(results_dir).glob("exp4_chunked_*.json")):
        # Infer model name from filename: exp4_chunked_<model>.json
        stem = fpath.stem  # e.g. exp4_chunked_gemma4_e2b
        model = stem.replace("exp4_chunked_", "")
        # Strip checkpoint suffix if present (e.g. _ckpt_1200)
        if "_ckpt_" in model:
            model = model[:model.index("_ckpt_")]
        with open(fpath) as f:
            results = json.load(f)
        for r in results:
            records.append({
                "model":               model,
                "method":              "chunked",
                "depth":               int(r["depth"]),
                "correct":             bool(r["correct"]),
                "corrections_applied": int(r.get("corrections_applied", 0)),
                "unverified_fallbacks":int(r.get("unverified_fallbacks", 0)),
                "extraction_failures": int(r.get("extraction_failures", 0)),
                "num_chunks":          int(r.get("num_chunks", 0)),
            })
    if not records:
        raise FileNotFoundError(f"No exp4_chunked_*.json files found in '{results_dir}'")
    df = pd.DataFrame(records)
    print(f"  Loaded {len(df):,} chunked records ({df['model'].nunique()} models)")
    return df


# ── Metric computation ───────────────────────────────────────────────────────

def compute_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, method, depth) accuracy with Wilson CI."""
    rows = []
    for (model, method, depth), grp in df.groupby(["model", "method", "depth"]):
        n = len(grp)
        k = grp["correct"].sum()
        acc = k / n if n > 0 else 0.0
        lo, hi = _wilson_ci(k, n)
        rows.append({
            "model": model, "method": method, "depth": depth,
            "n": n, "k": k, "acc": acc, "ci_lo": lo, "ci_hi": hi,
        })
    return pd.DataFrame(rows).sort_values(["model", "method", "depth"])


def compute_exp4_stats(df4: pd.DataFrame) -> pd.DataFrame:
    """Per (model, depth) Exp4 diagnostic stats (corrections, fallbacks, etc.)."""
    rows = []
    for (model, depth), grp in df4.groupby(["model", "depth"]):
        n = len(grp)
        rows.append({
            "model": model, "depth": depth,
            "corrections_per_prob":  grp["corrections_applied"].mean(),
            "fallbacks_per_prob":    grp["unverified_fallbacks"].mean(),
            "ext_fail_per_prob":     grp["extraction_failures"].mean(),
            "chunks_per_prob":       grp["num_chunks"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["model", "depth"])


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ── Figures ──────────────────────────────────────────────────────────────────

def fig1_acc_curves(acc_df: pd.DataFrame, output_dir: str):
    """
    Figure 1: Accuracy@depth — one subplot per model, three lines per subplot
    (Direct, CoT, Chunked).
    """
    models = sorted(acc_df["model"].unique(), key=lambda m: MODEL_LABELS.get(str(m), str(m)))
    depths = sorted(acc_df["depth"].unique())
    ncols = min(3, len(models))
    nrows = (len(models) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows),
                             sharey=True, sharex=True)
    axes_flat = np.array(axes).flatten() if len(models) > 1 else [axes]

    for ax_idx, model in enumerate(models):
        ax = axes_flat[ax_idx]
        m_df = acc_df[acc_df["model"] == model]

        for method in ["direct", "cot", "chunked"]:
            r_df = m_df[m_df["method"] == method].sort_values("depth")
            if r_df.empty:
                continue
            style = METHOD_STYLE[method]
            color = METHOD_COLORS[method]
            ax.plot(
                r_df["depth"], r_df["acc"],
                color=color, label=METHOD_LABELS[method],
                **style,
            )
            ax.fill_between(
                r_df["depth"], r_df["ci_lo"], r_df["ci_hi"],
                alpha=0.12, color=color,
            )

        ax.set_title(MODEL_LABELS.get(model, model), fontsize=11, fontweight="bold")
        ax.set_xlabel("Reasoning depth (steps)", fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=9)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks(depths)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[len(models):]:
        ax.set_visible(False)

    plt.tight_layout()
    _save(fig, output_dir, "fig1_acc_curves")


def fig2_method_bars(acc_df: pd.DataFrame, output_dir: str):
    """
    Figure 2: Grouped bar chart — mean accuracy across all depths,
    grouped by model, three bars per group (one per method).
    """
    models = sorted(acc_df["model"].unique(), key=lambda m: MODEL_LABELS.get(str(m), str(m)))
    methods = ["direct", "cot", "chunked"]
    n_models = len(models)
    n_methods = len(methods)
    x = np.arange(n_models)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(7, 2.8 * n_models), 5))

    for i, method in enumerate(methods):
        means = []
        for model in models:
            sub = acc_df[(acc_df["model"] == model) & (acc_df["method"] == method)]
            means.append(sub["acc"].mean() if not sub.empty else 0.0)
        bars = ax.bar(
            x + (i - 1) * width, means, width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="white", linewidth=0.6,
        )
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.1%}", ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=10)
    ax.set_ylabel("Mean accuracy (across all depths)", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9)
    ax.set_title("Mean Accuracy by Method and Model", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, output_dir, "fig2_method_bars")


def fig3_exp4_stats_heatmap(stats_df: pd.DataFrame, output_dir: str):
    """
    Figure 3: Heatmap of Exp4 diagnostic statistics per model × depth.
    Rows = metric, columns = depth, one panel per model.
    """
    models = sorted(stats_df["model"].unique(), key=lambda m: MODEL_LABELS.get(str(m), str(m)))
    depths = sorted(stats_df["depth"].unique())
    metrics = {
        "corrections_per_prob":  "Corrections / problem",
        "fallbacks_per_prob":    "Fallbacks / problem",
        "ext_fail_per_prob":     "Extract failures / problem",
    }

    fig, axes = plt.subplots(1, len(models),
                             figsize=(5 * len(models), 3.5),
                             sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        matrix = np.zeros((len(metrics), len(depths)))
        for j, depth in enumerate(depths):
            row = stats_df[(stats_df["model"] == model) & (stats_df["depth"] == depth)]
            for i, metric in enumerate(metrics):
                matrix[i, j] = row[metric].values[0] if not row.empty else 0.0

        im = ax.imshow(matrix, vmin=0, aspect="auto", cmap="OrRd")
        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels([f"d{d}" for d in depths], fontsize=8)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(list(metrics.values()), fontsize=8)
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=10, fontweight="bold")
        ax.set_xlabel("Depth", fontsize=8)

        for i in range(len(metrics)):
            for j in range(len(depths)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if matrix[i, j] < 0.5 else "white")

        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Exp4 Diagnostics per Depth (Chunked Method)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "fig3_exp4_stats_heatmap")


def fig4_delta_curves(acc_df: pd.DataFrame, output_dir: str):
    """
    Figure 4: Accuracy delta (Chunked − CoT) per depth, one line per model.
    Positive = chunked helps, negative = chunked hurts.
    """
    models = sorted(acc_df["model"].unique(), key=lambda m: MODEL_LABELS.get(str(m), str(m)))
    depths = sorted(acc_df["depth"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    for model in models:
        color = MODEL_COLORS.get(model, "black")
        label = MODEL_LABELS.get(model, model)
        deltas, plot_depths = [], []

        for depth in depths:
            cot_row = acc_df[(acc_df["model"] == model) &
                             (acc_df["method"] == "cot") &
                             (acc_df["depth"] == depth)]
            chk_row = acc_df[(acc_df["model"] == model) &
                             (acc_df["method"] == "chunked") &
                             (acc_df["depth"] == depth)]
            if cot_row.empty or chk_row.empty:
                continue
            deltas.append(chk_row["acc"].values[0] - cot_row["acc"].values[0])
            plot_depths.append(depth)

        if deltas:
            ax.plot(plot_depths, deltas, marker="o", color=color,
                    linewidth=2.2, label=label)

    ax.axhline(0, color="black", linewidth=1.0, linestyle="-")
    ax.fill_between(depths,  0.0,  0.3, alpha=0.04, color="green")
    ax.fill_between(depths, -0.3,  0.0, alpha=0.04, color="red")
    ax.set_xlabel("Reasoning depth (steps)", fontsize=10)
    ax.set_ylabel("Accuracy delta (Chunked − CoT)", fontsize=10)
    ax.set_title("Accuracy Gain / Loss of Chunked Method vs CoT Baseline",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(depths)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, output_dir, "fig4_delta_curves")


# ── Utility ──────────────────────────────────────────────────────────────────

def _save(fig, output_dir: str, name: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = Path(output_dir) / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {Path(output_dir) / name}.{{pdf,png}}")


def print_summary(acc_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("SUMMARY: Mean accuracy per model × method")
    print("=" * 70)
    pivot = acc_df.groupby(["model", "method"])["acc"].mean().unstack("method")
    pivot.index = [MODEL_LABELS.get(m, m) for m in pivot.index]
    # Reorder columns
    for col in ["direct", "cot", "chunked"]:
        if col not in pivot.columns:
            pivot[col] = float("nan")
    pivot = pivot[["direct", "cot", "chunked"]]
    pivot.columns = [METHOD_LABELS[c] for c in pivot.columns]
    print(pivot.round(4).to_string())

    print("\n" + "=" * 70)
    print("SUMMARY: Accuracy@depth (Chunked method)")
    print("=" * 70)
    chk = acc_df[acc_df["method"] == "chunked"].pivot_table(
        index="model", columns="depth", values="acc"
    )
    chk.index = [MODEL_LABELS.get(m, m) for m in chk.index]
    print(chk.round(3).to_string())


def save_corrections_by_model_csv(df4: pd.DataFrame, output_dir: str):
    """Save detailed corrections table per model × depth to CSV, with aggregate row."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build table: rows = depth, columns = model metrics
    rows = []
    for depth in sorted(df4["depth"].unique()):
        depth_data = df4[df4["depth"] == depth]
        row: dict[str, typing.Any] = {"Problem Type (Depth)": f"d{depth}"}

        for model in sorted(df4["model"].unique()):
            model_depth_data = depth_data[depth_data["model"] == model]
            if not model_depth_data.empty:
                total_corr = model_depth_data["corrections_applied"].sum()
                num_probs = len(model_depth_data)
                avg_corr = model_depth_data["corrections_applied"].mean()
                row[f"{MODEL_LABELS.get(model, model)} - Total"] = int(total_corr)
                row[f"{MODEL_LABELS.get(model, model)} - Count"] = num_probs
                row[f"{MODEL_LABELS.get(model, model)} - Avg"] = f"{avg_corr:.3f}"

        rows.append(row)

    # Add aggregate row (all depths combined)
    agg_row: dict[str, typing.Any] = {"Problem Type (Depth)": "TOTAL"}
    for model in sorted(df4["model"].unique()):
        model_data = df4[df4["model"] == model]
        total_corr = model_data["corrections_applied"].sum()
        num_probs = len(model_data)
        avg_corr = model_data["corrections_applied"].mean()
        agg_row[f"{MODEL_LABELS.get(model, model)} - Total"] = int(total_corr)
        agg_row[f"{MODEL_LABELS.get(model, model)} - Count"] = num_probs
        agg_row[f"{MODEL_LABELS.get(model, model)} - Avg"] = f"{avg_corr:.3f}"
    rows.append(agg_row)

    table_df = pd.DataFrame(rows)

    # Save to CSV
    csv_path = Path(output_dir) / "corrections_by_model.csv"
    table_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Also print to console
    print("\n" + "=" * 120)
    print("DETAILS: Corrections by Model and Problem Type")
    print("=" * 120)
    print(table_df.to_string(index=False))


# ── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description="Comparative analysis: Direct vs CoT vs Chunked"
    )
    p.add_argument("--exp2_dir",   default=str(root / "Experiment2" / "results"),
                   help="Directory with Experiment 2 JSONL baselines")
    p.add_argument("--exp4_dir",   default=str(root / "results"),
                   help="Directory with Experiment 4 JSON result files")
    p.add_argument("--output_dir", default=str(root / "Experiment4" / "figures"),
                   help="Where to save figures")
    p.add_argument("--exp4_file",  default=None,
                   help="Single Exp4 JSON file (overrides --exp4_dir)")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading data...")
    df2 = load_exp2(args.exp2_dir)

    # Support pointing at a single checkpoint file directly
    if args.exp4_file:
        exp4_dir = Path(args.exp4_file).parent
        # Rename temporarily so glob picks it up
        fpath = Path(args.exp4_file)
        if not fpath.name.startswith("exp4_chunked_"):
            print(f"  Warning: file name doesn't follow exp4_chunked_<model>.json "
                  f"convention — model name will be inferred as 'unknown'")
        df4 = load_exp4(str(exp4_dir))
    else:
        df4 = load_exp4(args.exp4_dir)

    # Combine into a single accuracy DataFrame
    print("Computing accuracy metrics...")
    df_all = pd.concat([
        df2[["model", "method", "depth", "correct"]],
        df4[["model", "method", "depth", "correct"]],
    ], ignore_index=True)
    acc_df = compute_accuracy(df_all)
    stats_df = compute_exp4_stats(df4)

    print_summary(acc_df)
   
    save_corrections_by_model_csv(df4, args.output_dir)

    print("\nGenerating figures...")
    fig1_acc_curves(acc_df, args.output_dir)
    fig2_method_bars(acc_df, args.output_dir)
    fig3_exp4_stats_heatmap(stats_df, args.output_dir)
    fig4_delta_curves(acc_df, args.output_dir)

    print(f"\nAll figures saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
