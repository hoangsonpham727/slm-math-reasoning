"""
Comparative Analysis: CoT vs Chunk-wise Incremental Solving with Verification

Supports two modes selected automatically at runtime:
  - Multi-seed (default when seed_* subdirectories are found): aggregates
    results from all seed variants, reporting mean ± std with 95% bootstrap
    confidence intervals across seeds.
  - Single-seed (fallback): uses a single results directory.

Figures generated:
  Fig 1  Accuracy@depth curves — mean line + per-seed thin lines + CI band
  Fig 2  Mean accuracy bar chart with ±std error bars across seeds
  Fig 3  CISV diagnostic statistics heatmap (corrections / fallbacks / failures)
  Fig 4  Accuracy delta (Chunked − CoT) with per-seed lines + CI band

Usage — multi-seed (auto-detected):
    python analyse_results.py \
        --exp2_dir ../Experiment2/results \
        --exp4_dir ../results \
        --output_dir figures

Usage — single-seed (explicit):
    python analyse_results.py --no_multi_seed \
        --exp2_dir ../Experiment2/results \
        --exp4_dir ../results \
        --output_dir figures
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


# ── Styling constants ─────────────────────────────────────────────────────────

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
    "chunked": "CISV (Ours)",
}
METHOD_COLORS = {
    "direct":  "#5B9BD5",   # steel blue
    "cot":     "#70AD47",   # green
    "chunked": "#ED7D31",   # orange
}

# Methods shown in multi-seed plots (paper focus: CoT vs CISV)
MULTISEED_METHODS = ["cot", "chunked"]


# ── Utility ───────────────────────────────────────────────────────────────────

def _has_seed_dirs(base_dir: str) -> bool:
    """Return True if base_dir contains at least one seed_* subdirectory."""
    base = Path(base_dir)
    if not base.exists():
        return False
    return any(d.is_dir() and d.name.startswith("seed_") for d in base.iterdir())


def _save(fig, output_dir: str, name: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = Path(output_dir) / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {Path(output_dir) / name}.{{pdf,png}}")


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score 95% CI for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _bootstrap_ci(values: list, n_bootstrap: int = 10_000) -> tuple:
    """Bootstrap 95% CI for the mean of a small sample (e.g. 3 seed accuracies)."""
    import random as _rng
    vals = list(values)
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (vals[0], vals[0])
    boot_means = [sum(_rng.choices(vals, k=n)) / n for _ in range(n_bootstrap)]
    boot_means.sort()
    return (boot_means[int(0.025 * n_bootstrap)], boot_means[int(0.975 * n_bootstrap)])


# ── Data loading ──────────────────────────────────────────────────────────────

def load_exp2(results_dir: str) -> pd.DataFrame:
    """Load Experiment 2 JSONL baselines (CoT / direct) into a DataFrame."""
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
                        "method":  r["regime"],
                        "depth":   int(r["depth"]),
                        "correct": bool(r["is_correct"]),
                    })
                except (json.JSONDecodeError, KeyError):
                    continue
    if not records:
        raise FileNotFoundError(f"No JSONL files found in '{results_dir}'")
    df = pd.DataFrame(records)
    print(f"  Loaded {len(df):,} baseline records from {results_dir}")
    return df


def load_exp4(results_dir: str) -> pd.DataFrame:
    """Load CISV JSON result files (one per model) into a DataFrame."""
    records = []
    for fpath in sorted(Path(results_dir).glob("exp4_chunked_*.json")):
        stem  = fpath.stem
        model = stem.replace("exp4_chunked_", "")
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
    print(f"  Loaded {len(df):,} CISV records from {results_dir} ({df['model'].nunique()} models)")
    return df


def load_multi_seed_exp2(exp2_base: str) -> dict:
    """Load Exp2 results from every seed_* subdirectory → {seed_label: DataFrame}."""
    base = Path(exp2_base)
    seed_dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith("seed_"))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* dirs found in '{exp2_base}'")
    print(f"  Found {len(seed_dirs)} seed dirs under {exp2_base}")
    return {sd.name: load_exp2(str(sd)) for sd in seed_dirs}


def load_multi_seed_exp4(exp4_base: str) -> dict:
    """Load CISV results from every seed_* subdirectory → {seed_label: DataFrame}."""
    base = Path(exp4_base)
    seed_dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith("seed_"))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* dirs found in '{exp4_base}'")
    print(f"  Found {len(seed_dirs)} seed dirs under {exp4_base}")
    return {sd.name: load_exp4(str(sd)) for sd in seed_dirs}


# ── Metric computation ────────────────────────────────────────────────────────

def compute_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, method, depth) accuracy with Wilson 95% CI."""
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
    """Per (model, depth) CISV diagnostic stats."""
    rows = []
    for (model, depth), grp in df4.groupby(["model", "depth"]):
        rows.append({
            "model": model, "depth": depth,
            "corrections_per_prob": grp["corrections_applied"].mean(),
            "fallbacks_per_prob":   grp["unverified_fallbacks"].mean(),
            "ext_fail_per_prob":    grp["extraction_failures"].mean(),
            "chunks_per_prob":      grp["num_chunks"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["model", "depth"])


def compute_multi_seed_accuracy(
    seed_df2s: dict, seed_df4s: dict
) -> tuple[pd.DataFrame, dict]:
    """
    Merge per-seed Exp2 + Exp4 data and aggregate across seeds.

    Returns
    -------
    ms_acc_df : DataFrame
        One row per (model, method, depth) with columns:
          acc       – mean accuracy across seeds
          std_acc   – std (ddof=1)
          ci_lo/hi  – 95% bootstrap CI on the mean
          n_seeds   – number of seeds contributing
    per_seed_acc : dict[seed_label → per-seed accuracy DataFrame]
        Retained for per-seed line plotting.
    """
    per_seed_acc: dict[str, pd.DataFrame] = {}
    for label in sorted(set(seed_df2s) | set(seed_df4s)):
        parts = []
        if label in seed_df2s:
            parts.append(seed_df2s[label][["model", "method", "depth", "correct"]])
        if label in seed_df4s:
            parts.append(seed_df4s[label][["model", "method", "depth", "correct"]])
        if parts:
            combined = pd.concat(parts, ignore_index=True)
            per_seed_acc[label] = compute_accuracy(combined)

    all_keys: set = set()
    for acc_df in per_seed_acc.values():
        for _, row in acc_df.iterrows():
            all_keys.add((row["model"], row["method"], row["depth"]))

    rows = []
    for (model, method, depth) in sorted(all_keys):
        seed_accs: dict[str, float] = {}
        for label, acc_df in per_seed_acc.items():
            match = acc_df[
                (acc_df["model"] == model) &
                (acc_df["method"] == method) &
                (acc_df["depth"] == depth)
            ]
            if not match.empty:
                seed_accs[label] = float(match["acc"].values[0])

        vals     = list(seed_accs.values())
        mean_acc = float(np.mean(vals)) if vals else float("nan")
        std_acc  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci_lo, ci_hi = _bootstrap_ci(vals)

        row: dict = {
            "model": model, "method": method, "depth": depth,
            "n_seeds": len(vals),
            "acc": mean_acc, "std_acc": std_acc,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
        }
        for label, acc in seed_accs.items():
            row[f"acc_{label}"] = acc
        rows.append(row)

    ms_acc_df = pd.DataFrame(rows).sort_values(["model", "method", "depth"])
    return ms_acc_df, per_seed_acc


# ── Single-seed figures ───────────────────────────────────────────────────────

def fig1_acc_curves(acc_df: pd.DataFrame, output_dir: str):
    """Fig 1 (single-seed): accuracy@depth curves with Wilson CI bands."""
    models = sorted(acc_df["model"].unique(), key=lambda m: MODEL_LABELS.get(str(m), str(m)))
    depths = sorted(acc_df["depth"].unique())
    ncols  = min(3, len(models))
    nrows  = (len(models) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows),
                             sharey=True, sharex=True)
    axes_flat = np.array(axes).flatten() if len(models) > 1 else [axes]

    for ax_idx, model in enumerate(models):
        ax   = axes_flat[ax_idx]
        m_df = acc_df[acc_df["model"] == model]

        for method in ["direct", "cot", "chunked"]:
            r_df = m_df[m_df["method"] == method].sort_values("depth")
            if r_df.empty:
                continue
            color = METHOD_COLORS[method]
            ax.plot(r_df["depth"], r_df["acc"], color=color,
                    label=METHOD_LABELS[method], **METHOD_STYLE[method])
            ax.fill_between(r_df["depth"], r_df["ci_lo"], r_df["ci_hi"],
                            alpha=0.12, color=color)

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
    """Fig 2 (single-seed): grouped bar chart — mean accuracy per method × model."""
    models  = sorted(acc_df["model"].unique(), key=lambda m: MODEL_LABELS.get(str(m), str(m)))
    methods = [m for m in ["direct", "cot", "chunked"] if m in acc_df["method"].unique()]
    x       = np.arange(len(models))
    width   = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(max(7, 2.8 * len(models)), 5))

    for i, method in enumerate(methods):
        means = [acc_df[(acc_df["model"] == m) & (acc_df["method"] == method)]["acc"].mean()
                 if not acc_df[(acc_df["model"] == m) & (acc_df["method"] == method)].empty
                 else 0.0 for m in models]
        offset = (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, label=METHOD_LABELS[method],
                      color=METHOD_COLORS[method], edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=10)
    ax.set_ylabel("Mean accuracy (across all depths)", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9)
    ax.set_title("Mean Accuracy by Method and Model", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save(fig, output_dir, "fig2_method_bars")


def fig3_exp4_stats_heatmap(stats_df: pd.DataFrame, output_dir: str):
    """Fig 3: heatmap of CISV diagnostic statistics per model × depth."""
    models  = sorted(stats_df["model"].unique(), key=lambda m: MODEL_LABELS.get(str(m), str(m)))
    depths  = sorted(stats_df["depth"].unique())
    metrics = {
        "corrections_per_prob": "Corrections / problem",
        "fallbacks_per_prob":   "Fallbacks / problem",
        "ext_fail_per_prob":    "Extract failures / problem",
    }

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.5), sharey=True)
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

    fig.suptitle("CISV Diagnostics per Depth", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "fig3_exp4_stats_heatmap")


def fig4_delta_curves(acc_df: pd.DataFrame, output_dir: str):
    """Fig 4 (single-seed): accuracy delta (Chunked − CoT) per depth."""
    models = sorted(acc_df["model"].unique(), key=lambda m: MODEL_LABELS.get(str(m), str(m)))
    depths = sorted(acc_df["depth"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    for model in models:
        color  = MODEL_COLORS.get(model, "black")
        deltas, plot_depths = [], []
        for depth in depths:
            cot_row = acc_df[(acc_df["model"] == model) & (acc_df["method"] == "cot")    & (acc_df["depth"] == depth)]
            chk_row = acc_df[(acc_df["model"] == model) & (acc_df["method"] == "chunked") & (acc_df["depth"] == depth)]
            if cot_row.empty or chk_row.empty:
                continue
            deltas.append(chk_row["acc"].values[0] - cot_row["acc"].values[0])
            plot_depths.append(depth)
        if deltas:
            ax.plot(plot_depths, deltas, marker="o", color=color,
                    linewidth=2.2, label=MODEL_LABELS.get(model, model))

    ax.axhline(0, color="black", linewidth=1.0)
    ax.fill_between(depths,  0.0,  0.3, alpha=0.04, color="green")
    ax.fill_between(depths, -0.3,  0.0, alpha=0.04, color="red")
    ax.set_xlabel("Reasoning depth (steps)", fontsize=10)
    ax.set_ylabel("Accuracy delta (CISV − CoT)", fontsize=10)
    ax.set_title("CISV Accuracy Gain / Loss vs CoT Baseline", fontsize=11, fontweight="bold")
    ax.set_xticks(depths)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save(fig, output_dir, "fig4_delta_curves")


# ── Multi-seed figures ────────────────────────────────────────────────────────

def fig1_acc_curves_multiseed(
    ms_acc_df: pd.DataFrame,
    per_seed_acc: dict,
    output_dir: str,
):
    """
    Fig 1 (multi-seed): accuracy@depth per model.

    Visual layers (back → front):
      1. Faint per-seed lines — show raw between-seed variability
      2. ±std shaded band     — summarises spread
      3. 95% bootstrap CI     — tighter inner band (statistical confidence)
      4. Bold mean line       — primary visual element
    """
    models   = sorted(ms_acc_df["model"].unique(), key=lambda m: MODEL_LABELS.get(str(m), str(m)))
    depths   = sorted(ms_acc_df["depth"].unique())
    n_seeds  = int(ms_acc_df["n_seeds"].max())
    ncols    = min(3, len(models))
    nrows    = (len(models) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows),
                             sharey=True, sharex=True)
    axes_flat = np.array(axes).flatten() if len(models) > 1 else [axes]

    for ax_idx, model in enumerate(models):
        ax   = axes_flat[ax_idx]
        m_df = ms_acc_df[ms_acc_df["model"] == model]

        for method in MULTISEED_METHODS:
            r_df = m_df[m_df["method"] == method].sort_values("depth")
            if r_df.empty:
                continue
            color = METHOD_COLORS[method]
            style = METHOD_STYLE[method]

            # ── Layer 1: individual seed lines ────────────────────────────────
            for seed_label, seed_acc_df in sorted(per_seed_acc.items()):
                s_df = seed_acc_df[
                    (seed_acc_df["model"]  == model) &
                    (seed_acc_df["method"] == method)
                ].sort_values("depth")
                if s_df.empty:
                    continue
                ax.plot(
                    s_df["depth"], s_df["acc"],
                    color=color, linewidth=0.9, alpha=0.25,
                    linestyle=style["linestyle"], marker=None,
                )

            # ── Layer 2: ±1 std band ──────────────────────────────────────────
            ax.fill_between(
                r_df["depth"],
                (r_df["acc"] - r_df["std_acc"]).clip(0),
                (r_df["acc"] + r_df["std_acc"]).clip(upper=1),
                alpha=0.12, color=color, label="_nolegend_",
            )

            # ── Layer 3: 95% bootstrap CI band ────────────────────────────────
            ax.fill_between(
                r_df["depth"], r_df["ci_lo"], r_df["ci_hi"],
                alpha=0.20, color=color, label="_nolegend_",
            )

            # ── Layer 4: bold mean line ───────────────────────────────────────
            ax.plot(
                r_df["depth"], r_df["acc"],
                color=color, label=f"{METHOD_LABELS[method]}",
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=5, zorder=3,
            )

        ax.set_title(
            f"{MODEL_LABELS.get(model, model)}\n"
            f"({n_seeds} seeds, shading = ±std / 95% CI)",
            fontsize=10, fontweight="bold",
        )
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
    _save(fig, output_dir, "fig1_acc_curves_multiseed")


def fig2_method_bars_multiseed(
    ms_acc_df: pd.DataFrame,
    per_seed_acc: dict,
    output_dir: str,
):
    """
    Fig 2 (multi-seed): grouped bar chart of mean accuracy ± std across seeds.

    Bar height  = mean of per-seed mean-accuracy-across-depths
    Error bar   = ±std of those per-seed means  (between-seed variability)
    Annotation  = "XX.X% ± Y.Y%"
    """
    models  = sorted(ms_acc_df["model"].unique(), key=lambda m: MODEL_LABELS.get(str(m), str(m)))
    methods = [m for m in MULTISEED_METHODS if m in ms_acc_df["method"].unique()]
    x       = np.arange(len(models))
    width   = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(max(7, 2.8 * len(models)), 5.5))

    for i, method in enumerate(methods):
        bar_means, bar_stds = [], []

        for model in models:
            # Collect per-seed mean (averaged across depths) for this (model, method)
            seed_means = []
            for seed_label, seed_acc_df in sorted(per_seed_acc.items()):
                sub = seed_acc_df[
                    (seed_acc_df["model"]  == model) &
                    (seed_acc_df["method"] == method)
                ]
                if not sub.empty:
                    seed_means.append(float(sub["acc"].mean()))

            if seed_means:
                bar_means.append(float(np.mean(seed_means)))
                bar_stds.append(float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0)
            else:
                bar_means.append(0.0)
                bar_stds.append(0.0)

        offset = (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(
            x + offset, bar_means, width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="white", linewidth=0.6,
            zorder=3,
        )
        # Error bars (±std across seeds)
        ax.errorbar(
            x + offset, bar_means,
            yerr=bar_stds,
            fmt="none", color="black",
            capsize=5, capthick=1.2, elinewidth=1.2,
            zorder=4,
        )
        # Annotations: mean ± std
        for bar, mean_val, std_val in zip(bars, bar_means, bar_stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(bar_stds) + 0.015,
                f"{mean_val:.1%}\n±{std_val:.1%}",
                ha="center", va="bottom", fontsize=7.5, linespacing=1.3,
            )

    n_seeds = len(per_seed_acc)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=10)
    ax.set_ylabel("Mean accuracy across all depths", fontsize=10)
    ax.set_ylim(0, 1.25)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9)
    ax.set_title(
        f"Mean Accuracy by Method and Model  ({n_seeds} seeds, error bars = ±std)",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save(fig, output_dir, "fig2_method_bars_multiseed")


def fig3_exp4_stats_heatmap_multiseed(seed_df4s: dict, output_dir: str):
    """
    Fig 3 (multi-seed): CISV diagnostic heatmap averaged across seeds.

    Cell values show mean across seeds; parenthetical values show ±std.
    """
    # Average stats DataFrames across seeds
    all_stats = [compute_exp4_stats(df) for df in seed_df4s.values()]
    n_seeds   = len(all_stats)

    # Collect all (model, depth) keys
    all_models = sorted({r for df in all_stats for r in df["model"].unique()},
                        key=lambda m: MODEL_LABELS.get(str(m), str(m)))
    all_depths = sorted({r for df in all_stats for r in df["depth"].unique()})
    metrics    = {
        "corrections_per_prob": "Corrections / problem",
        "fallbacks_per_prob":   "Fallbacks / problem",
        "ext_fail_per_prob":    "Extract failures / problem",
    }

    fig, axes = plt.subplots(1, len(all_models),
                             figsize=(5 * len(all_models), 3.8), sharey=True)
    if len(all_models) == 1:
        axes = [axes]

    for ax, model in zip(axes, all_models):
        mean_matrix = np.zeros((len(metrics), len(all_depths)))
        std_matrix  = np.zeros_like(mean_matrix)

        for j, depth in enumerate(all_depths):
            for i, metric in enumerate(metrics):
                vals = []
                for stats_df in all_stats:
                    row = stats_df[(stats_df["model"] == model) & (stats_df["depth"] == depth)]
                    if not row.empty:
                        vals.append(float(row[metric].values[0]))
                mean_matrix[i, j] = float(np.mean(vals)) if vals else 0.0
                std_matrix[i, j]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        im = ax.imshow(mean_matrix, vmin=0, aspect="auto", cmap="OrRd")
        ax.set_xticks(range(len(all_depths)))
        ax.set_xticklabels([f"d{d}" for d in all_depths], fontsize=8)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(list(metrics.values()), fontsize=8)
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=10, fontweight="bold")
        ax.set_xlabel("Depth", fontsize=8)

        for i in range(len(metrics)):
            for j in range(len(all_depths)):
                mean_v = mean_matrix[i, j]
                std_v  = std_matrix[i, j]
                text_color = "black" if mean_v < 0.5 else "white"
                label = f"{mean_v:.2f}" if n_seeds == 1 else f"{mean_v:.2f}\n±{std_v:.2f}"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=6.5, color=text_color, linespacing=1.3)

        plt.colorbar(im, ax=ax, fraction=0.046)

    seed_note = f"{n_seeds} seeds, cells = mean ± std"
    fig.suptitle(f"CISV Diagnostics per Depth  ({seed_note})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "fig3_exp4_stats_heatmap_multiseed")


def fig4_delta_curves_multiseed(
    ms_acc_df: pd.DataFrame,
    per_seed_acc: dict,
    output_dir: str,
):
    """
    Fig 4 (multi-seed): accuracy delta (CISV − CoT) per depth.

    Visual layers:
      1. Faint per-seed delta lines — raw between-seed variability
      2. 95% bootstrap CI band     — statistical confidence on the mean delta
      3. Bold mean delta line      — primary visual element
    """
    models = sorted(ms_acc_df["model"].unique(), key=lambda m: MODEL_LABELS.get(str(m), str(m)))
    depths = sorted(ms_acc_df["depth"].unique())

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for model in models:
        color = MODEL_COLORS.get(model, "black")
        m_df  = ms_acc_df[ms_acc_df["model"] == model]

        # ── Per-seed delta lines ──────────────────────────────────────────────
        for seed_label, seed_acc_df in sorted(per_seed_acc.items()):
            seed_deltas, seed_depths = [], []
            for depth in depths:
                cot_r = seed_acc_df[(seed_acc_df["model"] == model) &
                                    (seed_acc_df["method"] == "cot") &
                                    (seed_acc_df["depth"] == depth)]
                chk_r = seed_acc_df[(seed_acc_df["model"] == model) &
                                    (seed_acc_df["method"] == "chunked") &
                                    (seed_acc_df["depth"] == depth)]
                if cot_r.empty or chk_r.empty:
                    continue
                seed_deltas.append(float(chk_r["acc"].values[0]) - float(cot_r["acc"].values[0]))
                seed_depths.append(depth)
            if seed_deltas:
                ax.plot(seed_depths, seed_deltas, color=color,
                        linewidth=0.9, alpha=0.25, linestyle="--", marker=None)

        # ── Mean delta + bootstrap CI band ───────────────────────────────────
        mean_deltas, ci_lo_d, ci_hi_d, plot_depths = [], [], [], []
        for depth in depths:
            cot_row = m_df[(m_df["method"] == "cot")     & (m_df["depth"] == depth)]
            chk_row = m_df[(m_df["method"] == "chunked") & (m_df["depth"] == depth)]
            if cot_row.empty or chk_row.empty:
                continue

            # Collect per-seed deltas for CI
            seed_d = []
            for seed_label, seed_acc_df in per_seed_acc.items():
                c_r = seed_acc_df[(seed_acc_df["model"] == model) &
                                  (seed_acc_df["method"] == "cot") &
                                  (seed_acc_df["depth"] == depth)]
                k_r = seed_acc_df[(seed_acc_df["model"] == model) &
                                  (seed_acc_df["method"] == "chunked") &
                                  (seed_acc_df["depth"] == depth)]
                if not c_r.empty and not k_r.empty:
                    seed_d.append(float(k_r["acc"].values[0]) - float(c_r["acc"].values[0]))

            mean_d = float(chk_row["acc"].values[0]) - float(cot_row["acc"].values[0])
            lo, hi = _bootstrap_ci(seed_d) if seed_d else (mean_d, mean_d)

            mean_deltas.append(mean_d)
            ci_lo_d.append(lo)
            ci_hi_d.append(hi)
            plot_depths.append(depth)

        if mean_deltas:
            ax.fill_between(plot_depths, ci_lo_d, ci_hi_d,
                            alpha=0.18, color=color, label="_nolegend_")
            ax.plot(plot_depths, mean_deltas, marker="o", color=color,
                    linewidth=2.4, label=MODEL_LABELS.get(model, model), zorder=3)

    n_seeds = len(per_seed_acc)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.fill_between(depths,  0.0,  0.3, alpha=0.04, color="green")
    ax.fill_between(depths, -0.3,  0.0, alpha=0.04, color="red")
    ax.set_xlabel("Reasoning depth (steps)", fontsize=10)
    ax.set_ylabel("Accuracy delta (CISV − CoT)", fontsize=10)
    ax.set_title(
        f"CISV vs CoT Accuracy Delta  ({n_seeds} seeds, shading = 95% CI)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xticks(depths)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save(fig, output_dir, "fig4_delta_curves_multiseed")


# ── Console summaries ─────────────────────────────────────────────────────────

def print_summary(acc_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("SUMMARY: Mean accuracy per model × method")
    print("=" * 70)
    pivot = acc_df.groupby(["model", "method"])["acc"].mean().unstack("method")
    pivot.index = [MODEL_LABELS.get(m, m) for m in pivot.index]
    for col in ["direct", "cot", "chunked"]:
        if col not in pivot.columns:
            pivot[col] = float("nan")
    pivot = pivot[[c for c in ["direct", "cot", "chunked"] if c in pivot.columns]]
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


def print_multiseed_summary(ms_acc_df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("MULTI-SEED SUMMARY: Mean ± Std accuracy (CISV, per depth)")
    print("=" * 80)
    chk = ms_acc_df[ms_acc_df["method"] == "chunked"].copy()
    chk["mean±std"] = chk.apply(lambda r: f"{r['acc']:.3f} ± {r['std_acc']:.3f}", axis=1)
    pivot = chk.pivot_table(index="model", columns="depth", values="mean±std", aggfunc="first")
    pivot.index = [MODEL_LABELS.get(m, m) for m in pivot.index]
    print(pivot.to_string())

    print("\n" + "=" * 80)
    print("MULTI-SEED SUMMARY: CISV − CoT mean Δ ± std (averaged across depths)")
    print("=" * 80)
    for model in sorted(ms_acc_df["model"].unique()):
        m     = ms_acc_df[ms_acc_df["model"] == model]
        cot_d = m[m["method"] == "cot"].set_index("depth")["acc"]
        chk_d = m[m["method"] == "chunked"].set_index("depth")["acc"]
        common = sorted(set(cot_d.index) & set(chk_d.index))
        if not common:
            continue
        deltas = [chk_d[d] - cot_d[d] for d in common]
        print(f"  {MODEL_LABELS.get(model, model):<28} "
              f"Δ = {np.mean(deltas):+.3f} ± {np.std(deltas, ddof=1):.3f}")

    print("\n" + "=" * 80)
    print("MULTI-SEED SUMMARY: 95% Bootstrap CI on mean accuracy (CoT)")
    print("=" * 80)
    cot_df = ms_acc_df[ms_acc_df["method"] == "cot"]
    for model in sorted(cot_df["model"].unique()):
        m_df  = cot_df[cot_df["model"] == model]
        mean  = m_df["acc"].mean()
        lo    = m_df["ci_lo"].mean()
        hi    = m_df["ci_hi"].mean()
        print(f"  {MODEL_LABELS.get(model, model):<28} "
              f"mean = {mean:.3f}   CI = [{lo:.3f}, {hi:.3f}]")


def save_corrections_by_model_csv(df4: pd.DataFrame, output_dir: str):
    """Save corrections/fallbacks table per model × depth to CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for depth in sorted(df4["depth"].unique()):
        depth_data = df4[df4["depth"] == depth]
        row: dict[str, typing.Any] = {"Depth": f"d{depth}"}
        for model in sorted(df4["model"].unique()):
            md = depth_data[depth_data["model"] == model]
            if not md.empty:
                row[f"{MODEL_LABELS.get(model, model)} - Corrections"] = f"{md['corrections_applied'].mean():.3f}"
                row[f"{MODEL_LABELS.get(model, model)} - Fallbacks"]   = f"{md['unverified_fallbacks'].mean():.3f}"
        rows.append(row)
    table_df = pd.DataFrame(rows)
    csv_path = Path(output_dir) / "corrections_by_model.csv"
    table_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description="CISV comparative analysis: CoT vs Chunked (auto-detects multi-seed)"
    )
    p.add_argument("--exp2_dir",    default=str(root / "Experiment2" / "results"),
                   help="Base directory for Experiment 2 baselines. If seed_* "
                        "subdirs are found here, multi-seed mode is used automatically.")
    p.add_argument("--exp4_dir",    default=str(root / "CISV" /"results"),
                   help="Base directory for CISV results. If seed_* subdirs are "
                        "found here, multi-seed mode is used automatically.")
    p.add_argument("--output_dir",  default=str(root / "CISV" / "figures"),
                   help="Where to save figures and CSVs.")
    p.add_argument("--exp4_file",   default=None,
                   help="Single CISV JSON file (forces single-seed mode).")
    p.add_argument("--no_multi_seed", action="store_true",
                   help="Disable multi-seed mode even if seed_* dirs exist.")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Auto-detect multi-seed mode: use it whenever seed_* dirs exist in either
    # results directory, unless the user explicitly disabled it.
    use_multi = (
        not args.no_multi_seed
        and args.exp4_file is None
        and (_has_seed_dirs(args.exp2_dir) or _has_seed_dirs(args.exp4_dir))
    )

    if use_multi:
        # ── Multi-seed mode ──────────────────────────────────────────────────
        print(f"Multi-seed mode  (seed_* dirs detected)")
        print("─" * 50)

        print("\nLoading CoT baselines...")
        seed_df2s = load_multi_seed_exp2(args.exp2_dir)

        print("\nLoading CISV results...")
        seed_df4s = load_multi_seed_exp4(args.exp4_dir)

        print("\nAggregating across seeds...")
        ms_acc_df, per_seed_acc = compute_multi_seed_accuracy(seed_df2s, seed_df4s)

        print_multiseed_summary(ms_acc_df)

        # Save aggregated CSV
        ms_acc_df.to_csv(Path(args.output_dir) / "metrics_multiseed.csv", index=False)
        print(f"\n  Saved: {args.output_dir}/metrics_multiseed.csv")

        # Averaged diagnostic stats
        first_df4 = seed_df4s[sorted(seed_df4s.keys())[0]]
        save_corrections_by_model_csv(first_df4, args.output_dir)

        print("\nGenerating figures...")
        fig1_acc_curves_multiseed(ms_acc_df, per_seed_acc, args.output_dir)
        fig2_method_bars_multiseed(ms_acc_df, per_seed_acc, args.output_dir)
        fig3_exp4_stats_heatmap_multiseed(seed_df4s, args.output_dir)
        fig4_delta_curves_multiseed(ms_acc_df, per_seed_acc, args.output_dir)

    else:
        # ── Single-seed mode ─────────────────────────────────────────────────
        if use_multi is False and not args.no_multi_seed:
            print("Single-seed mode  (no seed_* dirs found, using flat results dirs)")
        else:
            print("Single-seed mode  (--no_multi_seed specified)")
        print("─" * 50)

        print("\nLoading data...")
        df2 = load_exp2(args.exp2_dir)

        if args.exp4_file:
            exp4_dir = Path(args.exp4_file).parent
            if not Path(args.exp4_file).name.startswith("exp4_chunked_"):
                print("  Warning: filename doesn't follow exp4_chunked_<model>.json "
                      "— model name inferred as 'unknown'")
            df4 = load_exp4(str(exp4_dir))
        else:
            df4 = load_exp4(args.exp4_dir)

        print("\nComputing metrics...")
        df_all   = pd.concat([df2[["model", "method", "depth", "correct"]],
                               df4[["model", "method", "depth", "correct"]]],
                              ignore_index=True)
        acc_df   = compute_accuracy(df_all)
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
