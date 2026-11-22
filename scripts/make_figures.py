#!/usr/bin/env python
"""Regenerate every report figure from the CSVs in ``results/`` (CPU, seconds).

    python scripts/make_figures.py --results results --out figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical palette (fixed order, never cycled) and text inks.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 2.0, "savefig.dpi": 300, "font.family": "DejaVu Sans",
})


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_reproduction(res, out):
    """Dot plot (paper vs ours, mean +- SD); dots, not bars, because the axis does not start at 0."""
    pl = pd.read_csv(res / "reproduction_personalllm.csv")
    rd = pd.read_csv(res / "reproduction_reddit.csv")
    paper = {("PL", 0.1): (96.0, 96.1, 0.7, 0.8), ("PL", 0.001): (94.3, 93.3, 0.9, 1.0),
             ("RD", 100): (65.0, 66.2, 1.1, 1.2), ("RD", 150): (66.2, 66.7, 1.0, 1.1)}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    panels = [("PersonalLLM", pl, "alpha", [0.1, 0.001], "PL", lambda v: f"α={v}", (91, 97.2)),
              ("Reddit TL;DR", rd, "n_seen_examples", [100, 150], "RD", lambda v: f"n={v}", (62.5, 68.2))]
    for ax, (title, df, key, vals, tag, fmt, ylim) in zip(axes, panels):
        labels, x = [], 0
        for v in vals:
            g = df[df[key] == v]
            ps, pu, pss, psu = paper[(tag, v)]
            for metric, pm, psd in [("seen_acc", ps, pss), ("unseen_acc", pu, psu)]:
                om, osd = g[metric].mean(), g[metric].std(ddof=1)
                ax.errorbar(x - 0.13, pm, yerr=psd, fmt="o", ms=6, color=C[0], mec="white", mew=1,
                            elinewidth=1.1, capsize=2.5, label="Paper (mean ± SD, 20 runs)" if x == 0 else None)
                ax.errorbar(x + 0.13, om, yerr=osd, fmt="s", ms=6, color=C[1], mec="white", mew=1,
                            elinewidth=1.1, capsize=2.5, label="Ours (mean ± SD, 5 seeds)" if x == 0 else None)
                ax.annotate(f"{om - pm:+.1f}", (x + 0.13, om - osd), xytext=(0, -4), textcoords="offset points",
                            ha="center", va="top", fontsize=7.5, color=INK2)
                labels.append(f"{'Seen' if metric == 'seen_acc' else 'Unseen'}\n{fmt(v)}")
                x += 1
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylim(*ylim)
        ax.set_title(title, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Pairwise accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out, "fig_reproduction")


# --------------------------------------------------------------------------- #
def fig_architecture(res, out):
    """Horizontal dot plot of unseen-user accuracy; trainable-parameter count in each label."""
    df = pd.read_csv(res / "architecture_ablation_summary.csv")
    order = ["Skywork reference", "PCA-5", "PCA-512 + Linear", "Random-512 + Linear", "Original Linear",
             "Fixed-Anchor-Residual", "MLP-8 (odd)", "MLP-64 (odd)", "Residual-16 (odd)"]
    short = {"Skywork reference": "Frozen Skywork reward", "PCA-5": "PCA-5 (fixed basis)",
             "PCA-512 + Linear": "PCA-512 + linear", "Random-512 + Linear": "Random-512 + linear",
             "Original Linear": "Linear (LoRe)", "Fixed-Anchor-Residual": "Anchor + 4 residual",
             "MLP-8 (odd)": "MLP-8 (odd)", "MLP-64 (odd)": "MLP-64 (odd)", "Residual-16 (odd)": "Residual-16 (odd)"}

    def k(n):
        return "0" if n == 0 else (f"{n / 1e3:.0f}k" if n < 1e6 else f"{n / 1e6:.1f}M")

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1))
    for ax, (bm, title, xlim) in zip(axes, [("PersonalLLM", "PersonalLLM (α = 0.1)", (79.5, 97)),
                                            ("RedditTLDR", "Reddit TL;DR (n = 150)", (46.5, 67.5))]):
        d = df[df.benchmark == bm].set_index("architecture").loc[order]
        y = np.arange(len(order))[::-1]
        lin = d.loc["Original Linear", "unseen_eval_mean"]
        ax.axvline(lin, color=C[1], linewidth=1, linestyle=":")
        for yi, (name, r) in zip(y, d.iterrows()):
            col = C[1] if name == "Original Linear" else (INK2 if name == "Skywork reference" else C[0])
            sd = 0 if pd.isna(r.unseen_eval_seed_sd) else r.unseen_eval_seed_sd
            ax.errorbar(r.unseen_eval_mean, yi, xerr=sd, fmt="o", ms=5.5, color=col, mec="white", mew=1,
                        elinewidth=1.1, capsize=2)
            ax.annotate(f"{r.unseen_eval_mean:.1f}", (r.unseen_eval_mean + sd, yi), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=7.5, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{short[n]}  [{k(d.loc[n, 'trainable_shared'])}]" for n in order] if bm == "PersonalLLM"
                           else [])
        ax.set_xlim(*xlim)
        ax.set_title(title, color=INK)
        ax.set_xlabel("Unseen-user accuracy (%)")
        ax.grid(axis="y", visible=False)
        if bm == "RedditTLDR":
            ax.axvline(50, color=INK2, linewidth=0.8)
            ax.text(50.3, 7.6, "chance", fontsize=7.5, color=INK2)
    fig.tight_layout()
    save(fig, out, "fig_architecture")


# --------------------------------------------------------------------------- #
METHOD_ORDER = ["random_logits", "uniform", "population_mean", "nearest_seen_user", "cluster_centroid"]
METHOD_LABEL = {"random_logits": "Random logits (LoRe)", "uniform": "Uniform", "population_mean": "Population mean",
                "nearest_seen_user": "Nearest seen user", "cluster_centroid": "Cluster centroid"}


def fig_init_curves(res, out):
    df = pd.read_csv(res / "init_ablation_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.9), sharex=True)
    for ax, n in zip(axes.flat, [5, 15, 50, 150]):
        d = df[df.support_size == n]
        for i, m in enumerate(METHOD_ORDER):
            g = d[d.initialization == m].sort_values("updates")
            ax.plot(g.updates, g.mean_query_accuracy, marker="o", ms=4.5, color=C[i], mec="white", mew=0.8,
                    label=METHOD_LABEL[m])
        ax.set_title(f"Support size N = {n}", color=INK)
        ax.set_xticks([0, 5, 20, 50])
        ax.set_ylim(89.5, 96.5)
    for ax in axes[1]:
        ax.set_xlabel("Gradient updates on the new user")
    for ax in axes[:, 0]:
        ax.set_ylabel("Query accuracy (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out, "fig_init_curves")


def fig_zero_update(res, out):
    df = pd.read_csv(res / "init_ablation_summary.csv")
    series = [("cluster_centroid", 0, "Cluster centroid, 0 updates", C[4], "o"),
              ("nearest_seen_user", 0, "Nearest seen user, 0 updates", C[3], "s"),
              ("random_logits", 50, "Random logits, 50 updates", C[0], "D"),
              ("uniform", 50, "Uniform, 50 updates", C[1], "^")]
    Ns = [5, 15, 50, 150]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for k, (m, u, lab, col, mk) in enumerate(series):
        y = [df[(df.support_size == n) & (df.initialization == m) & (df.updates == u)].mean_query_accuracy.iloc[0] for n in Ns]
        ax.plot(np.arange(4) + (k - 1.5) * 0.08, y, marker=mk, ms=6, color=col, mec="white", mew=0.8, label=lab,
                linewidth=1.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"N = {n}" for n in Ns])
    ax.set_ylabel("Query accuracy (%)")
    ax.set_xlabel("Support examples for the new user")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, out, "fig_zero_update")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_reproduction(res, out)
    fig_architecture(res, out)
    fig_init_curves(res, out)
    fig_zero_update(res, out)
    print("Figures written to", out)


if __name__ == "__main__":
    main()
