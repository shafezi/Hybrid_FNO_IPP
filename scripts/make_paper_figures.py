"""Generate paper figures from master_metrics.csv.

Produces:
  - bar_chart_<n_robots>bots.png  (Figure 4 candidate)
  - scaling_curve.png              (Figure 6)
  - pareto_<x>_vs_<y>.png          (Figure 5 candidates)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

OUT_DIR = os.path.join(ROOT, "results", "dynamic_ipp", "final", "figures_paper")
os.makedirs(OUT_DIR, exist_ok=True)

MASTER = os.path.join(ROOT, "results", "dynamic_ipp", "final", "master_metrics.csv")
df = pd.read_csv(MASTER)
print(f"Loaded {len(df)} rows")

METHODS = ["FNO-only", "FNO+GP", "Persist-only", "Persist+GP", "GP-only"]
COLORS = {"FNO-only": "#1f77b4", "FNO+GP": "#ff7f0e",
          "Persist-only": "#2ca02c", "Persist+GP": "#d62728",
          "GP-only": "#9467bd"}
KERNEL_MARKERS = {"matern05": "o", "matern15": "s", "matern25": "^", "rbf": "D"}

L = df["step"].max()
final_step = df[df["step"] == L]


# -----------------------------------------------------------------------------
# Figure 4: Bar chart at chosen config (default: 20 robots, all kernels/acq pooled)
# -----------------------------------------------------------------------------
def plot_bar_chart(n_robots):
    """Bar chart at one robot count, averaged over seeds and hyperparameters."""
    sub = final_step[final_step["n_robots"] == n_robots]
    metrics = ["RMSE", "ACC", "HF", "FSS"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for idx, met in enumerate(metrics):
        ax = axes[idx]
        means = []
        stds = []
        for m in METHODS:
            vals = sub[sub["method"] == m][met]
            means.append(vals.mean())
            stds.append(vals.std())

        x = np.arange(len(METHODS))
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                     color=[COLORS[m] for m in METHODS],
                     edgecolor="black", linewidth=0.7)

        if met == "HF":
            ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, linewidth=1)
            ax.text(0.02, 1.05, "ideal", fontsize=8, transform=ax.get_yaxis_transform())

        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, rotation=20, ha="right", fontsize=10)
        ax.set_title(met, fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylabel(met, fontsize=11)

    fig.suptitle(f"{n_robots} robots — final step (day {L * 4}), avg over 5 seeds × hyperparameters",
                fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"bar_chart_{n_robots}bots.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


for n in [5, 10, 20, 40]:
    plot_bar_chart(n)


# -----------------------------------------------------------------------------
# Figure 6: Robot-count scaling curve
# -----------------------------------------------------------------------------
def plot_scaling_curve():
    """RMSE/ACC/HF/FSS vs n_robots for each method, averaged over hyperparameters & seeds."""
    metrics = ["RMSE", "ACC", "HF", "FSS"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    for idx, met in enumerate(metrics):
        ax = axes[idx]
        for m in METHODS:
            sub_m = final_step[final_step["method"] == m]
            grouped = sub_m.groupby("n_robots")[met]
            means = grouped.mean()
            stds = grouped.std()
            ax.errorbar(means.index, means.values, yerr=stds.values,
                       label=m, color=COLORS[m], marker="o", capsize=4,
                       linewidth=2, markersize=7)

        if met == "HF":
            ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, linewidth=1)
        if met == "Bias":
            ax.axhline(y=0.0, color="black", linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xlabel("Number of robots", fontsize=11)
        ax.set_ylabel(met, fontsize=11)
        ax.set_title(met, fontsize=13, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 40])
        ax.set_xticklabels([5, 10, 20, 40])
        ax.grid(True, alpha=0.3, which="both")
        if idx == 0:
            ax.legend(loc="best", fontsize=9)

    fig.suptitle("Scaling with number of robots — final step (day {}), 5 seeds".format(L * 4),
                fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "scaling_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


plot_scaling_curve()


# -----------------------------------------------------------------------------
# Figure 5: Pareto plots (RMSE vs HF, RMSE vs FSS, etc.)
# -----------------------------------------------------------------------------
def plot_pareto_grid():
    """Big pareto grid: 5 metric-pairs × 4 robot counts.
    Only GP-using methods shown (FNO+GP, Persist+GP, GP-only).
    Color = method, Marker = kernel. All 4 kernels × 3 acquisitions × 5 seeds plotted.
    """
    metric_pairs = [
        ("RMSE", "ACC",   "lower is better", "higher is better"),
        ("RMSE", "HF",    "lower is better", "want = 1.0"),
        ("RMSE", "FSS",   "lower is better", "higher is better"),
        ("ACC",  "HF",    "higher is better","want = 1.0"),
        ("ACC",  "FSS",   "higher is better","higher is better"),
    ]
    GP_METHODS = ["FNO+GP", "Persist+GP", "GP-only"]

    fig, axes = plt.subplots(5, 4, figsize=(22, 24))

    for row, (xm, ym, x_dir, y_dir) in enumerate(metric_pairs):
        for col, n in enumerate([5, 10, 20, 40]):
            ax = axes[row, col]
            sub = final_step[final_step["n_robots"] == n]
            grp = sub.groupby(["method", "kernel_tag", "acquisition"]).agg(
                {xm: "mean", ym: "mean"}).reset_index()

            for m in GP_METHODS:
                rows_m = grp[grp["method"] == m]
                for _, r in rows_m.iterrows():
                    marker = KERNEL_MARKERS.get(r["kernel_tag"], "o")
                    ax.scatter(r[xm], r[ym], c=COLORS[m], marker=marker,
                              s=60, alpha=0.6, edgecolors="black", linewidths=0.4)

            if ym == "HF":
                ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, linewidth=1)
            if xm == "HF":
                ax.axvline(x=1.0, color="black", linestyle="--", alpha=0.5, linewidth=1)

            ax.set_xlabel(f"{xm} ({x_dir})", fontsize=10)
            ax.set_ylabel(f"{ym} ({y_dir})", fontsize=10)
            if row == 0:
                ax.set_title(f"{n} robots", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.text(-0.30, 0.5, f"{xm} vs {ym}",
                       transform=ax.transAxes, rotation=90, fontsize=12,
                       fontweight="bold", va="center", ha="center")

    # Method legend: lines (not markers) showing color = method
    method_handles = [Line2D([0], [0], color=COLORS[m], linewidth=4, label=m)
                      for m in GP_METHODS]
    # Kernel legend: gray markers showing shape = kernel
    kernel_labels_pretty = {"matern05": "Matérn ν=0.5",
                            "matern15": "Matérn ν=1.5",
                            "matern25": "Matérn ν=2.5",
                            "rbf":      "RBF"}
    kernel_handles = [Line2D([0], [0], marker=mk, color="gray", linestyle="None",
                             markersize=10, label=kernel_labels_pretty[k],
                             markeredgecolor="black")
                      for k, mk in KERNEL_MARKERS.items()]
    fig.legend(handles=method_handles + kernel_handles, loc="lower center",
              ncol=7, fontsize=11, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle("Pareto: each point = one (kernel × acquisition × seed). "
                 "Color = method, marker = kernel. Final step, 5 seeds.",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0.02, 0.02, 1, 0.99])
    out = os.path.join(OUT_DIR, "pareto_grid.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


plot_pareto_grid()


# -----------------------------------------------------------------------------
# Bonus: Trajectory plots over time (Avg RMSE evolution)
# -----------------------------------------------------------------------------
def plot_avg_rmse_over_time():
    """Avg RMSE over steps, per robot count, averaged over seeds & hyperparameters."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    for idx, n in enumerate([5, 10, 20, 40]):
        ax = axes[idx]
        sub = df[df["n_robots"] == n]
        for m in METHODS:
            sub_m = sub[sub["method"] == m]
            grouped = sub_m.groupby("step")["RMSE"]
            means = grouped.mean()
            stds = grouped.std()
            ax.plot(means.index, means.values, label=m,
                   color=COLORS[m], linewidth=2)
            ax.fill_between(means.index,
                          (means - stds).values, (means + stds).values,
                          color=COLORS[m], alpha=0.15)

        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("RMSE", fontsize=11)
        ax.set_title(f"{n} robots", fontsize=13, fontweight="bold")
        ax.set_xlim(1, L)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=9)

    fig.suptitle("RMSE evolution over rollout steps — 5 seeds × hyperparameters",
                fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "rmse_over_time.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


plot_avg_rmse_over_time()


# -----------------------------------------------------------------------------
# Significance tests: FNO+GP vs Persist+GP at each robot count
# -----------------------------------------------------------------------------
print("\n===== Significance: FNO+GP vs Persist+GP =====")
print(f"{'n':>3}  {'metric':<6}  {'FNO+GP':>9}  {'Per+GP':>9}  {'p':>8}  {'sig':<5}")
print("-" * 60)
for n in [5, 10, 20, 40]:
    for met in ["RMSE", "ACC", "HF", "FSS"]:
        a = final_step[(final_step["n_robots"] == n) &
                       (final_step["method"] == "FNO+GP")][met].values
        b = final_step[(final_step["n_robots"] == n) &
                       (final_step["method"] == "Persist+GP")][met].values
        # Independent samples since hyperparameter combos differ
        if len(a) == len(b) and len(a) > 1:
            t, p = stats.ttest_rel(a, b)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"{n:>3}  {met:<6}  {a.mean():>9.3f}  {b.mean():>9.3f}  {p:>8.4f}  {sig:<5}")

print("\nDone. Figures saved to:", OUT_DIR)
