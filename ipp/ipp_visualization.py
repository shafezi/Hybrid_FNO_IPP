"""
Visualization utilities for the single-robot IPP experiment.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.colors import Normalize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sym_vmax(arr, pct=97):
    finite = arr[np.isfinite(arr)]
    return float(np.percentile(np.abs(finite), pct)) if len(finite) > 0 else 1.0


def _mask_land(arr2d, ocean_mask):
    out = arr2d.copy().astype(float)
    out[~ocean_mask] = np.nan
    return out


# ---------------------------------------------------------------------------
# 1.  RMSE / MAE vs step number (policy comparison)
# ---------------------------------------------------------------------------

def plot_metric_vs_steps(records_df, metric="all_rmse", save_path=None,
                         title=None, fno_col=None):
    """
    Line plot of mean metric ± std across episodes, one line per policy.

    records_df columns: episode, policy, step, <metric>
    fno_col : column name for the FNO-only baseline (constant per sample).
              If provided, a horizontal dashed line is drawn.
    """
    import pandas as pd

    policies = records_df["policy"].unique()
    steps    = sorted(records_df["step"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    cmap   = cm.get_cmap("tab10", len(policies))
    colors = {p: cmap(i) for i, p in enumerate(policies)}

    for pol in policies:
        sub = records_df[records_df["policy"] == pol]
        grp = sub.groupby("step")[metric]
        means = grp.mean().reindex(steps).values
        stds  = grp.std().reindex(steps).fillna(0).values
        ax.plot(steps, means, "o-", color=colors[pol], label=pol, markersize=4)
        ax.fill_between(steps, means - stds, means + stds,
                        alpha=0.15, color=colors[pol])

    # FNO baseline (constant)
    if fno_col and fno_col in records_df.columns:
        fno_val = records_df[fno_col].mean()
        ax.axhline(fno_val, color="gray", linestyle="--", linewidth=1.5,
                   label="FNO only")

    ax.set_xlabel("Sensing step")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title(title or f"{metric} vs sensing step")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2.  Unobserved-only RMSE vs steps
# ---------------------------------------------------------------------------

def plot_unobs_rmse_vs_steps(records_df, save_path=None):
    plot_metric_vs_steps(
        records_df, metric="unobs_rmse",
        title="Unobserved-cell RMSE vs sensing step",
        fno_col="fno_rmse",
        save_path=save_path,
    )


# ---------------------------------------------------------------------------
# 3.  Robot trajectories (all episodes, one subplot per policy)
# ---------------------------------------------------------------------------

def plot_trajectories(trajectories, all_ocean_coords, cand_local_idx,
                      n_episodes, policies_ordered, save_path=None):
    """
    trajectories : dict (ep_i, pol_name) → (T+1, 2) array
    cand_local_idx : (N_cand,) for plotting candidate locations
    """
    n_pol = len(policies_ordered)
    cand_coords = all_ocean_coords[cand_local_idx]

    fig, axes = plt.subplots(1, n_pol, figsize=(5 * n_pol, 4),
                             squeeze=False)

    for j, pol in enumerate(policies_ordered):
        ax = axes[0, j]
        ax.scatter(all_ocean_coords[:, 1], all_ocean_coords[:, 0],
                   s=0.2, c="lightgray", zorder=0)
        ax.scatter(cand_coords[:, 1], cand_coords[:, 0],
                   s=0.5, c="steelblue", alpha=0.4, zorder=1)

        cmap = cm.get_cmap("plasma", n_episodes)
        for ep_i in range(n_episodes):
            key = (ep_i, pol)
            if key not in trajectories:
                continue
            traj = trajectories[key]   # (T+1, 2)
            ax.plot(traj[:, 1], traj[:, 0], "-", color=cmap(ep_i),
                    linewidth=0.8, alpha=0.6)
            ax.scatter(traj[0, 1], traj[0, 0], c="lime", s=20, zorder=5,
                       edgecolors="black", linewidths=0.3)
            ax.scatter(traj[-1, 1], traj[-1, 0], c="red", s=20, zorder=5,
                       marker="*", edgecolors="black", linewidths=0.3)

        ax.set_title(pol, fontsize=9)
        ax.set_xlabel("lon (norm.)")
        ax.set_ylabel("lat (norm.)")
        ax.invert_yaxis()
        ax.set_aspect("equal")

    fig.suptitle("Robot trajectories (green=start, red★=end)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4.  Qualitative panel for a single episode step sequence
# ---------------------------------------------------------------------------

def plot_ipp_qualitative(
    fno_flat,
    label_flat,
    qual_history,          # list of step dicts with gp_mean_all, gp_std_all
    trajectory,            # (T+1, 2)
    all_ocean_coords,      # (N_ocean, 2)
    ocean_mask,            # (H, W) bool
    H, W,
    steps_to_show,         # e.g. [1, 5, 10, 20]
    save_path=None,
    policy_name="",
):
    """
    For each step in steps_to_show, show a 3-panel row:
    [GP std (uncertainty) | GP corrected prediction | Remaining error]
    with robot path overlaid.
    """
    n_rows = len(steps_to_show)
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4 * n_rows),
                             squeeze=False, constrained_layout=True)

    fno_2d   = np.full((H * W,), np.nan)
    lab_2d   = np.full((H * W,), np.nan)
    ocean_flat_idx = np.where(ocean_mask.ravel())[0]

    fno_2d[ocean_flat_idx] = fno_flat
    lab_2d[ocean_flat_idx] = label_flat
    fno_2d = fno_2d.reshape(H, W)
    lab_2d = lab_2d.reshape(H, W)

    # --- compute global color limits across all shown steps (one per column) ---
    all_stds, all_corr, all_err = [], [], []
    for step in steps_to_show:
        rec      = qual_history[step - 1]
        gp_mean  = rec["gp_mean_all"]
        gp_std   = rec["gp_std_all"]
        all_stds.append(gp_std)
        all_corr.append(fno_flat + gp_mean)
        all_err.append(label_flat - (fno_flat + gp_mean))

    std_vmax  = float(np.nanpercentile(np.concatenate(all_stds), 99))
    ssh_vmax  = _sym_vmax(np.concatenate(all_corr))
    err_vmax  = _sym_vmax(np.concatenate(all_err))
    # -------------------------------------------------------------------------

    for row_i, step in enumerate(steps_to_show):
        rec = qual_history[step - 1]   # 0-indexed
        gp_mean = rec["gp_mean_all"]   # (N_ocean,)
        gp_std  = rec["gp_std_all"]    # (N_ocean,)

        corr_flat = fno_flat + gp_mean
        err_flat  = label_flat - corr_flat

        corr_2d = np.full((H * W,), np.nan)
        err_2d  = np.full((H * W,), np.nan)
        std_2d  = np.full((H * W,), np.nan)

        corr_2d[ocean_flat_idx] = corr_flat
        err_2d[ocean_flat_idx]  = err_flat
        std_2d[ocean_flat_idx]  = gp_std

        corr_2d = corr_2d.reshape(H, W)
        err_2d  = err_2d.reshape(H, W)
        std_2d  = std_2d.reshape(H, W)

        traj_so_far = trajectory[:step + 1]   # includes step positions

        # GP uncertainty
        ax = axes[row_i, 0]
        im = ax.imshow(std_2d, origin="upper", cmap="viridis", vmin=0, vmax=std_vmax)
        ax.plot(traj_so_far[:, 1] * W, traj_so_far[:, 0] * H,
                "w-o", markersize=2, linewidth=0.8)
        ax.scatter(traj_so_far[-1, 1] * W, traj_so_far[-1, 0] * H,
                   c="red", s=30, zorder=10, marker="*")
        ax.set_title(f"Step {step}: GP std (uncertainty)", fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.7)

        # Corrected prediction
        ax = axes[row_i, 1]
        im = ax.imshow(corr_2d, origin="upper", cmap="RdBu_r",
                       vmin=-ssh_vmax, vmax=ssh_vmax)
        ax.plot(traj_so_far[:, 1] * W, traj_so_far[:, 0] * H,
                "k-o", markersize=2, linewidth=0.8)
        ax.set_title(f"Step {step}: FNO+GP prediction", fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.7)

        # Remaining error
        ax = axes[row_i, 2]
        im = ax.imshow(err_2d, origin="upper", cmap="RdBu_r",
                       vmin=-err_vmax, vmax=err_vmax)
        ax.plot(traj_so_far[:, 1] * W, traj_so_far[:, 0] * H,
                "k-o", markersize=2, linewidth=0.8)
        ax.set_title(f"Step {step}: Remaining error", fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.7)

    fig.suptitle(f"IPP episode — {policy_name}", fontsize=11, fontweight="bold")
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5.  Final policy comparison bar chart
# ---------------------------------------------------------------------------

def plot_policy_comparison_bar(records_df, metric="all_rmse",
                                step=None, save_path=None):
    """
    Bar chart comparing mean final metric across policies.
    step : if None, use the last step of each episode.
    """
    import pandas as pd

    if step is None:
        sub = records_df.loc[records_df.groupby(["policy", "episode"])["step"].idxmax()]
    else:
        sub = records_df[records_df["step"] == step]

    grp   = sub.groupby("policy")[metric]
    means = grp.mean()
    stds  = grp.std().fillna(0)

    fig, ax = plt.subplots(figsize=(7, 4))
    policies = means.index.tolist()
    x = np.arange(len(policies))

    bars = ax.bar(x, means.values, yerr=stds.values, capsize=5,
                  color="steelblue", edgecolor="white", alpha=0.85)

    # FNO baseline line
    if "fno_rmse" in records_df.columns:
        fno_val = records_df["fno_rmse"].mean()
        ax.axhline(fno_val, color="tomato", linestyle="--", linewidth=1.5,
                   label=f"FNO only ({fno_val:.4f})")
        ax.legend(fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(metric.replace("_", " ").upper())
    step_label = f"(step {step})" if step else "(final step)"
    ax.set_title(f"Policy comparison — {metric.upper()} {step_label}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6.  Summary table image
# ---------------------------------------------------------------------------

def plot_ipp_summary_table(summary_df, save_path=None):
    """
    Render per-policy summary stats as a formatted table image.

    summary_df columns: policy, final_rmse_mean, final_rmse_std,
                        final_mae_mean, final_mae_std,
                        fno_rmse, pct_improvement_rmse
    """
    if summary_df is None or summary_df.empty:
        return

    display_cols = [c for c in [
        "policy", "final_rmse_mean", "final_rmse_std",
        "final_mae_mean", "final_mae_std",
        "fno_rmse", "pct_improvement_rmse",
    ] if c in summary_df.columns]

    sub = summary_df[display_cols].copy()
    for c in display_cols[1:]:
        sub[c] = sub[c].map(lambda x: f"{x:.4f}" if np.isfinite(x) else "—")

    col_labels = [c.replace("_", " ") for c in display_cols]

    fig, ax = plt.subplots(
        figsize=(len(display_cols) * 1.8 + 1, len(sub) * 0.55 + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=sub.values, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.8)
    ax.set_title("IPP policy comparison — aggregate results", pad=20)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7.  Cumulative distance vs RMSE (efficiency frontier)
# ---------------------------------------------------------------------------

def plot_distance_vs_rmse(records_df, save_path=None):
    """
    Scatter / line: cumulative travel distance vs RMSE improvement.
    One curve per policy (mean over episodes).
    """
    import pandas as pd

    policies = records_df["policy"].unique()
    fig, ax  = plt.subplots(figsize=(7, 5))
    cmap     = cm.get_cmap("tab10", len(policies))

    for i, pol in enumerate(policies):
        sub = records_df[records_df["policy"] == pol]
        grp = sub.groupby("step")[["cumulative_dist", "all_rmse"]].mean()
        ax.plot(grp["cumulative_dist"].values, grp["all_rmse"].values,
                "o-", color=cmap(i), label=pol, markersize=4)

    ax.set_xlabel("Mean cumulative travel distance (normalized)")
    ax.set_ylabel("Mean RMSE (all ocean cells)")
    ax.set_title("Travel efficiency: distance vs RMSE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
