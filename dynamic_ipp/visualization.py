"""
Visualization utilities for the dynamic rollout experiment.

All functions take pre-computed DataFrames or arrays and produce matplotlib figures.
No experiment logic here.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

_POLICY_COLORS = {
    "fno_only":        "#888888",
    "random":          "#1f77b4",
    "uncertainty_only":"#2ca02c",
    "hybrid_greedy":   "#d62728",
    "raster":          "#9467bd",
}

def _color(name):
    return _POLICY_COLORS.get(name, "#333333")

def _policy_label(name):
    return {
        "fno_only":         "FNO only",
        "random":           "Random",
        "uncertainty_only": "Uncertainty-only",
        "hybrid_greedy":    "Hybrid greedy",
        "raster":           "Raster",
    }.get(name, name)


# ---------------------------------------------------------------------------
# 1. RMSE vs time step
# ---------------------------------------------------------------------------

def plot_rmse_vs_steps(df, out_path, title_suffix="", days_per_step=5):
    """Rollout RMSE (mean ± std across episodes) vs forecast step."""
    policies = df["policy"].unique()
    steps    = sorted(df["step"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, ylabel in [
        (axes[0], "all_rmse",   "RMSE — all ocean cells"),
        (axes[1], "unobs_rmse", "RMSE — unobserved cells only"),
    ]:
        for pol in policies:
            sub   = df[df["policy"] == pol]
            grp   = sub.groupby("step")[col]
            means = grp.mean().reindex(steps).values
            stds  = grp.std(ddof=0).reindex(steps).fillna(0).values
            ax.plot(steps, means, "-o", color=_color(pol),
                    label=_policy_label(pol), markersize=3, linewidth=1.4)
            ax.fill_between(steps, means - stds, means + stds,
                            alpha=0.12, color=_color(pol))

        # Assimilation markers — from all policies (not just the first, which may be fno_only)
        assim_steps = sorted(df[df["assimilation"] == 1]["step"].unique())
        for s in assim_steps:
            ax.axvline(s, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)

        ax.set_xlabel(f"Forecast step (×{days_per_step} days)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Rollout RMSE vs forecast step{title_suffix}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. RMSE vs cumulative distance
# ---------------------------------------------------------------------------

def plot_rmse_vs_distance(df, out_path):
    """RMSE vs cumulative travel distance — travel efficiency."""
    policies = df["policy"].unique()

    fig, ax = plt.subplots(figsize=(7, 5))
    for pol in policies:
        sub = df[df["policy"] == pol]
        grp = sub.groupby("step")[["cumulative_dist", "all_rmse"]].mean()
        ax.plot(grp["cumulative_dist"].values, grp["all_rmse"].values,
                "-o", color=_color(pol), label=_policy_label(pol),
                markersize=3, linewidth=1.4)

    ax.set_xlabel("Mean cumulative travel distance (normalized [0,1]²)")
    ax.set_ylabel("Mean RMSE — all ocean")
    ax.set_title("Travel efficiency: RMSE vs distance traveled")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. RMSE vs number of observations
# ---------------------------------------------------------------------------

def plot_rmse_vs_nobs(df, out_path):
    """RMSE vs cumulative observations collected."""
    policies = df["policy"].unique()

    fig, ax = plt.subplots(figsize=(7, 5))
    for pol in policies:
        sub = df[df["policy"] == pol]
        grp = sub.groupby("step")[["n_meas_total", "all_rmse"]].mean()
        ax.plot(grp["n_meas_total"].values, grp["all_rmse"].values,
                "-o", color=_color(pol), label=_policy_label(pol),
                markersize=3, linewidth=1.4)

    ax.set_xlabel("Cumulative measurements (budget)")
    ax.set_ylabel("Mean RMSE — all ocean")
    ax.set_title("Observation efficiency: RMSE vs #observations")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Summary bar chart
# ---------------------------------------------------------------------------

def plot_summary_bar(summary_df, fno_rmse, out_path):
    """Final-step mean RMSE bar chart with FNO baseline."""
    fig, ax = plt.subplots(figsize=(8, 4))

    policies = summary_df["policy"].tolist()
    means    = summary_df["final_rmse_mean"].values
    stds     = summary_df["final_rmse_std"].values
    colors   = [_color(p) for p in policies]
    x        = np.arange(len(policies))

    ax.bar(x, means, yerr=stds, capsize=5, color=colors,
           edgecolor="white", alpha=0.85)
    ax.axhline(fno_rmse, color="gray", linestyle="--", lw=1.5,
               label=f"FNO only ({fno_rmse:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels([_policy_label(p) for p in policies],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Final-step mean RMSE")
    ax.set_title("Policy comparison — final forecast step")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Robot trajectories
# ---------------------------------------------------------------------------

def plot_trajectories(trajectories_dict, ocean_mask, all_ocean_coords, out_path):
    """All policies' episode-0 robot paths over the ocean domain."""
    policies = list(trajectories_dict.keys())
    n        = len(policies)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

    H, W = ocean_mask.shape
    land = ~ocean_mask

    for ax, pol in zip(axes[0], policies):
        # ocean background
        bg = np.zeros((H, W, 4))
        bg[ocean_mask]  = [0.85, 0.92, 0.99, 1.0]
        bg[land]        = [0.55, 0.55, 0.55, 1.0]
        ax.imshow(bg, origin="upper", aspect="auto")

        traj_list = trajectories_dict[pol]
        if traj_list:
            ep0 = traj_list[0]   # episode 0
            # Multi-robot: list of (T_r,2) arrays; single-robot: one (T,2) array
            robot_trajs = ep0 if isinstance(ep0, list) else [ep0]
            colors_r = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
            for r_i, traj in enumerate(robot_trajs):
                c = colors_r[r_i % len(colors_r)] if len(robot_trajs) > 1 else _color(pol)
                xs = traj[:, 1] * W
                ys = traj[:, 0] * H
                lbl = f"R{r_i}" if len(robot_trajs) > 1 else None
                ax.plot(xs, ys, "-", color=c, linewidth=0.8, alpha=0.8, label=lbl)
                ax.scatter(xs[0],  ys[0],  c="lime",  s=40, zorder=5)
                ax.scatter(xs[-1], ys[-1], c="red",   s=40, zorder=5, marker="*")

        ax.set_title(_policy_label(pol), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if pol == policies[0]:
            ax.legend(fontsize=7, loc="lower left")

    fig.suptitle("Robot trajectories — episode 0", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Qualitative panels (per policy, per time step)
# ---------------------------------------------------------------------------

def _field_to_2d(flat_vals, mask_2d, fill=np.nan):
    H, W = mask_2d.shape
    out  = np.full((H, W), fill, dtype=np.float32)
    out[mask_2d] = flat_vals
    return out


def plot_qualitative_panels(qual_frames, policy_name, ocean_mask, out_path):
    """
    For one policy, plot a multi-row panel: one row per qual step.
    Each row: [GT | FNO prior | GP correction | Corrected | GP std | Trajectory]
    """
    steps = sorted(qual_frames.keys())
    if not steps:
        return

    n_rows = len(steps)
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3.2 * n_rows),
                             squeeze=False)

    ssh_vmin, ssh_vmax = -0.9, 0.9
    err_abs = 0.25
    H, W = ocean_mask.shape

    for row, s in enumerate(steps):
        frame = qual_frames[s]
        y_true      = frame["y_true"]
        x_prior     = frame["x_prior"]
        x_corrected = frame["x_corrected"]
        gp_mean_map = frame["gp_mean_map"]   # (N_ocean,)
        gp_std_map  = frame["gp_std_map"]    # (N_ocean,)
        traj        = frame["trajectory"]    # (T, 2)
        obs_coords  = frame["obs_coords"]    # (K, 2)

        gp_mean_2d = _field_to_2d(gp_mean_map, ocean_mask)
        gp_std_2d  = _field_to_2d(gp_std_map,  ocean_mask)

        # Dynamic std colorbar: use p95 so tails don't compress the color range
        _std_vals = gp_std_map[np.isfinite(gp_std_map)]
        _noise_std = frame.get("gp_noise_std", None)
        if len(_std_vals) > 0:
            std_vmax = float(np.percentile(_std_vals, 95))
            std_vmax = max(std_vmax, 1e-6)   # guard against flat prior
            _std_stats = (f"med={np.median(_std_vals):.4f}  "
                          f"p95={np.percentile(_std_vals, 95):.4f}  "
                          f"max={_std_vals.max():.4f}")
            if _noise_std is not None:
                _std_stats += f"  noise_std={_noise_std:.4f}"
        else:
            std_vmax   = 0.10
            _std_stats = ""

        panels = [
            (y_true,      "GT SSH",          "RdBu_r", ssh_vmin, ssh_vmax),
            (x_prior,     "FNO prior error", "RdBu_r", -err_abs, err_abs),
            (gp_mean_2d,  "GP correction",   "RdBu_r", -err_abs, err_abs),
            (x_corrected, "Corrected error", "RdBu_r", -err_abs, err_abs),
            (gp_std_2d,   "GP std (uncert.)","YlOrRd",  0,       std_vmax),
        ]
        # Panels 1 and 3 are errors, not SSH
        for col, (field, title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row][col]
            if col == 1:
                field = x_prior - y_true
            elif col == 3:
                field = x_corrected - y_true

            im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax,
                           origin="upper", aspect="auto")
            ax.imshow(np.where(ocean_mask[..., None],
                               np.zeros((H, W, 4)),
                               np.array([[0.6, 0.6, 0.6, 1.0]])),
                      origin="upper", aspect="auto", alpha=0.5)
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

            # Overlay trajectory
            if traj is not None and len(traj) > 1:
                ax.plot(traj[:, 1] * W, traj[:, 0] * H,
                        "w-", linewidth=0.6, alpha=0.7)
            if obs_coords is not None and len(obs_coords) > 0:
                ax.scatter(obs_coords[:, 1] * W, obs_coords[:, 0] * H,
                           c="yellow", s=8, zorder=5, linewidths=0)

            if row == 0:
                ax.set_title(title, fontsize=9)
            # For the std panel, append stats to the row label
            if col == 4:
                ax.set_ylabel(f"Step {s}\n{_std_stats}", fontsize=7)
            else:
                ax.set_ylabel(f"Step {s}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"Dynamic rollout — {_policy_label(policy_name)}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Aggregate summary table image
# ---------------------------------------------------------------------------

def plot_summary_table(summary_df, fno_rmse, out_path):
    """Render summary table as a PNG."""
    disp = summary_df.copy()
    disp["policy"] = disp["policy"].map(_policy_label)
    disp = disp.round(4)

    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.45 * len(disp)))
    ax.axis("off")
    tbl = ax.table(
        cellText=disp.values,
        colLabels=disp.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.4)
    ax.set_title(f"Dynamic rollout summary  |  FNO RMSE = {fno_rmse:.4f}",
                 fontsize=10, pad=6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
