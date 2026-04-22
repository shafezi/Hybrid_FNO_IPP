"""For one config: load 5 method pkl files, generate 1 video + 5 timestep figures + 1 CSV.

Layout per timestep figure:
  Row 1 (fields, with robots/trajectory/Voronoi): GT | FNO-only | FNO+GP | Persist-only | Persist+GP | GP-only
  Row 2 (errors with metrics text):              [legend] | err1 | err2 | err3 | err4 | err5

Video: same layout, animated through L sub-steps with sub-frame interpolation.
"""
import argparse
import os
import sys
import time
import warnings
import pickle
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection

from _test_helpers import load_test_data
from make_comparison_video import (
    _rmse_over_ocean, _corr_over_ocean, _hf_energy_ratio, _fss_over_ocean,
    _field_to_display, _ensure_ffmpeg)


METHODS = ["FNO-only", "FNO+GP", "Persist-only", "Persist+GP", "GP-only"]


def method_safe(s):
    return s.replace(" ", "_").replace("+", "and")


def assign_region_per_ocean(ocean_mask, sites, H, W):
    """Voronoi region id per ocean cell. Returns (H, W) with NaN on land."""
    rows, cols = np.where(ocean_mask)
    coords = np.stack([rows / max(H - 1, 1), cols / max(W - 1, 1)], axis=1).astype(np.float32)
    dists = np.sum((coords[:, None, :] - sites[None, :, :]) ** 2, axis=2)
    region = np.argmin(dists, axis=1)
    region_img = np.full((H, W), np.nan, dtype=np.float32)
    region_img[rows, cols] = region.astype(np.float32)
    return region_img


def coord_to_pixel(coord, H, W):
    return coord[1] * (W - 1), coord[0] * (H - 1)


def draw_robots_voronoi(ax, robot_coords, full_traj, region_img, n_robots, robot_colors,
                       step_now, H, W, trail_len=10, marker_size=4, line_width=1.0):
    """Draw robots, trajectory trails, and Voronoi boundaries on an axis. Returns artist list."""
    artists = []

    # Voronoi boundaries
    if not np.all(np.isnan(region_img)):
        cs = ax.contour(region_img, levels=np.arange(n_robots - 1) + 0.5,
                       colors="black", linewidths=0.4, alpha=0.5)
        artists.append(cs)

    # For each robot
    for r in range(n_robots):
        traj = full_traj[r]  # (T+1, 2)
        history = traj[:step_now + 1]  # positions through current step

        # Trajectory trail (fade)
        if len(history) >= 2:
            xs = history[:, 1] * (W - 1)
            ys = history[:, 0] * (H - 1)
            trail_start = max(0, len(xs) - 1 - trail_len)
            xs = xs[trail_start:]
            ys = ys[trail_start:]
            if len(xs) >= 2:
                segs = np.stack([
                    np.column_stack([xs[:-1], ys[:-1]]),
                    np.column_stack([xs[1:],  ys[1:]]),
                ], axis=1)
                alphas_seg = np.linspace(0.2, 0.95, len(segs))
                base_rgb = robot_colors[r][:3]
                seg_colors = [(*base_rgb, a) for a in alphas_seg]
                lc = LineCollection(segs, colors=seg_colors, linewidths=line_width)
                ax.add_collection(lc)
                artists.append(lc)

        # Current position
        cx, cy = coord_to_pixel(history[-1], H, W)
        dot = ax.plot(cx, cy, marker="o", color=robot_colors[r],
                     markersize=marker_size, markeredgecolor="black",
                     markeredgewidth=0.4, linestyle="none", zorder=5)[0]
        artists.append(dot)

    return artists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Config tag, e.g., 10bots_matern15_uncertainty_only")
    parser.add_argument("--episodes_dir", type=str,
                        default=os.path.join(ROOT, "results", "dynamic_ipp", "final", "episodes"))
    parser.add_argument("--videos_dir", type=str,
                        default=os.path.join(ROOT, "results", "dynamic_ipp", "final", "videos"))
    parser.add_argument("--figures_dir", type=str,
                        default=os.path.join(ROOT, "results", "dynamic_ipp", "final", "figures"))
    parser.add_argument("--metrics_dir", type=str,
                        default=os.path.join(ROOT, "results", "dynamic_ipp", "final", "metrics"))
    parser.add_argument("--n_sub", type=int, default=4, help="Sub-frames per FNO step")
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()

    for d_ in [args.videos_dir, args.figures_dir, args.metrics_dir]:
        os.makedirs(d_, exist_ok=True)

    # Load 5 method pkl files
    data = {}
    for m in METHODS:
        path = os.path.join(args.episodes_dir, f"{args.config}_{method_safe(m)}.pkl")
        with open(path, "rb") as f:
            data[m] = pickle.load(f)

    L = data["FNO-only"]["L"]
    n_robots = data["FNO-only"]["n_robots"]
    t0 = data["FNO-only"]["t0"]

    # Load ocean mask (need it for plotting + metrics)
    d_test = load_test_data()
    ocean_mask = d_test["ocean_mask"]
    H, W = ocean_mask.shape

    # Per-step data extraction
    mask_land = lambda f: _field_to_display(f, ocean_mask)
    gt_by_step = {s: mask_land(data["FNO-only"]["qual_frames"][s]["y_true"])
                  for s in range(1, L + 1)}

    pred_by_step = {m: {} for m in METHODS}
    err_by_step = {m: {} for m in METHODS}
    metrics_per_step = {m: [] for m in METHODS}

    for m in METHODS:
        qf = data[m]["qual_frames"]
        for s in range(1, L + 1):
            p = qf[s]["x_corrected"]
            g = qf[s]["y_true"]
            pred_by_step[m][s] = mask_land(p)
            err_by_step[m][s] = mask_land(p - g)
            metrics_per_step[m].append({
                "step": s,
                "RMSE": _rmse_over_ocean(p, g, ocean_mask),
                "ACC":  _corr_over_ocean(p, g, ocean_mask),
                "HF":   _hf_energy_ratio(p, g, ocean_mask),
                "FSS":  _fss_over_ocean(p, g, ocean_mask),
            })

    # Avg RMSE = running mean
    for m in METHODS:
        rmses = [r["RMSE"] for r in metrics_per_step[m]]
        for i, r in enumerate(metrics_per_step[m]):
            r["AvgRMSE"] = float(np.mean(rmses[:i + 1]))

    # Save CSV
    csv_rows = []
    for m in METHODS:
        for r in metrics_per_step[m]:
            row = dict(r)
            row["method"] = m
            row["config"] = args.config
            csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(
        os.path.join(args.metrics_dir, f"{args.config}.csv"), index=False)

    # Methods that use a policy (have robots/trajectories/Voronoi)
    POLICY_METHODS = ["FNO+GP", "Persist+GP", "GP-only"]

    # Robot positions per step, per method (each policy method has its own trajectory)
    robot_pos_by_step = {m: {} for m in POLICY_METHODS}
    full_traj_by_method = {}
    region_img_by_step = {m: {} for m in POLICY_METHODS}

    for m in POLICY_METHODS:
        for s in range(1, L + 1):
            positions = np.stack(
                [data[m]["qual_frames"][s]["trajectories"][r][-1]
                 for r in range(n_robots)], axis=0)
            robot_pos_by_step[m][s] = positions
        robot_pos_by_step[m][0] = np.stack(
            [data[m]["qual_frames"][1]["trajectories"][r][0]
             for r in range(n_robots)], axis=0)
        full_traj_by_method[m] = data[m]["trajectories"]

        # Voronoi maps per step for this method
        for s in range(0, L + 1):
            sites_s = robot_pos_by_step[m][s]
            region_img_by_step[m][s] = assign_region_per_ocean(
                ocean_mask, sites_s, H, W)

    # Color ranges
    gt_vals = np.concatenate(
        [data["FNO-only"]["qual_frames"][s]["y_true"][ocean_mask]
         for s in range(1, L + 1)])
    ssh_vmin, ssh_vmax = np.percentile(gt_vals, [2, 98]).tolist()
    err_vals = []
    for m in METHODS:
        for s in range(1, L + 1):
            err_vals.append(
                (data[m]["qual_frames"][s]["x_corrected"]
                 - data[m]["qual_frames"][s]["y_true"])[ocean_mask])
    err_vals = np.concatenate(err_vals)
    err_absmax = float(np.percentile(np.abs(err_vals), 98))

    # Robot colors
    robot_colors = plt.cm.tab10(np.arange(n_robots) % 10)

    cmap_ssh = plt.cm.viridis.copy(); cmap_ssh.set_bad("lightgrey")
    cmap_err = plt.cm.RdBu_r.copy();  cmap_err.set_bad("lightgrey")
    kw_top = dict(cmap=cmap_ssh, vmin=ssh_vmin, vmax=ssh_vmax,
                  origin="upper", aspect="auto", interpolation="nearest")
    kw_bot = dict(cmap=cmap_err, vmin=-err_absmax, vmax=err_absmax,
                  origin="upper", aspect="auto", interpolation="nearest")

    # ── Generate 5 timestep figures ─────────────────────────────────────
    for s in range(1, L + 1):
        n_cols = 6  # GT + 5 methods
        fig = plt.figure(figsize=(4.0 * n_cols, 7.5))
        gs = fig.add_gridspec(
            2, n_cols + 1,
            width_ratios=[1] * n_cols + [0.04],
            wspace=0.05, hspace=0.10,
            left=0.02, right=0.96, top=0.90, bottom=0.04)

        ax_gt = fig.add_subplot(gs[0, 0])
        ax_preds = [fig.add_subplot(gs[0, i + 1]) for i in range(5)]
        ax_cb_top = fig.add_subplot(gs[0, n_cols])
        # NOTE: no legend cell under GT — leave that grid cell empty
        ax_errs = [fig.add_subplot(gs[1, i + 1]) for i in range(5)]
        ax_cb_bot = fig.add_subplot(gs[1, n_cols])

        for ax in [ax_gt] + ax_preds + ax_errs:
            ax.set_xticks([]); ax.set_yticks([])

        # Plot GT (no robot overlay - robots are method-specific)
        ax_gt.imshow(gt_by_step[s], **kw_top)
        ax_gt.set_title("Ground Truth", fontsize=11, fontweight="bold")

        for i, m in enumerate(METHODS):
            ax_preds[i].imshow(pred_by_step[m][s], **kw_top)
            ax_preds[i].set_title(m, fontsize=10, fontweight="bold")
            if m in POLICY_METHODS:
                draw_robots_voronoi(ax_preds[i], robot_pos_by_step[m][s],
                                  full_traj_by_method[m],
                                  region_img_by_step[m][s], n_robots,
                                  robot_colors, s, H, W)

            im_err = ax_errs[i].imshow(err_by_step[m][s], **kw_bot)
            ax_errs[i].set_title(f"{m} − GT", fontsize=9)

            r = metrics_per_step[m][s - 1]
            txt_kw = dict(fontsize=9, fontweight="bold", color="white",
                          path_effects=[path_effects.withStroke(linewidth=2, foreground="black")])
            ax_errs[i].text(0.02, 0.36, f"RMSE = {r['RMSE']:.3f}",
                          transform=ax_errs[i].transAxes, **txt_kw)
            ax_errs[i].text(0.02, 0.28, f"Avg RMSE = {r['AvgRMSE']:.3f}",
                          transform=ax_errs[i].transAxes, **txt_kw)
            ax_errs[i].text(0.02, 0.20, f"ACC = {r['ACC']:.3f}",
                          transform=ax_errs[i].transAxes, **txt_kw)
            ax_errs[i].text(0.02, 0.12, f"HF = {r['HF']:.3f}",
                          transform=ax_errs[i].transAxes, **txt_kw)
            ax_errs[i].text(0.02, 0.04, f"FSS = {r['FSS']:.3f}",
                          transform=ax_errs[i].transAxes, **txt_kw)

        plt.colorbar(ax_gt.images[0], cax=ax_cb_top).set_label("SSH (norm.)", fontsize=9)
        plt.colorbar(im_err, cax=ax_cb_bot).set_label("error", fontsize=9)

        fig.suptitle(f"{n_robots} robots   |   step {s}   |   day {s * 4}",
                    fontsize=14, fontweight="bold", y=0.98)
        out_fig = os.path.join(args.figures_dir, f"{args.config}_step{s}.png")
        plt.savefig(out_fig, dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ── Generate video (animated through L*n_sub frames) ──────────────────
    n_frames = L * args.n_sub
    fig = plt.figure(figsize=(4.0 * 6, 7.5))
    gs = fig.add_gridspec(2, 7, width_ratios=[1] * 6 + [0.04],
                          wspace=0.05, hspace=0.10,
                          left=0.02, right=0.96, top=0.90, bottom=0.04)
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_preds = [fig.add_subplot(gs[0, i + 1]) for i in range(5)]
    ax_cb_top = fig.add_subplot(gs[0, 6])
    # NOTE: no legend cell under GT
    ax_errs = [fig.add_subplot(gs[1, i + 1]) for i in range(5)]
    ax_cb_bot = fig.add_subplot(gs[1, 6])

    for ax in [ax_gt] + ax_preds + ax_errs:
        ax.set_xticks([]); ax.set_yticks([])

    im_gt = ax_gt.imshow(gt_by_step[1], **kw_top)
    ax_gt.set_title("Ground Truth", fontsize=11, fontweight="bold")
    im_preds = []
    im_errs = []
    for i, m in enumerate(METHODS):
        im_preds.append(ax_preds[i].imshow(pred_by_step[m][1], **kw_top))
        ax_preds[i].set_title(m, fontsize=10, fontweight="bold")
        im_errs.append(ax_errs[i].imshow(err_by_step[m][1], **kw_bot))
        ax_errs[i].set_title(f"{m} − GT", fontsize=9)
    plt.colorbar(im_gt, cax=ax_cb_top).set_label("SSH (norm.)", fontsize=9)
    plt.colorbar(im_errs[0], cax=ax_cb_bot).set_label("error", fontsize=9)

    txt_kw = dict(fontsize=9, fontweight="bold", color="white",
                  path_effects=[path_effects.withStroke(linewidth=2, foreground="black")])
    txt_metrics = []
    for i in range(5):
        method_txts = []
        for j, _ in enumerate(["RMSE", "Avg RMSE", "ACC", "HF", "FSS"]):
            y_pos = 0.36 - j * 0.08
            t = ax_errs[i].text(0.02, y_pos, "", transform=ax_errs[i].transAxes, **txt_kw)
            method_txts.append(t)
        txt_metrics.append(method_txts)

    suptitle = fig.suptitle("", fontsize=14, fontweight="bold", y=0.98)

    # Container for dynamic robot/voronoi artists
    dynamic_artists = {ax: [] for ax in [ax_gt] + ax_preds + ax_errs}

    def frame_context(f):
        step_next = f // args.n_sub + 1
        step_prev = step_next - 1
        alpha = (f % args.n_sub) / args.n_sub
        return step_prev, step_next, alpha

    def update(frame_idx):
        step_prev, step_next, alpha = frame_context(frame_idx)
        sp = max(step_prev, 1); sn = step_next

        def lerp(a, b, t): return (1 - t) * a + t * b

        im_gt.set_data(lerp(gt_by_step[sp], gt_by_step[sn], alpha))
        for i, m in enumerate(METHODS):
            im_preds[i].set_data(lerp(pred_by_step[m][sp], pred_by_step[m][sn], alpha))
            im_errs[i].set_data(lerp(err_by_step[m][sp], err_by_step[m][sn], alpha))

            # Lerp metrics
            r0 = metrics_per_step[m][sp - 1]
            r1 = metrics_per_step[m][sn - 1]
            for j, key in enumerate(["RMSE", "AvgRMSE", "ACC", "HF", "FSS"]):
                v = lerp(r0[key], r1[key], alpha)
                lbl = "Avg RMSE" if key == "AvgRMSE" else key
                txt_metrics[i][j].set_text(f"{lbl} = {v:.3f}")

        # Clear and redraw robot/voronoi
        for ax, arts in dynamic_artists.items():
            for art in arts:
                try:
                    art.remove()
                except Exception:
                    if hasattr(art, "collections"):
                        for c in art.collections:
                            try: c.remove()
                            except: pass
            dynamic_artists[ax] = []

        # Per-method robot drawing (each policy method has its own trajectory)
        for i, m in enumerate(METHODS):
            ax = ax_preds[i]
            arts = []
            if m in POLICY_METHODS:
                # Interpolate robot positions for this method
                pos_prev = robot_pos_by_step[m][step_prev]
                pos_next = robot_pos_by_step[m][step_next]
                pos_now = (1 - alpha) * pos_prev + alpha * pos_next

                # Voronoi region for this method
                region_now = (region_img_by_step[m][step_prev]
                             if alpha < 0.5 else region_img_by_step[m][step_next])
                if not np.all(np.isnan(region_now)):
                    cs = ax.contour(region_now,
                                  levels=np.arange(n_robots - 1) + 0.5,
                                  colors="black", linewidths=0.4, alpha=0.5)
                    arts.append(cs)

                # Trajectories for this method
                full_traj_m = full_traj_by_method[m]
                for r in range(n_robots):
                    traj = full_traj_m[r]
                    history = traj[:step_prev + 1]
                    if len(history) >= 1:
                        hist_ext = np.vstack([history, pos_now[r:r + 1]])
                        xs = hist_ext[:, 1] * (W - 1)
                        ys = hist_ext[:, 0] * (H - 1)
                        if len(xs) >= 2:
                            segs = np.stack([
                                np.column_stack([xs[:-1], ys[:-1]]),
                                np.column_stack([xs[1:],  ys[1:]]),
                            ], axis=1)
                            alphas_seg = np.linspace(0.2, 0.95, len(segs))
                            seg_colors = [(*robot_colors[r][:3], a) for a in alphas_seg]
                            lc = LineCollection(segs, colors=seg_colors, linewidths=1.0)
                            ax.add_collection(lc)
                            arts.append(lc)
                    cx, cy = coord_to_pixel(pos_now[r], H, W)
                    dot = ax.plot(cx, cy, marker="o", color=robot_colors[r],
                                markersize=4, markeredgecolor="black",
                                markeredgewidth=0.4, linestyle="none", zorder=5)[0]
                    arts.append(dot)
            dynamic_artists[ax] = arts

        day_idx = int(step_prev * 4 + alpha * 4)
        # Clean title: just robots, step (use range during transition), day
        if step_prev == 0:
            step_str = f"step {sn}"
        elif alpha < 1e-6:
            step_str = f"step {sp}"
        else:
            step_str = f"step {sp}-{sn}"
        suptitle.set_text(
            f"{n_robots} robots   |   {step_str}   |   day {day_idx}")

        all_txt = [t for mt in txt_metrics for t in mt]
        return [im_gt] + im_preds + im_errs + all_txt + [suptitle]

    update(0)
    mp4_path = os.path.join(args.videos_dir, f"{args.config}.mp4")
    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False,
                                  interval=1000 / args.fps)
    if _ensure_ffmpeg():
        anim.save(mp4_path, writer=animation.FFMpegWriter(fps=args.fps, bitrate=4000))
    plt.close(fig)
    print(f"Done: {args.config}")
    print(f"  Video:   {mp4_path}")
    print(f"  Figures: {args.figures_dir}/{args.config}_step*.png")
    print(f"  CSV:     {args.metrics_dir}/{args.config}.csv")


if __name__ == "__main__":
    main()
