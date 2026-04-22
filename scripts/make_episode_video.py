"""
Render a 5-panel animation of one rollout episode.

Layout (2 rows × 3 cols):

    ┌──────────┬──────────┬──────────┐
    │   GT     │  FNO     │  Hybrid  │  ← shared SSH colorbar
    │          │          │ +Voronoi │
    ├──────────┼──────────┼──────────┤
    │  legend  │ FNO-GT   │ hybrid-GT│  ← shared error colorbar (RdBu_r)
    │          │ RMSE=…   │ RMSE=…   │
    └──────────┴──────────┴──────────┘

Robots on the right panels: stable per-robot colours, fading trails,
observation markers, name labels.

Usage:
    python scripts/make_episode_video.py [--t0 N] [--n_sub 4] [--fps 5]

Output:
    results/dynamic_ipp/videos/episode_t0_<N>.mp4  (if ffmpeg available)
    results/dynamic_ipp/videos/episode_t0_<N>.gif
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
warnings.filterwarnings("ignore")

import yaml

from _test_helpers import load_test_data
from ipp.partitioning import build_voronoi_partition
from ipp.policies import build_policies
from ipp.simulator import build_candidates, build_eval_cells
from dynamic_ipp.rollout import run_dynamic_episode, _stable_hash


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _rmse_over_ocean(a, b, ocean_mask):
    diff = (a - b)[ocean_mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def _field_to_display(field, ocean_mask):
    """Return a copy with land set to NaN (for matplotlib's cmap.set_bad grey)."""
    disp = field.copy().astype(np.float32)
    disp[~ocean_mask] = np.nan
    return disp


def _compute_t0_list(cfg, N_samples):
    """Reproduce the same t0_list that run_all_dynamic_experiments uses."""
    L = cfg.get("episode_length", 15)
    rollout_lead_stride = cfg.get("rollout_lead_stride", cfg.get("lead", 1))
    max_t0 = N_samples - L * rollout_lead_stride - 1
    if max_t0 < 0:
        raise ValueError("Not enough samples for one episode.")
    n_ep = cfg.get("n_episodes", 50)
    seed = cfg.get("episode_seed_offset", 0)
    rng_ep = np.random.default_rng(seed)
    t0_list = sorted(
        rng_ep.choice(max_t0 + 1, size=min(n_ep, max_t0 + 1),
                      replace=False).tolist()
    )
    return t0_list


def _seed_for_policy(cfg, ep_i, pol_name):
    """Same formula used by run_all_dynamic_experiments."""
    return (cfg.get("episode_seed_offset", 0)
            + ep_i * 1000
            + _stable_hash(pol_name) % 1000)


def _assign_region_per_ocean(ocean_mask, sites, H, W):
    """
    For each ocean pixel, find its nearest Voronoi site.
    Returns region_img (H, W) float: region id on ocean, NaN on land.
    """
    rows, cols = np.where(ocean_mask)
    # Normalize to match candidate-coord convention
    coords = np.stack(
        [rows / max(H - 1, 1), cols / max(W - 1, 1)], axis=1
    ).astype(np.float32)
    # Pairwise squared distances (N_ocean, n_sites)
    dists = np.sum(
        (coords[:, None, :] - sites[None, :, :]) ** 2, axis=2)
    region = np.argmin(dists, axis=1)
    region_img = np.full((H, W), np.nan, dtype=np.float32)
    region_img[rows, cols] = region.astype(np.float32)
    return region_img


def _robot_pos_from_snapshot(snapshot, r):
    """Last trajectory point for robot r in a qual_frames[s] snapshot."""
    traj = snapshot["trajectories"][r]
    return np.asarray(traj[-1], dtype=np.float32)


def _ensure_ffmpeg():
    """Try to make ffmpeg available via imageio-ffmpeg; return True if MP4 writable."""
    if animation.FFMpegWriter.isAvailable():
        return True
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        return animation.FFMpegWriter.isAvailable()
    except Exception as e:
        print(f"  (ffmpeg unavailable: {e})")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0", type=int, default=-1,
                        help="Day-of-year start (-1 = use first seeded t0)")
    parser.add_argument("--n_sub", type=int, default=4,
                        help="Sub-frames per FNO step (4 × 15 = 60 total)")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(ROOT, "results", "dynamic_ipp", "videos"))
    parser.add_argument("--also_gif", action="store_true",
                        help="Also render GIF (MP4 always rendered unless --gif_only)")
    parser.add_argument("--gif_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    cfg_path = os.path.join(ROOT, "configs", "dynamic_rollout_ipp.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["n_episodes"] = 1
    cfg["debug"] = False

    L = cfg["episode_length"]
    n_robots = cfg["n_robots"]
    stride = cfg.get("rollout_lead_stride", cfg.get("lead", 1))

    print(f"Loading data + FNO …")
    d = load_test_data()
    inputs_np = d["inputs_np"]
    ocean_mask = d["ocean_mask"]
    all_ocean_coords = d["all_ocean_coords"]
    fno = d["fno"]
    device = d["device"]
    H, W = ocean_mask.shape

    # Build the shared candidate/eval sets
    cand_local_idx = build_candidates(
        all_ocean_coords,
        cfg.get("n_candidates", 2000),
        cfg.get("candidate_seed", 42))
    eval_local_idx = build_eval_cells(
        all_ocean_coords,
        cfg.get("n_eval_cells", 20000),
        cfg.get("eval_seed", 123))
    cand_coords = all_ocean_coords[cand_local_idx]

    # Pick t0
    t0_list = _compute_t0_list(cfg, inputs_np.shape[0])
    if args.t0 >= 0:
        t0 = args.t0
    else:
        t0 = t0_list[0]
    print(f"Using t0 = {t0} (day-of-year 2020)")

    # Episode seed (same formula as run_all_dynamic_experiments)
    pol_name = "uncertainty_only"
    episode_seed = _seed_for_policy(cfg, ep_i=0, pol_name=pol_name)
    print(f"episode_seed = {episode_seed}")

    # NOTE: Voronoi is now DYNAMIC (sites = current robot positions each step).
    # The initial seeded Voronoi is only used inside the rollout for cold-start
    # placement; after each step it's recomputed from robot positions.
    part_seed = episode_seed + cfg.get("partition_seed_offset", 0)
    region_id_initial, sites_initial = build_voronoi_partition(
        cand_coords, n_robots, seed=part_seed)

    # Build policy
    built = build_policies({"policies": {
        pol_name: {"type": pol_name, "lambda_dist": 0.0}
    }})
    policy = built[pol_name]

    # Run the episode with every step saved
    print(f"Running episode (this takes ~5 min on CPU) …")
    t0_wall = time.time()
    step_records, traj_arrays, qual_frames = run_dynamic_episode(
        inputs_np, ocean_mask, all_ocean_coords,
        cand_local_idx, eval_local_idx,
        policy, fno, device, cfg,
        episode_seed=episode_seed, t0=t0,
        save_qual_steps=set(range(1, L + 1)),
    )
    print(f"Episode complete in {time.time() - t0_wall:.0f}s")
    print(f"  Steps recorded: {sorted(qual_frames.keys())}")
    print(f"  Final RMSE (hybrid): {step_records[-1]['all_rmse']:.4f}")
    print(f"  Final RMSE (FNO):    {step_records[-1]['fno_rmse']:.4f}")

    # Per-step data
    def mask_land(f):
        return _field_to_display(f, ocean_mask)

    gt_by_step      = {s: mask_land(qual_frames[s]["y_true"])      for s in range(1, L+1)}
    fno_by_step     = {s: mask_land(qual_frames[s]["x_fno"])       for s in range(1, L+1)}
    hybrid_by_step  = {s: mask_land(qual_frames[s]["x_corrected"]) for s in range(1, L+1)}
    err_fno_by_step = {
        s: mask_land(qual_frames[s]["x_fno"] - qual_frames[s]["y_true"])
        for s in range(1, L+1)
    }
    err_hyb_by_step = {
        s: mask_land(qual_frames[s]["x_corrected"] - qual_frames[s]["y_true"])
        for s in range(1, L+1)
    }
    rmse_fno_by_step = {
        s: _rmse_over_ocean(qual_frames[s]["x_fno"],
                            qual_frames[s]["y_true"], ocean_mask)
        for s in range(1, L+1)
    }
    rmse_hyb_by_step = {
        s: _rmse_over_ocean(qual_frames[s]["x_corrected"],
                            qual_frames[s]["y_true"], ocean_mask)
        for s in range(1, L+1)
    }

    # Robot positions at each step: last entry of the snapshot's trajectories
    # (trajectories[r] has length s+1 at snapshot s — index 0 is cold-start,
    # index s is the step-s sample location)
    robot_pos_by_step = {}
    for s in range(1, L + 1):
        positions = np.stack(
            [qual_frames[s]["trajectories"][r][-1] for r in range(n_robots)],
            axis=0,
        )
        robot_pos_by_step[s] = positions
    # Step 0 = initial positions (cold-start, same as step-1 pos in lookahead mode)
    robot_pos_by_step[0] = np.stack(
        [qual_frames[1]["trajectories"][r][0] for r in range(n_robots)],
        axis=0,
    )

    # Full trajectory through step s (per robot)
    # Use the final episode trajectories and slice to step s
    full_traj = traj_arrays  # list of (L+1, 2) arrays (initial + L sample positions)

    # Per-step Voronoi region map: sites = robot positions at that step.
    # region_img_by_step[s] is (H, W) with region id on ocean, NaN on land.
    region_img_by_step = {}
    for s in range(0, L + 1):
        sites_s = robot_pos_by_step[s]   # (n_robots, 2) in normalized coords
        region_img_by_step[s] = _assign_region_per_ocean(
            ocean_mask, sites_s, H, W)

    # Colorbar ranges (fixed across frames).  Use GT-only 2/98 percentile
    # for SSH and hybrid-error 98 percentile for error, so FNO blow-ups
    # don't dominate the colorbar.  Saturation at extremes is desirable
    # (it shows "the FNO blew up here").
    gt_vals = np.concatenate(
        [qual_frames[s]["y_true"][ocean_mask] for s in range(1, L+1)])
    ssh_vmin, ssh_vmax = np.percentile(gt_vals, [2, 98]).tolist()

    err_hyb_vals = np.concatenate([
        (qual_frames[s]["x_corrected"] - qual_frames[s]["y_true"])[ocean_mask]
        for s in range(1, L+1)
    ])
    err_absmax = float(np.percentile(np.abs(err_hyb_vals), 98))

    print(f"  SSH range:   [{ssh_vmin:.3f}, {ssh_vmax:.3f}]")
    print(f"  Error range: ±{err_absmax:.3f}")

    # Running-average RMSE (mean of step 1..s).  Indexed by step s ∈ [1, L].
    avg_rmse_fno_by_step = {}
    avg_rmse_hyb_by_step = {}
    for s in range(1, L + 1):
        avg_rmse_fno_by_step[s] = float(np.mean(
            [rmse_fno_by_step[ss] for ss in range(1, s + 1)]))
        avg_rmse_hyb_by_step[s] = float(np.mean(
            [rmse_hyb_by_step[ss] for ss in range(1, s + 1)]))

    # Robot colours (tab10)
    robot_colors = plt.cm.tab10(np.arange(n_robots) % 10)

    # --- Interpolated frames ---
    n_frames = L * args.n_sub
    print(f"Building {n_frames} frames …")

    # Helper: for frame index f, determine (step_prev, step_next, alpha)
    def frame_context(f):
        step_next = f // args.n_sub + 1            # in [1, L]
        step_prev = step_next - 1                   # in [0, L-1]
        sub = f % args.n_sub                        # in [0, n_sub-1]
        alpha = sub / args.n_sub
        return step_prev, step_next, alpha

    # --- Build figure ---
    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(
        2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.08, hspace=0.15,
        left=0.02, right=0.96, top=0.87, bottom=0.05)

    ax_gt   = fig.add_subplot(gs[0, 0])
    ax_fno  = fig.add_subplot(gs[0, 1])
    ax_hyb  = fig.add_subplot(gs[0, 2])
    ax_cb_top = fig.add_subplot(gs[0, 3])

    ax_legend = fig.add_subplot(gs[1, 0]); ax_legend.axis("off")
    ax_err_fno = fig.add_subplot(gs[1, 1])
    ax_err_hyb = fig.add_subplot(gs[1, 2])
    ax_cb_bot = fig.add_subplot(gs[1, 3])

    for ax in [ax_gt, ax_fno, ax_hyb, ax_err_fno, ax_err_hyb]:
        ax.set_xticks([]); ax.set_yticks([])

    # Colormaps
    cmap_ssh = plt.cm.viridis.copy(); cmap_ssh.set_bad("lightgrey")
    cmap_err = plt.cm.RdBu_r.copy();  cmap_err.set_bad("lightgrey")
    cmap_region = plt.cm.tab10

    # imshow handles (updated each frame)
    kw_top = dict(cmap=cmap_ssh, vmin=ssh_vmin, vmax=ssh_vmax,
                  origin="upper", aspect="auto", interpolation="nearest")
    kw_bot = dict(cmap=cmap_err, vmin=-err_absmax, vmax=err_absmax,
                  origin="upper", aspect="auto", interpolation="nearest")

    im_gt   = ax_gt.imshow(  gt_by_step[1],     **kw_top)
    im_fno  = ax_fno.imshow( fno_by_step[1],    **kw_top)
    im_hyb  = ax_hyb.imshow( hybrid_by_step[1], **kw_top)
    im_err_fno = ax_err_fno.imshow(err_fno_by_step[1], **kw_bot)
    im_err_hyb = ax_err_hyb.imshow(err_hyb_by_step[1], **kw_bot)

    # Voronoi overlay: boundaries only (not filled) — drawn per-frame below.

    # Shared colorbars
    cb_top = fig.colorbar(im_hyb, cax=ax_cb_top); cb_top.set_label("SSH (norm.)", fontsize=9)
    cb_bot = fig.colorbar(im_err_hyb, cax=ax_cb_bot); cb_bot.set_label("error", fontsize=9)

    # Titles
    ax_gt.set_title("Ground Truth",     fontsize=11)
    ax_fno.set_title("FNO only",        fontsize=11)
    ax_hyb.set_title("FNO + GP hybrid", fontsize=11)
    ax_err_fno.set_title("FNO − GT",    fontsize=10)
    ax_err_hyb.set_title("Hybrid − GT", fontsize=10)

    # Static legend panel (adapts to many robots)
    ax_legend.text(0.02, 0.97, f"{n_robots} Robots",
                   transform=ax_legend.transAxes,
                   fontsize=12, fontweight="bold", va="top")
    # Dot grid showing colors (tab10 wraps after 10)
    if n_robots <= 15:
        # Vertical list with labels
        for r in range(n_robots):
            y = 0.88 - r * (0.80 / max(1, n_robots - 1))
            ax_legend.scatter([0.08], [y], s=55, color=robot_colors[r],
                              transform=ax_legend.transAxes, edgecolor="black",
                              linewidth=0.6, zorder=3)
            ax_legend.text(0.15, y, f"R{r}", transform=ax_legend.transAxes,
                           fontsize=9, va="center")
    else:
        # Compact color grid (10 per row) — shows there are many robots
        for r in range(n_robots):
            row, col = r // 10, r % 10
            x = 0.08 + col * 0.085
            y = 0.82 - row * 0.09
            ax_legend.scatter([x], [y], s=45, color=robot_colors[r],
                              transform=ax_legend.transAxes, edgecolor="black",
                              linewidth=0.4, zorder=3)
        ax_legend.text(0.02, 0.83 - ((n_robots // 10) + 1) * 0.09,
                       "(colors repeat every 10 robots)",
                       transform=ax_legend.transAxes, fontsize=8, va="center",
                       color="gray")

    # Symbol key at the bottom of the legend panel
    y0 = 0.30
    ax_legend.text(
        0.02, y0, "-- fading trail", transform=ax_legend.transAxes,
        fontsize=9, va="center")
    ax_legend.text(
        0.02, y0 - 0.08, "+   observation", transform=ax_legend.transAxes,
        fontsize=9, va="center")
    ax_legend.text(
        0.02, y0 - 0.16, "●   current pos", transform=ax_legend.transAxes,
        fontsize=9, va="center")

    # Containers for dynamic artists (we'll clear + redraw each frame)
    dynamic_artists = {"ax_hyb": [], "ax_err_hyb": []}

    # RMSE text (bottom-left of each error panel)
    rmse_txt_kw = dict(
        fontsize=11, fontweight="bold", color="white",
        path_effects=[path_effects.withStroke(linewidth=2, foreground="black")],
    )
    txt_rmse_fno = ax_err_fno.text(
        0.02, 0.10, "", transform=ax_err_fno.transAxes, **rmse_txt_kw)
    txt_rmse_hyb = ax_err_hyb.text(
        0.02, 0.10, "", transform=ax_err_hyb.transAxes, **rmse_txt_kw)
    txt_avg_fno = ax_err_fno.text(
        0.02, 0.03, "", transform=ax_err_fno.transAxes, **rmse_txt_kw)
    txt_avg_hyb = ax_err_hyb.text(
        0.02, 0.03, "", transform=ax_err_hyb.transAxes, **rmse_txt_kw)

    # Top banner / suptitle
    start_day = np.datetime64("2020-01-01") + np.timedelta64(int(t0), "D")
    suptitle = fig.suptitle("", fontsize=13, fontweight="bold", y=0.97)

    # ---- Frame update -----------------------------------------------------
    def _coord_to_pixel(coord):
        """(row_norm, col_norm) in [0,1]^2 -> (x_pixel, y_pixel) for imshow."""
        x = coord[1] * (W - 1)
        y = coord[0] * (H - 1)
        return x, y

    def update(frame_idx):
        step_prev, step_next, alpha = frame_context(frame_idx)
        # step_next in [1..L]. step_prev in [0..L-1].
        # At alpha=0: show state at step_prev; at alpha=1: show state at step_next.
        # For step_prev=0 we use the FIRST step's fields (the FNO has already
        # advanced 4 days by step 1; nothing observed at day 0).
        sp = max(step_prev, 1)
        sn = step_next

        # Interpolated fields
        def lerp(a, b, t):
            return (1 - t) * a + t * b

        gt_f   = lerp(gt_by_step[sp],     gt_by_step[sn],     alpha)
        fno_f  = lerp(fno_by_step[sp],    fno_by_step[sn],    alpha)
        hyb_f  = lerp(hybrid_by_step[sp], hybrid_by_step[sn], alpha)
        e_fno  = lerp(err_fno_by_step[sp], err_fno_by_step[sn], alpha)
        e_hyb  = lerp(err_hyb_by_step[sp], err_hyb_by_step[sn], alpha)
        r_fno  = lerp(rmse_fno_by_step[sp], rmse_fno_by_step[sn], alpha)
        r_hyb  = lerp(rmse_hyb_by_step[sp], rmse_hyb_by_step[sn], alpha)

        im_gt.set_data(gt_f)
        im_fno.set_data(fno_f)
        im_hyb.set_data(hyb_f)
        im_err_fno.set_data(e_fno)
        im_err_hyb.set_data(e_hyb)

        txt_rmse_fno.set_text(f"RMSE = {r_fno:.3f}")
        txt_rmse_hyb.set_text(f"RMSE = {r_hyb:.3f}")

        # Running-average RMSE (mean over steps 1..current)
        a_fno = lerp(avg_rmse_fno_by_step[sp], avg_rmse_fno_by_step[sn], alpha)
        a_hyb = lerp(avg_rmse_hyb_by_step[sp], avg_rmse_hyb_by_step[sn], alpha)
        txt_avg_fno.set_text(f"Avg RMSE = {a_fno:.3f}")
        txt_avg_hyb.set_text(f"Avg RMSE = {a_hyb:.3f}")

        # Clear previous dynamic artists (lines, dots, markers, contours)
        for key in dynamic_artists:
            for art in dynamic_artists[key]:
                try:
                    art.remove()
                except Exception:
                    # Older QuadContourSet: remove each collection manually
                    if hasattr(art, "collections"):
                        for coll in art.collections:
                            try:
                                coll.remove()
                            except Exception:
                                pass
            dynamic_artists[key] = []

        # Interpolate region image linearly between step_prev and step_next.
        # Since region_id is a discrete int field, interpolation doesn't really
        # make sense; pick the one closer in time.
        region_img_now = (region_img_by_step[step_prev]
                          if alpha < 0.5
                          else region_img_by_step[step_next])

        # Interpolated robot positions
        pos_prev = robot_pos_by_step[step_prev]     # (n_robots, 2)
        pos_next = robot_pos_by_step[step_next]
        pos_now  = (1 - alpha) * pos_prev + alpha * pos_next

        # For each robot, draw on BOTH right-column panels:
        #   - trail (past steps 0..step_prev) as LineCollection with fade
        #   - observation markers at every COMPLETED sampling location
        #     (steps 1..step_prev if alpha < 1, else 1..step_next)
        #   - current position dot + "R?" label
        completed_steps = step_prev if alpha < 1e-6 else step_prev
        # Simpler: show markers for steps 1..step_prev always, and ADD the
        # step_next marker once alpha>=0.99 (robot has arrived)
        markers_up_to = step_prev
        if alpha > 0.99:
            markers_up_to = step_next

        # Trail cap still applies (keeps fading trail readable) but markers
        # are shown in full and on both panels.
        TRAIL_HISTORY   = 5 if n_robots > 15 else 25  # steps of trail to show
        MARKER_HISTORY  = 9999                         # show all obs markers
        SHOW_OBS_ON_ERR = True                         # obs on both panels

        for ax_key, ax in [("ax_hyb", ax_hyb), ("ax_err_hyb", ax_err_hyb)]:
            # Voronoi boundaries as a single-colour contour
            if not np.all(np.isnan(region_img_now)):
                cs = ax.contour(
                    region_img_now,
                    levels=np.arange(n_robots - 1) + 0.5,
                    colors="black", linewidths=0.5, alpha=0.45)
                dynamic_artists[ax_key].append(cs)

            draw_obs_markers = (ax_key == "ax_hyb") or SHOW_OBS_ON_ERR

            for r in range(n_robots):
                traj = full_traj[r]                 # (L+1, 2)
                # Positions up through step_prev (inclusive)
                history = traj[: step_prev + 1]     # shape (step_prev+1, 2)
                history_ext = np.vstack([history, pos_now[r:r+1]])
                xs = history_ext[:, 1] * (W - 1)
                ys = history_ext[:, 0] * (H - 1)

                # Fading trail (cap to TRAIL_HISTORY most recent segments)
                if len(xs) >= 2:
                    trail_start = max(0, len(xs) - 1 - TRAIL_HISTORY)
                    segs = np.stack(
                        [np.column_stack([xs[trail_start:-1],
                                          ys[trail_start:-1]]),
                         np.column_stack([xs[trail_start + 1:],
                                          ys[trail_start + 1:]])], axis=1)
                    alphas_seg = np.linspace(0.10, 0.95, len(segs))
                    base_rgb = robot_colors[r][:3]
                    seg_colors = [(*base_rgb, a) for a in alphas_seg]
                    lc = LineCollection(segs, colors=seg_colors, linewidths=1.2)
                    ax.add_collection(lc)
                    dynamic_artists[ax_key].append(lc)

                # Observation markers (cap to last MARKER_HISTORY per robot).
                # Many-robot mode hides obs markers on the error panel.
                if draw_obs_markers:
                    ms_obs = 3 if n_robots > 15 else 4
                    mew_obs = 0.7 if n_robots > 15 else 1.0
                    first_marker = max(1, markers_up_to - MARKER_HISTORY + 1)
                    for ss in range(first_marker, markers_up_to + 1):
                        mx, my = _coord_to_pixel(traj[ss])
                        mk = ax.plot(
                            mx, my, marker="+", color=robot_colors[r],
                            markersize=ms_obs, markeredgewidth=mew_obs,
                            linestyle="none")[0]
                        dynamic_artists[ax_key].append(mk)

                # Current position dot (smaller)
                ms_dot = 4 if n_robots > 15 else 5
                cx, cy = _coord_to_pixel(pos_now[r])
                dot = ax.plot(
                    cx, cy, marker="o", color=robot_colors[r],
                    markersize=ms_dot, markeredgecolor="black",
                    markeredgewidth=0.5, linestyle="none", zorder=5)[0]
                dynamic_artists[ax_key].append(dot)

                # Name label (only on the hybrid panel to reduce clutter)
                # Drop labels when many robots (would overlap illegibly)
                if ax_key == "ax_hyb" and n_robots <= 15:
                    lbl = ax.text(
                        cx + 5, cy - 4, f"R{r}",
                        fontsize=6, color="white",
                        path_effects=[path_effects.withStroke(linewidth=1.2, foreground="black")],
                        zorder=6)
                    dynamic_artists[ax_key].append(lbl)

        # Top banner
        day_idx = int(step_prev * stride + alpha * stride)
        cur_date = (start_day + np.timedelta64(day_idx, "D"))
        # Progress bar (ASCII-ish)
        n_fill = int((frame_idx + 1) / n_frames * 20)
        bar = "█" * n_fill + "░" * (20 - n_fill)
        suptitle.set_text(
            f"Episode  t0={t0}  (2020)   |   "
            f"step {step_prev + (1 if alpha == 0 else 0)}-{step_next}/{L}   "
            f"day {day_idx}/{L*stride}   "
            f"{np.datetime_as_string(cur_date, unit='D')}   {bar}"
        )

        return [im_gt, im_fno, im_hyb, im_err_fno, im_err_hyb,
                txt_rmse_fno, txt_rmse_hyb,
                txt_avg_fno, txt_avg_hyb, suptitle]

    # Initial render
    update(0)

    # --- Save ---
    base_name = f"episode_t0_{t0}"
    gif_path = os.path.join(args.output_dir, base_name + ".gif")
    mp4_path = os.path.join(args.output_dir, base_name + ".mp4")

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, blit=False, interval=1000 / args.fps)

    # MP4 is the default output (cleaner colors than GIF).
    if not args.gif_only:
        if _ensure_ffmpeg():
            print(f"Saving MP4: {mp4_path}")
            t0_mp4 = time.time()
            anim.save(mp4_path, writer=animation.FFMpegWriter(
                fps=args.fps, bitrate=4000))
            print(f"  MP4 done in {time.time() - t0_mp4:.0f}s")
        else:
            print("  MP4 skipped (ffmpeg not available)")

    if args.also_gif or args.gif_only:
        print(f"Saving GIF: {gif_path}")
        t0_gif = time.time()
        anim.save(gif_path, writer=animation.PillowWriter(fps=args.fps))
        print(f"  GIF done in {time.time() - t0_gif:.0f}s")

    plt.close(fig)
    print("\nDone.")
    if not args.gif_only and _ensure_ffmpeg():
        print(f"  MP4: {mp4_path}")
    if args.also_gif or args.gif_only:
        print(f"  GIF: {gif_path}")


if __name__ == "__main__":
    main()
