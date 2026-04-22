"""
Side-by-side comparison video of all forecast modes.

Layout (2 rows x 5 cols + colorbar):

    ┌──────────┬──────────┬──────────┬──────────┬──────────┐
    │    GT    │ FNO-only │  FNO+GP  │Persist+GP│  GP-only │  <- shared SSH cbar
    ├──────────┼──────────┼──────────┼──────────┼──────────┤
    │  legend  │ FNO-GT   │FNO+GP-GT │Pers+GP-GT│GP-only-GT│  <- shared err cbar
    │          │ RMSE=... │ RMSE=... │ RMSE=... │ RMSE=... │
    └──────────┴──────────┴──────────┴──────────┴──────────┘

Usage:
    python scripts/make_comparison_video.py [--t0 N] [--n_robots 5]
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
warnings.filterwarnings("ignore")

import yaml

from _test_helpers import load_test_data
from ipp.policies import build_policies
from ipp.simulator import build_candidates, build_eval_cells
from dynamic_ipp.rollout import run_dynamic_episode, _stable_hash


# ---------------------------------------------------------------------------
# Utilities (from make_episode_video.py)
# ---------------------------------------------------------------------------

def _rmse_over_ocean(a, b, ocean_mask):
    diff = (a - b)[ocean_mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def _corr_over_ocean(a, b, ocean_mask):
    """Pearson spatial correlation over ocean cells."""
    a_flat = a[ocean_mask]
    b_flat = b[ocean_mask]
    a_mean = a_flat - a_flat.mean()
    b_mean = b_flat - b_flat.mean()
    num = np.sum(a_mean * b_mean)
    den = np.sqrt(np.sum(a_mean ** 2) * np.sum(b_mean ** 2))
    if den < 1e-12:
        return 0.0
    return float(num / den)


def _ssim_over_ocean(a, b, ocean_mask, win_size=11):
    """Windowed SSIM over ocean cells (Wang et al., 2004).

    Computes SSIM in local windows and averages over ocean cells.
    """
    from scipy.ndimage import uniform_filter
    a64 = np.where(ocean_mask, a, 0.0).astype(np.float64)
    b64 = np.where(ocean_mask, b, 0.0).astype(np.float64)
    ocf = ocean_mask.astype(np.float64)

    # Local statistics via uniform filter, normalized by ocean count
    oc_count = uniform_filter(ocf, size=win_size, mode='constant') * win_size * win_size
    valid = ocean_mask & (oc_count > 3)  # need enough ocean cells in window
    oc_count_safe = np.maximum(oc_count, 1.0)

    mu_a = uniform_filter(a64, size=win_size, mode='constant') * win_size * win_size / oc_count_safe
    mu_b = uniform_filter(b64, size=win_size, mode='constant') * win_size * win_size / oc_count_safe
    mu_a2 = mu_a ** 2
    mu_b2 = mu_b ** 2
    mu_ab = mu_a * mu_b

    sig_a2 = uniform_filter(a64 ** 2, size=win_size, mode='constant') * win_size * win_size / oc_count_safe - mu_a2
    sig_b2 = uniform_filter(b64 ** 2, size=win_size, mode='constant') * win_size * win_size / oc_count_safe - mu_b2
    sig_ab = uniform_filter(a64 * b64, size=win_size, mode='constant') * win_size * win_size / oc_count_safe - mu_ab

    # Clamp negative variances from numerical issues
    sig_a2 = np.maximum(sig_a2, 0.0)
    sig_b2 = np.maximum(sig_b2, 0.0)

    # Dynamic range from GT ocean values
    gt_ocean = a[ocean_mask]
    L = gt_ocean.max() - gt_ocean.min()
    if L < 1e-12:
        return 0.0
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    num = (2 * mu_ab + C1) * (2 * sig_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2)

    ssim_map = num / den
    return float(ssim_map[valid].mean())


def _hf_energy_ratio(pred, gt, ocean_mask):
    """Gradient energy ratio: sum|grad(pred)|^2 / sum|grad(gt)|^2 over ocean.

    Measures whether the prediction retains fine-scale spatial structure.
      1.0 = correct fine-scale energy, 0.0 = smooth blob, >1.0 = spurious noise.
    Uses squared gradient magnitude — no FFT, no land-fill artifacts.
    """
    # Compute gradients along both axes (np.gradient handles edges)
    gy_pred, gx_pred = np.gradient(pred.astype(np.float64))
    gy_gt, gx_gt     = np.gradient(gt.astype(np.float64))

    # Only include interior ocean cells (ocean AND all 4 neighbors are ocean)
    # to avoid land-ocean boundary gradients
    interior = ocean_mask.copy()
    interior[0, :] = False
    interior[-1, :] = False
    interior[:, 0] = False
    interior[:, -1] = False
    interior[1:, :] &= ocean_mask[:-1, :]   # neighbor above
    interior[:-1, :] &= ocean_mask[1:, :]   # neighbor below
    interior[:, 1:] &= ocean_mask[:, :-1]   # neighbor left
    interior[:, :-1] &= ocean_mask[:, 1:]   # neighbor right

    grad2_pred = (gx_pred ** 2 + gy_pred ** 2)[interior]
    grad2_gt   = (gx_gt ** 2 + gy_gt ** 2)[interior]

    E_gt = grad2_gt.sum()
    if E_gt < 1e-12:
        return 0.0
    return float(grad2_pred.sum() / E_gt)


def _bias_over_ocean(pred, gt, ocean_mask):
    """Mean error (pred - gt) over ocean cells."""
    return float((pred[ocean_mask] - gt[ocean_mask]).mean())


def _mae_over_ocean(pred, gt, ocean_mask):
    """Mean absolute error over ocean cells."""
    return float(np.abs(pred[ocean_mask] - gt[ocean_mask]).mean())


def _crmse_over_ocean(pred, gt, ocean_mask):
    """Centered RMSE: RMSE after removing mean bias from both fields."""
    p = pred[ocean_mask].astype(np.float64)
    g = gt[ocean_mask].astype(np.float64)
    p_anom = p - p.mean()
    g_anom = g - g.mean()
    return float(np.sqrt(np.mean((p_anom - g_anom) ** 2)))


def _emd_over_ocean(pred, gt, ocean_mask):
    """Earth Mover's Distance (Wasserstein-1) between value distributions."""
    p = np.sort(pred[ocean_mask].astype(np.float64))
    g = np.sort(gt[ocean_mask].astype(np.float64))
    # For equal-sized samples, W1 = mean of sorted absolute differences
    return float(np.mean(np.abs(p - g)))


def _sal_over_ocean(pred, gt, ocean_mask):
    """SAL (Structure-Amplitude-Location) score adapted for SSH anomalies.

    Uses threshold-based object identification (|field| > 1*std of GT).
    Returns (S, A, L) tuple. Each component is 0 for perfect match.
    S, A in [-2, 2], L in [0, 2].
    """
    from scipy.ndimage import label as ndlabel

    p_full = np.where(ocean_mask, pred.astype(np.float64), 0.0)
    g_full = np.where(ocean_mask, gt.astype(np.float64), 0.0)
    g_flat = gt[ocean_mask].astype(np.float64)
    p_flat = pred[ocean_mask].astype(np.float64)

    # A (Amplitude): use domain-integrated absolute values (robust near zero mean)
    D_p = np.abs(p_flat).mean()
    D_g = np.abs(g_flat).mean()
    denom_a = 0.5 * (D_p + D_g)
    A = float((D_p - D_g) / denom_a) if denom_a > 1e-12 else 0.0

    # Object identification: |field| > threshold
    thresh = g_flat.std()
    obj_p_mask = ocean_mask & (np.abs(p_full) > thresh)
    obj_g_mask = ocean_mask & (np.abs(g_full) > thresh)

    # L (Location): center of mass of thresholded objects
    rows, cols = np.where(ocean_mask)
    H, W = ocean_mask.shape
    d_max = np.sqrt(H ** 2 + W ** 2)

    rows_p, cols_p = np.where(obj_p_mask)
    rows_g, cols_g = np.where(obj_g_mask)

    if len(rows_p) > 0 and len(rows_g) > 0:
        w_p = np.abs(p_full[obj_p_mask])
        w_g = np.abs(g_full[obj_g_mask])
        com_p = np.array([np.average(rows_p, weights=w_p),
                          np.average(cols_p, weights=w_p)])
        com_g = np.array([np.average(rows_g, weights=w_g),
                          np.average(cols_g, weights=w_g)])
        L1 = float(np.linalg.norm(com_p - com_g) / d_max)

        coords_p = np.stack([rows_p, cols_p], axis=1).astype(np.float64)
        coords_g = np.stack([rows_g, cols_g], axis=1).astype(np.float64)
        r_p = float(np.average(np.linalg.norm(coords_p - com_p, axis=1), weights=w_p))
        r_g = float(np.average(np.linalg.norm(coords_g - com_g, axis=1), weights=w_g))
        L2 = 2.0 * abs(r_p - r_g) / (r_p + r_g) if (r_p + r_g) > 1e-12 else 0.0
    else:
        L1 = 0.0
        L2 = 0.0
    L = L1 + L2

    # S (Structure): compare object-based scaled volumes
    # For each labeled object, V = total_intensity / peak_intensity
    # Then aggregate: weighted mean V across objects
    def _object_V(field_full, obj_mask):
        labeled, n_obj = ndlabel(obj_mask)
        if n_obj == 0:
            return 0.0
        Vs = []
        Rs = []
        for i in range(1, n_obj + 1):
            obj_vals = np.abs(field_full[labeled == i])
            peak = obj_vals.max()
            total = obj_vals.sum()
            if peak > 1e-12:
                Vs.append(total / peak)
                Rs.append(peak)
        if not Rs:
            return 0.0
        Rs = np.array(Rs)
        Vs = np.array(Vs)
        return float(np.sum(Vs * Rs) / np.sum(Rs))

    V_p = _object_V(p_full, obj_p_mask)
    V_g = _object_V(g_full, obj_g_mask)
    denom_s = 0.5 * (V_p + V_g)
    S = float((V_p - V_g) / denom_s) if denom_s > 1e-12 else 0.0

    return (S, A, L)


def _fss_over_ocean(pred, gt, ocean_mask, threshold_std=1.0, radius=10):
    """Fractions Skill Score at a given threshold and neighborhood radius.

    threshold: cells with value > mean + threshold_std * std(gt) are "events".
    radius: neighborhood radius in pixels for fraction computation.
    Fractions are normalized by ocean cell count in each neighborhood.
    Returns FSS in [0, 1]. 1 = perfect, 0 = no skill.
    """
    from scipy.ndimage import uniform_filter
    g_flat = gt[ocean_mask]
    thresh = g_flat.mean() + threshold_std * g_flat.std()

    # Binary event fields (land = 0, won't count as events)
    bin_p = np.where(ocean_mask, (pred > thresh).astype(np.float64), 0.0)
    bin_g = np.where(ocean_mask, (gt > thresh).astype(np.float64), 0.0)
    ocean_f = ocean_mask.astype(np.float64)

    size = 2 * radius + 1
    # Sum of events and ocean cells in each neighborhood
    sum_p = uniform_filter(bin_p, size=size, mode='constant') * size * size
    sum_g = uniform_filter(bin_g, size=size, mode='constant') * size * size
    sum_ocean = uniform_filter(ocean_f, size=size, mode='constant') * size * size

    # Only evaluate at ocean cells with enough ocean neighbors
    valid = ocean_mask & (sum_ocean > 1.0)

    # Fractions normalized by ocean cell count in neighborhood
    fp = sum_p[valid] / sum_ocean[valid]
    fg = sum_g[valid] / sum_ocean[valid]

    mse = np.mean((fp - fg) ** 2)
    ref = np.mean(fp ** 2 + fg ** 2)
    if ref < 1e-12:
        return 1.0  # both fields empty at this threshold
    return float(1.0 - mse / ref)


def _field_to_display(field, ocean_mask):
    disp = field.copy().astype(np.float32)
    disp[~ocean_mask] = np.nan
    return disp


def _compute_t0_list(cfg, N_samples):
    L = cfg.get("episode_length", 15)
    rollout_lead_stride = cfg.get("rollout_lead_stride", cfg.get("lead", 1))
    max_t0 = N_samples - L * rollout_lead_stride - 1
    if max_t0 < 0:
        raise ValueError("Not enough samples for one episode.")
    n_ep = cfg.get("n_episodes", 50)
    seed = cfg.get("episode_seed_offset", 0)
    rng_ep = np.random.default_rng(seed)
    return sorted(
        rng_ep.choice(max_t0 + 1, size=min(n_ep, max_t0 + 1),
                      replace=False).tolist())


def _seed_for_policy(cfg, ep_i, pol_name):
    return (cfg.get("episode_seed_offset", 0)
            + ep_i * 1000
            + _stable_hash(pol_name) % 1000)


def _ensure_ffmpeg():
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
# Method definitions
# ---------------------------------------------------------------------------

METHODS = [
    # (label,         forecast_mode,  use_policy)
    ("FNO-only",      "fno",          False),
    ("FNO + GP",      "fno",          True),
    ("Persist-only",  "persistence",  False),
    ("Persist + GP",  "persistence",  True),
    ("GP-only",       "none",         True),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0", type=int, default=-1)
    parser.add_argument("--n_sub", type=int, default=4,
                        help="Sub-frames per FNO step")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--n_robots", type=int, default=5)
    parser.add_argument("--bg_mode", type=str, default="off")
    parser.add_argument("--n_init_observations", type=int, default=0)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(ROOT, "results", "dynamic_ipp", "videos"))
    parser.add_argument("--also_gif", action="store_true")
    parser.add_argument("--gif_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    cfg_path = os.path.join(ROOT, "configs", "dynamic_rollout_ipp.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["n_episodes"] = 1
    cfg["debug"] = False
    cfg["n_robots"] = args.n_robots
    cfg["bg_mode"] = args.bg_mode
    cfg["n_init_observations"] = args.n_init_observations

    L = cfg["episode_length"]
    n_robots = cfg["n_robots"]
    stride = cfg.get("rollout_lead_stride", cfg.get("lead", 1))

    print("Loading data + FNO ...")
    d = load_test_data()
    inputs_np = d["inputs_np"]
    ocean_mask = d["ocean_mask"]
    all_ocean_coords = d["all_ocean_coords"]
    fno = d["fno"]
    device = d["device"]
    H, W = ocean_mask.shape

    # Shared candidate/eval sets
    cand_local_idx = build_candidates(
        all_ocean_coords, cfg.get("n_candidates", 2000),
        cfg.get("candidate_seed", 42))
    eval_local_idx = build_eval_cells(
        all_ocean_coords, cfg.get("n_eval_cells", 20000),
        cfg.get("eval_seed", 123))

    # Pick t0
    t0_list = _compute_t0_list(cfg, inputs_np.shape[0])
    t0 = args.t0 if args.t0 >= 0 else t0_list[0]
    print(f"Using t0 = {t0}")

    # Build policy (shared across methods that use it)
    pol_name = "uncertainty_only"
    episode_seed = _seed_for_policy(cfg, ep_i=0, pol_name=pol_name)
    built = build_policies({"policies": {
        pol_name: {"type": pol_name, "lambda_dist": 0.0}
    }})
    policy = built[pol_name]

    # ── Run all 4 methods ─────────────────────────────────────────────────
    all_qual = {}   # method_label -> {step -> qual_frame}
    all_step_rec = {}

    for label, forecast_mode, use_policy in METHODS:
        run_cfg = dict(cfg)
        run_cfg["forecast_mode"] = forecast_mode
        run_policy = policy if use_policy else None
        print(f"\nRunning: {label} (forecast_mode={forecast_mode}, "
              f"policy={'yes' if use_policy else 'none'}) ...")
        t0w = time.time()
        step_records, traj_arrays, qual_frames = run_dynamic_episode(
            inputs_np, ocean_mask, all_ocean_coords,
            cand_local_idx, eval_local_idx,
            run_policy, fno, device, run_cfg,
            episode_seed=episode_seed, t0=t0,
            save_qual_steps=set(range(1, L + 1)),
        )
        elapsed = time.time() - t0w
        print(f"  Done in {elapsed:.0f}s  |  "
              f"final RMSE={step_records[-1]['all_rmse']:.4f}  "
              f"FNO={step_records[-1]['fno_rmse']:.4f}")
        all_qual[label] = qual_frames
        all_step_rec[label] = step_records

    # ── Extract per-step data ─────────────────────────────────────────────
    mask_land = lambda f: _field_to_display(f, ocean_mask)

    # Ground truth (same for all methods)
    gt_by_step = {s: mask_land(all_qual["FNO-only"][s]["y_true"])
                  for s in range(1, L + 1)}

    # Per-method metrics
    pred_by_step = {}
    err_by_step = {}
    M = {}  # M[metric_name][label][step] = float

    metric_names = ["RMSE", "AvgRMSE", "ACC", "SSIM", "HF",
                    "Bias", "MAE", "CRMSE", "EMD", "SAL", "FSS"]
    for mn in metric_names:
        M[mn] = {}

    for label, _, _ in METHODS:
        qf = all_qual[label]
        pred_by_step[label] = {
            s: mask_land(qf[s]["x_corrected"]) for s in range(1, L + 1)}
        err_by_step[label] = {
            s: mask_land(qf[s]["x_corrected"] - qf[s]["y_true"])
            for s in range(1, L + 1)}

        for mn in metric_names:
            M[mn][label] = {}

        for s in range(1, L + 1):
            p = qf[s]["x_corrected"]
            g = qf[s]["y_true"]
            M["RMSE"][label][s]  = _rmse_over_ocean(p, g, ocean_mask)
            M["ACC"][label][s]   = _corr_over_ocean(p, g, ocean_mask)
            M["SSIM"][label][s]  = _ssim_over_ocean(p, g, ocean_mask)
            M["HF"][label][s]    = _hf_energy_ratio(p, g, ocean_mask)
            M["Bias"][label][s]  = _bias_over_ocean(p, g, ocean_mask)
            M["MAE"][label][s]   = _mae_over_ocean(p, g, ocean_mask)
            M["CRMSE"][label][s] = _crmse_over_ocean(p, g, ocean_mask)
            M["EMD"][label][s]   = _emd_over_ocean(p, g, ocean_mask)
            sal = _sal_over_ocean(p, g, ocean_mask)
            M["SAL"][label][s]   = sal[0] + abs(sal[1]) + sal[2]  # scalar summary
            M["FSS"][label][s]   = _fss_over_ocean(p, g, ocean_mask)

        # Avg RMSE
        for s in range(1, L + 1):
            M["AvgRMSE"][label][s] = float(np.mean(
                [M["RMSE"][label][ss] for ss in range(1, s + 1)]))

    # ── Colorbar ranges (shared) ──────────────────────────────────────────
    gt_vals = np.concatenate(
        [all_qual["FNO-only"][s]["y_true"][ocean_mask] for s in range(1, L + 1)])
    ssh_vmin, ssh_vmax = np.percentile(gt_vals, [2, 98]).tolist()

    # Error range: use max across all methods so scale is fair
    all_err_vals = []
    for label, _, _ in METHODS:
        qf = all_qual[label]
        for s in range(1, L + 1):
            all_err_vals.append(
                (qf[s]["x_corrected"] - qf[s]["y_true"])[ocean_mask])
    all_err_vals = np.concatenate(all_err_vals)
    err_absmax = float(np.percentile(np.abs(all_err_vals), 98))

    print(f"\nSSH range:   [{ssh_vmin:.3f}, {ssh_vmax:.3f}]")
    print(f"Error range: +/-{err_absmax:.3f}")

    # ── Frame interpolation ───────────────────────────────────────────────
    n_frames = L * args.n_sub

    def frame_context(f):
        step_next = f // args.n_sub + 1
        step_prev = step_next - 1
        alpha = (f % args.n_sub) / args.n_sub
        return step_prev, step_next, alpha

    # ── Build figure ──────────────────────────────────────────────────────
    n_methods = len(METHODS)
    n_cols = 1 + n_methods  # GT + methods
    fig = plt.figure(figsize=(4.2 * n_cols, 7.5))
    gs = fig.add_gridspec(
        2, n_cols + 1,
        width_ratios=[1] * n_cols + [0.04],
        wspace=0.06, hspace=0.15,
        left=0.02, right=0.96, top=0.87, bottom=0.05)

    # Top row: GT + method prediction panels
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_preds = [fig.add_subplot(gs[0, i + 1]) for i in range(n_methods)]
    ax_cb_top = fig.add_subplot(gs[0, n_cols])

    # Bottom row: legend + method error panels
    ax_legend = fig.add_subplot(gs[1, 0]); ax_legend.axis("off")
    ax_errs = [fig.add_subplot(gs[1, i + 1]) for i in range(n_methods)]
    ax_cb_bot = fig.add_subplot(gs[1, n_cols])

    for ax in [ax_gt] + ax_preds + ax_errs:
        ax.set_xticks([]); ax.set_yticks([])

    # Colormaps
    cmap_ssh = plt.cm.viridis.copy(); cmap_ssh.set_bad("lightgrey")
    cmap_err = plt.cm.RdBu_r.copy();  cmap_err.set_bad("lightgrey")

    kw_top = dict(cmap=cmap_ssh, vmin=ssh_vmin, vmax=ssh_vmax,
                  origin="upper", aspect="auto", interpolation="nearest")
    kw_bot = dict(cmap=cmap_err, vmin=-err_absmax, vmax=err_absmax,
                  origin="upper", aspect="auto", interpolation="nearest")

    # Initial images
    im_gt = ax_gt.imshow(gt_by_step[1], **kw_top)
    im_preds = []
    im_errs = []
    for i, (label, _, _) in enumerate(METHODS):
        im_preds.append(ax_preds[i].imshow(pred_by_step[label][1], **kw_top))
        im_errs.append(ax_errs[i].imshow(err_by_step[label][1], **kw_bot))

    # Colorbars
    cb_top = fig.colorbar(im_gt, cax=ax_cb_top)
    cb_top.set_label("SSH (norm.)", fontsize=9)
    cb_bot = fig.colorbar(im_errs[0], cax=ax_cb_bot)
    cb_bot.set_label("error", fontsize=9)

    # Titles
    ax_gt.set_title("Ground Truth", fontsize=11, fontweight="bold")
    method_labels = [m[0] for m in METHODS]
    for i, label in enumerate(method_labels):
        ax_preds[i].set_title(label, fontsize=10, fontweight="bold")
        ax_errs[i].set_title(f"{label} - GT", fontsize=9)

    # Metrics text on error panels (11 lines, single column)
    # Display labels for each metric
    display_labels = ["RMSE", "Avg", "ACC", "SSIM", "HF",
                      "Bias", "MAE", "CRMSE", "EMD", "SAL", "FSS"]
    met_txt_kw = dict(
        fontsize=8, fontweight="bold", color="white",
        path_effects=[path_effects.withStroke(linewidth=2, foreground="black")],
    )
    # Create text artists: txt_metrics[i][j] = text for method i, metric j
    txt_metrics = []
    n_metrics = len(display_labels)
    for i in range(n_methods):
        method_txts = []
        for j in range(n_metrics):
            y_pos = 0.95 - j * 0.085
            t = ax_errs[i].text(0.02, y_pos, "", transform=ax_errs[i].transAxes,
                                **met_txt_kw)
            method_txts.append(t)
        txt_metrics.append(method_txts)

    # Legend panel
    ax_legend.text(0.05, 0.90, f"{n_robots} Robots", fontsize=11,
                   fontweight="bold", transform=ax_legend.transAxes, va="top")
    ax_legend.text(0.05, 0.75, f"bg_mode: {args.bg_mode}", fontsize=9,
                   transform=ax_legend.transAxes, va="top")
    ax_legend.text(0.05, 0.65, f"init_obs: {args.n_init_observations}", fontsize=9,
                   transform=ax_legend.transAxes, va="top")
    ax_legend.text(0.05, 0.55, f"gp_mode: {cfg.get('gp_mode', '?')}", fontsize=9,
                   transform=ax_legend.transAxes, va="top")
    ax_legend.text(0.05, 0.45, f"policy: {pol_name}", fontsize=9,
                   transform=ax_legend.transAxes, va="top")

    # Suptitle
    start_day = np.datetime64("2020-01-01") + np.timedelta64(int(t0), "D")
    suptitle = fig.suptitle("", fontsize=13, fontweight="bold", y=0.97)

    # ── Frame update ──────────────────────────────────────────────────────
    print(f"\nBuilding {n_frames} frames ...")

    def update(frame_idx):
        step_prev, step_next, alpha = frame_context(frame_idx)
        sp = max(step_prev, 1)
        sn = step_next

        def lerp(a, b, t):
            return (1 - t) * a + t * b

        # Ground truth
        im_gt.set_data(lerp(gt_by_step[sp], gt_by_step[sn], alpha))

        # Per-method fields + errors + all metrics
        for i, (label, _, _) in enumerate(METHODS):
            pred_f = lerp(pred_by_step[label][sp],
                          pred_by_step[label][sn], alpha)
            err_f = lerp(err_by_step[label][sp],
                         err_by_step[label][sn], alpha)
            im_preds[i].set_data(pred_f)
            im_errs[i].set_data(err_f)

            for j, mn in enumerate(metric_names):
                v = lerp(M[mn][label][sp], M[mn][label][sn], alpha)
                txt_metrics[i][j].set_text(f"{display_labels[j]}={v:+.3f}"
                                           if mn == "Bias"
                                           else f"{display_labels[j]}={v:.3f}")

        # Top banner
        day_idx = int(step_prev * stride + alpha * stride)
        cur_date = start_day + np.timedelta64(day_idx, "D")
        n_fill = int((frame_idx + 1) / n_frames * 20)
        bar = "\u2588" * n_fill + "\u2591" * (20 - n_fill)
        suptitle.set_text(
            f"Episode t0={t0} (2020)   |   "
            f"step {sp}/{L}   day {day_idx}/{L * stride}   "
            f"{np.datetime_as_string(cur_date, unit='D')}   {bar}"
        )

        all_txt = [t for method_txts in txt_metrics for t in method_txts]
        return [im_gt] + im_preds + im_errs + all_txt + [suptitle]

    # Initial render
    update(0)

    # ── Save ──────────────────────────────────────────────────────────────
    base_name = f"comparison_t0_{t0}_{n_robots}bots"
    mp4_path = os.path.join(args.output_dir, base_name + ".mp4")
    gif_path = os.path.join(args.output_dir, base_name + ".gif")

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, blit=False, interval=1000 / args.fps)

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

    # ── Save per-step metrics to CSV ──────────────────────────────────────
    csv_rows = []
    for label, _, _ in METHODS:
        for s in range(1, L + 1):
            row = {"method": label, "step": s, "n_robots": n_robots,
                   "t0": t0, "bg_mode": args.bg_mode,
                   "n_init_obs": args.n_init_observations}
            for mn in metric_names:
                row[mn] = M[mn][label][s]
            csv_rows.append(row)
    csv_path = os.path.join(args.output_dir, base_name + "_metrics.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"\nDone.")
    print(f"  Video:   {mp4_path}")
    print(f"  Metrics: {csv_path}")


if __name__ == "__main__":
    main()
