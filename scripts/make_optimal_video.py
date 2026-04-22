"""Generate comparison video where each method uses its own optimal hyperparameters."""
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
from make_comparison_video import (
    _rmse_over_ocean, _corr_over_ocean, _hf_energy_ratio, _fss_over_ocean,
    _ssim_over_ocean, _bias_over_ocean, _mae_over_ocean, _crmse_over_ocean,
    _emd_over_ocean, _sal_over_ocean, _field_to_display, _ensure_ffmpeg)
from run_optimal_episode import OPTIMAL_BY_NROBOTS
from optimal_hyperparameters import ALL_OPTIMAL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0", type=int, required=True)
    parser.add_argument("--n_robots", type=int, default=10)
    parser.add_argument("--n_sub", type=int, default=4)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(ROOT, "results", "dynamic_ipp", "optimal_videos"))
    parser.add_argument("--criterion", type=str, default="rmse",
                        choices=["rmse", "acc", "m2", "mstruct"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(ROOT, "configs/dynamic_rollout_ipp.yaml")) as f:
        base_cfg = yaml.safe_load(f)
    base_cfg.update(n_episodes=1, debug=False, bg_mode="off",
                    n_init_observations=0, n_robots=args.n_robots)

    L = base_cfg["episode_length"]
    stride = base_cfg.get("rollout_lead_stride", base_cfg.get("lead", 1))

    print("Loading data + FNO ...")
    d = load_test_data()
    inputs_np = d["inputs_np"]
    ocean_mask = d["ocean_mask"]
    fno = d["fno"]
    device = d["device"]
    H, W = ocean_mask.shape

    cand = build_candidates(d["all_ocean_coords"],
                            base_cfg.get("n_candidates", 2000),
                            base_cfg.get("candidate_seed", 42))
    evl = build_eval_cells(d["all_ocean_coords"],
                           base_cfg.get("n_eval_cells", 20000),
                           base_cfg.get("eval_seed", 123))

    # Run each method with its own optimal hyperparameters
    methods = list(ALL_OPTIMAL[args.criterion].get(args.n_robots, ALL_OPTIMAL[args.criterion][10]).keys())
    all_qual = {}

    for method in methods:
        opt = ALL_OPTIMAL[args.criterion].get(args.n_robots, ALL_OPTIMAL[args.criterion][10])[method]
        cfg = dict(base_cfg)
        cfg["forecast_mode"] = opt["forecast_mode"]
        cfg["gp_kernel"] = opt["kernel"]
        cfg["gp_matern_nu"] = opt["nu"]
        cfg["gp_length_scale_init"] = opt["ls"]
        cfg["gp_length_scale_bounds"] = [0.01, max(0.5, opt["ls"] * 5)]

        pol = None
        if opt["use_policy"]:
            pol = build_policies({"policies": {
                "uncertainty_only": {"type": "uncertainty_only", "lambda_dist": 0.0}
            }})["uncertainty_only"]

        seed = cfg.get("episode_seed_offset", 0) + _stable_hash("uncertainty_only") % 1000
        print(f"Running {method} (kernel={opt['kernel']} nu={opt['nu']} LS={opt['ls']}) ...")
        t0w = time.time()
        sr, _, qf = run_dynamic_episode(
            inputs_np, ocean_mask, d["all_ocean_coords"],
            cand, evl, pol, fno, device, cfg,
            episode_seed=seed, t0=args.t0,
            save_qual_steps=set(range(1, L + 1)))
        print(f"  done in {time.time() - t0w:.0f}s, "
              f"final RMSE={sr[-1]['all_rmse']:.3f}")
        all_qual[method] = qf

    # Per-step data
    mask_land = lambda f: _field_to_display(f, ocean_mask)
    gt_by_step = {s: mask_land(all_qual["FNO-only"][s]["y_true"])
                  for s in range(1, L + 1)}

    pred_by_step = {}
    err_by_step = {}
    M = {}
    metric_names = ["RMSE", "AvgRMSE", "ACC", "SSIM", "HF",
                    "Bias", "MAE", "CRMSE", "EMD", "SAL", "FSS"]
    for mn in metric_names:
        M[mn] = {}

    for method in methods:
        qf = all_qual[method]
        pred_by_step[method] = {s: mask_land(qf[s]["x_corrected"]) for s in range(1, L + 1)}
        err_by_step[method] = {s: mask_land(qf[s]["x_corrected"] - qf[s]["y_true"])
                               for s in range(1, L + 1)}
        for mn in metric_names:
            M[mn][method] = {}
        for s in range(1, L + 1):
            p = qf[s]["x_corrected"]
            g = qf[s]["y_true"]
            M["RMSE"][method][s]  = _rmse_over_ocean(p, g, ocean_mask)
            M["ACC"][method][s]   = _corr_over_ocean(p, g, ocean_mask)
            M["SSIM"][method][s]  = _ssim_over_ocean(p, g, ocean_mask)
            M["HF"][method][s]    = _hf_energy_ratio(p, g, ocean_mask)
            M["Bias"][method][s]  = _bias_over_ocean(p, g, ocean_mask)
            M["MAE"][method][s]   = _mae_over_ocean(p, g, ocean_mask)
            M["CRMSE"][method][s] = _crmse_over_ocean(p, g, ocean_mask)
            M["EMD"][method][s]   = _emd_over_ocean(p, g, ocean_mask)
            sal = _sal_over_ocean(p, g, ocean_mask)
            M["SAL"][method][s]   = sal[0] + abs(sal[1]) + sal[2]
            M["FSS"][method][s]   = _fss_over_ocean(p, g, ocean_mask)
        for s in range(1, L + 1):
            M["AvgRMSE"][method][s] = float(np.mean(
                [M["RMSE"][method][ss] for ss in range(1, s + 1)]))

    # Colorbar ranges
    gt_vals = np.concatenate(
        [all_qual["FNO-only"][s]["y_true"][ocean_mask] for s in range(1, L + 1)])
    ssh_vmin, ssh_vmax = np.percentile(gt_vals, [2, 98]).tolist()
    all_err_vals = np.concatenate([
        (all_qual[m][s]["x_corrected"] - all_qual[m][s]["y_true"])[ocean_mask]
        for m in methods for s in range(1, L + 1)])
    err_absmax = float(np.percentile(np.abs(all_err_vals), 98))

    # Figure
    n_methods = len(methods)
    n_cols = 1 + n_methods
    fig = plt.figure(figsize=(4.2 * n_cols, 7.5))
    gs = fig.add_gridspec(2, n_cols + 1,
                          width_ratios=[1] * n_cols + [0.04],
                          wspace=0.06, hspace=0.15,
                          left=0.02, right=0.96, top=0.87, bottom=0.05)

    ax_gt = fig.add_subplot(gs[0, 0])
    ax_preds = [fig.add_subplot(gs[0, i + 1]) for i in range(n_methods)]
    ax_cb_top = fig.add_subplot(gs[0, n_cols])
    ax_legend = fig.add_subplot(gs[1, 0]); ax_legend.axis("off")
    ax_errs = [fig.add_subplot(gs[1, i + 1]) for i in range(n_methods)]
    ax_cb_bot = fig.add_subplot(gs[1, n_cols])

    for ax in [ax_gt] + ax_preds + ax_errs:
        ax.set_xticks([]); ax.set_yticks([])

    cmap_ssh = plt.cm.viridis.copy(); cmap_ssh.set_bad("lightgrey")
    cmap_err = plt.cm.RdBu_r.copy();  cmap_err.set_bad("lightgrey")
    kw_top = dict(cmap=cmap_ssh, vmin=ssh_vmin, vmax=ssh_vmax,
                  origin="upper", aspect="auto", interpolation="nearest")
    kw_bot = dict(cmap=cmap_err, vmin=-err_absmax, vmax=err_absmax,
                  origin="upper", aspect="auto", interpolation="nearest")

    im_gt = ax_gt.imshow(gt_by_step[1], **kw_top)
    im_preds = [ax_preds[i].imshow(pred_by_step[m][1], **kw_top) for i, m in enumerate(methods)]
    im_errs = [ax_errs[i].imshow(err_by_step[m][1], **kw_bot) for i, m in enumerate(methods)]

    fig.colorbar(im_gt, cax=ax_cb_top).set_label("SSH (norm.)", fontsize=9)
    fig.colorbar(im_errs[0], cax=ax_cb_bot).set_label("error", fontsize=9)

    ax_gt.set_title("Ground Truth", fontsize=11, fontweight="bold")
    for i, m in enumerate(methods):
        opt = ALL_OPTIMAL[args.criterion].get(args.n_robots, ALL_OPTIMAL[args.criterion][10])[m]
        title = f"{m}\n{opt['kernel']} nu={opt['nu']} LS={opt['ls']}" if opt["use_policy"] else m
        ax_preds[i].set_title(title, fontsize=9, fontweight="bold")
        ax_errs[i].set_title(f"{m} - GT", fontsize=8)

    display_labels = ["RMSE", "Avg", "ACC", "SSIM", "HF",
                      "Bias", "MAE", "CRMSE", "EMD", "SAL", "FSS"]
    met_txt_kw = dict(fontsize=8, fontweight="bold", color="white",
                      path_effects=[path_effects.withStroke(linewidth=2, foreground="black")])
    txt_metrics = []
    for i in range(n_methods):
        method_txts = []
        for j in range(len(display_labels)):
            y_pos = 0.95 - j * 0.085
            t = ax_errs[i].text(0.02, y_pos, "", transform=ax_errs[i].transAxes, **met_txt_kw)
            method_txts.append(t)
        txt_metrics.append(method_txts)

    ax_legend.text(0.05, 0.90, f"{args.n_robots} Robots", fontsize=11, fontweight="bold",
                   transform=ax_legend.transAxes, va="top")
    ax_legend.text(0.05, 0.75, "Optimal hyperparams\nper method (10 bots)", fontsize=8,
                   transform=ax_legend.transAxes, va="top")

    start_day = np.datetime64("2020-01-01") + np.timedelta64(int(args.t0), "D")
    suptitle = fig.suptitle("", fontsize=13, fontweight="bold", y=0.97)

    n_frames = L * args.n_sub
    print(f"Rendering {n_frames} frames ...")

    def update(frame_idx):
        step_next = frame_idx // args.n_sub + 1
        step_prev = step_next - 1
        alpha = (frame_idx % args.n_sub) / args.n_sub
        sp = max(step_prev, 1); sn = step_next

        def lerp(a, b, t):
            return (1 - t) * a + t * b

        im_gt.set_data(lerp(gt_by_step[sp], gt_by_step[sn], alpha))
        for i, m in enumerate(methods):
            im_preds[i].set_data(lerp(pred_by_step[m][sp], pred_by_step[m][sn], alpha))
            im_errs[i].set_data(lerp(err_by_step[m][sp], err_by_step[m][sn], alpha))
            for j, mn in enumerate(metric_names):
                v = lerp(M[mn][m][sp], M[mn][m][sn], alpha)
                txt_metrics[i][j].set_text(f"{display_labels[j]}={v:+.3f}"
                                           if mn == "Bias"
                                           else f"{display_labels[j]}={v:.3f}")

        day_idx = int(step_prev * stride + alpha * stride)
        cur_date = start_day + np.timedelta64(day_idx, "D")
        n_fill = int((frame_idx + 1) / n_frames * 20)
        bar = "\u2588" * n_fill + "\u2591" * (20 - n_fill)
        suptitle.set_text(
            f"Episode t0={args.t0} (2020) | step {sp}/{L} day {day_idx}/{L * stride} "
            f"{np.datetime_as_string(cur_date, unit='D')} {bar}")

        all_txt = [t for mt in txt_metrics for t in mt]
        return [im_gt] + im_preds + im_errs + all_txt + [suptitle]

    update(0)
    base_name = f"optimal_t0_{args.t0}_{args.n_robots}bots_{args.criterion}"
    mp4_path = os.path.join(args.out_dir, base_name + ".mp4")

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000 / args.fps)
    if _ensure_ffmpeg():
        print(f"Saving {mp4_path}")
        anim.save(mp4_path, writer=animation.FFMpegWriter(fps=args.fps, bitrate=4000))
    plt.close(fig)
    print(f"Done: {mp4_path}")


if __name__ == "__main__":
    main()
