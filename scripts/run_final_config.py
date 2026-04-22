"""Run all 5 methods for one (n_robots, kernel, acquisition) config.

Default mode: save full pickles (qual_frames + step records) for asset generation.
With --metrics_only: save just one CSV with per-step metrics for all 5 methods.
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
import yaml
from _test_helpers import load_test_data
from ipp.policies import build_policies
from ipp.simulator import build_candidates, build_eval_cells
from dynamic_ipp.rollout import run_dynamic_episode, _stable_hash
from make_comparison_video import (
    _rmse_over_ocean, _corr_over_ocean, _hf_energy_ratio, _fss_over_ocean,
    _ssim_over_ocean, _bias_over_ocean, _mae_over_ocean, _crmse_over_ocean)


# Theoretical kernel length scales (target correlation 0.4 at robot spacing)
# Bounds = [LS/4, LS*3] (set in main function)
LS_TABLE = {
    ("matern", 0.5): {5: 0.49, 10: 0.35, 20: 0.24, 40: 0.17},
    ("matern", 1.5): {5: 0.38, 10: 0.27, 20: 0.19, 40: 0.14},
    ("matern", 2.5): {5: 0.36, 10: 0.26, 20: 0.18, 40: 0.13},
    ("rbf", 1.5):    {5: 0.33, 10: 0.23, 20: 0.16, 40: 0.12},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_robots", type=int, required=True)
    parser.add_argument("--kernel", type=str, required=True,
                        choices=["matern", "rbf"])
    parser.add_argument("--nu", type=float, default=1.5)
    parser.add_argument("--acquisition", type=str, required=True,
                        choices=["uncertainty_only", "hybrid_ucb", "mi"])
    parser.add_argument("--t0", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for episode (initial positions, partitioning)")
    parser.add_argument("--episode_length", type=int, default=5)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(ROOT, "results", "dynamic_ipp", "final", "episodes"))
    parser.add_argument("--metrics_only", action="store_true",
                        help="Skip pickle save; just save per-step metrics CSV")
    parser.add_argument("--metrics_dir", type=str,
                        default=os.path.join(ROOT, "results", "dynamic_ipp", "final", "metrics"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load config
    with open(os.path.join(ROOT, "configs/dynamic_rollout_ipp.yaml")) as f:
        base_cfg = yaml.safe_load(f)

    base_cfg.update(
        n_episodes=1, debug=False, bg_mode="off",
        n_init_observations=0, n_robots=args.n_robots,
        episode_length=args.episode_length,
        init_position="random",
        gp_kernel=args.kernel, gp_matern_nu=args.nu,
        acquisition_mode="lookahead",
        partition_dynamic=True,
    )

    # Length scale from theory table
    ls_init = LS_TABLE[(args.kernel, args.nu)][args.n_robots]
    base_cfg["gp_length_scale_init"] = ls_init
    base_cfg["gp_length_scale_bounds"] = [ls_init / 4, ls_init * 3]

    # Build a config tag including seed and t0 for unambiguous filenames
    kernel_tag = f"{args.kernel}{int(args.nu*10):02d}" if args.kernel == "matern" else "rbf"
    config_tag = (f"seed{args.seed}_t0{args.t0}_{args.n_robots}bots_"
                  f"{kernel_tag}_{args.acquisition}")

    print(f"Config: {config_tag}  LS={ls_init:.3f} bounds={base_cfg['gp_length_scale_bounds']}")

    # Load shared data
    d = load_test_data()
    om = d["ocean_mask"]
    cand = build_candidates(d["all_ocean_coords"],
                            base_cfg.get("n_candidates", 2000),
                            base_cfg.get("candidate_seed", 42))
    evl = build_eval_cells(d["all_ocean_coords"],
                           base_cfg.get("n_eval_cells", 20000),
                           base_cfg.get("eval_seed", 123))

    # Methods: (label, forecast_mode, use_policy)
    METHODS = [
        ("FNO-only",     "fno",         False),
        ("FNO+GP",       "fno",         True),
        ("Persist-only", "persistence", False),
        ("Persist+GP",   "persistence", True),
        ("GP-only",      "none",        True),
    ]

    L = args.episode_length

    # Collect per-method per-step metrics for combined CSV (used in --metrics_only mode)
    csv_rows = []

    for label, fm, use_pol in METHODS:
        cfg = dict(base_cfg)
        cfg["forecast_mode"] = fm

        pol = None
        if use_pol:
            if args.acquisition == "mi":
                pol_cfg = {"type": "mi", "kernel": args.kernel,
                           "nu": args.nu, "length_scale": ls_init,
                           "lambda_dist": 0.0}
            elif args.acquisition == "hybrid_ucb":
                pol_cfg = {"type": "hybrid_ucb", "delta": 0.1, "input_dim": 3}
            else:  # uncertainty_only
                pol_cfg = {"type": "uncertainty_only", "lambda_dist": 0.0}
            pol = build_policies({"policies": {args.acquisition: pol_cfg}})[args.acquisition]

        # Use the specified seed for reproducibility
        seed = args.seed
        tw = time.time()
        sr, traj_arrays, qf = run_dynamic_episode(
            d["inputs_np"], om, d["all_ocean_coords"],
            cand, evl, pol, d["fno"], d["device"], cfg,
            episode_seed=seed, t0=args.t0,
            save_qual_steps=set(range(1, L + 1)),
        )
        elapsed = time.time() - tw
        print(f"  {label}: {elapsed:.0f}s  final RMSE={sr[-1]['all_rmse']:.3f}")

        if args.metrics_only:
            # Compute and append per-step metrics to csv_rows
            rmses = []
            for s in range(1, L + 1):
                p = qf[s]["x_corrected"]
                g = qf[s]["y_true"]
                rmses.append(_rmse_over_ocean(p, g, om))
            for s in range(1, L + 1):
                p = qf[s]["x_corrected"]
                g = qf[s]["y_true"]
                row = {
                    "seed": args.seed,
                    "t0": args.t0,
                    "n_robots": args.n_robots,
                    "kernel": args.kernel,
                    "nu": args.nu,
                    "ls_init": ls_init,
                    "acquisition": args.acquisition,
                    "method": label,
                    "step": s,
                    "RMSE":    _rmse_over_ocean(p, g, om),
                    "AvgRMSE": float(np.mean(rmses[:s])),
                    "ACC":     _corr_over_ocean(p, g, om),
                    "HF":      _hf_energy_ratio(p, g, om),
                    "FSS":     _fss_over_ocean(p, g, om),
                    "CRMSE":   _crmse_over_ocean(p, g, om),
                    "Bias":    _bias_over_ocean(p, g, om),
                    "MAE":     _mae_over_ocean(p, g, om),
                    "SSIM":    _ssim_over_ocean(p, g, om),
                }
                csv_rows.append(row)
        else:
            # Save full pickle for asset generation
            method_safe = label.replace(" ", "_").replace("+", "and")
            out_path = os.path.join(args.out_dir, f"{config_tag}_{method_safe}.pkl")
            save_data = {
                "qual_frames": qf, "step_records": sr,
                "trajectories": traj_arrays,
                "config": config_tag, "method": label,
                "n_robots": args.n_robots, "kernel": args.kernel,
                "nu": args.nu, "ls_init": ls_init,
                "acquisition": args.acquisition,
                "t0": args.t0, "L": L,
            }
            with open(out_path, "wb") as f:
                pickle.dump(save_data, f)

    if args.metrics_only and csv_rows:
        os.makedirs(args.metrics_dir, exist_ok=True)
        out_csv = os.path.join(args.metrics_dir, f"{config_tag}.csv")
        pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
        print(f"Done: {config_tag}  -> {out_csv}")
    else:
        print(f"Done: {config_tag}")


if __name__ == "__main__":
    main()
