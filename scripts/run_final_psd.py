"""Run all 5 methods at a chosen config and save 1D PSDs (Hanning-windowed, ocean-only rows)."""
import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import yaml
from _test_helpers import load_test_data
from ipp.policies import build_policies
from ipp.simulator import build_candidates, build_eval_cells
from dynamic_ipp.rollout import run_dynamic_episode, _stable_hash
from compute_psd_1d import find_ocean_rows, psd_1d_hanning


# Length scale table (matches run_final_config.py)
LS_TABLE = {
    ("matern", 0.5): {5: 0.49, 10: 0.35, 20: 0.24, 40: 0.17},
    ("matern", 1.5): {5: 0.38, 10: 0.27, 20: 0.19, 40: 0.14},
    ("matern", 2.5): {5: 0.36, 10: 0.26, 20: 0.18, 40: 0.13},
    ("rbf", 1.5):    {5: 0.33, 10: 0.23, 20: 0.16, 40: 0.12},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_robots", type=int, required=True)
    parser.add_argument("--kernel", type=str, required=True)
    parser.add_argument("--nu", type=float, default=1.5)
    parser.add_argument("--acquisition", type=str, default="uncertainty_only")
    parser.add_argument("--t0", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episode_length", type=int, default=10)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(ROOT, "results", "dynamic_ipp", "final", "psd"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(ROOT, "configs/dynamic_rollout_ipp.yaml")) as f:
        base_cfg = yaml.safe_load(f)
    base_cfg.update(
        n_episodes=1, debug=False, bg_mode="off",
        n_init_observations=0, n_robots=args.n_robots,
        episode_length=args.episode_length,
        init_position="random",
        gp_kernel=args.kernel, gp_matern_nu=args.nu,
        acquisition_mode="lookahead", partition_dynamic=True,
    )
    ls_init = LS_TABLE[(args.kernel, args.nu)][args.n_robots]
    base_cfg["gp_length_scale_init"] = ls_init
    base_cfg["gp_length_scale_bounds"] = [ls_init / 4, ls_init * 3]

    kernel_tag = f"{args.kernel}{int(args.nu*10):02d}" if args.kernel == "matern" else "rbf"
    config_tag = (f"seed{args.seed}_t0{args.t0}_{args.n_robots}bots_"
                  f"{kernel_tag}_{args.acquisition}")

    d = load_test_data()
    om = d["ocean_mask"]
    H, W = om.shape
    cand = build_candidates(d["all_ocean_coords"],
                            base_cfg.get("n_candidates", 2000),
                            base_cfg.get("candidate_seed", 42))
    evl = build_eval_cells(d["all_ocean_coords"],
                           base_cfg.get("n_eval_cells", 20000),
                           base_cfg.get("eval_seed", 123))
    rows_info = find_ocean_rows(om, min_length=50)

    METHODS = [
        ("FNO-only",     "fno",         False),
        ("FNO+GP",       "fno",         True),
        ("Persist-only", "persistence", False),
        ("Persist+GP",   "persistence", True),
        ("GP-only",      "none",        True),
    ]

    L = args.episode_length
    results = {}

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
            else:
                pol_cfg = {"type": "uncertainty_only", "lambda_dist": 0.0}
            pol = build_policies({"policies": {args.acquisition: pol_cfg}})[args.acquisition]

        sr, _, qf = run_dynamic_episode(
            d["inputs_np"], om, d["all_ocean_coords"],
            cand, evl, pol, d["fno"], d["device"], cfg,
            episode_seed=args.seed, t0=args.t0,
            save_qual_steps=set(range(1, L + 1)))

        method_safe = label.replace(" ", "_").replace("+", "and")
        for s in range(1, L + 1):
            k, psd = psd_1d_hanning(qf[s]["x_corrected"], om, rows_info)
            results[f"{method_safe}_step{s}"] = psd
            if label == "FNO-only":
                _, psd_gt = psd_1d_hanning(qf[s]["y_true"], om, rows_info)
                results[f"GT_step{s}"] = psd_gt
        results["k"] = k
        print(f"  {label} done")

    out = os.path.join(args.out_dir, f"{config_tag}.npz")
    np.savez(out, **results)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
