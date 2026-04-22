"""Run an episode at specific hyperparameters and save PSD."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0", type=int, required=True)
    parser.add_argument("--n_robots", type=int, required=True)
    parser.add_argument("--forecast_mode", type=str, required=True,
                        choices=["fno", "persistence", "none"])
    parser.add_argument("--use_policy", action="store_true")
    parser.add_argument("--gp_kernel", type=str, default="matern")
    parser.add_argument("--gp_matern_nu", type=float, default=1.5)
    parser.add_argument("--gp_length_scale_init", type=float, default=0.1)
    parser.add_argument("--tag", type=str, required=True,
                        help="Tag for filename, e.g. 'fno_rmseopt' or 'fno_accopt'")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(ROOT, "results", "dynamic_ipp", "psd_optimal"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(ROOT, "configs/dynamic_rollout_ipp.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg.update(n_episodes=1, debug=False, bg_mode="off",
               n_init_observations=0, n_robots=args.n_robots,
               forecast_mode=args.forecast_mode)
    cfg["gp_kernel"] = args.gp_kernel
    cfg["gp_matern_nu"] = args.gp_matern_nu
    cfg["gp_length_scale_init"] = args.gp_length_scale_init
    cfg["gp_length_scale_bounds"] = [0.01, max(0.5, args.gp_length_scale_init * 5)]

    d = load_test_data()
    om = d["ocean_mask"]
    H, W = om.shape
    cand = build_candidates(d["all_ocean_coords"], cfg.get("n_candidates", 2000),
                            cfg.get("candidate_seed", 42))
    evl = build_eval_cells(d["all_ocean_coords"], cfg.get("n_eval_cells", 20000),
                           cfg.get("eval_seed", 123))

    pol = None
    if args.use_policy:
        pol = build_policies({"policies": {
            "uncertainty_only": {"type": "uncertainty_only", "lambda_dist": 0.0}
        }})["uncertainty_only"]

    seed = cfg.get("episode_seed_offset", 0) + _stable_hash("uncertainty_only") % 1000
    L = cfg["episode_length"]
    sr, _, qf = run_dynamic_episode(
        d["inputs_np"], om, d["all_ocean_coords"],
        cand, evl, pol, d["fno"], d["device"], cfg,
        episode_seed=seed, t0=args.t0, save_qual_steps=set(range(1, L + 1)))

    rows_info = find_ocean_rows(om, min_length=50)
    results = {}
    for s in range(1, L + 1):
        k, psd = psd_1d_hanning(qf[s]["x_corrected"], om, rows_info)
        results[f"pred_step{s}"] = psd
        _, psd_gt = psd_1d_hanning(qf[s]["y_true"], om, rows_info)
        results[f"gt_step{s}"] = psd_gt
    results["k"] = k
    results["L"] = np.array(L)

    out = os.path.join(args.out_dir,
                       f"psd_t0_{args.t0}_{args.n_robots}bots_{args.tag}.npz")
    np.savez(out, **results)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
