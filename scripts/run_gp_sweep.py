"""Sweep GP hyperparameters and report metrics."""
import argparse
import os
import sys
import time
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
from make_comparison_video import (
    _rmse_over_ocean, _corr_over_ocean, _hf_energy_ratio, _fss_over_ocean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0", type=int, required=True)
    parser.add_argument("--n_robots", type=int, default=20)
    parser.add_argument("--forecast_mode", type=str, default="fno")
    parser.add_argument("--gp_kernel", type=str, default="matern")
    parser.add_argument("--gp_matern_nu", type=float, default=1.5)
    parser.add_argument("--gp_length_scale_init", type=float, default=0.1)
    parser.add_argument("--gp_noise_upper", type=float, default=0.01)
    args = parser.parse_args()

    with open(os.path.join(ROOT, "configs/dynamic_rollout_ipp.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg.update(n_episodes=1, debug=False, bg_mode="off",
               n_init_observations=0, n_robots=args.n_robots,
               forecast_mode=args.forecast_mode)
    cfg["gp_kernel"] = args.gp_kernel
    cfg["gp_matern_nu"] = args.gp_matern_nu
    cfg["gp_length_scale_init"] = args.gp_length_scale_init
    cfg["gp_length_scale_bounds"] = [0.01, max(0.5, args.gp_length_scale_init * 5)]
    cfg["gp_noise_bounds"] = [1e-6, args.gp_noise_upper]

    d = load_test_data()
    om = d["ocean_mask"]
    cand = build_candidates(d["all_ocean_coords"],
                            cfg.get("n_candidates", 2000),
                            cfg.get("candidate_seed", 42))
    evl = build_eval_cells(d["all_ocean_coords"],
                           cfg.get("n_eval_cells", 20000),
                           cfg.get("eval_seed", 123))

    pol = build_policies({"policies": {
        "uncertainty_only": {"type": "uncertainty_only", "lambda_dist": 0.0}
    }})["uncertainty_only"]

    seed = cfg.get("episode_seed_offset", 0) + _stable_hash("uncertainty_only") % 1000
    L = cfg["episode_length"]
    all_steps = set(range(1, L + 1))
    tw = time.time()
    sr, _, qf = run_dynamic_episode(
        d["inputs_np"], om, d["all_ocean_coords"],
        cand, evl, pol, d["fno"], d["device"], cfg,
        episode_seed=seed, t0=args.t0, save_qual_steps=all_steps)
    elapsed = time.time() - tw

    # Trajectory-averaged metrics over all 25 steps
    rmses, accs, hfs, fsss = [], [], [], []
    for s in range(1, L + 1):
        p, g = qf[s]["x_corrected"], qf[s]["y_true"]
        rmses.append(_rmse_over_ocean(p, g, om))
        accs.append(_corr_over_ocean(p, g, om))
        hfs.append(_hf_energy_ratio(p, g, om))
        fsss.append(_fss_over_ocean(p, g, om))

    import numpy as np
    rmse_mean = np.mean(rmses)
    acc_mean = np.mean(accs)
    hf_mean = np.mean(hfs)
    fss_mean = np.mean(fsss)

    print(f"{args.t0},{args.forecast_mode},{args.gp_kernel},"
          f"{args.gp_matern_nu},{args.gp_length_scale_init},"
          f"{args.gp_noise_upper},{rmse_mean:.4f},{acc_mean:.4f},"
          f"{hf_mean:.4f},{fss_mean:.4f},{elapsed:.0f}")


if __name__ == "__main__":
    main()
