"""Run one episode with method-specific optimal hyperparameters.

Saves per-step metrics to a CSV so trajectory-averaged comparisons can be made.
"""
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
    _rmse_over_ocean, _corr_over_ocean, _hf_energy_ratio, _fss_over_ocean,
    _ssim_over_ocean, _bias_over_ocean, _mae_over_ocean, _crmse_over_ocean,
    _emd_over_ocean, _sal_over_ocean)


# Optimal hyperparameters per robot count
# Selected by M2 = RMSE × (1 + |1 - HF|) -- rewards point-wise + spectral fidelity
OPTIMAL_BY_NROBOTS = {
    5: {
        "FNO-only":     {"forecast_mode": "fno",         "use_policy": False,
                         "kernel": "matern", "nu": 1.5, "ls": 0.1},
        "FNO + GP":     {"forecast_mode": "fno",         "use_policy": True,
                         "kernel": "matern", "nu": 2.5, "ls": 0.02},
        "Persist-only": {"forecast_mode": "persistence", "use_policy": False,
                         "kernel": "matern", "nu": 1.5, "ls": 0.1},
        "Persist + GP": {"forecast_mode": "persistence", "use_policy": True,
                         "kernel": "matern", "nu": 0.5, "ls": 0.10},
        "GP-only":      {"forecast_mode": "none",        "use_policy": True,
                         "kernel": "matern", "nu": 1.5, "ls": 0.05},
    },
    10: {
        "FNO-only":     {"forecast_mode": "fno",         "use_policy": False,
                         "kernel": "matern", "nu": 1.5, "ls": 0.1},
        "FNO + GP":     {"forecast_mode": "fno",         "use_policy": True,
                         "kernel": "matern", "nu": 0.5, "ls": 0.02},
        "Persist-only": {"forecast_mode": "persistence", "use_policy": False,
                         "kernel": "matern", "nu": 1.5, "ls": 0.1},
        "Persist + GP": {"forecast_mode": "persistence", "use_policy": True,
                         "kernel": "matern", "nu": 1.5, "ls": 0.20},
        "GP-only":      {"forecast_mode": "none",        "use_policy": True,
                         "kernel": "matern", "nu": 2.5, "ls": 0.05},
    },
    20: {
        "FNO-only":     {"forecast_mode": "fno",         "use_policy": False,
                         "kernel": "matern", "nu": 1.5, "ls": 0.1},
        "FNO + GP":     {"forecast_mode": "fno",         "use_policy": True,
                         "kernel": "matern", "nu": 2.5, "ls": 0.05},
        "Persist-only": {"forecast_mode": "persistence", "use_policy": False,
                         "kernel": "matern", "nu": 1.5, "ls": 0.1},
        "Persist + GP": {"forecast_mode": "persistence", "use_policy": True,
                         "kernel": "matern", "nu": 0.5, "ls": 0.50},
        "GP-only":      {"forecast_mode": "none",        "use_policy": True,
                         "kernel": "rbf",    "nu": 1.5, "ls": 0.05},
    },
    40: {
        "FNO-only":     {"forecast_mode": "fno",         "use_policy": False,
                         "kernel": "matern", "nu": 1.5, "ls": 0.1},
        "FNO + GP":     {"forecast_mode": "fno",         "use_policy": True,
                         "kernel": "rbf",    "nu": 1.5, "ls": 0.05},
        "Persist-only": {"forecast_mode": "persistence", "use_policy": False,
                         "kernel": "matern", "nu": 1.5, "ls": 0.1},
        "Persist + GP": {"forecast_mode": "persistence", "use_policy": True,
                         "kernel": "matern", "nu": 1.5, "ls": 0.30},
        "GP-only":      {"forecast_mode": "none",        "use_policy": True,
                         "kernel": "rbf",    "nu": 1.5, "ls": 0.30},
    },
}

# Default for backward compat
OPTIMAL = OPTIMAL_BY_NROBOTS[10]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0", type=int, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--n_robots", type=int, default=10)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(ROOT, "results", "dynamic_ipp", "optimal"))
    parser.add_argument("--criterion", type=str, default=None,
                        choices=["rmse", "acc", "m2", "mstruct"],
                        help="Use criterion-specific optima from optimal_hyperparameters.py")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.criterion is not None:
        from optimal_hyperparameters import ALL_OPTIMAL
        opt_table = ALL_OPTIMAL[args.criterion].get(args.n_robots, ALL_OPTIMAL[args.criterion][10])
    else:
        opt_table = OPTIMAL_BY_NROBOTS.get(args.n_robots, OPTIMAL_BY_NROBOTS[10])
    opt = opt_table[args.method]

    with open(os.path.join(ROOT, "configs/dynamic_rollout_ipp.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg.update(n_episodes=1, debug=False, bg_mode="off",
               n_init_observations=0, n_robots=args.n_robots,
               forecast_mode=opt["forecast_mode"])
    cfg["gp_kernel"] = opt["kernel"]
    cfg["gp_matern_nu"] = opt["nu"]
    cfg["gp_length_scale_init"] = opt["ls"]
    cfg["gp_length_scale_bounds"] = [0.01, max(0.5, opt["ls"] * 5)]

    d = load_test_data()
    om = d["ocean_mask"]
    cand = build_candidates(d["all_ocean_coords"],
                            cfg.get("n_candidates", 2000),
                            cfg.get("candidate_seed", 42))
    evl = build_eval_cells(d["all_ocean_coords"],
                           cfg.get("n_eval_cells", 20000),
                           cfg.get("eval_seed", 123))

    pol = None
    if opt["use_policy"]:
        pol = build_policies({"policies": {
            "uncertainty_only": {"type": "uncertainty_only", "lambda_dist": 0.0}
        }})["uncertainty_only"]

    seed = cfg.get("episode_seed_offset", 0) + _stable_hash("uncertainty_only") % 1000
    L = cfg["episode_length"]
    sr, _, qf = run_dynamic_episode(
        d["inputs_np"], om, d["all_ocean_coords"],
        cand, evl, pol, d["fno"], d["device"], cfg,
        episode_seed=seed, t0=args.t0, save_qual_steps=set(range(1, L + 1)))

    # Save per-step metrics
    rows = []
    for s in range(1, L + 1):
        p, g = qf[s]["x_corrected"], qf[s]["y_true"]
        sal = _sal_over_ocean(p, g, om)
        rows.append({
            "method": args.method, "step": s, "t0": args.t0, "n_robots": args.n_robots,
            "RMSE": _rmse_over_ocean(p, g, om),
            "ACC": _corr_over_ocean(p, g, om),
            "SSIM": _ssim_over_ocean(p, g, om),
            "HF": _hf_energy_ratio(p, g, om),
            "Bias": _bias_over_ocean(p, g, om),
            "MAE": _mae_over_ocean(p, g, om),
            "CRMSE": _crmse_over_ocean(p, g, om),
            "EMD": _emd_over_ocean(p, g, om),
            "SAL": sal[0] + abs(sal[1]) + sal[2],
            "FSS": _fss_over_ocean(p, g, om),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    method_safe = args.method.replace(" ", "_").replace("+", "and")
    out = os.path.join(args.out_dir,
                       f"optimal_t0_{args.t0}_{args.n_robots}bots_{method_safe}.csv")
    df.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
