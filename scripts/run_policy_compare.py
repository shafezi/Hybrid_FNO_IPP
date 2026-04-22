"""Run a single (t0, policy) episode and print metrics."""
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
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--n_robots", type=int, default=20)
    parser.add_argument("--acquisition_mode", type=str, default=None,
                        choices=["myopic", "lookahead"])
    args = parser.parse_args()

    with open(os.path.join(ROOT, "configs/dynamic_rollout_ipp.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg.update(n_episodes=1, debug=False, bg_mode="off",
               n_init_observations=0, n_robots=args.n_robots,
               forecast_mode="fno")
    if args.acquisition_mode is not None:
        cfg["acquisition_mode"] = args.acquisition_mode

    d = load_test_data()
    om = d["ocean_mask"]
    cand = build_candidates(d["all_ocean_coords"],
                            cfg.get("n_candidates", 2000),
                            cfg.get("candidate_seed", 42))
    evl = build_eval_cells(d["all_ocean_coords"],
                           cfg.get("n_eval_cells", 20000),
                           cfg.get("eval_seed", 123))

    pol_params = {"type": args.policy, "lambda_dist": 0.0}
    if args.policy == "gp_ucb":
        pol_params = {"type": "gp_ucb", "delta": 0.1, "input_dim": 3}
    pol = build_policies({"policies": {
        args.policy: pol_params
    }})[args.policy]

    seed = cfg.get("episode_seed_offset", 0) + _stable_hash(args.policy) % 1000
    tw = time.time()
    sr, _, qf = run_dynamic_episode(
        d["inputs_np"], om, d["all_ocean_coords"],
        cand, evl, pol, d["fno"], d["device"], cfg,
        episode_seed=seed, t0=args.t0, save_qual_steps={25})
    elapsed = time.time() - tw

    s = 25
    p, g = qf[s]["x_corrected"], qf[s]["y_true"]
    rmse = _rmse_over_ocean(p, g, om)
    acc = _corr_over_ocean(p, g, om)
    hf = _hf_energy_ratio(p, g, om)
    fss = _fss_over_ocean(p, g, om)
    acq_mode = cfg.get("acquisition_mode", "lookahead")
    print(f"{args.t0},{args.policy},{acq_mode},{rmse:.4f},{acc:.4f},{hf:.4f},{fss:.4f},{elapsed:.0f}")


if __name__ == "__main__":
    main()
