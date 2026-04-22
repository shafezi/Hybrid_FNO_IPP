"""
Partition + multi-robot scaling sweep.

Runs a grid of configs varying n_robots and use_partitioning while keeping
total observations per assimilation constant (= 20).

Usage:
    python scripts/run_partition_sweep.py [--n_episodes 20]

Output:
    results/dynamic_ipp/sweeps/partition_sweep.csv
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from _test_helpers import make_test_config, load_test_data
from ipp.policies import build_policies
from dynamic_ipp.rollout import run_all_dynamic_experiments


TOTAL_OBS_PER_ASSIM = 20  # keep budget constant across configs
POLICY = "uncertainty_only"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=20)
    args = parser.parse_args()

    out_dir = os.path.join(ROOT, "results", "dynamic_ipp", "sweeps")
    os.makedirs(out_dir, exist_ok=True)

    d = load_test_data()
    built = build_policies({"policies": {
        POLICY: {"type": POLICY, "lambda_dist": 0.0},
    }})
    policies_dict = {
        "fno_only": None,
        POLICY: built[POLICY],
    }

    # Also run FNO baseline once
    grid = []
    for n_robots in [1, 2, 4]:
        obs_per_robot = TOTAL_OBS_PER_ASSIM // n_robots
        for use_part in [False, True]:
            if n_robots == 1 and use_part:
                continue  # partitioning meaningless for 1 robot
            grid.append((n_robots, obs_per_robot, use_part))

    print(f"Partition sweep: {len(grid)} configs × {args.n_episodes} episodes")
    print(f"Total obs/assim = {TOTAL_OBS_PER_ASSIM} (constant)")
    print()

    rows = []
    for n_robots, obs_per_robot, use_part in grid:
        label = (f"R={n_robots} obs/r={obs_per_robot} "
                 f"part={'on' if use_part else 'off'}")
        print(f"  {label} …", end="", flush=True)
        t0 = time.time()

        cfg = make_test_config(
            n_episodes=args.n_episodes,
            n_robots=n_robots,
            obs_per_robot_per_assim=obs_per_robot,
            gp_use_time=True,
            gp_mode="cumulative",
            bg_mode="off",
            init_position="random",
            use_partitioning=use_part,
            partition_strict=True,
        )

        records, _, _, _, _ = run_all_dynamic_experiments(
            d["inputs_np"], d["ocean_mask"], d["all_ocean_coords"],
            policies_dict, d["fno"], d["device"], cfg, verbose=False,
        )
        df = pd.DataFrame(records)
        elapsed = time.time() - t0

        # FNO baseline RMSE
        fno_final = df[(df["policy"] == "fno_only")].groupby("episode")["step"].idxmax()
        fno_rmse = df.loc[fno_final, "all_rmse"].mean()

        # Policy final-step metrics
        pol_df = df[df["policy"] == POLICY]
        final_idx = pol_df.groupby("episode")["step"].idxmax()
        final = pol_df.loc[final_idx]

        rmse_mean = final["all_rmse"].mean()
        rmse_std  = final["all_rmse"].std(ddof=0)
        pct_imp   = 100 * (fno_rmse - rmse_mean) / fno_rmse if fno_rmse > 0 else 0
        dist_mean = final["dist_per_robot_mean"].mean()
        spill_mean = final["partition_spillover_count_total"].mean()

        rows.append(dict(
            n_robots=n_robots,
            obs_per_robot=obs_per_robot,
            use_partitioning=use_part,
            final_rmse_mean=round(rmse_mean, 4),
            final_rmse_std=round(rmse_std, 4),
            fno_rmse=round(fno_rmse, 4),
            pct_improvement=round(pct_imp, 2),
            dist_per_robot_mean=round(dist_mean, 4),
            spillover_total_mean=round(spill_mean, 2),
        ))
        print(f"  {elapsed:.0f}s  RMSE={rmse_mean:.4f}  "
              f"({pct_imp:+.1f}% vs FNO)")

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "partition_sweep.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print()
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()