"""
Budget-scaling sweep.

Unlike the partition sweep (which held total-obs-per-assim constant),
this sweep holds obs_per_robot constant and lets total observations
scale linearly with n_robots.  This isolates "does extra budget help?"
from "does splitting the budget help?".

Grid:
    n_robots   ∈ {1, 2, 4, 8}
    use_partitioning ∈ {False, True}   (skipped for n_robots=1)
    selection_mode   ∈ {"batch", "sequential"}  (skipped for n_robots=1)

obs_per_robot_per_assim = 20 (constant), so total_obs_per_assim scales
with n_robots.

Usage:
    python scripts/run_budget_scaling_sweep.py [--n_episodes 20] [--n_robots 1,2,4,8]

Output:
    results/dynamic_ipp/sweeps/budget_scaling_sweep.csv
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


OBS_PER_ROBOT = 20  # constant across configs
POLICY = "uncertainty_only"


def _parse_robots_arg(s):
    return [int(x) for x in s.split(",") if x.strip()]


def _build_grid(n_robots_list):
    """
    Build the list of (n_robots, use_partitioning, selection_mode) configs.
    Skip meaningless combinations for n_robots=1.
    """
    grid = []
    for n_robots in n_robots_list:
        if n_robots == 1:
            grid.append((1, False, "batch"))
            continue
        for use_part in [False, True]:
            for sel_mode in ["batch", "sequential"]:
                grid.append((n_robots, use_part, sel_mode))
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument(
        "--n_robots", type=_parse_robots_arg, default=[1, 2, 4, 8],
        help="Comma-separated list of n_robots values (default: 1,2,4,8)",
    )
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

    grid = _build_grid(args.n_robots)
    print(f"Budget-scaling sweep: {len(grid)} configs × {args.n_episodes} episodes")
    print(f"obs_per_robot = {OBS_PER_ROBOT} (constant); "
          f"total_obs_per_assim scales with n_robots")
    print()

    rows = []
    for n_robots, use_part, sel_mode in grid:
        total_obs = n_robots * OBS_PER_ROBOT
        label = (f"R={n_robots} total={total_obs} "
                 f"part={'on' if use_part else 'off'} sel={sel_mode}")
        print(f"  {label} …", end="", flush=True)
        t0 = time.time()

        cfg = make_test_config(
            n_episodes=args.n_episodes,
            n_robots=n_robots,
            obs_per_robot_per_assim=OBS_PER_ROBOT,
            gp_use_time=True,
            gp_mode="cumulative",
            bg_mode="off",
            init_position="random",
            use_partitioning=use_part,
            partition_strict=True,
            selection_mode=sel_mode,
            episode_seed_offset=0,   # identical t0_list across configs
        )

        records, _, _, _, _ = run_all_dynamic_experiments(
            d["inputs_np"], d["ocean_mask"], d["all_ocean_coords"],
            policies_dict, d["fno"], d["device"], cfg, verbose=False,
        )
        df = pd.DataFrame(records)
        elapsed = time.time() - t0

        # FNO baseline RMSE (same for every config but measured here for safety)
        fno_final_idx = df[(df["policy"] == "fno_only")].groupby("episode")["step"].idxmax()
        fno_rmse = df.loc[fno_final_idx, "all_rmse"].mean()

        # Policy final-step metrics
        pol_df = df[df["policy"] == POLICY]
        final_idx = pol_df.groupby("episode")["step"].idxmax()
        final = pol_df.loc[final_idx]

        rmse_mean  = final["all_rmse"].mean()
        rmse_std   = final["all_rmse"].std(ddof=0)
        pct_imp    = 100 * (fno_rmse - rmse_mean) / fno_rmse if fno_rmse > 0 else 0
        dist_mean  = final["dist_per_robot_mean"].mean()
        spill_mean = final["partition_spillover_count_total"].mean()

        rows.append(dict(
            n_robots=n_robots,
            obs_per_robot=OBS_PER_ROBOT,
            use_partitioning=use_part,
            selection_mode=sel_mode,
            total_obs_per_assim=total_obs,
            final_rmse_mean=round(rmse_mean, 4),
            final_rmse_std=round(rmse_std, 4),
            fno_rmse=round(fno_rmse, 4),
            pct_improvement=round(pct_imp, 2),
            dist_per_robot_mean=round(dist_mean, 4),
            spillover_total_mean=round(spill_mean, 2),
            runtime_sec=round(elapsed, 1),
        ))
        print(f"  {elapsed:.0f}s  RMSE={rmse_mean:.4f}  "
              f"({pct_imp:+.1f}% vs FNO)")

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "budget_scaling_sweep.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print()
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
