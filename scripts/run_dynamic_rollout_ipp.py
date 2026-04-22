"""
Dynamic Rollout IPP Experiment
==============================

FNO autoregressive rollout with periodic GP assimilation.

Usage
-----
    python scripts/run_dynamic_rollout_ipp.py --config configs/dynamic_rollout_ipp.yaml

Outputs saved to cfg["output_dir"]:
    per_step_metrics.csv
    per_episode_metrics.csv
    aggregate_metrics.csv
    plots/
    qualitative_examples/
    trajectories/
    findings_summary.md
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

# ── project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from experiments.fno_model     import load_fno_model
from experiments.data_utils    import load_experiment_data, normalize
from experiments.gp_correction import build_ocean_coords
from ipp.policies               import build_policies

from dynamic_ipp.rollout       import run_all_dynamic_experiments, _fno_step
from dynamic_ipp.visualization import (
    plot_rmse_vs_steps,
    plot_rmse_vs_distance,
    plot_rmse_vs_nobs,
    plot_summary_bar,
    plot_summary_table,
    plot_trajectories,
    plot_qualitative_panels,
)


# =============================================================================
# Helpers
# =============================================================================

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(cfg_device):
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def build_policies_dict(cfg):
    """
    Build the {name: policy_or_None} dict from config.
    'fno_only' is represented as None (no assimilation).

    build_policies() expects cfg["policies"] as a nested dict; it reads
    alpha/beta/lambda_dist from each policy's own sub-dict, so hyperparameters
    from the YAML are correctly applied.
    """
    policies_cfg = cfg.get("policies", {})
    result = {}

    # Separate fno_only (not a real policy, just means no assimilation)
    non_fno = {k: v for k, v in policies_cfg.items()
               if v.get("type", k) != "fno_only"}

    if non_fno:
        # Pass a sub-config that has the "policies" key build_policies expects
        built = build_policies({"policies": non_fno})
        result.update(built)

    # Insert fno_only entries as None, preserving YAML order
    ordered = {}
    for name, params in policies_cfg.items():
        if params.get("type", name) == "fno_only":
            ordered[name] = None
        else:
            ordered[name] = result[name]
    return ordered


def inputs_to_numpy(inputs_tensor, mean_field, std_field):
    """
    Normalize and convert inputs to (N, H, W) float32 numpy array.
    inputs_tensor shape: (N, 1, H, W) raw SSH (load_experiment_data convention).
    Normalization: (raw - mean) / std  (same as FNO training).
    """
    # Normalize in torch (keeps device handling simple)
    # inputs shape: (N, 1, H, W); mean/std shape: (H, W)
    inp = inputs_tensor.float()
    # Broadcast mean/std over batch and channel dims
    m = mean_field.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
    s = std_field.unsqueeze(0).unsqueeze(0)    # (1,1,H,W)
    inp_norm = (inp - m) / s                   # (N,1,H,W)
    arr = inp_norm.cpu().numpy()[:, 0]         # (N,H,W)
    return arr.astype(np.float32)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Dynamic rollout IPP experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = resolve_device(cfg.get("device", "cpu"))

    out_dir       = cfg["output_dir"]
    plots_dir     = os.path.join(out_dir, "plots")
    qual_dir      = os.path.join(out_dir, "qualitative_examples")
    traj_dir      = os.path.join(out_dir, "trajectories")
    for d in [out_dir, plots_dir, qual_dir, traj_dir]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Load FNO ──────────────────────────────────────────────────────────
    print("[1/5] Loading FNO model …")
    fno = load_fno_model(
        cfg["checkpoint_path"],
        modes1=cfg.get("fno_modes1", 128),
        modes2=cfg.get("fno_modes2", 128),
        width =cfg.get("fno_width",  20),
        device=device,
    )
    fno.eval()
    print(f"      FNO loaded on {device}")

    # ── 2. Load data ─────────────────────────────────────────────────────────
    print("[2/5] Loading test data …")
    (inputs, labels, ocean_mask, mean_field, std_field,
     ocean_grid_size, lat_rho, lon_rho) = load_experiment_data(
        data_dir  = cfg["data_dir"],
        test_year = cfg.get("test_year", 2020),
        lead      = cfg.get("lead", 4),
        device    = device,
    )

    # Normalize to the same space the FNO was trained in: (raw - mean) / std
    inputs_np     = inputs_to_numpy(inputs, mean_field, std_field)   # (N, H, W)
    ocean_mask_np = ocean_mask.astype(bool)

    N, H, W = inputs_np.shape
    print(f"      Loaded {N} samples  |  grid {H}×{W}  "
          f"|  ocean cells {ocean_mask_np.sum():,}")

    # Sanity check: need enough samples for episodes
    L                   = cfg.get("episode_length", 30)
    n_episodes          = cfg.get("n_episodes", 5)
    rollout_lead_stride = cfg.get("rollout_lead_stride", cfg.get("lead", 1))
    days_per_step       = 5 * rollout_lead_stride
    needed              = L * rollout_lead_stride + 2
    if N < needed:
        raise ValueError(
            f"episode_length={L} × rollout_lead_stride={rollout_lead_stride} "
            f"requires at least {needed} samples, but only {N} loaded."
        )

    # ── 3. Ocean coordinate grid ──────────────────────────────────────────────
    print("[3/5] Building ocean coordinate grid …")
    all_ocean_coords = build_ocean_coords(ocean_mask_np)   # (N_ocean, 2)
    print(f"      Ocean cells: {len(all_ocean_coords):,}")

    # ── 3b. Stride sanity check ───────────────────────────────────────────
    print("[3b/5] Stride sanity check …")
    delta_t_diag = cfg.get("delta_t", 1.0)
    n_diag       = min(10, N - rollout_lead_stride - 1)
    rng_diag     = np.random.default_rng(0)
    t_samples    = rng_diag.choice(N - rollout_lead_stride - 1, size=n_diag, replace=False)
    rmse_stride1 = []
    rmse_strideK = []
    for t in t_samples:
        pred = _fno_step(fno, inputs_np[t], delta_t_diag, device)
        rmse_stride1.append(float(np.sqrt(np.mean((pred - inputs_np[t + 1]) ** 2))))
        rmse_strideK.append(float(np.sqrt(np.mean((pred - inputs_np[t + rollout_lead_stride]) ** 2))))
    print(f"      PECstep vs t+1   (stride=1, {5} days):  mean RMSE = {np.mean(rmse_stride1):.4f}")
    print(f"      PECstep vs t+{rollout_lead_stride} (stride={rollout_lead_stride}, {5*rollout_lead_stride} days): mean RMSE = {np.mean(rmse_strideK):.4f}")
    if np.mean(rmse_strideK) < np.mean(rmse_stride1):
        print(f"      → stride={rollout_lead_stride} matches PECstep better  (config is correct)")
    else:
        print(f"      → stride=1 matches PECstep better  (consider setting rollout_lead_stride=1)")

    # ── 4. Build policies ────────────────────────────────────────────────────
    print("[4/5] Building policies …")
    policies_dict = build_policies_dict(cfg)
    print(f"      Policies: {list(policies_dict.keys())}")

    # ── 5. Run experiments ────────────────────────────────────────────────────
    print(f"[5/5] Running dynamic rollout  "
          f"(L={L}, K={cfg.get('assimilation_interval',5)}, "
          f"obs/assim={cfg.get('obs_per_assimilation',5)}, "
          f"mode={cfg.get('gp_mode','windowed')}) …")
    t0_wall = time.time()

    all_records, trajectories, qual_data, cand_local_idx, eval_local_idx = \
        run_all_dynamic_experiments(
            inputs_np, ocean_mask_np, all_ocean_coords,
            policies_dict, fno, device, cfg, verbose=True,
        )

    elapsed = time.time() - t0_wall
    print(f"      Done in {elapsed:.1f}s")

    # ── 6. Save metrics ───────────────────────────────────────────────────────
    print("Saving results …")
    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(out_dir, "per_step_metrics.csv"), index=False)

    # Per-episode summary (final step of each episode)
    final_df = df.loc[df.groupby(["policy", "episode"])["step"].idxmax()]
    final_df.to_csv(os.path.join(out_dir, "per_episode_metrics.csv"), index=False)

    # Baseline: final-step RMSE of fno_only policy (same slice as compared metrics)
    if "fno_only" in final_df["policy"].values:
        fno_baseline_rmse = final_df[final_df["policy"] == "fno_only"]["all_rmse"].mean()
    else:
        # Fallback: mean of fno_rmse column at final step across all policies
        fno_baseline_rmse = final_df["fno_rmse"].mean()

    agg_rows = []
    for pol in df["policy"].unique():
        sub = final_df[final_df["policy"] == pol]
        agg_rows.append(dict(
            policy               = pol,
            final_rmse_mean      = sub["all_rmse"].mean(),
            final_rmse_std       = sub["all_rmse"].std(ddof=0),
            final_mae_mean       = sub["all_mae"].mean(),
            final_mae_std        = sub["all_mae"].std(ddof=0),
            unobs_rmse_mean      = sub["unobs_rmse"].mean(),
            total_meas_mean       = sub["n_meas_total"].mean(),
            total_meas_robot_mean = sub["n_meas_robot_total"].mean(),
            total_meas_bg_mean    = sub["n_meas_bg_total"].mean(),
            total_unique_mean     = sub["n_unique_total"].mean(),
            total_unique_robot    = sub["n_unique_robot_sites"].mean(),
            total_dist_mean      = sub["cumulative_dist"].mean(),
            fno_only_rmse        = fno_baseline_rmse,
            pct_improvement_rmse = 100 * (fno_baseline_rmse - sub["all_rmse"].mean())
                                   / fno_baseline_rmse,
        ))
    summary_df = pd.DataFrame(agg_rows)
    summary_df.to_csv(os.path.join(out_dir, "aggregate_metrics.csv"), index=False)

    print("\nAggregate results:")
    print(summary_df[["policy", "final_rmse_mean", "final_rmse_std",
                       "pct_improvement_rmse"]].to_string(index=False))

    # ── 7. Save trajectories ──────────────────────────────────────────────────
    for pol_name, traj_list in trajectories.items():
        for ep_i, robot_trajs in enumerate(traj_list):
            # robot_trajs is a list of (T_r, 2) arrays, one per robot
            if isinstance(robot_trajs, list):
                for r_i, traj in enumerate(robot_trajs):
                    np.save(
                        os.path.join(traj_dir, f"traj_{pol_name}_ep{ep_i}_r{r_i}.npy"), traj)
            else:
                # Backward compat: single array
                np.save(
                    os.path.join(traj_dir, f"traj_{pol_name}_ep{ep_i}.npy"), robot_trajs)

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    print("Generating plots …")

    plot_rmse_vs_steps(
        df, os.path.join(plots_dir, "rollout_rmse_vs_steps.png"),
        days_per_step=days_per_step)
    plot_rmse_vs_distance(
        df, os.path.join(plots_dir, "rmse_vs_distance.png"))
    plot_rmse_vs_nobs(
        df, os.path.join(plots_dir, "rmse_vs_nobs.png"))
    plot_summary_bar(
        summary_df, fno_baseline_rmse,
        os.path.join(plots_dir, "summary_bar.png"))
    plot_summary_table(
        summary_df, fno_baseline_rmse,
        os.path.join(plots_dir, "summary_table.png"))
    plot_trajectories(
        trajectories, ocean_mask_np, all_ocean_coords,
        os.path.join(plots_dir, "trajectories.png"))

    # Qualitative panels
    for pol_name, frames in qual_data.items():
        if frames:
            plot_qualitative_panels(
                frames, pol_name, ocean_mask_np,
                os.path.join(qual_dir, f"qual_{pol_name}.png"))

    # ── 9. findings_summary.md ────────────────────────────────────────────────
    best   = summary_df.loc[summary_df["final_rmse_mean"].idxmin()]
    fno_final_rmse = fno_baseline_rmse

    with open(os.path.join(out_dir, "findings_summary.md"), "w") as f:
        f.write("# Dynamic Rollout IPP — Findings Summary\n\n")
        f.write(f"**Episode length:** {L} steps (×{days_per_step} days/step = {L*days_per_step} days)  \n")
        f.write(f"**Assimilation interval:** every {cfg.get('assimilation_interval',5)} steps  \n")
        f.write(f"**Obs per assimilation (robot):** {cfg.get('obs_per_assimilation',5)}  \n")
        f.write(f"**Background sensors (n_init_observations):** {cfg.get('n_init_observations',0)}  \n")
        f.write(f"**GP mode:** {cfg.get('gp_mode','windowed')}  \n")
        f.write(f"**Episodes:** {n_episodes}  \n\n")
        f.write("## Observation Budget Note\n\n")
        f.write("> **Measurement count** (`n_meas_*`) = total GP observations added "
                "(assimilation_events × obs_per_assimilation).  \n")
        f.write("> **Unique sites** (`n_unique_*`) = distinct spatial locations visited.  \n")
        f.write("> In **windowed** mode the robot may revisit the same locations across "
                "windows (each window resets), so unique sites < measurement count.  \n")
        f.write("> In **cumulative** mode revisits are suppressed, so they are equal.  \n\n")
        f.write("## Aggregate Results\n\n")
        f.write(summary_df[["policy", "final_rmse_mean", "final_rmse_std",
                             "pct_improvement_rmse",
                             "total_meas_mean", "total_unique_robot"]].to_string(index=False))
        f.write(f"\n\n**FNO-only final RMSE:** {fno_final_rmse:.4f}  \n")
        f.write(f"**Best policy:** {best['policy']}  "
                f"(RMSE={best['final_rmse_mean']:.4f}, "
                f"{best['pct_improvement_rmse']:.1f}% improvement)  \n")

    print(f"\nAll outputs saved to: {out_dir}")
    print(f"  per_step_metrics.csv, aggregate_metrics.csv, plots/, qualitative_examples/")
    print(f"\nBest policy: {best['policy']}  "
          f"(+{best['pct_improvement_rmse']:.1f}% over FNO rollout)")


if __name__ == "__main__":
    main()
