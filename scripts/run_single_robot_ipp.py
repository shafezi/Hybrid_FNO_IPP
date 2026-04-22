#!/usr/bin/env python
"""
Single-robot informative path planning (IPP) experiment.

Usage
-----
    python scripts/run_single_robot_ipp.py --config configs/single_robot_ipp.yaml

What it does
------------
1. Load FNO checkpoint and run inference on a held-out test year.
2. For each episode (one SSH snapshot) and each policy, run a sequential
   sensing simulation where the robot:
     a. Uses the FNO prediction as a prior mean.
     b. Maintains a residual GP that is updated after each observation.
     c. Picks the next waypoint using the policy's acquisition function.
3. Save per-step metrics, trajectories, and qualitative visualizations.
"""

import os
import sys
import argparse
import warnings
import yaml
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.fno_model     import load_fno_model, PECstep
from experiments.data_utils    import load_experiment_data, normalize, denormalize
from experiments.gp_correction import build_ocean_coords
from ipp.policies              import build_policies
from ipp.simulator             import run_all_experiments
from ipp.ipp_visualization     import (
    plot_metric_vs_steps,
    plot_unobs_rmse_vs_steps,
    plot_trajectories,
    plot_ipp_qualitative,
    plot_policy_comparison_bar,
    plot_ipp_summary_table,
    plot_distance_vs_rmse,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Path to YAML config file")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== Single-Robot IPP ===")
    print(f"Output: {out_dir}\n")

    # ------------------------------------------------------------------
    # 1. Device
    # ------------------------------------------------------------------
    import torch
    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 2. Load FNO and run inference
    # ------------------------------------------------------------------
    print("\n[1/5] Loading FNO model …")
    model = load_fno_model(
        cfg["checkpoint_path"],
        modes1=cfg.get("fno_modes1", 128),
        modes2=cfg.get("fno_modes2", 128),
        width=cfg.get("fno_width",  20),
        device=device,
    )
    model.eval()

    print("[2/5] Loading data and running FNO inference …")
    (inputs, labels, ocean_mask, mean_field, std_field,
     ocean_grid_size, lat_rho, lon_rho) = load_experiment_data(
        cfg["data_dir"], cfg["test_year"], cfg["lead"], device,
    )

    H, W = ocean_mask.shape
    n_test = inputs.shape[0]

    # Run FNO inference in batches via PECstep
    # inputs shape: (N, 1, H, W) raw (not normalized)
    # PECstep expects: (B, H, W, 1) normalized, and appends x/y coordinate grids
    bs = cfg.get("inference_batch_size", 8)
    delta_t = cfg.get("delta_t", 1.0)
    preds_list, labs_list = [], []
    with torch.no_grad():
        for start in range(0, n_test, bs):
            inp = inputs[start:start + bs].to(device)   # (B, 1, H, W)
            inp_norm = normalize(inp[:, 0], mean_field, std_field)  # (B, H, W)
            inp_norm = inp_norm.unsqueeze(-1)                        # (B, H, W, 1)
            pred_norm = PECstep(model, inp_norm, delta_t)            # (B, H, W, 1)
            pred_phys = denormalize(pred_norm[..., 0],
                                    mean_field, std_field)           # (B, H, W)
            lab_raw   = labels[start:start + bs].to(device)[:, 0]   # (B, H, W)
            preds_list.append(pred_phys.cpu())
            labs_list.append(lab_raw.cpu())

    preds_phys  = torch.cat(preds_list, dim=0).numpy()   # (N, H, W)
    labels_phys = torch.cat(labs_list,  dim=0).numpy()   # (N, H, W)

    print(f"  FNO inference done: {preds_phys.shape}")

    # ------------------------------------------------------------------
    # 3. Ocean coordinates
    # ------------------------------------------------------------------
    all_ocean_coords   = build_ocean_coords(ocean_mask)    # (N_ocean, 2)
    flat_ocean_indices = np.where(ocean_mask.ravel())[0]   # (N_ocean,)
    print(f"  Ocean cells: {len(all_ocean_coords):,}")

    # ------------------------------------------------------------------
    # 4. Build policies
    # ------------------------------------------------------------------
    print("\n[3/5] Building policies …")
    policies = build_policies(cfg)
    print(f"  Policies: {list(policies.keys())}")

    # ------------------------------------------------------------------
    # 5. Run IPP simulation
    # ------------------------------------------------------------------
    print("\n[4/5] Running IPP simulation …")
    (all_records, trajectories, qual_records,
     cand_local_idx, eval_local_idx,
     episode_sample_idx) = run_all_experiments(
        preds_phys, labels_phys,
        ocean_mask, all_ocean_coords, flat_ocean_indices,
        policies, cfg, verbose=True,
    )

    # ------------------------------------------------------------------
    # 6. Save raw records
    # ------------------------------------------------------------------
    df = pd.DataFrame(all_records)
    # Drop large array columns before saving CSV
    csv_df = df.drop(columns=[c for c in ["gp_mean_all", "gp_std_all"]
                               if c in df.columns], errors="ignore")
    records_path = os.path.join(out_dir, "ipp_records.csv")
    csv_df.to_csv(records_path, index=False)
    print(f"\n  Records saved → {records_path}  ({len(csv_df):,} rows)")

    # ------------------------------------------------------------------
    # 7. Summary stats per policy
    # ------------------------------------------------------------------
    print("\n[5/5] Generating plots and summary …")

    # Final-step metrics
    final_df = csv_df.loc[csv_df.groupby(["policy", "episode"])["step"].idxmax()]
    summary_rows = []
    for pol in policies:
        sub = final_df[final_df["policy"] == pol]
        fno_rmse = sub["fno_rmse"].mean()
        fin_rmse = sub["all_rmse"].mean()
        row = {
            "policy":              pol,
            "final_rmse_mean":     fin_rmse,
            "final_rmse_std":      sub["all_rmse"].std(),
            "final_mae_mean":      sub["all_mae"].mean(),
            "final_mae_std":       sub["all_mae"].std(),
            "fno_rmse":            fno_rmse,
            "pct_improvement_rmse": 100 * (fno_rmse - fin_rmse) / fno_rmse,
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "ipp_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n  ── Policy summary ──")
    for _, r in summary_df.iterrows():
        print(f"  {r['policy']:22s}  "
              f"RMSE={r['final_rmse_mean']:.4f} ± {r['final_rmse_std']:.4f}  "
              f"FNO={r['fno_rmse']:.4f}  "
              f"Δ={r['pct_improvement_rmse']:+.1f}%")

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    pol_list = list(policies.keys())

    # 8a. RMSE vs steps
    plot_metric_vs_steps(
        csv_df, metric="all_rmse",
        title="All-ocean RMSE vs sensing step",
        fno_col="fno_rmse",
        save_path=os.path.join(out_dir, "rmse_vs_steps.png"),
    )

    # 8b. Unobserved RMSE vs steps
    plot_unobs_rmse_vs_steps(
        csv_df,
        save_path=os.path.join(out_dir, "unobs_rmse_vs_steps.png"),
    )

    # 8c. MAE vs steps
    plot_metric_vs_steps(
        csv_df, metric="all_mae",
        title="All-ocean MAE vs sensing step",
        fno_col="fno_mae",
        save_path=os.path.join(out_dir, "mae_vs_steps.png"),
    )

    # 8d. Trajectories
    plot_trajectories(
        trajectories, all_ocean_coords, cand_local_idx,
        n_episodes=cfg["n_episodes"],
        policies_ordered=pol_list,
        save_path=os.path.join(out_dir, "trajectories.png"),
    )

    # 8e. Policy comparison bar chart
    plot_policy_comparison_bar(
        csv_df, metric="all_rmse",
        save_path=os.path.join(out_dir, "policy_comparison_bar.png"),
    )

    # 8f. Distance vs RMSE
    plot_distance_vs_rmse(
        csv_df,
        save_path=os.path.join(out_dir, "distance_vs_rmse.png"),
    )

    # 8g. Summary table
    plot_ipp_summary_table(
        summary_df,
        save_path=os.path.join(out_dir, "summary_table.png"),
    )

    # 8h. Qualitative panels for each policy (episode 0)
    qual_steps = [s for s in cfg.get("qual_steps", [1, 5, 10, 20, 50])
                  if s <= cfg["sensing_budget"]]
    if not qual_steps:
        qual_steps = [cfg["sensing_budget"]]

    sample_idx_ep0 = int(episode_sample_idx[0])
    fno_flat_ep0   = preds_phys[sample_idx_ep0][ocean_mask]
    label_flat_ep0 = labels_phys[sample_idx_ep0][ocean_mask]

    for pol_name, history in qual_records.items():
        if not history or "gp_mean_all" not in history[0]:
            continue
        traj = trajectories.get((0, pol_name))
        if traj is None:
            continue
        save_qual = os.path.join(out_dir, f"qual_{pol_name}.png")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_ipp_qualitative(
                fno_flat_ep0, label_flat_ep0,
                history, traj,
                all_ocean_coords, ocean_mask, H, W,
                steps_to_show=qual_steps,
                save_path=save_qual,
                policy_name=pol_name,
            )
        print(f"  Qualitative plot → {save_qual}")

    # ------------------------------------------------------------------
    # 9. Optional shuffled-residual control
    # ------------------------------------------------------------------
    if cfg.get("run_shuffle_control", False):
        print("\n  Running shuffled-residual control …")
        from ipp.simulator import run_all_experiments as _run

        # Only run the main policy for the control
        ctrl_policies = {k: v for k, v in policies.items()
                         if k == "hybrid_greedy"}
        ctrl_cfg = dict(cfg, output_dir=out_dir)

        (ctrl_records, *_) = _run(
            preds_phys, labels_phys,
            ocean_mask, all_ocean_coords, flat_ocean_indices,
            ctrl_policies, ctrl_cfg, verbose=False,
        )
        # Re-run with shuffle
        from ipp.simulator import (
            build_candidates, build_eval_cells, run_single_policy_episode
        )
        cand_idx = build_candidates(all_ocean_coords, cfg["n_candidates"],
                                    cfg.get("candidate_seed", 42))
        eval_idx = build_eval_cells(all_ocean_coords,
                                    cfg.get("n_eval_cells", 20000),
                                    cfg.get("eval_seed", 123))
        rng_ep = np.random.default_rng(cfg.get("episode_seed_offset", 0))
        ep_indices = rng_ep.choice(preds_phys.shape[0],
                                   size=cfg["n_episodes"], replace=False)
        shuf_records = []
        pol_obj = list(ctrl_policies.values())[0]
        pol_name = list(ctrl_policies.keys())[0]
        for ep_i, si in enumerate(ep_indices):
            ep_seed = cfg.get("episode_seed_offset", 0) * 10_000 + ep_i * 100
            hist, _ = run_single_policy_episode(
                preds_phys[si][ocean_mask],
                labels_phys[si][ocean_mask],
                all_ocean_coords, cand_idx, eval_idx,
                pol_obj, cfg["sensing_budget"], cfg,
                episode_seed=ep_seed,
                shuffle_residuals=True,
            )
            for rec in hist:
                rec["episode"]    = ep_i
                rec["sample_idx"] = int(si)
                rec["policy"]     = pol_name + "_shuffle"
                shuf_records.append(rec)

        shuf_df = pd.DataFrame(shuf_records)
        shuf_path = os.path.join(out_dir, "ipp_shuffle_control.csv")
        shuf_df.drop(columns=["gp_mean_all", "gp_std_all"],
                     errors="ignore").to_csv(shuf_path, index=False)

        # Compare real vs shuffled on same plot
        combined = pd.concat(
            [csv_df[csv_df["policy"] == "hybrid_greedy"],
             shuf_df.drop(columns=["gp_mean_all", "gp_std_all"],
                          errors="ignore")],
            ignore_index=True,
        )
        plot_metric_vs_steps(
            combined, metric="all_rmse",
            title="Real vs shuffled residuals — RMSE vs step",
            fno_col="fno_rmse",
            save_path=os.path.join(out_dir, "shuffle_control_rmse.png"),
        )
        print(f"  Shuffle control saved → {shuf_path}")

    print(f"\n=== Done. All outputs in {out_dir} ===\n")


if __name__ == "__main__":
    main()
