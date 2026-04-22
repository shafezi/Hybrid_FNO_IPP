#!/usr/bin/env python
"""
Residual GP Experiment
======================
Tests whether a GP can learn FNO residual structure from sparse observations
and improve one-step-ahead SSH prediction.

Usage
-----
    python scripts/run_residual_gp_experiment.py --config configs/residual_gp.yaml

Outputs (all in config.output_dir):
    aggregate_metrics.csv          — mean/std per budget over all samples+seeds
    per_sample_metrics.csv         — full record for every (sample, budget, seed)
    rmse_vs_budget.png
    mae_vs_budget.png
    improvement_vs_budget.png
    histogram_of_improvement.png
    residual_structure_diagnostic.png
    aggregate_table.png
    qualitative_examples/          — up to n_qualitative_examples panels
    findings_summary.md            — auto-generated interpretation
"""

import argparse
import os
import sys
import time
import json

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Make project root importable regardless of working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.fno_model   import load_fno_model, PECstep
from experiments.data_utils  import load_experiment_data, normalize, denormalize
from experiments.gp_correction import (
    build_ocean_coords, fit_residual_gp, predict_residual_mean, gp_correct_prediction
)
from experiments.metrics      import compute_metrics, improvement_stats
from experiments.visualization import (
    plot_qualitative_example,
    plot_metric_vs_budget,
    plot_improvement_vs_budget,
    plot_improvement_histogram,
    plot_residual_structure,
    plot_aggregate_table,
)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path):
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with:  pip install pyyaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# FNO inference helpers
# ---------------------------------------------------------------------------

def run_fno_predictions(model, inputs_raw, labels_raw, sample_indices,
                         mean_field, std_field, device, delta_t, batch_size=8):
    """
    Run FNO (one PEC step) for a list of sample indices.

    Returns
    -------
    preds_np  : (N, H, W) float32 numpy  predictions in physical units
    labels_np : (N, H, W) float32 numpy  ground truth in physical units
    """
    preds, labs = [], []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(sample_indices), batch_size):
            idxs = sample_indices[start: start + batch_size]
            inp  = inputs_raw[idxs].to(device)   # (B, 1, H, W) raw

            # Normalize
            inp_norm = normalize(inp[:, 0], mean_field, std_field)   # (B, H, W)
            inp_norm = inp_norm.unsqueeze(-1)                         # (B, H, W, 1)

            # PEC step
            pred_norm = PECstep(model, inp_norm, delta_t)             # (B, H, W, 1)

            # Denormalize
            pred_phys = denormalize(pred_norm[..., 0], mean_field, std_field)  # (B, H, W)
            lab_raw   = labels_raw[idxs].to(device)[:, 0]                     # (B, H, W)

            preds.append(pred_phys.cpu().numpy())
            labs.append(lab_raw.cpu().numpy())

    return np.concatenate(preds, axis=0), np.concatenate(labs, axis=0)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(cfg):
    t0 = time.time()

    # ---- setup -------------------------------------------------------------
    dev_cfg = cfg.get("device", "auto")
    if dev_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_cfg)
    print(f"Device: {device}")

    out_dir   = cfg["output_dir"]
    qual_dir  = os.path.join(out_dir, "qualitative_examples")
    os.makedirs(qual_dir, exist_ok=True)

    # ---- load model --------------------------------------------------------
    print("\n[1/5] Loading FNO model...")
    ckpt = cfg["checkpoint_path"]
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(_PROJECT_ROOT, ckpt)
    model = load_fno_model(
        ckpt,
        modes1=cfg["fno_modes1"],
        modes2=cfg["fno_modes2"],
        width=cfg["fno_width"],
        device=device,
    )

    # ---- load data ---------------------------------------------------------
    print("\n[2/5] Loading data...")
    data_dir = cfg["data_dir"]
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(_PROJECT_ROOT, data_dir)

    inputs_raw, labels_raw, ocean_mask, mean_field, std_field, _, lat_rho, lon_rho = \
        load_experiment_data(data_dir, cfg["test_year"], cfg["lead"], device)

    H, W = ocean_mask.shape
    N_total = inputs_raw.shape[0]
    print(f"  Grid: {H}×{W}  |  Ocean cells: {ocean_mask.sum():,}  |  Total test pairs: {N_total}")

    # Build ocean coordinates and flat index array
    all_ocean_coords   = build_ocean_coords(ocean_mask)          # (N_ocean, 2)
    flat_ocean_indices = np.where(ocean_mask.ravel())[0]         # (N_ocean,) int

    # ---- select test samples -----------------------------------------------
    n_samples = min(cfg["n_test_samples"], N_total)
    rng_sample = np.random.default_rng(cfg.get("sample_seed", 0))
    sample_indices = rng_sample.choice(N_total, size=n_samples, replace=False)
    sample_indices = np.sort(sample_indices)
    print(f"  Selected {n_samples} test samples (indices {sample_indices[:5]}...)")

    # ---- run FNO predictions -----------------------------------------------
    print("\n[3/5] Running FNO one-step predictions...")
    preds_np, labels_np = run_fno_predictions(
        model, inputs_raw, labels_raw, sample_indices,
        mean_field, std_field, device,
        delta_t=cfg["delta_t"],
        batch_size=cfg.get("inference_batch_size", 8),
    )
    print(f"  FNO predictions done  ({time.time()-t0:.1f}s total so far)")

    # Flatten to ocean cells only for each sample
    preds_ocean  = preds_np[:, ocean_mask]    # (N, N_ocean)
    labels_ocean = labels_np[:, ocean_mask]   # (N, N_ocean)

    # ---- collect all residuals for structure diagnostics -------------------
    residuals_all = labels_ocean - preds_ocean   # (N, N_ocean)

    # ---- GP correction loop ------------------------------------------------
    print("\n[4/5] Running GP correction for all budgets and seeds...")
    budgets = cfg["observation_budgets"]
    n_seeds = cfg["n_seeds"]

    gp_kwargs = dict(
        kernel_type       = cfg.get("gp_kernel",            "matern"),
        matern_nu         = cfg.get("gp_matern_nu",         1.5),
        length_scale_init = cfg.get("gp_length_scale_init", 0.1),
        normalize_y       = cfg.get("gp_normalize_y",       True),
        n_restarts        = cfg.get("gp_n_restarts",         0),
    )

    records = []   # one row per (sample, budget, seed)

    # Storage for qualitative examples: (sample_idx, budget, seed, r_gp_flat, obs_idx)
    qual_storage = []
    n_qual = cfg.get("n_qualitative_examples", 6)

    total_iters = n_samples * len(budgets) * n_seeds
    done = 0

    for s_i, _ in enumerate(sample_indices):
        pred_flat  = preds_ocean[s_i]     # (N_ocean,)
        label_flat = labels_ocean[s_i]    # (N_ocean,)

        # FNO-only metrics (same for all budgets/seeds)
        m_fno = compute_metrics(label_flat, pred_flat)

        for budget in budgets:
            for seed in range(n_seeds):
                rng_obs = np.random.default_rng(seed * 10000 + s_i)
                obs_idx = rng_obs.choice(len(label_flat), size=budget, replace=False)

                corrected_flat, r_gp_flat, gp = gp_correct_prediction(
                    pred_flat, label_flat, all_ocean_coords, obs_idx, gp_kwargs
                )

                m_corr = compute_metrics(label_flat, corrected_flat)
                impr   = improvement_stats(m_fno["mae"], m_fno["rmse"], m_corr["mae"], m_corr["rmse"])

                records.append({
                    "sample_id":           s_i,
                    "sample_data_idx":     int(sample_indices[s_i]),
                    "budget":              budget,
                    "seed":                seed,
                    "fno_mae":             m_fno["mae"],
                    "fno_rmse":            m_fno["rmse"],
                    "corrected_mae":       m_corr["mae"],
                    "corrected_rmse":      m_corr["rmse"],
                    "delta_mae":           impr["delta_mae"],
                    "delta_rmse":          impr["delta_rmse"],
                    "pct_mae":             impr["pct_mae"],
                    "pct_rmse":            impr["pct_rmse"],
                })

                # Store for qualitative plotting (first few samples, first seed)
                if seed == 0 and len(qual_storage) < n_qual:
                    qual_storage.append({
                        "s_i":    s_i,
                        "budget": budget,
                        "seed":   seed,
                        "r_gp":   r_gp_flat.copy(),
                        "obs_idx": obs_idx.copy(),
                    })

                done += 1
                if done % max(1, total_iters // 20) == 0:
                    elapsed = time.time() - t0
                    print(f"  {done}/{total_iters} ({100*done/total_iters:.0f}%)  |  {elapsed:.1f}s elapsed")

    df = pd.DataFrame(records)

    # ---- save per-sample CSV -----------------------------------------------
    per_sample_path = os.path.join(out_dir, "per_sample_metrics.csv")
    df.to_csv(per_sample_path, index=False)
    print(f"\nSaved per-sample metrics → {per_sample_path}")

    # ---- aggregate -----------------------------------------------------------
    print("\n[5/5] Aggregating, plotting, and saving results...")

    agg_records = []
    for budget in budgets:
        sub = df[df["budget"] == budget]
        agg_records.append({
            "budget":               budget,
            "fno_mae_mean":         sub["fno_mae"].mean(),
            "fno_mae_std":          sub["fno_mae"].std(),
            "fno_rmse_mean":        sub["fno_rmse"].mean(),
            "fno_rmse_std":         sub["fno_rmse"].std(),
            "corrected_mae_mean":   sub["corrected_mae"].mean(),
            "corrected_mae_std":    sub["corrected_mae"].std(),
            "corrected_rmse_mean":  sub["corrected_rmse"].mean(),
            "corrected_rmse_std":   sub["corrected_rmse"].std(),
            "delta_mae_mean":       sub["delta_mae"].mean(),
            "delta_mae_std":        sub["delta_mae"].std(),
            "delta_rmse_mean":      sub["delta_rmse"].mean(),
            "delta_rmse_std":       sub["delta_rmse"].std(),
            "pct_mae_mean":         sub["pct_mae"].mean(),
            "pct_rmse_mean":        sub["pct_rmse"].mean(),
            "pct_improved_rmse":    100.0 * (sub["delta_rmse"] > 0).mean(),
        })

    agg_df = pd.DataFrame(agg_records)
    agg_path = os.path.join(out_dir, "aggregate_metrics.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"Saved aggregate metrics → {agg_path}")

    # ---- plots ---------------------------------------------------------------
    plot_metric_vs_budget(agg_df, "rmse",
                          os.path.join(out_dir, "rmse_vs_budget.png"))
    plot_metric_vs_budget(agg_df, "mae",
                          os.path.join(out_dir, "mae_vs_budget.png"))
    plot_improvement_vs_budget(agg_df,
                               os.path.join(out_dir, "improvement_vs_budget.png"))

    hist_budget = cfg.get("histogram_budget", budgets[len(budgets)//2])
    plot_improvement_histogram(df, hist_budget,
                               os.path.join(out_dir, "histogram_of_improvement.png"))

    plot_aggregate_table(agg_df, os.path.join(out_dir, "aggregate_table.png"))

    # ---- residual structure -------------------------------------------------
    struct_stats = plot_residual_structure(
        [residuals_all[i] for i in range(residuals_all.shape[0])],
        all_ocean_coords,
        os.path.join(out_dir, "residual_structure_diagnostic.png"),
    )
    print(f"Residual structure stats: {struct_stats}")

    # ---- qualitative examples -----------------------------------------------
    print(f"Saving {len(qual_storage)} qualitative examples...")
    for q in qual_storage:
        s_i    = q["s_i"]
        budget = q["budget"]
        seed   = q["seed"]

        # Reconstruct full 2D fields
        def flat_to_2d(arr_flat):
            out = np.full(H * W, np.nan, dtype=np.float32)
            out[flat_ocean_indices] = arr_flat
            return out.reshape(H, W)

        label_2d     = flat_to_2d(labels_ocean[s_i])
        pred_2d      = flat_to_2d(preds_ocean[s_i])
        r_gp_2d      = flat_to_2d(q["r_gp"])
        corrected_2d = flat_to_2d(preds_ocean[s_i] + q["r_gp"])

        fname = f"sample{s_i:03d}_budget{budget}_seed{seed}.png"
        plot_qualitative_example(
            label_2d, pred_2d, corrected_2d, ocean_mask, r_gp_2d,
            obs_coords_idx   = q["obs_idx"],
            all_ocean_flat_idx = flat_ocean_indices,
            H=H, W=W,
            sample_id=s_i,
            budget=budget,
            seed=seed,
            save_path=os.path.join(qual_dir, fname),
        )

    # ---- text summary -------------------------------------------------------
    summary_path = os.path.join(out_dir, "findings_summary.md")
    _write_findings_summary(summary_path, cfg, agg_df, struct_stats, df, budgets, n_samples, n_seeds)
    print(f"Saved findings summary → {summary_path}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s  |  Results in: {out_dir}")
    print(agg_df[["budget", "fno_rmse_mean", "corrected_rmse_mean",
                  "delta_rmse_mean", "pct_rmse_mean", "pct_improved_rmse"]].to_string(index=False))

    return agg_df, df


# ---------------------------------------------------------------------------
# Auto-generated findings summary
# ---------------------------------------------------------------------------

def _write_findings_summary(path, cfg, agg_df, struct_stats, df, budgets, n_samples, n_seeds):
    # Best budget
    best_row = agg_df.loc[agg_df["delta_rmse_mean"].idxmax()]
    worst_budget = budgets[0]

    with open(path, "w") as f:
        f.write("# Residual GP Experiment — Findings Summary\n\n")
        f.write(f"**Generated automatically by run_residual_gp_experiment.py**\n\n")
        f.write("## Experiment Setup\n\n")
        f.write(f"- Test year: {cfg['test_year']}\n")
        f.write(f"- Prediction horizon: lead={cfg['lead']} × 5 days = {cfg['lead']*5} days\n")
        f.write(f"- Test samples: {n_samples}\n")
        f.write(f"- Seeds per run: {n_seeds}\n")
        f.write(f"- Observation budgets: {budgets}\n")
        f.write(f"- GP kernel: {cfg.get('gp_kernel','matern')} (ν={cfg.get('gp_matern_nu',1.5)})\n\n")

        f.write("## Aggregate Results\n\n")
        f.write(agg_df[["budget", "fno_rmse_mean", "corrected_rmse_mean",
                         "delta_rmse_mean", "pct_rmse_mean",
                         "pct_improved_rmse"]].to_markdown(index=False))
        f.write("\n\n")

        f.write("## Key Findings\n\n")

        # Viability
        best_pct = float(best_row["pct_rmse_mean"])
        best_budget = int(best_row["budget"])
        best_pct_imp = float(best_row["pct_improved_rmse"])

        if best_pct > 5:
            verdict = "**VIABLE** — GP correction provides meaningful improvement."
        elif best_pct > 1:
            verdict = "**MARGINALLY VIABLE** — GP correction provides modest improvement."
        else:
            verdict = "**QUESTIONABLE** — GP correction provides negligible improvement."

        f.write(f"**Project viability verdict:** {verdict}\n\n")
        f.write(f"- Best budget: {best_budget} observations → {best_pct:.1f}% RMSE reduction, "
                f"{best_pct_imp:.0f}% of samples improved.\n")

        if struct_stats.get("corr_length_normalized"):
            cl = struct_stats["corr_length_normalized"]
            f.write(f"- Estimated residual correlation length: {cl:.3f} (normalized units)\n")
            f.write(f"  → Residuals are {'spatially structured (long range)' if cl > 0.05 else 'relatively uncorrelated (short range)'}\n")

        f.write("\n## Interpretation\n\n")
        f.write("The FNO residual represents the systematic error the neural operator\n")
        f.write("cannot capture in one step. If this residual is spatially correlated,\n")
        f.write("a GP can learn it from sparse observations and improve predictions.\n\n")
        f.write("A positive RMSE improvement (> few %) with a growing trend across\n")
        f.write("budgets strongly supports the FNO-prior + residual-GP project direction.\n\n")

        f.write("## Limitations\n\n")
        f.write("- GP uses grid-index coordinates (not physical lat/lon km distance).\n")
        f.write("- Kernel hyperparameters optimized per run; may not reflect global structure.\n")
        f.write("- Only 2020 validation year evaluated; generalization to other years not tested.\n")
        f.write("- No multi-step or autoregressive GP correction implemented in this version.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Residual GP Experiment")
    parser.add_argument("--config", default="configs/residual_gp.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_PROJECT_ROOT, config_path)

    print(f"Loading config: {config_path}")
    cfg = load_config(config_path)

    # Resolve relative output dir
    if not os.path.isabs(cfg["output_dir"]):
        cfg["output_dir"] = os.path.join(_PROJECT_ROOT, cfg["output_dir"])

    run_experiment(cfg)


if __name__ == "__main__":
    main()
