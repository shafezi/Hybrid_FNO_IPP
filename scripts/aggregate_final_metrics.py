"""Aggregate all per-config metric CSVs into one master DataFrame, then summarize."""
import os
import sys
import glob
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np


def parse_config_from_filename(fname):
    """Extract (seed, t0, n_robots, kernel_label, acquisition) from filename."""
    base = os.path.basename(fname).replace(".csv", "")
    # Pattern: seed{S}_t0{T}_{N}bots_{kernel}_{acq}
    m = re.match(r"seed(\d+)_t0(\d+)_(\d+)bots_([a-z0-9]+)_(.+)", base)
    if not m:
        return None
    return {
        "seed": int(m.group(1)),
        "t0": int(m.group(2)),
        "n_robots": int(m.group(3)),
        "kernel_tag": m.group(4),
        "acquisition": m.group(5),
    }


def main():
    metrics_dir = os.path.join(ROOT, "results", "dynamic_ipp", "final", "metrics")
    out_path = os.path.join(ROOT, "results", "dynamic_ipp", "final", "master_metrics.csv")

    files = sorted(glob.glob(os.path.join(metrics_dir, "*.csv")))
    print(f"Found {len(files)} CSV files")

    rows = []
    for f in files:
        info = parse_config_from_filename(f)
        if info is None:
            print(f"  skipping (bad name): {f}")
            continue
        df = pd.read_csv(f)
        # The CSV may already have these cols (from --metrics_only mode) or not (from older runs)
        # Override with parsed values for safety
        for k, v in info.items():
            if k not in df.columns:
                df[k] = v
        rows.append(df)

    master = pd.concat(rows, ignore_index=True)
    print(f"\nMaster DataFrame: {len(master)} rows")
    print(f"Columns: {list(master.columns)}")

    master.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # ── Quick summary ──
    print("\n===== UNIQUE VALUES =====")
    for col in ["seed", "n_robots", "kernel_tag", "acquisition", "method"]:
        if col in master.columns:
            vals = sorted(master[col].unique())
            print(f"  {col}: {vals}")

    # ── Final-step (last step) average per (n_robots, method) across seeds ──
    L = master["step"].max()
    print(f"\n===== FINAL STEP (step={L}) average across seeds & configs =====")
    final = master[master["step"] == L]

    summary = final.groupby(["n_robots", "method"]).agg(
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        ACC_mean=("ACC", "mean"),
        ACC_std=("ACC", "std"),
        HF_mean=("HF", "mean"),
        HF_std=("HF", "std"),
        FSS_mean=("FSS", "mean"),
        FSS_std=("FSS", "std"),
    ).round(3).reset_index()

    print(summary.to_string(index=False))

    summary.to_csv(out_path.replace("master_metrics.csv", "summary_final_step.csv"),
                   index=False)
    print(f"\nSaved summary: {out_path.replace('master_metrics.csv', 'summary_final_step.csv')}")


if __name__ == "__main__":
    main()
