"""Compute 1D radially-averaged PSD using ocean-only rows with Hanning window.

For each timestep, extracts rows that are fully ocean, applies a Hanning
window, computes 1D FFT along each row, and averages the power spectra.
This is the standard approach in SSH spectral analysis (e.g., Xu et al. 2022).

Runs one t0 and saves results to npz.
"""
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


def find_ocean_rows(ocean_mask, min_length=50):
    """Find rows that have contiguous ocean runs of at least min_length."""
    H, W = ocean_mask.shape
    rows_info = []  # (row_idx, col_start, col_end)
    for r in range(H):
        row = ocean_mask[r]
        # Find contiguous runs of True
        changes = np.diff(np.concatenate(([False], row, [False])).astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        for s, e in zip(starts, ends):
            if e - s >= min_length:
                rows_info.append((r, s, e))
    return rows_info


def psd_1d_hanning(field, ocean_mask, rows_info):
    """Compute averaged 1D PSD along ocean-only rows with Hanning window."""
    all_psds = []
    all_lengths = []

    for r, c_start, c_end in rows_info:
        segment = field[r, c_start:c_end].astype(np.float64)
        N = len(segment)

        # Apply Hanning window
        window = np.hanning(N)
        windowed = segment * window

        # 1D FFT
        fft_vals = np.fft.rfft(windowed)
        psd = np.abs(fft_vals) ** 2 / N

        # Normalize by window power to correct for energy loss
        psd /= np.mean(window ** 2)

        all_psds.append(psd)
        all_lengths.append(N)

    if not all_psds:
        return np.array([]), np.array([])

    # Interpolate all PSDs to a common wavenumber grid
    # Use the median length for the grid
    N_ref = int(np.median(all_lengths))
    n_freqs = N_ref // 2 + 1
    k_ref = np.fft.rfftfreq(N_ref)  # normalized wavenumber [0, 0.5]

    interpolated = []
    for psd, N in zip(all_psds, all_lengths):
        k_orig = np.fft.rfftfreq(N)
        # Interpolate to common grid
        psd_interp = np.interp(k_ref, k_orig, psd)
        interpolated.append(psd_interp)

    avg_psd = np.mean(interpolated, axis=0)
    return k_ref, avg_psd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0", type=int, required=True)
    parser.add_argument("--n_robots", type=int, default=20)
    parser.add_argument("--min_row_length", type=int, default=50)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(ROOT, "results", "dynamic_ipp", "psd"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(ROOT, "configs/dynamic_rollout_ipp.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg.update(n_episodes=1, debug=False, bg_mode="off",
               n_init_observations=0, n_robots=args.n_robots)

    d = load_test_data()
    ocean_mask = d["ocean_mask"]
    cand = build_candidates(d["all_ocean_coords"],
                            cfg.get("n_candidates", 2000),
                            cfg.get("candidate_seed", 42))
    evl = build_eval_cells(d["all_ocean_coords"],
                           cfg.get("n_eval_cells", 20000),
                           cfg.get("eval_seed", 123))
    pol = build_policies({"policies": {
        "uncertainty_only": {"type": "uncertainty_only", "lambda_dist": 0.0}
    }})["uncertainty_only"]
    seed = (cfg.get("episode_seed_offset", 0)
            + _stable_hash("uncertainty_only") % 1000)

    # Find ocean-only rows
    rows_info = find_ocean_rows(ocean_mask, min_length=args.min_row_length)
    print(f"Found {len(rows_info)} ocean-only row segments "
          f"(min length {args.min_row_length})")

    METHODS = [
        ("FNO-only", "fno", False),
        ("FNO+GP", "fno", True),
        ("Persist-only", "persistence", False),
        ("Persist+GP", "persistence", True),
        ("GP-only", "none", True),
    ]
    steps = list(range(1, cfg["episode_length"] + 1))

    results = {}
    for label, fm, use_pol in METHODS:
        c = dict(cfg)
        c["forecast_mode"] = fm
        print(f"Running {label}...", flush=True)
        sr, _, qf = run_dynamic_episode(
            d["inputs_np"], ocean_mask, d["all_ocean_coords"],
            cand, evl, pol if use_pol else None,
            d["fno"], d["device"], c,
            episode_seed=seed, t0=args.t0,
            save_qual_steps=set(steps),
        )
        for s in steps:
            k, psd = psd_1d_hanning(qf[s]["x_corrected"], ocean_mask, rows_info)
            results[f"{label}_step{s}"] = psd
            if label == "FNO-only":
                _, psd_gt = psd_1d_hanning(qf[s]["y_true"], ocean_mask, rows_info)
                results[f"GT_step{s}"] = psd_gt

    results["k"] = k
    results["steps"] = np.array(steps)
    results["n_rows"] = len(rows_info)

    out = os.path.join(args.out_dir, f"psd1d_t0_{args.t0}_{args.n_robots}bots.npz")
    np.savez(out, **results)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
