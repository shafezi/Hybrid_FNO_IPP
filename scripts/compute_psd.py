"""Compute radially-averaged PSD for one t0 and save to npz."""
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


def radial_psd(field, ocean_mask, H, W):
    filled = np.where(ocean_mask, field, 0.0).astype(np.float64)
    P = np.abs(np.fft.fft2(filled)) ** 2
    ky = np.fft.fftfreq(H).reshape(-1, 1)
    kx = np.fft.fftfreq(W).reshape(1, -1)
    kr = np.sqrt(kx ** 2 + ky ** 2)
    n_bins = min(H, W) // 2
    bin_edges = np.linspace(0, 0.5, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    psd = np.zeros(n_bins)
    for i in range(n_bins):
        mask_k = (kr >= bin_edges[i]) & (kr < bin_edges[i + 1])
        if mask_k.any():
            psd[i] = P[mask_k].mean()
    return bin_centers, psd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0", type=int, required=True)
    parser.add_argument("--n_robots", type=int, default=20)
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
    H, W = ocean_mask.shape
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
        sr, _, qf = run_dynamic_episode(
            d["inputs_np"], ocean_mask, d["all_ocean_coords"],
            cand, evl, pol if use_pol else None,
            d["fno"], d["device"], c,
            episode_seed=seed, t0=args.t0,
            save_qual_steps=set(steps),
        )
        for s in steps:
            k, psd = radial_psd(qf[s]["x_corrected"], ocean_mask, H, W)
            results[f"{label}_step{s}"] = psd
            if label == "FNO-only":
                _, psd_gt = radial_psd(qf[s]["y_true"], ocean_mask, H, W)
                results[f"GT_step{s}"] = psd_gt

    results["k"] = k
    results["steps"] = np.array(steps)

    out = os.path.join(args.out_dir, f"psd_t0_{args.t0}_{args.n_robots}bots.npz")
    np.savez(out, **results)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
