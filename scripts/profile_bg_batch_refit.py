"""
Profile batch vs loop GP refit for background observations.

Compares:
  (a) Loop: call add_static_observation() N times  (N refits)
  (b) Batch: call add_static_observations() once    (1 refit)

Usage:
    python scripts/profile_bg_batch_refit.py
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ipp.gp_state import GPState


def make_gp():
    return GPState(
        prior_std=0.07,
        gp_kwargs=dict(
            kernel_type="matern",
            matern_nu=1.5,
            length_scale_init=0.1,
            length_scale_bounds=[0.01, 1.0],
            noise_bounds=[1e-5, 0.1],
            constant_value_bounds=[0.01, 10.0],
            normalize_y=True,
            n_restarts=0,
        ),
    )


def run_benchmark(n_bg, n_query=200, seed=42):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1, size=(n_bg, 2)).astype(np.float32)
    residuals = rng.normal(0, 0.05, size=n_bg).astype(np.float32)
    query = rng.uniform(0, 1, size=(n_query, 2)).astype(np.float32)

    # (a) Loop style
    gp_loop = make_gp()
    t0 = time.perf_counter()
    for i in range(n_bg):
        gp_loop.add_static_observation(coords[i], residuals[i])
    t_loop = time.perf_counter() - t0
    mean_loop, std_loop = gp_loop.predict(query)

    # (b) Batch style
    gp_batch = make_gp()
    t0 = time.perf_counter()
    gp_batch.add_static_observations(coords, residuals)
    t_batch = time.perf_counter() - t0
    mean_batch, std_batch = gp_batch.predict(query)

    # Compare outputs
    mean_diff = float(np.max(np.abs(mean_loop - mean_batch)))
    std_diff = float(np.max(np.abs(std_loop - std_batch)))

    return t_loop, t_batch, mean_diff, std_diff


def main():
    print("Background obs batch refit profiling")
    print("=" * 55)
    for n in [50, 100]:
        t_loop, t_batch, md, sd = run_benchmark(n)
        speedup = t_loop / t_batch if t_batch > 0 else float("inf")
        print(f"\n  N={n:3d} bg observations:")
        print(f"    Loop  (N refits): {t_loop:7.3f}s")
        print(f"    Batch (1 refit):  {t_batch:7.3f}s")
        print(f"    Speedup:          {speedup:7.1f}x")
        print(f"    Max |mean diff|:  {md:.2e}")
        print(f"    Max |std  diff|:  {sd:.2e}")
    print()


if __name__ == "__main__":
    main()
