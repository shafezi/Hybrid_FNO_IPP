"""
Voronoi-style spatial partitioning for multi-robot IPP.

Assigns each candidate point to the nearest of n_regions site locations,
creating non-overlapping spatial regions for robot assignment.
"""

import numpy as np


def build_voronoi_partition(coords, n_regions, seed=0):
    """
    Partition spatial coordinates into n_regions Voronoi cells.

    Sites are chosen via k-means++ initialization for good spatial spread.

    Parameters
    ----------
    coords    : (N, 2) float32  normalized [0,1]^2 coordinates
    n_regions : int              number of regions
    seed      : int              random seed for site selection

    Returns
    -------
    region_id : (N,) int  region assignment in [0, n_regions-1]
    sites     : (n_regions, 2) float32  the Voronoi site locations
    """
    coords = np.asarray(coords, dtype=np.float32)
    N = len(coords)
    assert n_regions > 0
    if n_regions >= N:
        return np.arange(N, dtype=int), coords.copy()

    rng = np.random.default_rng(seed)
    sites = _kmeans_plus_plus_init(coords, n_regions, rng)

    # Assign each point to the nearest site
    region_id = _assign_nearest(coords, sites)
    return region_id, sites


def _kmeans_plus_plus_init(coords, k, rng):
    """
    K-means++ initialization: pick k sites with probability proportional
    to squared distance from the nearest existing site.
    """
    N = len(coords)
    sites = np.empty((k, 2), dtype=np.float32)

    # First site: uniform random
    sites[0] = coords[rng.integers(N)]

    for i in range(1, k):
        # Squared distance from each point to the nearest existing site
        dists = np.min(
            np.sum((coords[:, None, :] - sites[None, :i, :]) ** 2, axis=2),
            axis=1,
        )
        # Probability proportional to squared distance
        probs = dists / dists.sum()
        idx = rng.choice(N, p=probs)
        sites[i] = coords[idx]

    return sites


def _assign_nearest(coords, sites):
    """Assign each coord to the nearest site. Returns (N,) int."""
    # (N, k) distance matrix
    dists = np.sqrt(
        np.sum((coords[:, None, :] - sites[None, :, :]) ** 2, axis=2)
    )
    return np.argmin(dists, axis=1).astype(int)


def partition_candidates(cand_coords, n_robots, seed=0):
    """
    Partition candidate coordinates into n_robots Voronoi regions.

    Parameters
    ----------
    cand_coords : (N_cand, 2) float32
    n_robots    : int
    seed        : int

    Returns
    -------
    region_id : (N_cand,) int  in [0, n_robots-1]
    """
    region_id, _ = build_voronoi_partition(cand_coords, n_robots, seed=seed)
    return region_id