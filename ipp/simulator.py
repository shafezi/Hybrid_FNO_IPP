"""
Single-robot IPP episode simulator.

Design
------
Each episode corresponds to one held-out SSH snapshot (one "day").
The robot sequentially chooses observation locations from a fixed
set of candidate waypoints.  After each observation, the residual GP
is updated and the corrected field is re-evaluated.

Separation of concerns
----------------------
- candidates: downsampled subset of ocean cells used as action space
- eval cells:  (larger) subset used for per-step RMSE/MAE computation
- full ocean:  used only for final per-episode evaluation and visualization

Scientific correctness
----------------------
- Ground truth is ONLY used at the chosen observation location (no leakage).
- Unobserved metrics exclude all cells that have been visited.
- Shuffled-residual control experiment supported via shuffle_residuals=True.
- All episodes/policies are deterministic given a seed.
"""

import os
import sys
import warnings
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.metrics import compute_metrics
from ipp.gp_state import GPState


# ---------------------------------------------------------------------------
# Candidate / eval cell builders
# ---------------------------------------------------------------------------

def build_candidates(all_ocean_coords, n_candidates, seed):
    """
    Downsample ocean cells to N candidate waypoints.

    Returns
    -------
    cand_local_idx : (N_cand,) int   indices into all_ocean_coords
    """
    N = len(all_ocean_coords)
    n = min(n_candidates, N)
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=n, replace=False)
    return np.sort(idx)


def build_init_observations(cand_coords, cand_local_idx, n_init, strategy, rng):
    """
    Select n_init candidate indices for GP warm-start.

    strategy : "random"  — uniform random from candidates
               "lh"      — Latin Hypercube (space-filling); requires scipy
    Returns
    -------
    local_indices : (n_init,) int  indices into cand_coords / cand_local_idx
    """
    n = min(n_init, len(cand_coords))
    if n == 0:
        return np.array([], dtype=int)

    if strategy == "lh":
        try:
            from scipy.stats.qmc import LatinHypercube, scale
            sampler = LatinHypercube(d=2, seed=int(rng.integers(0, 2**31)))
            lh_pts  = sampler.random(n=n)            # (n, 2) in [0,1]^2
            # Find nearest candidate to each LH point
            chosen = set()
            for pt in lh_pts:
                dists = np.linalg.norm(cand_coords - pt, axis=1)
                # skip already chosen
                dists_masked = dists.copy()
                while True:
                    idx = int(np.argmin(dists_masked))
                    if idx not in chosen:
                        chosen.add(idx)
                        break
                    dists_masked[idx] = np.inf
            return np.array(sorted(chosen), dtype=int)
        except ImportError:
            pass  # fall through to random

    # random fallback
    return rng.choice(len(cand_coords), size=n, replace=False).astype(int)


def build_eval_cells(all_ocean_coords, n_eval_cells, seed):
    """
    Downsample ocean cells for fast per-step metric evaluation.
    Returns all cells if n_eval_cells <= 0 or >= N_ocean.
    """
    N = len(all_ocean_coords)
    if n_eval_cells <= 0 or n_eval_cells >= N:
        return np.arange(N)
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=n_eval_cells, replace=False)
    return np.sort(idx)


def get_init_position(cand_coords, strategy, rng):
    """Choose initial robot position index in candidate array."""
    if strategy == "random":
        return int(rng.integers(0, len(cand_coords)))
    elif strategy == "top_left":
        return int(np.argmin(np.linalg.norm(cand_coords - [0., 0.], axis=1)))
    elif strategy == "center":
        return int(np.argmin(np.linalg.norm(cand_coords - [0.5, 0.5], axis=1)))
    else:
        return int(rng.integers(0, len(cand_coords)))


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------

def run_single_policy_episode(
    fno_flat,           # (N_ocean,) FNO prediction in physical units
    label_flat,         # (N_ocean,) ground truth in physical units
    all_ocean_coords,   # (N_ocean, 2) normalized coords
    cand_local_idx,     # (N_cand,)   indices into all_ocean_coords
    eval_local_idx,     # (N_eval,)   indices into all_ocean_coords
    policy,
    T,
    cfg,
    episode_seed,
    save_full_fields=False,   # if True, save GP mean/std on ALL ocean cells every step
    shuffle_residuals=False,  # if True, shuffle residuals at obs points (control experiment)
):
    """
    Run one episode with one policy for T sensing steps.

    Returns
    -------
    history : list of T dicts, one per step
    trajectory : (T+1, 2) array  robot positions (init + T steps)
    """
    rng = np.random.default_rng(episode_seed)

    max_dist   = cfg.get("max_step_distance", 0.0)
    gp_kwargs  = dict(
        kernel_type          = cfg.get("gp_kernel",                 "matern"),
        matern_nu            = cfg.get("gp_matern_nu",              1.5),
        length_scale_init    = cfg.get("gp_length_scale_init",      0.05),
        length_scale_bounds   = cfg.get("gp_length_scale_bounds",    [1e-4, 1.0]),
        noise_bounds          = cfg.get("gp_noise_bounds",           [1e-8, 1e-1]),
        constant_value_bounds = cfg.get("gp_constant_value_bounds",  [0.01, 10.0]),
        normalize_y          = cfg.get("gp_normalize_y",            True),
        n_restarts           = cfg.get("gp_n_restarts",             0),
    )

    # Subsets
    cand_coords    = all_ocean_coords[cand_local_idx]    # (N_cand, 2)
    eval_coords    = all_ocean_coords[eval_local_idx]    # (N_eval, 2)
    fno_eval       = fno_flat[eval_local_idx]
    label_eval     = label_flat[eval_local_idx]

    # GP state
    gp = GPState(prior_std=cfg.get("gp_prior_std", 0.07), gp_kwargs=gp_kwargs)

    # FNO-only metrics (constant; computed on eval cells)
    fno_rmse = float(np.sqrt(np.mean((label_eval - fno_eval) ** 2)))
    fno_mae  = float(np.mean(np.abs(label_eval - fno_eval)))

    # Robot init
    init_cand_idx = get_init_position(cand_coords, cfg.get("init_position", "random"), rng)
    robot_coord   = cand_coords[init_cand_idx].copy()
    trajectory    = [robot_coord.copy()]

    # Raster needs its order set before step 0
    if hasattr(policy, "set_order"):
        policy.set_order(cand_coords)

    visited_set      = set()       # set of candidate local indices
    cumulative_dist  = 0.0

    # For shuffled control: pre-shuffle residuals at all ocean cells
    _shuffled_residuals = None
    if shuffle_residuals:
        true_residuals = label_flat - fno_flat
        perm = rng.permutation(len(true_residuals))
        _shuffled_residuals = true_residuals[perm]

    # -------------------------------------------------------------------
    # Warm-start: pre-observe K fixed points to initialise the GP before
    # the adaptive loop begins.  These represent "existing sensor data"
    # and are NOT counted in the T-step sensing budget.
    # -------------------------------------------------------------------
    n_init        = cfg.get("n_init_observations", 0)
    init_strategy = cfg.get("init_strategy", "random")

    if n_init > 0:
        init_local = build_init_observations(
            cand_coords, cand_local_idx, n_init, init_strategy, rng
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for local_idx in init_local:
                ocean_idx = int(cand_local_idx[local_idx])
                if shuffle_residuals:
                    r = float(_shuffled_residuals[ocean_idx])
                else:
                    r = float(label_flat[ocean_idx] - fno_flat[ocean_idx])
                gp.add_observation(cand_coords[local_idx], r)
                visited_set.add(int(local_idx))

    history = []

    for t in range(T):
        # --- 1. GP predictions on candidates (for policy scoring) ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp_mean_cand, gp_std_cand = gp.predict(cand_coords)

        # --- 2. Policy chooses next location ---
        chosen_local = policy.choose(
            robot_coord, cand_coords, gp_mean_cand, gp_std_cand,
            visited_set, rng, max_dist,
        )

        # --- 3. Move robot ---
        next_coord   = cand_coords[chosen_local].copy()
        step_dist    = float(np.linalg.norm(next_coord - robot_coord))
        cumulative_dist += step_dist
        robot_coord  = next_coord
        trajectory.append(robot_coord.copy())

        # --- 4. Observe ground truth residual at chosen location ---
        ocean_idx = int(cand_local_idx[chosen_local])    # index in flat ocean array
        if shuffle_residuals:
            r_obs = float(_shuffled_residuals[ocean_idx])
        else:
            r_obs = float(label_flat[ocean_idx] - fno_flat[ocean_idx])

        # --- 5. Update GP ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.add_observation(robot_coord, r_obs)
        visited_set.add(chosen_local)

        # --- 6. GP predictions on eval cells ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp_mean_eval, gp_std_eval = gp.predict(eval_coords)

        corrected_eval = fno_eval + gp_mean_eval

        # --- 7. Unobserved eval cells mask ---
        obs_ocean_set = {int(cand_local_idx[i]) for i in visited_set}
        unobs_eval_mask = np.array(
            [int(eval_local_idx[i]) not in obs_ocean_set for i in range(len(eval_local_idx))],
            dtype=bool,
        )

        # --- 8. Metrics ---
        all_m   = compute_metrics(label_eval, corrected_eval)
        unobs_m = (compute_metrics(label_eval[unobs_eval_mask], corrected_eval[unobs_eval_mask])
                   if unobs_eval_mask.any() else {"rmse": float("nan"), "mae": float("nan")})

        record = {
            "step":             t + 1,
            "chosen_cand_idx":  int(chosen_local),
            "chosen_ocean_idx": ocean_idx,
            "robot_row":        float(robot_coord[0]),
            "robot_col":        float(robot_coord[1]),
            "step_dist":        step_dist,
            "cumulative_dist":  cumulative_dist,
            "r_obs":            r_obs,
            "n_obs":            gp.n_obs,
            "all_rmse":         all_m["rmse"],
            "all_mae":          all_m["mae"],
            "unobs_rmse":       unobs_m["rmse"],
            "unobs_mae":        unobs_m["mae"],
            "fno_rmse":         fno_rmse,
            "fno_mae":          fno_mae,
        }

        # --- 9. Optionally save full GP fields (for visualization) ---
        if save_full_fields:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp_mean_all, gp_std_all = gp.predict(all_ocean_coords)
            record["gp_mean_all"] = gp_mean_all
            record["gp_std_all"]  = gp_std_all

        history.append(record)

    return history, np.array(trajectory)


# ---------------------------------------------------------------------------
# Multi-episode / multi-policy runner
# ---------------------------------------------------------------------------

def run_all_experiments(
    preds_np,           # (N, H, W) FNO predictions (physical units)
    labels_np,          # (N, H, W) ground truth
    ocean_mask,         # (H, W) bool
    all_ocean_coords,   # (N_ocean, 2)
    flat_ocean_indices, # (N_ocean,) flat indices in H×W
    policies,           # dict: name → policy object
    cfg,
    verbose=True,
):
    """
    Run all policies on all episodes.

    Returns
    -------
    all_records         : list of dicts (every step, every policy, every episode)
    trajectories        : dict: (ep_i, policy_name) → (T+1, 2) array
    qual_records        : dict: policy_name → list of step dicts with full GP fields
                          (first episode only, all ocean cells)
    cand_local_idx      : (N_cand,) shared across episodes
    eval_local_idx      : (N_eval,) shared across episodes
    episode_sample_idx  : (n_episodes,) which test samples were used
    """
    n_episodes    = cfg["n_episodes"]
    T             = cfg["sensing_budget"]
    cand_seed     = cfg.get("candidate_seed",    42)
    eval_seed     = cfg.get("eval_seed",        123)
    seed_offset   = cfg.get("episode_seed_offset", 0)

    cand_local_idx = build_candidates(all_ocean_coords, cfg["n_candidates"], cand_seed)
    eval_local_idx = build_eval_cells(all_ocean_coords, cfg.get("n_eval_cells", 20000), eval_seed)

    if verbose:
        print(f"  Candidates: {len(cand_local_idx):,}  "
              f"| Eval cells: {len(eval_local_idx):,}  "
              f"| All ocean: {len(all_ocean_coords):,}")

    # Select episode samples (fixed random subset of test set)
    rng_ep = np.random.default_rng(seed_offset)
    episode_sample_idx = rng_ep.choice(preds_np.shape[0], size=n_episodes, replace=False)

    all_records  = []
    trajectories = {}
    qual_records = {}   # first episode only, for qualitative visualization

    for ep_i, sample_idx in enumerate(episode_sample_idx):
        fno_flat   = preds_np[sample_idx][ocean_mask]
        label_flat = labels_np[sample_idx][ocean_mask]

        if verbose:
            print(f"\n  Episode {ep_i+1}/{n_episodes}  (sample {sample_idx})")

        for pol_name, policy in policies.items():
            ep_seed     = seed_offset * 10_000 + ep_i * 100
            is_first_ep = (ep_i == 0)

            history, traj = run_single_policy_episode(
                fno_flat, label_flat,
                all_ocean_coords, cand_local_idx, eval_local_idx,
                policy, T, cfg,
                episode_seed    = ep_seed,
                save_full_fields= is_first_ep,   # save full fields for ep0 only
            )

            # Store trajectory
            trajectories[(ep_i, pol_name)] = traj

            # Store first-episode qual records
            if is_first_ep:
                qual_records[pol_name] = history

            # Flatten records
            for rec in history:
                rec["episode"]    = ep_i
                rec["sample_idx"] = int(sample_idx)
                rec["policy"]     = pol_name
                all_records.append(rec)

            if verbose:
                print(f"    {pol_name:20s}  final RMSE={history[-1]['all_rmse']:.4f}  "
                      f"FNO={history[-1]['fno_rmse']:.4f}  "
                      f"dist={history[-1]['cumulative_dist']:.3f}")

    return (all_records, trajectories, qual_records,
            cand_local_idx, eval_local_idx, episode_sample_idx)
