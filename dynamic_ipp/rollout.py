"""
Dynamic rollout episode runner: FNO autoregressive forecast + periodic GP assimilation.

Supports single-robot and multi-robot (synchronous batch) modes.

Episode structure
-----------------
  t0  : start index into inputs_np (the ground-truth SSH time series)
  L   : episode length (number of forecast steps)
  K   : assimilation interval (assimilate every K steps)

At each forecast step s = 1..L:
  1. x_hat_prior = PECstep(fno, x_hat)          # one-step FNO forecast
  2. if s % K == 0:  (assimilation step)
       background sensors add observations to GP (if bg_mode != off)
       synchronous robot observation rounds:
         for each round:
           compute GP posterior at candidates (same for all robots)
           each robot picks 1 point (mutual exclusion within round)
           batch-add all M robot observations and refit GP once
       x_hat = x_hat_prior + gain * GP_mean(all ocean cells)
     else:
       x_hat = x_hat_prior                       # free forecast
  3. metrics vs ground truth inputs_np[t0+s]

Multi-robot config
------------------
  n_robots                : number of robots (default 1)
  obs_per_robot_per_assim : observations per robot per assimilation event
  Total observations per assimilation = n_robots * obs_per_robot_per_assim.
"""

import hashlib

import numpy as np
import torch

from experiments.fno_model import PECstep
from ipp.gp_state import GPState
from ipp.partitioning import partition_candidates
from ipp.simulator import build_candidates, build_eval_cells, get_init_position


def _stable_hash(s):
    """Deterministic hash independent of PYTHONHASHSEED."""
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def _mae(a, b):
    return float(np.mean(np.abs(a - b)))

def _fno_step(fno, arr_2d, delta_t, device):
    """Apply one PECstep to a (H, W) numpy array. Returns (H, W) numpy array."""
    t = torch.tensor(arr_2d[None, :, :, None], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = PECstep(fno, t, delta_t)
    return out[0, :, :, 0].cpu().numpy()


def _sample_init_indices(rng, n_cand, n_samples, strategy, cand_coords=None):
    """Sample n_samples indices from [0, n_cand) for the background sensor network."""
    n_samples = min(n_samples, n_cand)
    if n_samples <= 0:
        return np.array([], dtype=int)
    if strategy == "lh" and cand_coords is not None:
        order = np.lexsort((cand_coords[:, 1], cand_coords[:, 0]))
        intervals = np.array_split(order, n_samples)
        return np.array([int(rng.choice(iv)) for iv in intervals])
    else:
        return rng.choice(n_cand, size=n_samples, replace=False)


def _compute_max_step_from_glider_speed(cfg, all_ocean_coords):
    """
    If cfg['glider_max_speed_mps'] > 0, compute normalized max_step_distance from
    physical glider reach over one FNO step.

    Assumes the 2D candidate/ocean coords are normalized over the domain [0,1]^2.
    Domain physical extents are inferred from the coords + a fixed mapping:
    latitude span 33-45.76 deg, longitude span -79.6 to -55.64 deg.  Uses cos(lat)
    for longitude scaling at mean latitude.

    Returns max_step_distance as a float in normalized-coord units, or None if
    glider_max_speed_mps is not set.
    """
    speed_mps = float(cfg.get("glider_max_speed_mps", 0) or 0)
    if speed_mps <= 0:
        return None
    # days per FNO step = rollout_lead_stride × 1 day (data is daily)
    step_days = float(cfg.get("rollout_lead_stride", cfg.get("lead", 1)))
    max_reach_km = speed_mps * step_days * 86400.0 / 1000.0
    # Domain extent in km (Gulf Stream region; data_utils hardcodes this region)
    lat_span_deg = 45.76 - 33.0
    lon_span_deg = 79.6 - 55.64
    mean_lat = (33.0 + 45.76) / 2.0
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.deg2rad(mean_lat))
    domain_km_ns = lat_span_deg * km_per_deg_lat
    domain_km_ew = lon_span_deg * km_per_deg_lon
    # Normalize by the larger extent (conservative)
    return max_reach_km / max(domain_km_ns, domain_km_ew)


def _dynamic_partition(robot_coords, cand_coords, n_robots, robot_region):
    """
    Compute Voronoi partition using current robot positions as sites.

    Robot r's region contains candidates closest to robot r's position.
    Returns (region_id, out_of_region, in_region_indices).
    """
    sites = np.stack(
        [np.asarray(rc).ravel()[:2] for rc in robot_coords]).astype(np.float32)
    dists_sq = np.sum(
        (cand_coords[:, None, :] - sites[None, :, :]) ** 2, axis=2)
    region_id = np.argmin(dists_sq, axis=1).astype(int)
    out_of_region = []
    in_region_indices = []
    for r in range(n_robots):
        rgn = robot_region[r]
        oor = frozenset(int(i) for i in range(len(cand_coords))
                        if region_id[i] != rgn)
        out_of_region.append(oor)
        in_region_indices.append(np.where(region_id == rgn)[0])
    return region_id, out_of_region, in_region_indices


def _init_robot_positions(cand_coords, cfg, n_robots, rng):
    """
    Initialize positions for n_robots.

    Returns list of n_robots candidate indices.
    """
    positions_cfg = cfg.get("robot_init_positions", cfg.get("init_position", "center"))

    if isinstance(positions_cfg, list) and len(positions_cfg) == n_robots:
        # Explicit normalized coords: find nearest candidate for each
        indices = []
        for coord in positions_cfg:
            coord = np.asarray(coord, dtype=np.float32)
            idx = int(np.argmin(np.linalg.norm(cand_coords - coord, axis=1)))
            indices.append(idx)
        return indices

    # String strategy: spread robots using get_init_position with different seeds
    strategy = positions_cfg if isinstance(positions_cfg, str) else "center"
    if n_robots == 1:
        return [get_init_position(cand_coords, strategy, rng)]

    # For multiple robots: use "random" to spread them, or repeat strategy
    indices = []
    used = set()
    for r in range(n_robots):
        sub_rng = np.random.default_rng(rng.integers(0, 2**31) + r)
        idx = get_init_position(cand_coords, strategy, sub_rng)
        # Avoid duplicate starting positions
        attempts = 0
        while idx in used and attempts < 100:
            idx = int(sub_rng.integers(0, len(cand_coords)))
            attempts += 1
        used.add(idx)
        indices.append(idx)
    return indices


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

def run_dynamic_episode(
    inputs_np,          # (N_samples, H, W)  normalized SSH time series
    ocean_mask,         # (H, W) bool
    all_ocean_coords,   # (N_ocean, 2)  normalized [0,1]^2 coords
    cand_local_idx,     # (N_cand,)  indices into all_ocean_coords
    eval_local_idx,     # (N_eval,)  indices into all_ocean_coords
    policy,             # BasePolicy or None  (None = FNO-only, no assimilation)
    fno,                # pretrained FNO2d
    device,
    cfg,
    episode_seed,
    t0,                 # start index into inputs_np
    save_qual_steps=(), # timesteps at which to save full-field data for visualization
):
    """
    Run one dynamic rollout episode with 1 or more robots.

    Returns
    -------
    step_records  : list[dict]          one dict per forecast step (s = 1..L)
    trajectories  : list of (T_r, 2)    per-robot trajectories
    qual_frames   : dict[step] -> dict  field snapshots (only for save_qual_steps)
    """
    L                   = cfg.get("episode_length", 30)
    K                   = cfg.get("assimilation_interval", 5)
    gp_mode             = cfg.get("gp_mode", "windowed")
    max_dist            = cfg.get("max_step_distance", 0.0)
    delta_t             = cfg.get("delta_t", 1.0)
    rollout_lead_stride = cfg.get("rollout_lead_stride", cfg.get("lead", 1))
    assimilation_gain   = cfg.get("assimilation_gain", 1.0)
    n_init_obs          = cfg.get("n_init_observations", 0)
    init_strategy       = cfg.get("init_strategy", "random")
    bg_mode             = cfg.get("bg_mode", "off")
    gp_use_time         = cfg.get("gp_use_time", False)
    debug               = cfg.get("debug", False)
    forecast_mode       = cfg.get("forecast_mode", "fno")  # fno | persistence | none
    do_assimilate       = (policy is not None)

    # Multi-robot config
    n_robots = cfg.get("n_robots", 1)
    obs_per_robot = cfg.get("obs_per_robot_per_assim",
                            cfg.get("obs_per_assimilation", 5))
    obs_per_assim_total = n_robots * obs_per_robot
    selection_mode = cfg.get("selection_mode", "batch")
    if selection_mode not in ("batch", "sequential"):
        raise ValueError(
            f"Unknown selection_mode '{selection_mode}'. "
            f"Choose 'batch' or 'sequential'.")

    # Acquisition mode (myopic = current; lookahead = plan at end of step s
    # based on GP std at (s+1)/L)
    acquisition_mode = cfg.get("acquisition_mode", "myopic")
    cold_start_mode  = cfg.get("cold_start_mode", "random_in_region")
    allow_revisit    = cfg.get("allow_revisit", False)
    if acquisition_mode not in ("myopic", "lookahead"):
        raise ValueError(
            f"Unknown acquisition_mode '{acquisition_mode}'. "
            f"Choose 'myopic' or 'lookahead'.")
    if acquisition_mode == "lookahead":
        if obs_per_robot != 1 or K != 1:
            raise ValueError(
                "acquisition_mode='lookahead' requires "
                "obs_per_robot_per_assim=1 and assimilation_interval=1")
        if not gp_use_time:
            raise ValueError(
                "acquisition_mode='lookahead' requires gp_use_time=true")

    # Physics-based reach: override max_step_distance from glider speed if set.
    _glider_max_step = _compute_max_step_from_glider_speed(cfg, all_ocean_coords)
    if _glider_max_step is not None:
        max_dist = _glider_max_step
        if debug:
            print(f"  [glider] max_step_distance = {max_dist:.4f} "
                  f"(from {cfg.get('glider_max_speed_mps')} m/s)")

    rng         = np.random.default_rng(episode_seed)
    cand_coords = all_ocean_coords[cand_local_idx]   # (N_cand, 2)
    land_mask   = ~ocean_mask                         # (H, W)

    # -- GP ----------------------------------------------------------------
    gp_kwargs = dict(
        kernel_type           = cfg.get("gp_kernel",                 "matern"),
        matern_nu             = cfg.get("gp_matern_nu",              1.5),
        length_scale_init     = cfg.get("gp_length_scale_init",      0.1),
        length_scale_bounds   = cfg.get("gp_length_scale_bounds",    [1e-4, 1.0]),
        noise_bounds          = cfg.get("gp_noise_bounds",           [1e-8, 1e-1]),
        constant_value_bounds = cfg.get("gp_constant_value_bounds",  [0.01, 10.0]),
        normalize_y           = cfg.get("gp_normalize_y",            True),
        n_restarts            = cfg.get("gp_n_restarts",             0),
        use_time              = gp_use_time,
        kernel_type_time      = cfg.get("gp_time_kernel",            "matern"),
        matern_nu_time        = cfg.get("gp_time_matern_nu",         1.5),
        time_length_scale_init   = cfg.get("gp_time_length_scale_init",   0.2),
        time_length_scale_bounds = cfg.get("gp_time_length_scale_bounds", [1e-3, 2.0]),
        optimizer             = cfg.get("optimizer",                    "fmin_l_bfgs_b"),
    )
    gp = GPState(prior_std=cfg.get("gp_prior_std", 0.07), gp_kwargs=gp_kwargs)

    # -- Initial state (enforce land = 0 from the start) -------------------
    x_hat_np = inputs_np[t0].copy()
    x_fno_np = inputs_np[t0].copy()
    x_hat_np[land_mask] = 0.0
    x_fno_np[land_mask] = 0.0

    # -- Robot init --------------------------------------------------------
    init_indices = _init_robot_positions(cand_coords, cfg, n_robots, rng)
    robot_coords = [cand_coords[idx].copy() for idx in init_indices]
    trajectories = [[rc.copy()] for rc in robot_coords]
    cumulative_dists = [0.0] * n_robots

    visited_cand_set  = set()   # global across all robots for the episode
    visited_ocean_set = set()

    # -- Voronoi partitioning (optional) -----------------------------------
    use_partitioning = cfg.get("use_partitioning", False) and n_robots > 1
    partition_strict = cfg.get("partition_strict", True)
    partition_allow_spillover = cfg.get("partition_allow_spillover", False)
    spillover_total = 0

    if use_partitioning:
        part_seed = episode_seed + cfg.get("partition_seed_offset", 0)
        cand_region_id = partition_candidates(cand_coords, n_robots, seed=part_seed)
        robot_region = cfg.get("robot_region_map", list(range(n_robots)))
        # Pre-compute per-region masks
        out_of_region = []
        in_region_indices = []
        for r in range(n_robots):
            rgn = robot_region[r]
            oor = frozenset(int(i) for i in range(len(cand_coords))
                            if cand_region_id[i] != rgn)
            out_of_region.append(oor)
            in_region_indices.append(
                np.where(cand_region_id == rgn)[0])
        if debug:
            for r in range(n_robots):
                print(f"  [partition] robot {r} -> region {robot_region[r]}  "
                      f"({len(in_region_indices[r])} candidates)")
    else:
        cand_region_id = None
        out_of_region = [frozenset()] * n_robots
        in_region_indices = None

    # -- Cold-start override for lookahead -----------------------------
    # In lookahead mode, step 1 is a cold start: each robot is placed at a
    # random candidate within its Voronoi region, and it samples there at
    # step 1 without moving.  These positions also serve as the step-1
    # planned_locations.
    if acquisition_mode == "lookahead" and cold_start_mode == "random_in_region":
        cs_indices = []
        used = set()
        if use_partitioning and in_region_indices is not None:
            for r in range(n_robots):
                pool = [int(i) for i in in_region_indices[r] if i not in used]
                if not pool:
                    raise RuntimeError(
                        f"Cold-start: robot {r} has no in-region candidates.")
                pick = int(rng.choice(pool))
                used.add(pick)
                cs_indices.append(pick)
        else:
            # No partitioning: spread randomly with mutex
            pool = list(range(len(cand_coords)))
            cs_indices = rng.choice(pool, size=n_robots, replace=False).tolist()
        init_indices = cs_indices
        robot_coords = [cand_coords[idx].copy() for idx in init_indices]
        trajectories = [[rc.copy()] for rc in robot_coords]
        if debug:
            print(f"  [cold-start] robots placed randomly in their regions: "
                  f"{init_indices}")

    # planned_locations: where each robot will sample NEXT.  For lookahead,
    # step-1 planned = cold-start positions (robots don't move t0 -> step 1).
    # For myopic, this field is unused.
    planned_locations = list(init_indices) if acquisition_mode == "lookahead" else None

    # -- Background sensor network -----------------------------------------
    if bg_mode != "off" and n_init_obs > 0 and do_assimilate:
        init_local_indices = _sample_init_indices(
            rng, len(cand_local_idx), n_init_obs, init_strategy, cand_coords)
        init_ocean_set = {int(cand_local_idx[i]) for i in init_local_indices}
    else:
        init_local_indices = np.array([], dtype=int)
        init_ocean_set     = set()

    step_records = []
    qual_frames  = {}
    n_assim_done      = 0
    bg_meas_total     = 0
    robot_meas_total  = 0

    last_gp_mean_all  = np.zeros(len(all_ocean_coords), dtype=np.float32)
    last_gp_std_all   = np.full(len(all_ocean_coords),
                                cfg.get("gp_prior_std", 0.07), dtype=np.float32)
    last_gp_noise_std = 0.0
    last_assim_step   = None

    for s in range(1, L + 1):

        # ── 1. One-step FNO forecast ───────────────────────────────────────
        x_hat_np[land_mask] = 0.0
        x_fno_np[land_mask] = 0.0
        # Always advance x_fno_np for comparison metrics
        x_fno_np = _fno_step(fno, x_fno_np, delta_t, device)
        x_fno_np[land_mask] = 0.0

        if forecast_mode == "fno":
            x_hat_prior = _fno_step(fno, x_hat_np, delta_t, device)
        elif forecast_mode == "persistence":
            x_hat_prior = x_hat_np.copy()
        else:  # "none" — GP-only, no prior
            x_hat_prior = np.zeros_like(x_hat_np)
        x_hat_prior[land_mask] = 0.0

        # ── 2. Ground truth ───────────────────────────────────────────────
        gt_idx = t0 + s * rollout_lead_stride
        if debug:
            assert gt_idx < len(inputs_np), (
                f"gt_idx={gt_idx} out of bounds (N={len(inputs_np)}, "
                f"t0={t0}, s={s}, stride={rollout_lead_stride})")

        y_true_np    = inputs_np[gt_idx]
        y_true_flat  = y_true_np[ocean_mask]
        x_prior_flat = x_hat_prior[ocean_mask]

        # ── 3. Assimilation ───────────────────────────────────────────────
        newly_visited_cand = []
        is_assimilation    = do_assimilate and (s % K == 0)
        n_bg_added         = 0
        spillover_step     = 0

        # Update GP-UCB beta_t if policy supports it
        if hasattr(policy, 'set_step'):
            policy.set_step(s)
        # Reset MI policy's picks-this-round (for greedy sequential conditioning)
        if hasattr(policy, 'reset_picks'):
            policy.reset_picks()

        if is_assimilation:
            t_norm = s / L if gp_use_time else None

            if gp_use_time:
                cand_coords_gp = GPState.make_query_coords_st(cand_coords, t_norm)
                all_coords_gp  = GPState.make_query_coords_st(all_ocean_coords, t_norm)
            else:
                cand_coords_gp = cand_coords
                all_coords_gp  = all_ocean_coords

            if gp_mode == "windowed":
                gp.reset(keep_static=(bg_mode == "once"))
                window_visited = set(init_local_indices.tolist())
            else:
                window_visited = visited_cand_set | set(init_local_indices.tolist())

            # Voronoi partition: either dynamic (recompute from current robot
            # positions each step) or static (use the seeded partition from
            # episode init).  Default: dynamic.
            if use_partitioning and cfg.get("partition_dynamic", True):
                cand_region_id, out_of_region, in_region_indices = \
                    _dynamic_partition(
                        robot_coords, cand_coords, n_robots, robot_region)

            # Background observations
            add_bg_this_step = False
            if bg_mode == "each_assim":
                add_bg_this_step = True
            elif bg_mode == "once" and n_assim_done == 0:
                add_bg_this_step = True

            if add_bg_this_step and len(init_local_indices) > 0:
                bg_ocean_idx = cand_local_idx[init_local_indices].astype(int)
                if debug:
                    assert np.all(bg_ocean_idx < len(y_true_flat))
                bg_coords_batch = cand_coords_gp[init_local_indices] if gp_use_time \
                    else cand_coords[init_local_indices]
                if forecast_mode == "none":
                    bg_residuals_batch = y_true_flat[bg_ocean_idx]
                else:
                    bg_residuals_batch = (y_true_flat[bg_ocean_idx]
                                          - x_prior_flat[bg_ocean_idx])
                if bg_mode == "once":
                    gp.add_static_observations(bg_coords_batch, bg_residuals_batch)
                else:
                    gp.add_observations(bg_coords_batch, bg_residuals_batch)
                n_bg_added     = len(init_local_indices)
                bg_meas_total += n_bg_added

            if debug:
                print(f"  [assim step={s}]  n_robots={n_robots}  "
                      f"obs_per_robot={obs_per_robot}  bg_mode={bg_mode}  "
                      f"bg_added={n_bg_added}  bg_total={bg_meas_total}  "
                      f"robot_total={robot_meas_total}"
                      + (f"  t_norm={t_norm:.4f}" if gp_use_time else ""))

            # ── Observation rounds ──────────────────────────────────────
            # acquisition_mode == "lookahead": robots sample at pre-planned
            #   locations, then plan the next step using std at (s+1)/L.
            # acquisition_mode == "myopic":   existing batch/sequential logic
            #   where robots pick based on current-step std.
            if acquisition_mode == "lookahead":
              # 1) Sample at planned_locations (set at end of previous step,
              #    or cold-start for step 1).
              batch_coords_la   = []
              batch_residuals_la = []
              for r in range(n_robots):
                  next_local = planned_locations[r]
                  ocean_idx  = int(cand_local_idx[next_local])
                  if debug:
                      assert ocean_idx < len(y_true_flat)
                  if forecast_mode == "none":
                      r_obs = float(y_true_flat[ocean_idx])
                  else:
                      r_obs = float(y_true_flat[ocean_idx] - x_prior_flat[ocean_idx])
                  obs_coord = cand_coords_gp[next_local] if gp_use_time \
                      else cand_coords[next_local]
                  batch_coords_la.append(obs_coord)
                  batch_residuals_la.append(r_obs)

                  # Travel cost from current position to sampling point.
                  # For step 1 (cold start), robot is already at next_local,
                  # so step_dist = 0.
                  step_dist = float(np.linalg.norm(
                      cand_coords[next_local] - robot_coords[r]))
                  cumulative_dists[r] += step_dist
                  robot_coords[r] = cand_coords[next_local].copy()
                  trajectories[r].append(robot_coords[r].copy())

                  if not allow_revisit:
                      visited_cand_set.add(next_local)
                  visited_ocean_set.add(ocean_idx)
                  window_visited.add(next_local)
                  newly_visited_cand.append(next_local)

              gp.add_observations(
                  np.array(batch_coords_la, dtype=np.float32),
                  np.array(batch_residuals_la, dtype=np.float32),
              )
              robot_meas_total += n_robots

              # 2) Plan for step s+1 using GP std at future t_norm.
              if s < L:
                  t_next = (s + 1) / L
                  cand_coords_future = GPState.make_query_coords_st(
                      cand_coords, t_next)
                  gp_mean_f, gp_std_total_f = gp.predict(cand_coords_future)
                  noise_var_f = gp.get_noise_variance()
                  gp_std_epi_f = np.sqrt(
                      np.maximum(
                          gp_std_total_f.astype(np.float64) ** 2 - noise_var_f,
                          0.0)
                  ).astype(np.float32)

                  round_reserved = set()
                  new_planned = [None] * n_robots
                  for r in range(n_robots):
                      excl = visited_cand_set | round_reserved | out_of_region[r]
                      policy_failed = False
                      try:
                          nl = policy.choose(
                              robot_coords[r], cand_coords,
                              gp_mean_f, gp_std_epi_f,
                              excl, rng, max_dist,
                          )
                      except RuntimeError:
                          policy_failed = True
                          nl = -1

                      # Partition enforcement (mirror of batch path)
                      needs_override = (
                          policy_failed
                          or (use_partitioning
                              and cand_region_id is not None
                              and cand_region_id[nl] != robot_region[r])
                      )
                      if use_partitioning and needs_override:
                          blocked = visited_cand_set | round_reserved
                          avail = [i for i in in_region_indices[r]
                                   if i not in blocked]
                          if partition_strict:
                              if not avail:
                                  raise RuntimeError(
                                      f"Robot {r}: no in-region candidates "
                                      f"remain for planning step {s+1}")
                              dists_ir = np.linalg.norm(
                                  cand_coords[avail] - robot_coords[r], axis=1)
                              nl = int(avail[np.argmin(dists_ir)])
                          else:
                              # Spillover fallback (rare given strict default)
                              if avail:
                                  dists_ir = np.linalg.norm(
                                      cand_coords[avail] - robot_coords[r],
                                      axis=1)
                                  nl = int(avail[np.argmin(dists_ir)])
                              elif not policy_failed:
                                  spillover_total += 1
                                  spillover_step  += 1
                              else:
                                  global_avail = [i for i in range(len(cand_coords))
                                                  if i not in blocked]
                                  if not global_avail:
                                      raise RuntimeError(
                                          f"Robot {r}: no candidates remain "
                                          f"at planning step {s+1}")
                                  dists_g = np.linalg.norm(
                                      cand_coords[global_avail]
                                      - robot_coords[r], axis=1)
                                  nl = int(global_avail[np.argmin(dists_g)])
                                  spillover_total += 1
                                  spillover_step  += 1
                      elif policy_failed:
                          raise RuntimeError(
                              f"Robot {r}: no unvisited candidates for "
                              f"planning step {s+1}")

                      round_reserved.add(nl)
                      new_planned[r] = nl

                  planned_locations = new_planned

            elif selection_mode == "batch":
              for round_i in range(obs_per_robot):
                # Compute GP posterior once for this round (shared by all robots)
                gp_mean_cands, gp_std_total = gp.predict(cand_coords_gp)
                noise_var  = gp.get_noise_variance()
                gp_std_epi = np.sqrt(
                    np.maximum(gp_std_total.astype(np.float64) ** 2 - noise_var, 0.0)
                ).astype(np.float32)

                if debug:
                    _ve = gp_std_epi[np.isfinite(gp_std_epi)]
                    print(f"    [round {round_i}]  std_epi  "
                          f"min={_ve.min():.5f}  med={np.median(_ve):.5f}  "
                          f"max={_ve.max():.5f}")

                # Each robot chooses simultaneously (mutual exclusion within round)
                round_reserved = set()
                round_choices  = []   # (robot_id, cand_local_idx)

                for r in range(n_robots):
                    # Visited + round reserved + out-of-region (Voronoi)
                    excl = window_visited | round_reserved | out_of_region[r]

                    # The policy may raise RuntimeError if excl covers
                    # all candidates (exhausted region).  Handle below.
                    policy_failed = False
                    try:
                        next_local = policy.choose(
                            robot_coords[r], cand_coords,
                            gp_mean_cands, gp_std_epi,
                            excl, rng, max_dist,
                        )
                    except RuntimeError:
                        policy_failed = True
                        next_local = -1

                    # Partition enforcement / recovery
                    needs_override = (
                        policy_failed
                        or (use_partitioning
                            and cand_region_id is not None
                            and cand_region_id[next_local] != robot_region[r])
                    )
                    if use_partitioning and needs_override:
                        blocked = window_visited | round_reserved
                        avail = [i for i in in_region_indices[r]
                                 if i not in blocked]
                        if partition_strict:
                            if not avail:
                                raise RuntimeError(
                                    f"Robot {r}: no in-region candidates remain "
                                    f"(region {robot_region[r]}, "
                                    f"{len(in_region_indices[r])} total, "
                                    f"{len(blocked)} blocked)")
                            dists_ir = np.linalg.norm(
                                cand_coords[avail] - robot_coords[r], axis=1)
                            next_local = int(avail[np.argmin(dists_ir)])
                            if debug:
                                print(f"      robot {r}: strict override -> "
                                      f"cand={next_local}")
                        else:
                            # Spillover mode
                            if avail:
                                # Pick nearest in-region anyway (best effort)
                                dists_ir = np.linalg.norm(
                                    cand_coords[avail] - robot_coords[r], axis=1)
                                next_local = int(avail[np.argmin(dists_ir)])
                            elif not policy_failed:
                                # Policy found something out-of-region; accept
                                spillover_total += 1
                                spillover_step  += 1
                                if debug:
                                    print(f"      robot {r}: SPILLOVER to "
                                          f"region {cand_region_id[next_local]} "
                                          f"(own={robot_region[r]})")
                            else:
                                # No in-region and policy failed: pick any
                                # unvisited candidate globally
                                global_avail = [i for i in range(len(cand_coords))
                                                if i not in blocked]
                                if not global_avail:
                                    raise RuntimeError(
                                        f"Robot {r}: no candidates remain at all")
                                dists_g = np.linalg.norm(
                                    cand_coords[global_avail] - robot_coords[r],
                                    axis=1)
                                next_local = int(global_avail[np.argmin(dists_g)])
                                spillover_total += 1
                                spillover_step  += 1
                                if debug:
                                    print(f"      robot {r}: SPILLOVER (exhausted) "
                                          f"-> cand={next_local}")
                    elif policy_failed:
                        # No partitioning but policy failed (all visited)
                        raise RuntimeError(
                            f"Robot {r}: no unvisited candidates remain")

                    round_reserved.add(next_local)
                    round_choices.append((r, next_local))

                    if debug:
                        rgn = (f"  region={cand_region_id[next_local]}"
                               if use_partitioning else "")
                        print(f"      robot {r}: cand={next_local}  "
                              f"epi={gp_std_epi[next_local]:.5f}{rgn}")

                # Batch-add all robot observations for this round
                batch_coords = []
                batch_residuals = []
                for r, next_local in round_choices:
                    ocean_idx = int(cand_local_idx[next_local])
                    if debug:
                        assert ocean_idx < len(y_true_flat)
                    if forecast_mode == "none":
                        r_obs = float(y_true_flat[ocean_idx])
                    else:
                        r_obs = float(y_true_flat[ocean_idx] - x_prior_flat[ocean_idx])
                    obs_coord = cand_coords_gp[next_local] if gp_use_time \
                        else cand_coords[next_local]
                    batch_coords.append(obs_coord)
                    batch_residuals.append(r_obs)

                    # Update robot position and bookkeeping
                    step_dist = float(np.linalg.norm(
                        cand_coords[next_local] - robot_coords[r]))
                    cumulative_dists[r] += step_dist
                    robot_coords[r] = cand_coords[next_local].copy()
                    trajectories[r].append(robot_coords[r].copy())

                    if not allow_revisit:
                      visited_cand_set.add(next_local)
                    visited_ocean_set.add(ocean_idx)
                    window_visited.add(next_local)
                    newly_visited_cand.append(next_local)

                # Single GP refit for the whole round batch
                gp.add_observations(
                    np.array(batch_coords, dtype=np.float32),
                    np.array(batch_residuals, dtype=np.float32),
                )
                robot_meas_total += len(round_choices)

            else:  # selection_mode == "sequential"
              # Round-robin: one pick + one refit at a time.
              for i in range(obs_per_assim_total):
                r = i % n_robots

                gp_mean_cands, gp_std_total = gp.predict(cand_coords_gp)
                noise_var  = gp.get_noise_variance()
                gp_std_epi = np.sqrt(
                    np.maximum(gp_std_total.astype(np.float64) ** 2 - noise_var, 0.0)
                ).astype(np.float32)

                # No round_reserved here: each pick refits immediately and
                # updates window_visited, which prevents re-selection.
                excl = window_visited | out_of_region[r]

                policy_failed = False
                try:
                    next_local = policy.choose(
                        robot_coords[r], cand_coords,
                        gp_mean_cands, gp_std_epi,
                        excl, rng, max_dist,
                    )
                except RuntimeError:
                    policy_failed = True
                    next_local = -1

                needs_override = (
                    policy_failed
                    or (use_partitioning
                        and cand_region_id is not None
                        and cand_region_id[next_local] != robot_region[r])
                )
                if use_partitioning and needs_override:
                    blocked = window_visited
                    avail = [j for j in in_region_indices[r]
                             if j not in blocked]
                    if partition_strict:
                        if not avail:
                            raise RuntimeError(
                                f"Robot {r}: no in-region candidates remain "
                                f"(region {robot_region[r]}, "
                                f"{len(in_region_indices[r])} total, "
                                f"{len(blocked)} blocked)")
                        dists_ir = np.linalg.norm(
                            cand_coords[avail] - robot_coords[r], axis=1)
                        next_local = int(avail[np.argmin(dists_ir)])
                    else:
                        if avail:
                            dists_ir = np.linalg.norm(
                                cand_coords[avail] - robot_coords[r], axis=1)
                            next_local = int(avail[np.argmin(dists_ir)])
                        elif not policy_failed:
                            spillover_total += 1
                            spillover_step  += 1
                        else:
                            global_avail = [j for j in range(len(cand_coords))
                                            if j not in blocked]
                            if not global_avail:
                                raise RuntimeError(
                                    f"Robot {r}: no candidates remain at all")
                            dists_g = np.linalg.norm(
                                cand_coords[global_avail] - robot_coords[r],
                                axis=1)
                            next_local = int(global_avail[np.argmin(dists_g)])
                            spillover_total += 1
                            spillover_step  += 1
                elif policy_failed:
                    raise RuntimeError(
                        f"Robot {r}: no unvisited candidates remain")

                # Add single observation (refits GP immediately)
                ocean_idx = int(cand_local_idx[next_local])
                if forecast_mode == "none":
                    r_obs = float(y_true_flat[ocean_idx])
                else:
                    r_obs = float(y_true_flat[ocean_idx] - x_prior_flat[ocean_idx])
                obs_coord = cand_coords_gp[next_local] if gp_use_time \
                    else cand_coords[next_local]
                gp.add_observation(obs_coord, r_obs)

                step_dist = float(np.linalg.norm(
                    cand_coords[next_local] - robot_coords[r]))
                cumulative_dists[r] += step_dist
                robot_coords[r] = cand_coords[next_local].copy()
                trajectories[r].append(robot_coords[r].copy())

                if not allow_revisit:
                    visited_cand_set.add(next_local)
                visited_ocean_set.add(ocean_idx)
                window_visited.add(next_local)
                newly_visited_cand.append(next_local)
                robot_meas_total += 1

                if debug:
                    rgn = (f"  region={cand_region_id[next_local]}"
                           if use_partitioning else "")
                    print(f"    [seq i={i}]  robot {r}: cand={next_local}  "
                          f"epi={gp_std_epi[next_local]:.5f}{rgn}")

            # Correct the field
            gp_mean_all, gp_std_all_total = gp.predict(all_coords_gp)
            _noise_var_all = gp.get_noise_variance()
            gp_std_all_epi = np.sqrt(
                np.maximum(gp_std_all_total.astype(np.float64) ** 2 - _noise_var_all, 0.0)
            ).astype(np.float32)
            gp_noise_std_scalar = float(np.sqrt(max(_noise_var_all, 0.0)))

            if debug:
                _vae = gp_std_all_epi[np.isfinite(gp_std_all_epi)]
                print(f"  [std_epi_all  step={s}]  "
                      f"min={_vae.min():.5f}  med={np.median(_vae):.5f}  "
                      f"max={_vae.max():.5f}  noise_std={gp_noise_std_scalar:.5f}")

            last_gp_mean_all  = gp_mean_all.copy()
            last_gp_std_all   = gp_std_all_epi.copy()
            last_gp_noise_std = gp_noise_std_scalar
            last_assim_step   = s

            if forecast_mode == "none":
                corrected_flat = assimilation_gain * gp_mean_all
            else:
                corrected_flat = x_prior_flat + assimilation_gain * gp_mean_all
            x_hat_np          = x_hat_prior.copy()
            x_hat_np[ocean_mask] = corrected_flat
            x_hat_np[land_mask]  = 0.0

            n_assim_done += 1
        else:
            x_hat_np = x_hat_prior

        # ── 4. Metrics ────────────────────────────────────────────────────
        y_eval     = y_true_flat[eval_local_idx]
        x_hat_eval = x_hat_np[ocean_mask][eval_local_idx]
        x_fno_eval = x_fno_np[ocean_mask][eval_local_idx]

        all_observed = visited_ocean_set | (init_ocean_set if bg_mode != "off" else set())
        unobs_mask = np.array(
            [eval_local_idx[i] not in all_observed
             for i in range(len(eval_local_idx))],
            dtype=bool,
        )

        total_dist_sum = sum(cumulative_dists)
        rec = dict(
            step                 = s,
            t_idx                = gt_idx,
            assimilation         = int(is_assimilation),
            n_robots             = n_robots,
            n_meas_robot_total   = robot_meas_total,
            n_meas_bg_total      = bg_meas_total,
            n_meas_total         = robot_meas_total + bg_meas_total,
            n_meas_step          = len(newly_visited_cand),
            n_unique_robot_sites = len(visited_ocean_set),
            n_unique_bg_sites    = len(init_ocean_set) if bg_mode != "off" else 0,
            n_unique_total       = len(visited_ocean_set) + (len(init_ocean_set) if bg_mode != "off" else 0),
            cumulative_dist      = total_dist_sum,
            dist_per_robot_mean  = total_dist_sum / n_robots if n_robots > 0 else 0.0,
            all_rmse          = _rmse(y_eval, x_hat_eval),
            all_mae           = _mae(y_eval, x_hat_eval),
            fno_rmse          = _rmse(y_eval, x_fno_eval),
            fno_mae           = _mae(y_eval, x_fno_eval),
            unobs_rmse        = _rmse(y_eval[unobs_mask], x_hat_eval[unobs_mask])
                                if unobs_mask.any() else float("nan"),
            unobs_mae         = _mae(y_eval[unobs_mask], x_hat_eval[unobs_mask])
                                if unobs_mask.any() else float("nan"),
            partition_spillover_count_step  = spillover_step,
            partition_spillover_count_total = spillover_total,
        )
        step_records.append(rec)

        # ── 5. Qualitative snapshot ───────────────────────────────────────
        if s in save_qual_steps:
            H, W = y_true_np.shape
            qual_frames[s] = dict(
                y_true          = y_true_np.copy(),
                x_prior         = x_hat_prior.copy(),
                x_corrected     = x_hat_np.copy(),
                x_fno           = x_fno_np.copy(),
                gp_mean_map     = last_gp_mean_all.copy(),
                gp_std_map      = last_gp_std_all.copy(),
                gp_noise_std    = last_gp_noise_std,
                is_assimilation = is_assimilation,
                last_assim_step = last_assim_step,
                trajectories    = [np.array(t) for t in trajectories],
                obs_coords      = (all_ocean_coords[list(visited_ocean_set)]
                                   if visited_ocean_set else np.zeros((0, 2))),
                bg_coords       = (all_ocean_coords[list(init_ocean_set)]
                                   if init_ocean_set else np.zeros((0, 2))),
                ocean_mask      = ocean_mask,
                H=H, W=W,
            )

    # Return per-robot trajectories as list of arrays
    traj_arrays = [np.array(t) for t in trajectories]
    return step_records, traj_arrays, qual_frames


# ---------------------------------------------------------------------------
# Multi-policy / multi-episode runner
# ---------------------------------------------------------------------------

def run_all_dynamic_experiments(
    inputs_np,          # (N_samples, H, W)
    ocean_mask,         # (H, W) bool
    all_ocean_coords,   # (N_ocean, 2)
    policies_dict,      # {"name": policy_obj_or_None, ...}
    fno,
    device,
    cfg,
    verbose=True,
):
    """
    Run all policies across n_episodes.

    Returns
    -------
    all_records    : list[dict]  step-level, with 'policy' and 'episode' columns
    trajectories   : dict[policy_name] -> list of list-of-(T_r,2) per episode
    qual_data      : dict[policy_name] -> dict[step] -> field snapshots (episode 0 only)
    cand_local_idx, eval_local_idx
    """
    n_episodes   = cfg.get("n_episodes", 5)
    L            = cfg.get("episode_length", 30)
    qual_ep      = cfg.get("qual_episode", 0)
    qual_steps   = set(cfg.get("qual_steps", []))
    episode_seed_offset = cfg.get("episode_seed_offset", 0)

    cand_local_idx = build_candidates(
        all_ocean_coords, cfg.get("n_candidates", 2000), cfg.get("candidate_seed", 42))
    eval_local_idx = build_eval_cells(
        all_ocean_coords, cfg.get("n_eval_cells", 20000), cfg.get("eval_seed", 123))

    rollout_lead_stride = cfg.get("rollout_lead_stride", cfg.get("lead", 1))
    N      = inputs_np.shape[0]
    max_t0 = N - L * rollout_lead_stride - 1
    if max_t0 < 0:
        raise ValueError(
            f"Not enough data: need at least {L * rollout_lead_stride + 2} samples, have {N}.")
    if max_t0 < n_episodes - 1:
        raise ValueError(
            f"Need {n_episodes} episodes but only {max_t0+1} valid start positions.")

    rng_ep  = np.random.default_rng(episode_seed_offset)
    t0_list = sorted(
        rng_ep.choice(max_t0 + 1, size=min(n_episodes, max_t0 + 1),
                      replace=False).tolist()
    )

    all_records  = []
    trajectories = {name: [] for name in policies_dict}
    qual_data    = {name: {} for name in policies_dict}

    for ep_i, t0 in enumerate(t0_list):
        if verbose:
            print(f"  Episode {ep_i+1}/{len(t0_list)}  (t0={t0})")

        save_qs = qual_steps if ep_i == qual_ep else set()

        for pol_name, policy in policies_dict.items():
            seed    = episode_seed_offset + ep_i * 1000 + _stable_hash(pol_name) % 1000
            records, traj_list, qual_frames = run_dynamic_episode(
                inputs_np, ocean_mask, all_ocean_coords,
                cand_local_idx, eval_local_idx,
                policy, fno, device, cfg,
                episode_seed=seed, t0=t0,
                save_qual_steps=save_qs,
            )
            for r in records:
                r["policy"]  = pol_name
                r["episode"] = ep_i
                r["t0"]      = t0
            all_records.extend(records)
            trajectories[pol_name].append(traj_list)
            if ep_i == qual_ep:
                qual_data[pol_name] = qual_frames

    return all_records, trajectories, qual_data, cand_local_idx, eval_local_idx