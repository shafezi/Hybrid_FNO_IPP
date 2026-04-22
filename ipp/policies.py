"""
IPP acquisition policies for single-robot sensing.

All policies share the same interface: given the current GP state and robot
position, choose the index of the next candidate location to visit.

Acquisition function (hybrid greedy):
    score(x) = beta * sigma_GP(x) + alpha * |mu_GP(x)| - lambda_dist * dist(robot, x)

Policies
--------
HybridGreedyPolicy   — main method: balances exploration, exploitation, movement
UncertaintyOnlyPolicy — explores highest-uncertainty regions
ResidualMeanPolicy    — targets locations with largest predicted bias
RandomPolicy          — uniform random baseline
RasterPolicy          — fixed lawnmower coverage baseline (non-adaptive)
"""

import numpy as np
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BasePolicy(ABC):
    name = "base"

    @abstractmethod
    def choose(self, robot_coord, cand_coords, gp_mean, gp_std,
               visited_set, rng, max_step_distance=0.0):
        """
        Choose the next candidate index.

        Parameters
        ----------
        robot_coord       : (2,)      current robot position (normalized)
        cand_coords       : (N, 2)    all candidate positions
        gp_mean           : (N,)      GP residual mean at candidates
        gp_std            : (N,)      GP residual std at candidates
        visited_set       : set[int]  already-visited candidate indices
        rng               : np.random.Generator
        max_step_distance : float     hard distance constraint (0 = none)

        Returns
        -------
        int  index into cand_coords
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _valid_mask(self, robot_coord, cand_coords, visited_set, max_step_distance):
        """Boolean mask: unvisited AND within max_step_distance."""
        valid = np.ones(len(cand_coords), dtype=bool)
        if visited_set:
            valid[list(visited_set)] = False
        if max_step_distance > 0:
            dists = np.linalg.norm(cand_coords - robot_coord, axis=1)
            valid &= (dists <= max_step_distance)
        return valid

    def _fallback_nearest_unvisited(self, robot_coord, cand_coords, visited_set):
        """Emergency fallback: nearest unvisited candidate (ignore distance limit)."""
        unvisited = np.array([i for i in range(len(cand_coords))
                              if i not in visited_set])
        if len(unvisited) == 0:
            raise RuntimeError("No unvisited candidates remain.")
        dists = np.linalg.norm(cand_coords[unvisited] - robot_coord, axis=1)
        return int(unvisited[np.argmin(dists)])

    def _argmax(self, scores, rng, eps=1e-9):
        """Argmax with random tie-breaking among near-equal top scores."""
        best = np.max(scores)
        top  = np.where(scores >= best - eps)[0]
        return int(rng.choice(top))


# ---------------------------------------------------------------------------
# Concrete policies
# ---------------------------------------------------------------------------

class HybridGreedyPolicy(BasePolicy):
    """
    Main method.

    score(x) = beta * sigma_GP(x) + alpha * |mu_GP(x)| - lambda_dist * d(x)

    - sigma term: encourages exploring uncertain regions
    - |mu| term:  encourages correcting locations with large predicted bias
    - distance penalty: avoids unnecessary travel
    """
    name = "hybrid_greedy"

    def __init__(self, alpha=1.0, beta=1.0, lambda_dist=0.5):
        self.alpha       = alpha
        self.beta        = beta
        self.lambda_dist = lambda_dist

    def choose(self, robot_coord, cand_coords, gp_mean, gp_std,
               visited_set, rng, max_step_distance=0.0):
        valid = self._valid_mask(robot_coord, cand_coords, visited_set, max_step_distance)
        if not valid.any():
            return self._fallback_nearest_unvisited(robot_coord, cand_coords, visited_set)

        dists  = np.linalg.norm(cand_coords - robot_coord, axis=1)
        scores = (self.beta  * gp_std
                  + self.alpha * np.abs(gp_mean)
                  - self.lambda_dist * dists)
        scores[~valid] = -np.inf
        return self._argmax(scores, rng)


class UncertaintyOnlyPolicy(BasePolicy):
    """
    Uncertainty sampling.

    score(x) = sigma_GP(x) - lambda_dist * d(x)

    Pure exploration baseline — ignores predicted residual magnitude.
    """
    name = "uncertainty_only"

    def __init__(self, lambda_dist=0.5):
        self.lambda_dist = lambda_dist

    def choose(self, robot_coord, cand_coords, gp_mean, gp_std,
               visited_set, rng, max_step_distance=0.0):
        valid = self._valid_mask(robot_coord, cand_coords, visited_set, max_step_distance)
        if not valid.any():
            return self._fallback_nearest_unvisited(robot_coord, cand_coords, visited_set)

        dists  = np.linalg.norm(cand_coords - robot_coord, axis=1)
        scores = gp_std - self.lambda_dist * dists
        scores[~valid] = -np.inf
        return self._argmax(scores, rng)


class ResidualMeanPolicy(BasePolicy):
    """
    Exploitation-only: target locations with largest predicted bias.

    score(x) = |mu_GP(x)| - lambda_dist * d(x)
    """
    name = "residual_mean"

    def __init__(self, lambda_dist=0.5):
        self.lambda_dist = lambda_dist

    def choose(self, robot_coord, cand_coords, gp_mean, gp_std,
               visited_set, rng, max_step_distance=0.0):
        valid = self._valid_mask(robot_coord, cand_coords, visited_set, max_step_distance)
        if not valid.any():
            return self._fallback_nearest_unvisited(robot_coord, cand_coords, visited_set)

        dists  = np.linalg.norm(cand_coords - robot_coord, axis=1)
        scores = np.abs(gp_mean) - self.lambda_dist * dists
        scores[~valid] = -np.inf
        return self._argmax(scores, rng)


class MIPolicy(BasePolicy):
    """
    Greedy Mutual Information sensor placement
    (Krause & Guestrin 2005; Ma et al. IROS 2016).

    score(x) = sum_y K(x, y) * sigma(y)
        where K is the kernel and sigma(y) is the GP posterior std at y.

    This is the kernel-weighted informativeness: locations that are uncertain
    AND well-correlated with many other uncertain candidates are scored highest.

    For multi-robot batch picks, the policy maintains a list of locations
    already picked in the current round (via reset_picks() at start of round).
    Subsequent picks subtract the redundancy with previously-picked locations.
    """
    name = "mi"

    def __init__(self, length_scale=0.1, kernel="matern", nu=1.5,
                 lambda_dist=0.0):
        self.length_scale = length_scale
        self.kernel = kernel
        self.nu = nu
        self.lambda_dist = lambda_dist
        self._picks_this_round = []  # candidate indices picked by other robots

    def reset_picks(self):
        """Call at the start of a multi-robot picking round."""
        self._picks_this_round = []

    def _kernel_matrix(self, A, B):
        """Compute K(A, B) for the configured kernel."""
        from scipy.spatial.distance import cdist
        D = cdist(A, B)
        r = D / max(self.length_scale, 1e-6)
        if self.kernel == "rbf":
            return np.exp(-0.5 * r ** 2)
        # Matern
        if abs(self.nu - 0.5) < 1e-6:
            return np.exp(-r)
        if abs(self.nu - 1.5) < 1e-6:
            return (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
        if abs(self.nu - 2.5) < 1e-6:
            return (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r)
        return np.exp(-r)  # fallback

    def choose(self, robot_coord, cand_coords, gp_mean, gp_std,
               visited_set, rng, max_step_distance=0.0):
        valid = self._valid_mask(robot_coord, cand_coords, visited_set, max_step_distance)
        if not valid.any():
            return self._fallback_nearest_unvisited(robot_coord, cand_coords, visited_set)

        # Kernel-weighted informativeness: K @ gp_std
        K = self._kernel_matrix(cand_coords, cand_coords)
        scores = K @ gp_std

        # Subtract redundancy with locations already picked this round
        # For each picked p, reduce score at x by K(x, p) * gp_std(p)
        for p in self._picks_this_round:
            scores -= K[:, p] * gp_std[p]

        # Distance penalty
        if self.lambda_dist > 0:
            dists = np.linalg.norm(cand_coords - robot_coord, axis=1)
            scores -= self.lambda_dist * dists

        scores[~valid] = -np.inf
        idx = self._argmax(scores, rng)
        self._picks_this_round.append(idx)
        return idx


class GPUCBPolicy(BasePolicy):
    """
    GP-UCB (Srinivas et al., 2012) applied to the residual field.

    score(x) = mu_GP(x) + beta_t^{1/2} * sigma_GP(x)

    where beta_t = 2 * log(|D| * t^2 * pi^2 / (6 * delta)).

    This is the standard GP-UCB with theoretical regret bounds.
    Call set_step(t) before each choose() to update beta_t.
    """
    name = "gp_ucb"

    def __init__(self, delta=0.1, input_dim=2):
        self.delta = delta
        self.input_dim = input_dim
        self._step = 1
        self._beta_t = self._compute_beta(1)

    def _compute_beta(self, t):
        return 2.0 * np.log(self.input_dim * (t ** 2) * (np.pi ** 2) / (6.0 * self.delta))

    def set_step(self, t):
        """Update the exploration parameter for trial t (call before choose)."""
        self._step = max(t, 1)
        self._beta_t = self._compute_beta(self._step)

    def choose(self, robot_coord, cand_coords, gp_mean, gp_std,
               visited_set, rng, max_step_distance=0.0):
        valid = self._valid_mask(robot_coord, cand_coords, visited_set, max_step_distance)
        if not valid.any():
            return self._fallback_nearest_unvisited(robot_coord, cand_coords, visited_set)

        scores = gp_mean + np.sqrt(self._beta_t) * gp_std
        scores[~valid] = -np.inf
        return self._argmax(scores, rng)


class HybridUCBPolicy(BasePolicy):
    """
    Hybrid GP-UCB adapted for residual correction.

    score(x) = |mu_GP(x)| + beta_t^{1/2} * sigma_GP(x)

    Like GP-UCB but uses |mu| instead of mu, since for residual correction
    we want to target large errors in either direction.
    Inherits the growing beta_t schedule from Srinivas et al. (2012).
    """
    name = "hybrid_ucb"

    def __init__(self, delta=0.1, input_dim=2):
        self.delta = delta
        self.input_dim = input_dim
        self._step = 1
        self._beta_t = self._compute_beta(1)

    def _compute_beta(self, t):
        return 2.0 * np.log(self.input_dim * (t ** 2) * (np.pi ** 2) / (6.0 * self.delta))

    def set_step(self, t):
        self._step = max(t, 1)
        self._beta_t = self._compute_beta(self._step)

    def choose(self, robot_coord, cand_coords, gp_mean, gp_std,
               visited_set, rng, max_step_distance=0.0):
        valid = self._valid_mask(robot_coord, cand_coords, visited_set, max_step_distance)
        if not valid.any():
            return self._fallback_nearest_unvisited(robot_coord, cand_coords, visited_set)

        scores = np.abs(gp_mean) + np.sqrt(self._beta_t) * gp_std
        scores[~valid] = -np.inf
        return self._argmax(scores, rng)


class RandomPolicy(BasePolicy):
    """
    Random baseline: uniformly samples a valid unvisited candidate.
    Ignores all GP information.
    """
    name = "random"

    def choose(self, robot_coord, cand_coords, gp_mean, gp_std,
               visited_set, rng, max_step_distance=0.0):
        valid = self._valid_mask(robot_coord, cand_coords, visited_set, max_step_distance)
        if not valid.any():
            return self._fallback_nearest_unvisited(robot_coord, cand_coords, visited_set)
        valid_idx = np.where(valid)[0]
        return int(rng.choice(valid_idx))


class RasterPolicy(BasePolicy):
    """
    Non-adaptive lawnmower (boustrophedon) coverage baseline.

    Visits candidates in a fixed row-by-row order with alternating column
    direction.  Distance constraint is NOT enforced — the robot teleports
    to the next raster position.  This represents the best systematic
    coverage without any adaptation.

    Note: the raster order is fixed at the start of each episode via
    set_order(cand_coords).  The policy object is stateless between episodes
    (order is recomputed each time set_order is called).
    """
    name = "raster"

    def __init__(self):
        self._order = None   # list of candidate indices in raster order

    def set_order(self, cand_coords):
        """Precompute boustrophedon traversal order for these candidates."""
        rows = cand_coords[:, 0]
        cols = cand_coords[:, 1]

        # Divide into horizontal strips
        n_strips = max(1, int(np.ceil(np.sqrt(len(cand_coords)) / 2)))
        strip_h  = 1.0 / n_strips
        strip_id = np.floor(rows / strip_h).astype(int)
        strip_id = np.clip(strip_id, 0, n_strips - 1)

        order = []
        for s in sorted(np.unique(strip_id)):
            in_strip = np.where(strip_id == s)[0]
            if s % 2 == 0:
                in_strip = in_strip[np.argsort(cols[in_strip])]       # left→right
            else:
                in_strip = in_strip[np.argsort(-cols[in_strip])]      # right→left
            order.extend(in_strip.tolist())
        self._order = order

    def choose(self, robot_coord, cand_coords, gp_mean, gp_std,
               visited_set, rng, max_step_distance=0.0):
        if self._order is None:
            self.set_order(cand_coords)

        for idx in self._order:
            if idx not in visited_set:
                return int(idx)

        # Fallback if all visited (shouldn't happen if budget < n_candidates)
        return self._fallback_nearest_unvisited(robot_coord, cand_coords, visited_set)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_policies(cfg):
    """
    Instantiate requested policies from config dict.

    Supports two config formats:

    1. Nested (preferred — from single_robot_ipp.yaml):
        policies:
          hybrid_greedy:
            type: hybrid_greedy
            beta: 1.0
            alpha: 1.0
            lambda_dist: 0.5
          random:
            type: random

    2. Flat (legacy — single set of params for all policies):
        alpha: 1.0
        beta:  1.0
        lambda_dist: 0.5
        policies: [hybrid_greedy, random, raster]
    """
    pol_cfg = cfg.get("policies", None)

    # ---- nested dict format -------------------------------------------------
    if isinstance(pol_cfg, dict):
        result = {}
        for name, pcfg in pol_cfg.items():
            ptype = pcfg.get("type", name)
            if ptype == "hybrid_greedy":
                result[name] = HybridGreedyPolicy(
                    alpha=pcfg.get("alpha", 1.0),
                    beta=pcfg.get("beta",  1.0),
                    lambda_dist=pcfg.get("lambda_dist", 0.5),
                )
            elif ptype == "uncertainty_only":
                result[name] = UncertaintyOnlyPolicy(
                    lambda_dist=pcfg.get("lambda_dist", 0.5))
            elif ptype == "residual_mean":
                result[name] = ResidualMeanPolicy(
                    lambda_dist=pcfg.get("lambda_dist", 0.5))
            elif ptype == "mi":
                result[name] = MIPolicy(
                    length_scale=pcfg.get("length_scale", 0.1),
                    kernel=pcfg.get("kernel", "matern"),
                    nu=pcfg.get("nu", 1.5),
                    lambda_dist=pcfg.get("lambda_dist", 0.0))
            elif ptype == "gp_ucb":
                result[name] = GPUCBPolicy(
                    delta=pcfg.get("delta", 0.1),
                    input_dim=pcfg.get("input_dim", 2))
            elif ptype == "hybrid_ucb":
                result[name] = HybridUCBPolicy(
                    delta=pcfg.get("delta", 0.1),
                    input_dim=pcfg.get("input_dim", 2))
            elif ptype == "random":
                result[name] = RandomPolicy()
            elif ptype == "raster":
                result[name] = RasterPolicy()
        return result

    # ---- flat / list format (legacy) ----------------------------------------
    alpha       = cfg.get("alpha",       1.0)
    beta        = cfg.get("beta",        1.0)
    lambda_dist = cfg.get("lambda_dist", 0.5)

    available = {
        "hybrid_greedy":    HybridGreedyPolicy(alpha=alpha, beta=beta, lambda_dist=lambda_dist),
        "uncertainty_only": UncertaintyOnlyPolicy(lambda_dist=lambda_dist),
        "residual_mean":    ResidualMeanPolicy(lambda_dist=lambda_dist),
        "gp_ucb":           GPUCBPolicy(),
        "hybrid_ucb":       HybridUCBPolicy(),
        "random":           RandomPolicy(),
        "raster":           RasterPolicy(),
    }

    if isinstance(pol_cfg, list):
        return {k: available[k] for k in pol_cfg if k in available}

    return available
