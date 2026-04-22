"""
Sequential GP state for online residual correction.

The GP is refit from scratch after each new observation (exact GP).
This is safe and exact for small N_obs (≤ ~200 typical episode budgets).

Before any observations: returns mean=0, std=prior_std everywhere.
After k observations: returns full GP posterior mean and std.
"""

import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.gp_correction import (
    fit_residual_gp,
    predict_residual_mean,
    predict_residual_with_std,
)


class GPState:
    """
    Online GP state manager for sequential residual correction.

    Usage
    -----
    gp = GPState(prior_std=0.07, gp_kwargs={...})
    gp.add_observation(coord, residual)   # call once per step
    mean, std = gp.predict(query_coords)  # call for acquisition + metrics
    """

    def __init__(self, prior_std=0.07, gp_kwargs=None):
        """
        Parameters
        ----------
        prior_std  : float  uniform std returned before any observations
        gp_kwargs  : dict   passed verbatim to fit_residual_gp
        """
        self.prior_std = prior_std
        self.gp_kwargs = gp_kwargs or {}
        self._static_coords    = []   # static (warm-start) observations
        self._static_residuals = []
        self._obs_coords    = []   # dynamic observations (robot, per-window bg, etc.)
        self._obs_residuals = []   # list of float scalars
        self._gp            = None # fitted GaussianProcessRegressor or None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_obs(self):
        return len(self._static_coords) + len(self._obs_coords)

    @property
    def n_static(self):
        return len(self._static_coords)

    @property
    def n_dynamic(self):
        return len(self._obs_coords)

    @property
    def obs_coords_array(self):
        """Return (n_obs, D) float32 array, or (0, D) if empty (D=2 or 3)."""
        all_coords = self._static_coords + self._obs_coords
        if all_coords:
            return np.array(all_coords, dtype=np.float32)
        return np.zeros((0, 2), dtype=np.float32)

    @property
    def obs_residuals_array(self):
        all_res = self._static_residuals + self._obs_residuals
        return np.array(all_res, dtype=np.float32)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def add_observations(self, coords, residuals):
        """
        Batch-add dynamic observations and refit the GP once.

        Parameters
        ----------
        coords    : array-like (N, D)  normalized coords (D=2 spatial, D=3 spatio-temporal)
        residuals : array-like (N,)    observed residuals = y_true - y_fno
        """
        coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        residuals = np.asarray(residuals, dtype=np.float32).ravel()
        for c, r in zip(coords, residuals):
            self._obs_coords.append(c)
            self._obs_residuals.append(float(r))
        self._refit()

    def add_observation(self, coord, residual):
        """Add one dynamic observation and refit the GP."""
        self.add_observations(
            np.asarray(coord, dtype=np.float32).reshape(1, -1),
            [residual],
        )

    def add_static_observations(self, coords, residuals):
        """
        Batch-add static (warm-start) observations and refit the GP once.

        Static observations persist across reset(keep_static=True) calls.

        Parameters
        ----------
        coords    : array-like (N, D)  normalized coords (D=2 spatial, D=3 spatio-temporal)
        residuals : array-like (N,)    observed residuals
        """
        coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        residuals = np.asarray(residuals, dtype=np.float32).ravel()
        for c, r in zip(coords, residuals):
            self._static_coords.append(c)
            self._static_residuals.append(float(r))
        self._refit()

    def add_static_observation(self, coord, residual):
        """Add one static (warm-start) observation and refit the GP."""
        self.add_static_observations(
            np.asarray(coord, dtype=np.float32).reshape(1, -1),
            [residual],
        )

    def _refit(self):
        """Refit GP from all (static + dynamic) observations."""
        all_coords = self._static_coords + self._obs_coords
        all_vals   = self._static_residuals + self._obs_residuals
        if not all_coords:
            self._gp = None
            return
        coords = np.array(all_coords, dtype=np.float32)
        vals   = np.array(all_vals, dtype=np.float32)
        self._gp = fit_residual_gp(coords, vals, **self.gp_kwargs)

    def predict(self, query_coords):
        """
        Predict GP residual (mean, std).

        Before any observations: returns (0, prior_std) everywhere.

        Parameters
        ----------
        query_coords : (M, 2) float32

        Returns
        -------
        mean : (M,) float32
        std  : (M,) float32
        """
        if self._gp is None:
            n = len(query_coords)
            return (np.zeros(n, dtype=np.float32),
                    np.full(n, self.prior_std, dtype=np.float32))
        return predict_residual_with_std(self._gp, query_coords)

    def get_noise_variance(self):
        """
        Return the fitted WhiteKernel noise variance in prediction (output) scale.

        Kernel structure: (ConstantKernel * base) + WhiteKernel
          kernel_.k2.noise_level  is in normalized-y space when normalize_y=True.
        We un-normalize by multiplying with _y_train_std^2 so the result is in
        the same units as the values returned by predict().

        Returns 0.0 before any observations (GP not yet fitted).
        """
        if self._gp is None:
            return 0.0
        try:
            noise_level = float(self._gp.kernel_.k2.noise_level)
        except AttributeError:
            return 0.0
        # Account for sklearn's internal y-normalization
        y_std = getattr(self._gp, '_y_train_std',
                getattr(self._gp, '_y_std', np.array([1.0])))
        y_std = float(np.ravel(y_std)[0]) if hasattr(y_std, '__len__') else float(y_std)
        return noise_level * y_std ** 2

    def predict_epistemic_std(self, query_coords):
        """
        Epistemic (signal-only) std: total predictive std with noise variance removed.

        epistemic_var = max(total_var - noise_var, 0)

        Returns the same mean as predict() but with noise subtracted from the variance.
        Useful for acquisition: the robot should visit high-epistemic-std locations,
        not locations that merely have high noise.
        """
        mean, std_total = self.predict(query_coords)
        noise_var = self.get_noise_variance()
        epi_std = np.sqrt(
            np.maximum(std_total.astype(np.float64) ** 2 - noise_var, 0.0)
        ).astype(np.float32)
        return mean, epi_std

    def predict_mean_only(self, query_coords):
        """Faster mean-only prediction (no std computation)."""
        if self._gp is None:
            return np.zeros(len(query_coords), dtype=np.float32)
        return predict_residual_mean(self._gp, query_coords)

    @staticmethod
    def make_query_coords_st(space_coords_2d, t_norm):
        """
        Append a constant time column to 2D spatial coordinates.

        Parameters
        ----------
        space_coords_2d : (N, 2) float32  spatial (row, col)
        t_norm          : float            normalized time value

        Returns
        -------
        coords_3d : (N, 3) float32  (row, col, t_norm)
        """
        n = len(space_coords_2d)
        t_col = np.full((n, 1), t_norm, dtype=np.float32)
        return np.hstack([np.asarray(space_coords_2d, dtype=np.float32), t_col])

    def reset(self, keep_static=False):
        """
        Reset dynamic observations.

        Parameters
        ----------
        keep_static : bool
            If True, keep static observations and refit GP from them only.
            If False, clear everything (full reset to prior).
        """
        self._obs_coords    = []
        self._obs_residuals = []
        if keep_static:
            # Refit from static obs only (or set _gp=None if none exist)
            self._refit()
        else:
            self._static_coords    = []
            self._static_residuals = []
            self._gp               = None
