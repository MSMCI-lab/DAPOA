"""Simulation of binary safeguard decisions from specified class-conditional copula models (for examples and tests)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import stats

from .data import CalibrationData


def _stable_positive(alpha: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Positive stable variates with index ``alpha`` in (0, 1) (Chambers-Mallows-Stuck)."""
    U = rng.uniform(0, np.pi, size); W = rng.exponential(1.0, size)
    return (np.sin(alpha * U) / np.sin(U) ** (1 / alpha)) * (np.sin((1 - alpha) * U) / W) ** ((1 - alpha) / alpha)


def sample_copula(family: str, n: int, m: int, rng: np.random.Generator, correlation: Optional[np.ndarray] = None, nu: Optional[float] = None, theta: Optional[float] = None) -> np.ndarray:
    """Uniform(0,1) samples with the given copula."""
    if family == "independence":
        return rng.uniform(size=(n, m))
    if family in ("gaussian", "student_t"):
        R = np.eye(m) if correlation is None else np.asarray(correlation, float); Z = rng.multivariate_normal(np.zeros(m), R, size=n)
        if family == "gaussian":
            return stats.norm.cdf(Z)
        V = rng.chisquare(nu, size=n) / nu
        return stats.t.cdf(Z / np.sqrt(V)[:, None], df=nu)
    if family == "clayton":
        V = rng.gamma(1 / theta, 1.0, size=n); E = rng.exponential(1.0, size=(n, m))
        return (1 + E / V[:, None]) ** (-1 / theta)
    if family == "gumbel":
        if theta <= 1 + 1e-12:
            return rng.uniform(size=(n, m))
        V = _stable_positive(1 / theta, n, rng); E = rng.exponential(1.0, size=(n, m))
        return np.exp(-(E / V[:, None]) ** (1 / theta))
    raise ValueError(f"unknown family {family}")


def simulate_decisions(n_authorized: int, n_unauthorized: int, block_prob_y0: Sequence[float], block_prob_y1: Sequence[float], family_y0: str = "independence", family_y1: str = "gaussian",
                       params_y0: Optional[dict] = None, params_y1: Optional[dict] = None, components: Optional[Sequence[str]] = None, n_groups: Optional[int] = None, seed: int = 0) -> CalibrationData:
    """Simulate labeled binary decisions: ``D_j = 1{U_j > 1 - p_j}`` with ``U`` drawn from the class copula.

    ``params_y`` may contain ``correlation`` (Gaussian / Student-t), ``nu`` (Student-t) and ``theta`` (Clayton / Gumbel).
    """
    rng = np.random.default_rng(seed); p0 = np.asarray(block_prob_y0, float); p1 = np.asarray(block_prob_y1, float); m = len(p0)
    U0 = sample_copula(family_y0, n_authorized, m, rng, **(params_y0 or {})); U1 = sample_copula(family_y1, n_unauthorized, m, rng, **(params_y1 or {}))
    D = np.vstack([(U0 > 1 - p0).astype(int), (U1 > 1 - p1).astype(int)]); y = np.r_[np.zeros(n_authorized, int), np.ones(n_unauthorized, int)]
    groups = None if n_groups is None else rng.integers(0, n_groups, size=len(y))
    names = list(components) if components is not None else [f"D{j + 1}" for j in range(m)]
    return CalibrationData(D, y, names, groups)
