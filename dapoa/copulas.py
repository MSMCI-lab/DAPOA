"""Class-conditional joint decision probabilities: copula families, estimation and model selection.

For a class ``y`` with empirical blocking probabilities ``p_j`` and a copula ``C``, the probability of the
binary pattern ``d`` is the rectangle probability (finite difference) of eq. (2.17) in the paper::

    q(d) = sum_{eps in {0,1}^m} (-1)^{|eps|} C(F_1(d_1 - eps_1), ..., F_m(d_m - eps_m)),

with ``F_j(-1) = 0``, ``F_j(0) = 1 - p_j`` (ALLOW probability) and ``F_j(1) = 1``.  Elliptical families use
rectangle probabilities of the latent vector (deterministic quasi-Monte Carlo with a fixed generator);
Archimedean families use the explicit copula.  Margins are the empirical class proportions and are shared
by every family (including the independence comparator).  Dependence parameters are estimated by
maximizing the grouped pattern log-likelihood (eq. C.4); families are compared by BIC (eq. C.5) by default.

Coordinates whose empirical blocking probability is 0 or 1 are constant in the class; they are excluded
from dependence estimation and restored as constants when the full pattern table is formed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Optional, Sequence

import numpy as np
from scipy import optimize, stats

from .data import CalibrationData, pattern_matrix

FAMILIES = ("independence", "gaussian", "student_t", "clayton", "gumbel")
NU_GRID = (3.0, 5.0, 8.0, 15.0, 30.0, 100.0)
_EPS = 1e-12


# ====================================================================================== pattern probabilities
def product_patterns(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, dtype=float); D = pattern_matrix(len(p))
    return np.prod(np.where(D == 1, p, 1 - p), axis=1)


def empirical_patterns(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    return counts / counts.sum()


def _elliptical_patterns(p: np.ndarray, R: np.ndarray, nu: Optional[float], seed: int, maxpts: int, abseps: float) -> np.ndarray:
    """Rectangle probabilities of a latent Gaussian (nu=None) or Student-t vector at the binary thresholds."""
    m = len(p); D = pattern_matrix(m)
    if m == 1:
        return np.array([1 - p[0], p[0]])
    t = stats.norm.ppf(1 - p) if nu is None else stats.t.ppf(1 - p, df=nu)
    # ALLOW (d_j = 0): X_j <= t_j ; BLOCK (d_j = 1): X_j > t_j
    out = np.zeros(1 << m)
    for d in range(1 << m):
        lower = np.where(D[d] == 1, t, -np.inf); upper = np.where(D[d] == 1, np.inf, t)
        rng = np.random.default_rng(seed + d)
        if nu is None:
            out[d] = stats.multivariate_normal.cdf(upper, mean=np.zeros(m), cov=R, lower_limit=lower, maxpts=maxpts, abseps=abseps, releps=abseps, rng=rng)
        else:
            out[d] = stats.multivariate_t.cdf(upper, loc=np.zeros(m), shape=R, df=nu, lower_limit=lower, maxpts=maxpts, random_state=rng)
    return out


def gaussian_patterns(p, R, seed: int = 0, maxpts: int = 200_000, abseps: float = 1e-9) -> np.ndarray:
    return _elliptical_patterns(np.asarray(p, float), np.asarray(R, float), None, seed, maxpts, abseps)


def student_t_patterns(p, R, nu: float, seed: int = 0, maxpts: int = 200_000, abseps: float = 1e-9) -> np.ndarray:
    return _elliptical_patterns(np.asarray(p, float), np.asarray(R, float), float(nu), seed, maxpts, abseps)


def _archimedean_patterns(p: np.ndarray, C) -> np.ndarray:
    """Finite differences of an explicit copula ``C(u)`` (eq. 2.17)."""
    m = len(p); D = pattern_matrix(m); F = {-1: np.zeros(m), 0: 1 - p, 1: np.ones(m)}
    out = np.zeros(1 << m)
    for d in range(1 << m):
        total = 0.0
        for eps in product((0, 1), repeat=m):
            u = np.array([F[int(D[d, j]) - eps[j]][j] for j in range(m)])
            total += (-1) ** sum(eps) * (0.0 if (u <= 0).any() else C(u))
        out[d] = total
    return out


def clayton_patterns(p, theta: float) -> np.ndarray:
    p = np.asarray(p, float); m = len(p)
    if theta <= 1e-10:
        return product_patterns(p)
    return _archimedean_patterns(p, lambda u: (np.sum(u ** (-theta)) - m + 1) ** (-1 / theta))


def gumbel_patterns(p, theta: float) -> np.ndarray:
    p = np.asarray(p, float)
    if theta <= 1 + 1e-12:
        return product_patterns(p)
    return _archimedean_patterns(p, lambda u: np.exp(-(np.sum((-np.log(u)) ** theta)) ** (1 / theta)))


# ====================================================================================== correlation parameterization
def partial_to_corr(P: np.ndarray) -> np.ndarray:
    """Map canonical partial correlations (strict upper triangle of ``P``, each in (-1,1)) to a
    positive-definite correlation matrix (Lewandowski, Kurowicka and Joe, 2009)."""
    n = P.shape[0]; R = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = P[i, j]
            for k in range(i - 1, -1, -1):
                r = r * np.sqrt((1 - P[k, i] ** 2) * (1 - P[k, j] ** 2)) + P[k, i] * P[k, j]
            R[i, j] = R[j, i] = r
    return R


def _unpack_corr(z: np.ndarray, n: int) -> np.ndarray:
    P = np.zeros((n, n)); iu = np.triu_indices(n, 1); P[iu] = np.tanh(z)
    return partial_to_corr(P)


def corr_to_partial(R: np.ndarray) -> np.ndarray:
    """Inverse of :func:`partial_to_corr` (used to warm-start from a sample correlation)."""
    n = R.shape[0]; P = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = R[i, j]
            for k in range(i):
                r = (r - P[k, i] * P[k, j]) / np.sqrt(max((1 - P[k, i] ** 2) * (1 - P[k, j] ** 2), 1e-12))
            P[i, j] = np.clip(r, -0.999, 0.999)
    return P


# ====================================================================================== fitted model container
@dataclass
class PatternModel:
    """Fitted class-conditional pattern probabilities for one class."""
    family: str
    y: int
    components: list[str]
    margins: np.ndarray                     # empirical blocking probabilities, all components
    q: np.ndarray                           # pattern probabilities, length 2**m (full component set)
    params: dict = field(default_factory=dict)
    nonconstant: list[int] = field(default_factory=list)
    loglik: float = np.nan
    n_params: int = 0
    n_obs: int = 0
    checks: dict = field(default_factory=dict)
    fit_info: dict = field(default_factory=dict)

    @property
    def bic(self) -> float:
        return -2 * self.loglik + self.n_params * np.log(max(self.n_obs, 1))

    @property
    def aic(self) -> float:
        return -2 * self.loglik + 2 * self.n_params

    def to_dict(self) -> dict:
        return {"family": self.family, "y": self.y, "components": self.components, "margins": self.margins.tolist(), "q": self.q.tolist(),
                "params": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.params.items()}, "nonconstant": [self.components[j] for j in self.nonconstant],
                "loglik": self.loglik, "n_params": self.n_params, "n_obs": self.n_obs, "bic": self.bic, "aic": self.aic, "checks": self.checks, "fit_info": self.fit_info}


def _expand(q_sub: np.ndarray, m: int, nonconstant: list[int], margins: np.ndarray) -> np.ndarray:
    """Insert constant coordinates (p = 0 -> always ALLOW, p = 1 -> always BLOCK) into a sub-pattern table."""
    D = pattern_matrix(m); q = np.zeros(1 << m); Ds = pattern_matrix(len(nonconstant))
    const = [j for j in range(m) if j not in nonconstant]
    for ds in range(len(q_sub)):
        d = 0
        for k, j in enumerate(nonconstant):
            d |= int(Ds[ds, k]) << j
        for j in const:
            d |= int(round(margins[j])) << j
        q[d] = q_sub[ds]
    return q


def _check(q: np.ndarray, margins: np.ndarray, tol: float = 1e-7) -> tuple[np.ndarray, dict]:
    """Nonnegativity, normalization and marginal checks; repairs numerical residues below ``tol``."""
    D = pattern_matrix(len(margins)); raw_min = float(q.min()); raw_sum = float(q.sum())
    if raw_min < -tol:
        raise ValueError(f"pattern probabilities contain a negative entry {raw_min:.3e} beyond tolerance {tol}")
    qc = np.clip(q, 0, None); qc = qc / qc.sum()
    marg_err = float(np.max(np.abs(qc @ D - margins)))
    return qc, {"raw_min": raw_min, "raw_sum": raw_sum, "n_clipped": int((q < 0).sum()), "l1_adjustment": float(np.abs(qc - q).sum()), "margin_error_max": marg_err, "margins_reproduced": marg_err < 1e-5}


def _loglik(q: np.ndarray, counts: np.ndarray) -> float:
    mask = counts > 0
    if (q[mask] <= 0).any():
        return -np.inf
    return float(np.sum(counts[mask] * np.log(q[mask])))


# ====================================================================================== estimation
def fit_family(data: CalibrationData, y: int, family: str, seed: int = 0, nu_grid: Sequence[float] = NU_GRID,
               fit_maxpts: int = 5_000, fit_abseps: float = 1e-5, final_maxpts: int = 500_000, final_abseps: float = 1e-10,
               maxiter: int = 200) -> PatternModel:
    """Fit one family to class ``y`` of ``data`` by maximizing the grouped pattern log-likelihood with fixed empirical margins.

    Elliptical rectangle probabilities use a fixed quasi-Monte Carlo generator: a coarse tolerance (``fit_*``) during the
    likelihood optimization and a tight one (``final_*``) for the reported pattern table.  Student-t degrees of freedom are
    profiled over ``nu_grid`` (one extra parameter in BIC).
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from {FAMILIES}")
    m = data.m; counts = data.counts(y); margins = data.margins(y); n = int(counts.sum())
    nonconst = [j for j in range(m) if 0 < margins[j] < 1]
    p = margins[nonconst]; Ds = pattern_matrix(len(nonconst))
    # counts of the sub-patterns over nonconstant coordinates
    full_idx = np.arange(1 << m); D = pattern_matrix(m)
    sub_idx = np.zeros(1 << m, dtype=int)
    for k, j in enumerate(nonconst):
        sub_idx |= D[:, j].astype(int) << k
    sub_counts = np.bincount(sub_idx, weights=counts, minlength=1 << len(nonconst))
    info = {"n_nonconstant": len(nonconst), "converged": True, "boundary": False, "message": ""}
    params: dict = {}; k_par = 0
    if len(nonconst) < 2 or family == "independence":
        q_sub = product_patterns(p)
        if family != "independence":
            info["message"] = "fewer than two nonconstant coordinates: no dependence parameter estimated"
    elif family in ("clayton", "gumbel"):
        pf = clayton_patterns if family == "clayton" else gumbel_patterns
        lo, hi = (1e-6, 50.0) if family == "clayton" else (1.0 + 1e-9, 50.0)
        obj = lambda th: -_loglik(np.clip(pf(p, th), _EPS, None), sub_counts)
        res = optimize.minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6, "maxiter": maxiter})
        theta = float(res.x); params = {"theta": theta}; k_par = 1; q_sub = pf(p, theta)
        info.update({"converged": bool(res.success), "boundary": bool(theta > hi * 0.98 or theta < lo + 1e-5), "message": str(res.message)})
    else:
        ms = len(nonconst); n_corr = ms * (ms - 1) // 2
        # warm start: tetrachoric-like start from the sample correlation of the binary decisions
        X = data.subset(y)[:, nonconst].astype(float); Rs = np.corrcoef(X, rowvar=False); Rs = np.nan_to_num(Rs, nan=0.0)
        z0 = np.arctanh(np.clip(corr_to_partial(Rs)[np.triu_indices(ms, 1)], -0.9, 0.9))

        def nll(z, nu):
            R = _unpack_corr(z, ms)
            q = _elliptical_patterns(p, R, nu, seed, fit_maxpts, fit_abseps)
            return -_loglik(np.clip(q, _EPS, None), sub_counts)
        if family == "gaussian":
            res = optimize.minimize(lambda z: nll(z, None), z0, method="L-BFGS-B", bounds=[(-4.0, 4.0)] * n_corr, options={"maxiter": maxiter, "eps": 1e-5, "ftol": 1e-10, "gtol": 1e-7})
            R = _unpack_corr(res.x, ms); params = {"correlation": R, "nu": None}; k_par = n_corr; nu = None
            info.update({"converged": bool(res.success), "message": str(res.message)})
        else:  # student_t: profile over a fixed nu grid, correlations refitted for each nu
            best = None
            for nu_c in nu_grid:
                res = optimize.minimize(lambda z: nll(z, nu_c), z0, method="L-BFGS-B", bounds=[(-4.0, 4.0)] * n_corr, options={"maxiter": maxiter, "eps": 1e-5, "ftol": 1e-10, "gtol": 1e-7})
                if best is None or res.fun < best[0]:
                    best = (res.fun, nu_c, res)
                z0 = res.x
            _, nu, res = best; R = _unpack_corr(res.x, ms); params = {"correlation": R, "nu": float(nu)}; k_par = n_corr + 1
            info.update({"converged": bool(res.success), "message": str(res.message), "nu_grid": list(nu_grid), "nu_at_grid_boundary": bool(nu in (min(nu_grid), max(nu_grid)))})
        info["boundary"] = bool(np.max(np.abs(R[np.triu_indices(ms, 1)])) > 0.99)
        q_sub = _elliptical_patterns(p, R, nu, seed, final_maxpts, final_abseps)
    q_full = _expand(q_sub, m, nonconst, margins)
    q_full, checks = _check(q_full, margins)
    ll = _loglik(q_full, counts)
    return PatternModel(family, y, list(data.components), margins, q_full, params, nonconst, ll, k_par, n, checks, info)


def empirical_model(data: CalibrationData, y: int) -> PatternModel:
    counts = data.counts(y); margins = data.margins(y); q = empirical_patterns(counts)
    nonconst = [j for j in range(data.m) if 0 < margins[j] < 1]
    k = (1 << len(nonconst)) - 1 - len(nonconst)   # saturated joint table beyond the margins
    _, checks = _check(q, margins)
    return PatternModel("empirical", y, list(data.components), margins, q, {}, nonconst, _loglik(q, counts), max(k, 0), int(counts.sum()), checks, {"message": "raw calibration pattern frequencies (nonparametric comparator; not an oracle)"})


def fit_all(data: CalibrationData, y: int, families: Sequence[str] = FAMILIES, seed: int = 0, **kw) -> dict[str, PatternModel]:
    out = {f: fit_family(data, y, f, seed=seed, **kw) for f in families}
    out["empirical"] = empirical_model(data, y)
    return out


def _cv_score(data: CalibrationData, y: int, family: str, n_folds: int, seed: int, **kw) -> float:
    """Out-of-fold pattern log score (higher is better); folds keep groups together."""
    folds = data.split_by_group(n_folds, seed)[0]; cls = data.y == y; score = 0.0
    for f in range(n_folds):
        train = cls & (folds != f); test = cls & (folds == f)
        if test.sum() == 0 or train.sum() < 2:
            continue
        # training container: the fold's class-y actions plus one action of the other class (required by the validator; unused)
        sub = CalibrationData(np.vstack([data.decisions[train], data.decisions[data.y != y][:1]]), np.r_[np.full(train.sum(), y), [1 - y]], data.components)
        model = fit_family(sub, y, family, seed=seed, **kw)
        q = np.clip(model.q, 1e-12, None); q = q / q.sum()
        test_idx = (data.decisions[test] * (1 << np.arange(data.m))).sum(axis=1)
        score += float(np.sum(np.log(q[test_idx])))
    return score


def select_model(data: CalibrationData, y: int, families: Sequence[str] = FAMILIES, criterion: str = "bic", seed: int = 0,
                 n_folds: int = 5, include_empirical: bool = False, **kw) -> tuple[PatternModel, dict[str, PatternModel], dict]:
    """Fit the candidate families and select one.

    criterion: ``"bic"`` (default, eq. C.5), ``"aic"``, or ``"cv"`` (grouped out-of-fold log score).
    Returns (selected model, all fitted models, selection table).
    """
    models = fit_all(data, y, families, seed=seed, **kw)
    cands = list(families) + (["empirical"] if include_empirical else [])
    table = {}
    for f in cands:
        mdl = models[f]
        if criterion == "bic":
            table[f] = {"score": mdl.bic, "loglik": mdl.loglik, "n_params": mdl.n_params}
        elif criterion == "aic":
            table[f] = {"score": mdl.aic, "loglik": mdl.loglik, "n_params": mdl.n_params}
        elif criterion == "cv":
            table[f] = {"score": -_cv_score(data, y, f, n_folds, seed, **kw) if f != "empirical" else np.inf, "loglik": mdl.loglik, "n_params": mdl.n_params}
        else:
            raise ValueError("criterion must be 'bic', 'aic' or 'cv'")
    # lower score is better for every criterion as stored; ties resolved in the order of ``families`` (independence first)
    best = min(cands, key=lambda f: (table[f]["score"], cands.index(f)))
    sel = {"criterion": criterion, "selected": best, "table": table,
           "note": ("fewer than two coordinates vary in this class; all families coincide with the product" if len(models[best].nonconstant) < 2 else "")}
    return models[best], models, sel
