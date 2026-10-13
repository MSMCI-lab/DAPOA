"""Exact dependence-aware policy search (Algorithms 1 and 2 of the paper).

* :func:`layered_opt` -- Algorithm 1 (LayeredOpt): subset dynamic programme over ``2**m`` states giving the
  minimum operating cost ``K*(S)`` and an optimal invocation order for every subset, followed by constrained
  subset selection.  No intermediate state is pruned by the error limits.
* :func:`routing_opt` -- the binary routing programme (3.25) for one first-stage/reviewer pair, solved globally
  by scipy's HiGHS mixed-integer solver (``mip_rel_gap = 0``), or by exhaustive enumeration on request.
* :func:`design` -- Algorithm 2: LayeredOpt plus every admitted pair; returns the global optimum of the
  total-cost problem (2.26) or of the budget problem (2.27) within the admitted class, or INFEASIBLE.

Ties are resolved by the fixed hierarchy (J, M, B, K, canonical index) for minimum total cost and
(M, K, B, canonical index) for minimum miss, with a relative tolerance ``tie_rel * max(1, |key|)``.
A solver time limit is reported as an unresolved status, never as infeasibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .data import pattern_matrix
from .policies import LayeredPolicy, PolicyClass, VerificationPolicy, flag_to_review, policy_metrics


@dataclass
class Admissible:
    """Operational restrictions on the policy class (all optional)."""
    allow_layered: bool = True
    allow_verification: bool = True
    flag_to_review_only: bool = False            # restrict verification to the fixed flag-to-review cascades (BASE class)
    max_layered_size: Optional[int] = None
    required_components: tuple[int, ...] = ()    # must be invoked (layered: in the subset; verification: in F or as reviewer)
    allowed_reviewers: Optional[tuple[int, ...]] = None
    allowed_first_stage_sizes: Optional[tuple[int, ...]] = None
    veto_when_all_block: bool = False            # force automatic BLOCK on the all-BLOCK first-stage pattern
    forced_routes: dict = field(default_factory=dict)   # {(F, h): {w: 'A'|'B'|'R'}} fixed routing decisions


@dataclass
class Requirements:
    """Operating requirements of the design problem."""
    costs: Sequence[float]
    omega: float = 0.5
    objective: str = "min_total_cost"            # or "min_miss"
    alpha: Optional[float] = 0.05                # miss limit (min_total_cost)
    beta: float = 0.10                           # false-block limit (both problems)
    budget: Optional[float] = None               # operating-cost budget on K (min_miss)
    loss_miss: float = 100.0
    loss_false_block: float = 10.0
    admissible: Admissible = field(default_factory=Admissible)
    feas_tol: float = 1e-10
    tie_rel: float = 1e-10

    def __post_init__(self):
        if self.objective not in ("min_total_cost", "min_miss"):
            raise ValueError("objective must be 'min_total_cost' or 'min_miss'")
        if self.objective == "min_total_cost" and self.alpha is None:
            raise ValueError("alpha is required for min_total_cost")
        if self.objective == "min_miss" and self.budget is None:
            raise ValueError("budget is required for min_miss")
        if not 0 <= self.omega <= 1:
            raise ValueError("omega must lie in [0, 1]")
        if any(c < 0 for c in self.costs):
            raise ValueError("costs must be nonnegative")


@dataclass
class DesignResult:
    status: str                                   # "optimal", "infeasible", "unresolved"
    policy: Optional[object] = None
    metrics: dict = field(default_factory=dict)
    objective_value: float = np.inf
    best_layered: Optional[dict] = None
    best_verification: Optional[dict] = None
    pair_results: list = field(default_factory=list)
    layered_table: list = field(default_factory=list)
    feasible_without_tolerance: Optional[bool] = None
    n_unresolved_pairs: int = 0
    cross_check: Optional[dict] = None

    def describe(self, names: Sequence[str]) -> str:
        return self.policy.describe(names) if self.policy is not None else self.status


# ====================================================================================== helpers
def _joint_allow(q: np.ndarray, m: int) -> dict[int, float]:
    """a_y(S) = sum_{d: d_j = 0 for j in S} q(d) for every subset bitmask S (a(empty) = 1)."""
    D = pattern_matrix(m); out = {}
    for s in range(1 << m):
        memb = [j for j in range(m) if (s >> j) & 1]
        allow = np.all(D[:, memb] == 0, axis=1) if memb else np.ones(1 << m, bool)
        out[s] = float(q[allow].sum())
    return out


def _feasible(req: Requirements, M: float, B: float, K: float, strict: bool = False) -> bool:
    tol = 0.0 if strict else req.feas_tol
    if req.objective == "min_total_cost":
        return M <= req.alpha + tol and B <= req.beta + tol
    return K <= req.budget + tol and B <= req.beta + tol


def _keys(req: Requirements, M: float, B: float, K: float, J: float) -> tuple:
    return (J, M, B, K) if req.objective == "min_total_cost" else (M, K, B)


def _better(req: Requirements, a: tuple, b: tuple) -> bool:
    """True if key tuple ``a`` beats ``b`` under the tolerance hierarchy (b may be None)."""
    if b is None:
        return True
    for x, y in zip(a, b):
        tol = req.tie_rel * max(1.0, abs(y))
        if x < y - tol:
            return True
        if x > y + tol:
            return False
    return False


# ====================================================================================== Algorithm 1
def layered_opt(q1: np.ndarray, q0: np.ndarray, req: Requirements) -> dict:
    """Algorithm 1: exact selection and ordering of sequential layered screening.

    Returns a dict with the best admitted subset/order (or ``status='infeasible'``) and the full K*(S) table.
    """
    c = np.asarray(req.costs, float); m = len(c); adm = req.admissible; omega = req.omega
    a1 = _joint_allow(q1, m); a0 = _joint_allow(q0, m); abar = {s: omega * a1[s] + (1 - omega) * a0[s] for s in a1}
    Kstar = {0: 0.0}; last = {}
    for s in range(1, 1 << m):                     # states in increasing bitmask order: every proper subset precedes its supersets
        best, bj = np.inf, None
        for j in range(m):
            if (s >> j) & 1:
                v = Kstar[s ^ (1 << j)] + c[j] * abar[s ^ (1 << j)]
                if v < best - 1e-15 or (abs(v - best) <= 1e-15 and bj is not None and j < bj):
                    best, bj = v, j
        Kstar[s] = best; last[s] = bj
    table, best_keys, best = [], None, None
    for s in range(1, 1 << m):
        memb = [j for j in range(m) if (s >> j) & 1]
        if adm.max_layered_size is not None and len(memb) > adm.max_layered_size:
            continue
        if any(r not in memb for r in adm.required_components):
            continue
        t, order = s, []
        while t:
            j = last[t]; order.insert(0, j); t ^= 1 << j
        M, B, K = a1[s], 1 - a0[s], Kstar[s]; J = K + omega * req.loss_miss * M + (1 - omega) * req.loss_false_block * B
        feas = _feasible(req, M, B, K); row = {"subset": memb, "order": order, "M": M, "B": B, "K_star": K, "J": J, "feasible": feas, "feasible_strict": _feasible(req, M, B, K, True)}
        table.append(row)
        if feas and adm.allow_layered:
            keys = _keys(req, M, B, K, J)
            if _better(req, keys, best_keys):
                best_keys, best = keys, row
    if best is None:
        return {"status": "infeasible", "table": table}
    pol = LayeredPolicy(tuple(best["order"]))
    return {"status": "optimal", "policy": pol, "metrics": policy_metrics(pol, q1, q0, req.costs, omega, req.loss_miss, req.loss_false_block), "table": table, "K_star": Kstar,
            "feasible_without_tolerance": best["feasible_strict"]}


# ====================================================================================== routing programme (3.25)
def _pair_coefficients(q1, q0, F, h):
    m = int(np.log2(len(q1))); D = pattern_matrix(m); n = 1 << len(F); w = (D[:, list(F)] * (1 << np.arange(len(F)))).sum(1).astype(int)
    s1 = np.bincount(w, weights=q1, minlength=n); s0 = np.bincount(w, weights=q0, minlength=n)
    t1 = np.bincount(w, weights=q1 * D[:, h], minlength=n); t0 = np.bincount(w, weights=q0 * D[:, h], minlength=n)
    return n, s1, s0, t1, t0


def routing_opt(q1: np.ndarray, q0: np.ndarray, req: Requirements, F: Sequence[int], h: int, method: str = "milp", time_limit: Optional[float] = None) -> dict:
    """Solve the binary routing programme for first stage ``F`` and reviewer ``h``.

    method: ``"milp"`` (HiGHS, global) or ``"enumerate"`` (all 3**(2**|F|) maps; exact tie hierarchy).
    """
    F = tuple(sorted(F)); c = np.asarray(req.costs, float); omega = req.omega; lm, lb = req.loss_miss, req.loss_false_block
    n, s1, s0, t1, t0 = _pair_coefficients(q1, q0, F, h); cF = float(c[list(F)].sum()); rk = c[h] * (omega * s1 + (1 - omega) * s0)
    adm = req.admissible; forced = dict(adm.forced_routes.get((F, h), {}))
    if adm.veto_when_all_block:
        forced[n - 1] = "B"
    if method == "enumerate":
        from itertools import product as _prod
        best, best_keys = None, None
        for route in _prod("ABR", repeat=n):
            if any(route[w] != a for w, a in forced.items()):
                continue
            x = np.array([ch == "B" for ch in route], float); z = np.array([ch == "R" for ch in route], float)
            M = 1 - float((s1 * x + t1 * z).sum()); B = float((s0 * x + t0 * z).sum()); K = cF + float((rk * z).sum()); J = K + omega * lm * M + (1 - omega) * lb * B
            if not _feasible(req, M, B, K):
                continue
            keys = _keys(req, M, B, K, J)
            if _better(req, keys, best_keys):
                best, best_keys = "".join(route), keys
        if best is None:
            return {"status": "infeasible", "F": F, "h": h}
        pol = VerificationPolicy(F, h, best)
        return {"status": "optimal", "F": F, "h": h, "policy": pol, "metrics": policy_metrics(pol, q1, q0, req.costs, omega, lm, lb), "solver": "enumeration", "gap": 0.0}
    # ---- MILP: variables x_w (BLOCK), z_w (REVIEW); x_w + z_w <= 1
    if req.objective == "min_total_cost":
        obj = np.r_[(1 - omega) * lb * s0 - omega * lm * s1, rk + (1 - omega) * lb * t0 - omega * lm * t1]
    else:
        obj = -np.r_[s1, t1]
    A = np.zeros((n, 2 * n)); A[np.arange(n), np.arange(n)] = 1; A[np.arange(n), n + np.arange(n)] = 1
    rows = [A, np.r_[s0, t0][None, :]]; lo = [np.full(n, -np.inf), [-np.inf]]; hi = [np.ones(n), [req.beta + req.feas_tol]]
    if req.objective == "min_total_cost":
        rows.append(np.r_[s1, t1][None, :]); lo.append([1 - req.alpha - req.feas_tol]); hi.append([np.inf])
    else:
        rows.append(np.r_[np.zeros(n), rk][None, :]); lo.append([-np.inf]); hi.append([req.budget - cF + req.feas_tol])
    lb_v = np.zeros(2 * n); ub_v = np.ones(2 * n)
    for w, a in forced.items():
        lb_v[w] = ub_v[w] = 1.0 if a == "B" else 0.0; lb_v[n + w] = ub_v[n + w] = 1.0 if a == "R" else 0.0
    opts = {"mip_rel_gap": 0.0, "presolve": True}
    if time_limit is not None:
        opts["time_limit"] = time_limit
    res = milp(obj, integrality=np.ones(2 * n), bounds=Bounds(lb_v, ub_v), constraints=LinearConstraint(np.vstack(rows), np.concatenate(lo), np.concatenate(hi)), options=opts)
    if res.status == 2:
        return {"status": "infeasible", "F": F, "h": h, "solver": "milp"}
    if res.status != 0:
        return {"status": "unresolved", "F": F, "h": h, "solver": "milp", "message": str(res.message), "solver_status": int(res.status)}
    x = np.rint(res.x[:n]); z = np.rint(res.x[n:]); route = "".join("ABR"[int(v)] for v in (x + 2 * z))
    pol = VerificationPolicy(F, h, route); met = policy_metrics(pol, q1, q0, req.costs, omega, lm, lb)
    return {"status": "optimal", "F": F, "h": h, "policy": pol, "metrics": met, "solver": "milp", "gap": float(getattr(res, "mip_gap", 0.0) or 0.0),
            "feasible_without_tolerance": _feasible(req, met["M"], met["B"], met["K"], True)}


# ====================================================================================== Algorithm 2
def design(q1: np.ndarray, q0: np.ndarray, req: Requirements, method: str = "milp", cross_check: bool = False, time_limit: Optional[float] = None) -> DesignResult:
    """Algorithm 2: exact policy search over the admitted layered and selective-verification classes.

    ``cross_check=True`` re-solves by exhaustive enumeration of every specification (m <= 4 recommended) and
    records the agreement of the optimum objective and feasibility status.
    """
    c = np.asarray(req.costs, float); m = len(c); adm = req.admissible; omega = req.omega; lm, lb = req.loss_miss, req.loss_false_block
    if len(q1) != 1 << m or len(q0) != 1 << m:
        raise ValueError("pattern tables must have length 2**m with m = len(costs)")
    lay = layered_opt(q1, q0, req)
    best, best_keys, best_kind = None, None, None
    if lay["status"] == "optimal":
        mt = lay["metrics"]; best, best_keys, best_kind = lay, _keys(req, mt["M"], mt["B"], mt["K"], mt["J"]), "layered"
    pair_results = []; best_ver = None; best_ver_keys = None; n_unres = 0
    if adm.allow_verification:
        for f in range(1, m):
            if adm.allowed_first_stage_sizes is not None and f not in adm.allowed_first_stage_sizes:
                continue
            for F in combinations(range(m), f):
                for h in range(m):
                    if h in F or (adm.allowed_reviewers is not None and h not in adm.allowed_reviewers):
                        continue
                    if any(r not in F and r != h for r in adm.required_components):
                        continue
                    if adm.flag_to_review_only:
                        pol = flag_to_review(F, h); met = policy_metrics(pol, q1, q0, req.costs, omega, lm, lb)
                        r = ({"status": "optimal", "F": F, "h": h, "policy": pol, "metrics": met, "solver": "fixed cascade", "gap": 0.0, "feasible_without_tolerance": _feasible(req, met["M"], met["B"], met["K"], True)}
                             if _feasible(req, met["M"], met["B"], met["K"]) else {"status": "infeasible", "F": F, "h": h, "solver": "fixed cascade"})
                    else:
                        r = routing_opt(q1, q0, req, F, h, method=method, time_limit=time_limit)
                    pair_results.append(r)
                    if r["status"] == "unresolved":
                        n_unres += 1; continue
                    if r["status"] == "optimal":
                        mt = r["metrics"]; keys = _keys(req, mt["M"], mt["B"], mt["K"], mt["J"])
                        if _better(req, keys, best_ver_keys):
                            best_ver, best_ver_keys = r, keys
                        if _better(req, keys, best_keys):      # layered wins exact ties (canonical order)
                            best, best_keys, best_kind = r, keys, "verification"
    if best is None:
        status = "unresolved" if n_unres else "infeasible"
        return DesignResult(status, None, {}, np.inf, lay if lay["status"] == "optimal" else None, best_ver, pair_results, lay["table"], None, n_unres)
    res = DesignResult("optimal", best["policy"], best["metrics"], best_keys[0], lay if lay["status"] == "optimal" else None, best_ver, pair_results, lay["table"],
                       best.get("feasible_without_tolerance"), n_unres)
    if n_unres:
        res.status = "optimal_with_unresolved_pairs"
    if cross_check:
        res.cross_check = enumeration_check(q1, q0, req, res)
    return res


def enumeration_check(q1: np.ndarray, q0: np.ndarray, req: Requirements, res: DesignResult) -> dict:
    """Exhaustive enumeration of the admitted class (m <= 4 recommended) compared with the returned optimum."""
    m = len(req.costs); adm = req.admissible
    P = PolicyClass(m, include_all_maps=not adm.flag_to_review_only, max_subset=adm.max_layered_size)
    M, B, K, J = P.metrics(q1, q0, req.costs, req.omega, req.loss_miss, req.loss_false_block)
    mask = np.ones(len(P), bool)
    for i, sp in enumerate(P.specs):
        if sp.kind == "layered" and (not adm.allow_layered or any(r not in sp.order for r in adm.required_components)):
            mask[i] = False
        if sp.kind == "verification" and (not adm.allow_verification or (adm.allowed_reviewers is not None and sp.reviewer not in adm.allowed_reviewers)
                                          or (adm.allowed_first_stage_sizes is not None and len(sp.first_stage) not in adm.allowed_first_stage_sizes)
                                          or any(r not in sp.first_stage and r != sp.reviewer for r in adm.required_components)):
            mask[i] = False
        if sp.kind == "verification" and adm.veto_when_all_block and sp.routing[-1] != "B":
            mask[i] = False
    feas = mask & np.array([_feasible(req, M[i], B[i], K[i]) for i in range(len(P))])
    if not feas.any():
        return {"n_enumerated": int(mask.sum()), "enumeration_status": "infeasible", "agrees": res.status == "infeasible"}
    ids = np.flatnonzero(feas); best_i, best_k = None, None
    for i in ids:
        k = _keys(req, M[i], B[i], K[i], J[i])
        if _better(req, k, best_k):
            best_i, best_k = i, k
    agree = res.status.startswith("optimal") and abs(best_k[0] - res.objective_value) <= 1e-9 * max(1.0, abs(best_k[0]))
    return {"n_enumerated": int(mask.sum()), "n_feasible": int(feas.sum()), "enumeration_status": "optimal", "enumeration_objective": float(best_k[0]), "returned_objective": float(res.objective_value),
            "abs_diff": float(abs(best_k[0] - res.objective_value)) if np.isfinite(res.objective_value) else np.inf, "agrees": bool(agree), "enumeration_policy": P.specs[best_i]}
