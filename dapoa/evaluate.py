"""Evaluation of fixed policies on labeled held-out decisions (replay with fixed invocation charges) and
paired group-level bootstrap intervals for policy differences.  Nothing here feeds back into model
fitting or policy selection."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .data import CalibrationData, pattern_index


def evaluate_policy(policy, data: CalibrationData, costs: Sequence[float], omega: float, loss_miss: float, loss_false_block: float, weights: Optional[np.ndarray] = None) -> dict:
    """Replay ``policy`` on every action of ``data``: class-specific error rates from integer counts, fixed-charge
    operating cost from the components actually invoked, and total cost under the supplied omega and losses."""
    m = data.m; c = np.asarray(costs, float); g, inv = policy.tables(m); d = pattern_index(data.decisions); y = data.y
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    gd = g[d]; invd = inv[d]; kd = invd @ c; n1 = w[y == 1].sum(); n0 = w[y == 0].sum()
    M = float((w * (1 - gd) * (y == 1)).sum() / n1); B = float((w * gd * (y == 0)).sum() / n0)
    K = omega * float((w * kd * (y == 1)).sum() / n1) + (1 - omega) * float((w * kd * (y == 0)).sum() / n0)
    inv1 = (w[:, None] * invd * (y == 1)[:, None]).sum(0) / n1; inv0 = (w[:, None] * invd * (y == 0)[:, None]).sum(0) / n0
    return {"M": M, "B": B, "K": K, "J": K + omega * loss_miss * M + (1 - omega) * loss_false_block * B,
            "n_miss": float((w * (1 - gd) * (y == 1)).sum()), "n_false_block": float((w * gd * (y == 0)).sum()), "n1": float(n1), "n0": float(n0),
            "P1_invoked": inv1.tolist(), "P0_invoked": inv0.tolist(), "K_identity_check": float(abs(K - c @ (omega * inv1 + (1 - omega) * inv0))), "decisions": gd.astype(int)}


def paired_bootstrap(policy_a, policy_b, data: CalibrationData, costs, omega, loss_miss, loss_false_block, n_boot: int = 500, seed: int = 0, level: float = 0.95) -> dict:
    """Paired percentile intervals for the evaluation differences (a minus b) of M, B, K, J, resampling groups
    (or actions when no groups are given) with replacement; the same draws are used for both policies."""
    rng = np.random.default_rng(seed); units = data.groups if data.groups is not None else np.arange(len(data.y)); uniq = np.unique(units)
    diffs = {q: [] for q in ("M", "B", "K", "J")}
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True); cnt = {}
        for u in draw:
            cnt[u] = cnt.get(u, 0) + 1
        w = np.array([cnt.get(u, 0) for u in units], float)
        if w[data.y == 1].sum() == 0 or w[data.y == 0].sum() == 0:
            continue
        ra = evaluate_policy(policy_a, data, costs, omega, loss_miss, loss_false_block, w); rb = evaluate_policy(policy_b, data, costs, omega, loss_miss, loss_false_block, w)
        for q in diffs:
            diffs[q].append(ra[q] - rb[q])
    ra0 = evaluate_policy(policy_a, data, costs, omega, loss_miss, loss_false_block); rb0 = evaluate_policy(policy_b, data, costs, omega, loss_miss, loss_false_block)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2; out = {}
    for q, v in diffs.items():
        v = np.array(v); out[q] = {"point": ra0[q] - rb0[q], "ci_lo": float(np.quantile(v, lo)), "ci_hi": float(np.quantile(v, hi)), "n_resamples": int(len(v))}
    out["n_discordant_actions"] = int((ra0["decisions"] != rb0["decisions"]).sum()); out["unit"] = "group" if data.groups is not None else "action"
    return out
