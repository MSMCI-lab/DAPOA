"""Safeguard-combination policies: sequential layered screening and selective verification.

Both classes are evaluated from the class-conditional pattern tables ``q1`` (unauthorized) and ``q0``
(authorized) via eqs. (2.18)-(2.19):  ``M = sum_d [1 - g(d)] q1(d)``,  ``B = sum_d g(d) q0(d)``,
``K = sum_d c(S(d)) [omega q1(d) + (1-omega) q0(d)]``,  ``J = K + omega lM M + (1-omega) lB B``.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations, product
from typing import Callable, Optional, Sequence

import numpy as np

from .data import pattern_matrix

ROUTE = {"A": 0, "B": 1, "R": 2}
ROUTE_NAME = {"A": "ALLOW", "B": "BLOCK", "R": "REVIEW"}


@dataclass(frozen=True)
class LayeredPolicy:
    """Invoke ``order`` sequentially; BLOCK at the first BLOCK; ALLOW if every invoked safeguard allows."""
    order: tuple[int, ...]           # component indices in invocation order

    @property
    def subset(self) -> frozenset:
        return frozenset(self.order)

    @property
    def kind(self) -> str:
        return "layered"

    def describe(self, names: Sequence[str]) -> str:
        return "seq[" + ">".join(names[j] for j in self.order) + "]"

    def tables(self, m: int) -> tuple[np.ndarray, np.ndarray]:
        """(g, inv): final BLOCK indicator per pattern and (2**m, m) invocation indicator per pattern."""
        D = pattern_matrix(m); active = np.ones(1 << m, bool); g = np.zeros(1 << m); inv = np.zeros((1 << m, m))
        for j in self.order:
            inv[:, j] = active; g = np.maximum(g, active * D[:, j]); active &= D[:, j] == 0
        return g, inv

    def decide(self, invoke: Callable[[int], int]) -> tuple[int, list[int]]:
        """Deployment rule: ``invoke(j)`` must return the decision of safeguard ``j`` (0/1). Returns (final decision, invoked)."""
        used = []
        for j in self.order:
            used.append(j)
            if int(invoke(j)) == 1:
                return 1, used
        return 0, used

    def to_dict(self, names: Sequence[str]) -> dict:
        return {"kind": "layered", "invocation_order": [names[j] for j in self.order], "rule": "stop at the first BLOCK; ALLOW if all allow"}


@dataclass(frozen=True)
class VerificationPolicy:
    """Invoke all of ``first_stage``; route the first-stage pattern to ALLOW/BLOCK/REVIEW; the reviewer decides on REVIEW."""
    first_stage: tuple[int, ...]     # sorted ascending
    reviewer: int
    routing: str                     # one letter A/B/R per first-stage pattern w = sum_k D[F_k] 2**k

    @property
    def kind(self) -> str:
        return "verification"

    def __post_init__(self):
        if tuple(sorted(self.first_stage)) != tuple(self.first_stage):
            raise ValueError("first_stage must be sorted ascending")
        if self.reviewer in self.first_stage:
            raise ValueError("reviewer must not belong to the first stage")
        if len(self.routing) != 1 << len(self.first_stage) or any(ch not in ROUTE for ch in self.routing):
            raise ValueError("routing must contain one of A/B/R for each first-stage pattern")

    def describe(self, names: Sequence[str]) -> str:
        return f"ver[F={'+'.join(names[j] for j in self.first_stage)};h={names[self.reviewer]};{self.routing}]"

    def pattern_of(self, decisions: np.ndarray) -> np.ndarray:
        """First-stage pattern index ``w`` of full decision vectors (rows) or a (2**m, m) pattern matrix."""
        F = list(self.first_stage)
        return (np.asarray(decisions)[:, F] * (1 << np.arange(len(F)))).sum(axis=1).astype(int)

    def tables(self, m: int) -> tuple[np.ndarray, np.ndarray]:
        D = pattern_matrix(m); w = self.pattern_of(D); r = np.array([ROUTE[ch] for ch in self.routing])[w]
        block = (r == 1).astype(float); review = (r == 2).astype(float)
        g = block + review * D[:, self.reviewer]; inv = np.zeros((1 << m, m)); inv[:, list(self.first_stage)] = 1; inv[:, self.reviewer] = review
        return g, inv

    def routing_table(self, names: Sequence[str]) -> dict:
        F = list(self.first_stage)
        return {"".join(str((w >> k) & 1) for k in range(len(F))): ROUTE_NAME[self.routing[w]] for w in range(1 << len(F))}

    def decide(self, invoke: Callable[[int], int]) -> tuple[int, list[int]]:
        used = list(self.first_stage); w = sum(int(invoke(j)) << k for k, j in enumerate(self.first_stage)); act = self.routing[w]
        if act == "A":
            return 0, used
        if act == "B":
            return 1, used
        used.append(self.reviewer)
        return int(invoke(self.reviewer)), used

    def to_dict(self, names: Sequence[str]) -> dict:
        return {"kind": "verification", "first_stage": [names[j] for j in self.first_stage], "reviewer": names[self.reviewer], "routing_vector": self.routing,
                "routing_by_pattern": self.routing_table(names), "pattern_order": "bits list first-stage decisions in the order of first_stage (1 = BLOCK)"}


def flag_to_review(first_stage: Sequence[int], reviewer: int) -> VerificationPolicy:
    """The fixed flag-to-review cascade: ALLOW the all-ALLOW pattern, REVIEW every flagged pattern."""
    n = 1 << len(first_stage)
    return VerificationPolicy(tuple(sorted(first_stage)), reviewer, "A" + "R" * (n - 1))


def policy_metrics(policy, q1: np.ndarray, q0: np.ndarray, costs: Sequence[float], omega: float, loss_miss: float, loss_false_block: float) -> dict:
    """Exact M, B, K, J (and review/invocation rates) of one policy under the pattern tables."""
    m = len(costs); c = np.asarray(costs, float); g, inv = policy.tables(m); mix = omega * q1 + (1 - omega) * q0
    M = float((1 - g) @ q1); B = float(g @ q0); K = float((inv @ c) @ mix)
    out = {"M": M, "B": B, "K": K, "J": K + omega * loss_miss * M + (1 - omega) * loss_false_block * B,
           "P1_invoked": (inv.T @ q1).tolist(), "P0_invoked": (inv.T @ q0).tolist()}
    if policy.kind == "verification":
        rv = inv[:, policy.reviewer]; out["review_rate_y1"] = float(rv @ q1); out["review_rate_y0"] = float(rv @ q0)
    return out


class PolicyClass:
    """Explicit enumeration of a policy class with vectorized metrics (used for exact cross-checks and m <= 4 searches).

    BASE = all layered orders + flag-to-review cascades; FULL = all layered orders + all deterministic routing maps.
    Canonical order: layered policies (subset size, lexicographic order), then verification (sorted F, reviewer, routing A<B<R).
    """

    def __init__(self, m: int, include_all_maps: bool = True, max_subset: Optional[int] = None):
        self.m = m; D = pattern_matrix(m); specs, G, INV, base = [], [], [], []
        for s in range(1, (max_subset or m) + 1):
            for order in sorted(permutations(range(m), s)):
                pol = LayeredPolicy(order); g, inv = pol.tables(m); specs.append(pol); G.append(g); INV.append(inv); base.append(True)
        self.n_layered = len(specs); self.pairs = []
        for f in range(1, m):
            for F in combinations(range(m), f):
                n = 1 << f
                for h in range(m):
                    if h in F:
                        continue
                    start = len(specs)
                    routes = product("ABR", repeat=n) if include_all_maps else [tuple("A" + "R" * (n - 1))]
                    for route in routes:
                        pol = VerificationPolicy(F, h, "".join(route)); g, inv = pol.tables(m); specs.append(pol); G.append(g); INV.append(inv); base.append("".join(route) == "A" + "R" * (n - 1))
                    self.pairs.append({"F": F, "h": h, "start": start, "end": len(specs)})
        self.specs = specs; self.G = np.array(G); self.INV = np.array(INV); self.base = np.array(base)

    def __len__(self) -> int:
        return len(self.specs)

    def metrics(self, q1, q0, costs, omega, loss_miss, loss_false_block):
        c = np.asarray(costs, float); mix = omega * q1 + (1 - omega) * q0
        M = 1 - self.G @ q1; B = self.G @ q0; K = (self.INV @ c) @ mix
        return M, B, K, K + omega * loss_miss * M + (1 - omega) * loss_false_block * B
