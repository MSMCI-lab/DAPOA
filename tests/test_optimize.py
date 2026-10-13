"""Exactness checks and reproduction of published values of the paper's numerical pool (Section 4)."""
import numpy as np
import pytest

from dapoa import Requirements, Admissible, design, layered_opt, routing_opt, product_patterns, gaussian_patterns, policy_metrics, LayeredPolicy, VerificationPolicy, PolicyClass
from dapoa.optimize import enumeration_check

MISS = np.array([0.160, 0.080, 0.100, 0.035]); FB = np.array([0.030, 0.120, 0.050, 0.025]); COST = [0.20, 0.35, 0.65, 2.00]
P1 = 1 - MISS; P0 = FB
REQ = Requirements(COST, omega=0.10, objective="min_total_cost", alpha=0.05, beta=0.10, loss_miss=100, loss_false_block=10)


def test_layered_order_cost_published():
    q1, q0 = product_patterns(P1), product_patterns(P0)
    k13 = policy_metrics(LayeredPolicy((0, 2)), q1, q0, COST, 0.10, 100, 10)["K"]; k31 = policy_metrics(LayeredPolicy((2, 0)), q1, q0, COST, 0.10, 100, 10)["K"]
    assert k13 == pytest.approx(0.77785, abs=1e-6) and k31 == pytest.approx(0.82300, abs=1e-6)


def test_independence_scenario_S0_published():
    q1, q0 = product_patterns(P1), product_patterns(P0)
    res = design(q1, q0, REQ, cross_check=True)
    assert res.status == "optimal" and res.cross_check["agrees"]
    pol = res.policy; assert isinstance(pol, VerificationPolicy)
    assert pol.first_stage == (0, 1) and pol.reviewer == 2                     # R1: F = {1,2}, h = 3
    assert pol.routing_table(["D1", "D2", "D3", "D4"]) == {"00": "ALLOW", "10": "REVIEW", "01": "REVIEW", "11": "BLOCK"}
    m = res.metrics
    assert m["M"] == pytest.approx(0.03424, abs=5e-6) and m["B"] == pytest.approx(0.01074, abs=5e-6)
    assert m["K"] == pytest.approx(0.647474, abs=1e-6) and m["J"] == pytest.approx(1.086534, abs=1e-6)
    base = design(q1, q0, Requirements(COST, 0.10, "min_total_cost", 0.05, 0.10, None, 100, 10, Admissible(flag_to_review_only=True)))
    assert base.metrics["J"] == pytest.approx(1.5174, abs=1e-4)                # best baseline C(1,2;4)


def test_gaussian_scenario_S1_published():
    rho = np.sin(np.pi * 0.30 / 2); R = np.full((4, 4), rho); np.fill_diagonal(R, 1)
    q1, q0 = gaussian_patterns(P1, R), gaussian_patterns(P0, R)
    res = design(q1, q0, REQ)
    pol = res.policy; assert pol.first_stage == (0, 1) and pol.reviewer == 3   # R2: reviewer D4
    assert pol.routing_table(list("abcd")) == {"00": "ALLOW", "10": "REVIEW", "01": "REVIEW", "11": "BLOCK"}
    assert res.metrics["J"] == pytest.approx(1.5017, abs=2e-4) and res.metrics["M"] == pytest.approx(0.0488, abs=2e-4)
    # independence-selected R1 violates the miss limit under S1 (Figure 1)
    r1 = VerificationPolicy((0, 1), 2, "ARRB"); assert policy_metrics(r1, q1, q0, COST, 0.10, 100, 10)["M"] > 0.05


def test_dp_matches_enumeration_random_tables():
    rng = np.random.default_rng(3)
    for _ in range(5):
        q1 = rng.dirichlet(np.ones(16)); q0 = rng.dirichlet(np.ones(16) * 3); req = Requirements(rng.uniform(0.1, 2, 4), 0.3, "min_total_cost", 1.0, 1.0, None, 50, 5)
        lay = layered_opt(q1, q0, req); P = PolicyClass(4, include_all_maps=False); M, B, K, J = P.metrics(q1, q0, req.costs, 0.3, 50, 5)
        for row in lay["table"]:
            ids = [i for i, sp in enumerate(P.specs[:P.n_layered]) if set(sp.order) == set(row["subset"])]
            assert row["K_star"] == pytest.approx(K[ids].min(), abs=1e-12)
        assert design(q1, q0, req, cross_check=True).cross_check["agrees"]


def test_milp_matches_enumeration_per_pair():
    rng = np.random.default_rng(7); q1 = rng.dirichlet(np.ones(16)); q0 = rng.dirichlet(np.ones(16) * 4)
    for req in (Requirements([0.2, 0.3, 0.6, 2.0], 0.2, "min_total_cost", 0.3, 0.3, None, 100, 10), Requirements([0.2, 0.3, 0.6, 2.0], 0.2, "min_miss", None, 0.3, 1.5, 100, 10)):
        for F in [(0,), (1, 2), (0, 2, 3)]:
            for h in range(4):
                if h in F:
                    continue
                a = routing_opt(q1, q0, req, F, h, "milp"); b = routing_opt(q1, q0, req, F, h, "enumerate")
                assert a["status"] == b["status"]
                if a["status"] == "optimal":
                    key = "J" if req.objective == "min_total_cost" else "M"
                    assert a["metrics"][key] == pytest.approx(b["metrics"][key], abs=1e-9)


def test_infeasible_and_restrictions():
    q1, q0 = product_patterns(P1), product_patterns(P0)
    assert design(q1, q0, Requirements(COST, 0.1, "min_total_cost", 1e-6, 1e-6, None, 100, 10)).status == "infeasible"
    res = design(q1, q0, Requirements(COST, 0.1, "min_total_cost", 0.05, 0.10, None, 100, 10, Admissible(allow_verification=False)))
    assert isinstance(res.policy, LayeredPolicy)
    res = design(q1, q0, Requirements(COST, 0.1, "min_total_cost", 0.05, 0.10, None, 100, 10, Admissible(allowed_reviewers=(3,))))
    assert res.policy.kind == "layered" or res.policy.reviewer == 3
    res = design(q1, q0, Requirements(COST, 0.1, "min_miss", None, 0.10, 2.0, 100, 10))
    assert res.status == "optimal" and res.metrics["K"] <= 2.0 + 1e-9
