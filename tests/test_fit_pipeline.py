import json

import numpy as np
import pytest

from dapoa import CalibrationData, Requirements, run_design, simulate_decisions, fit_family, select_model, evaluate_policy, LayeredPolicy
from dapoa.cli import main

P0 = [0.03, 0.12, 0.05]; P1 = [0.80, 0.90, 0.85]


def _data(seed=0, n=1500, rho=0.5):
    R = np.full((3, 3), rho); np.fill_diagonal(R, 1)
    return simulate_decisions(n, n, P0, P1, "gaussian", "gaussian", {"correlation": R}, {"correlation": R}, n_groups=100, seed=seed)


def test_fit_recovers_gaussian_correlation():
    data = _data(); mdl = fit_family(data, 1, "gaussian")
    R = mdl.params["correlation"]; off = R[np.triu_indices(3, 1)]
    assert np.all(np.abs(off - 0.5) < 0.15)
    assert mdl.checks["margins_reproduced"] and abs(mdl.q.sum() - 1) < 1e-9


def test_selection_prefers_dependence_over_product():
    data = _data(); sel, allm, tab = select_model(data, 1, ("independence", "gaussian", "clayton", "gumbel"), "bic")
    assert sel.family != "independence"
    assert tab["table"]["gaussian"]["score"] < tab["table"]["independence"]["score"]


def test_constant_coordinate_handling():
    D = np.array([[0, 1, 1], [0, 0, 1], [0, 1, 0], [0, 0, 0]] * 20 + [[0, 0, 0]] * 80); y = np.r_[np.ones(80, int), np.zeros(80, int)]
    data = CalibrationData(D, y, ["a", "b", "c"]); mdl = fit_family(data, 1, "gaussian")
    assert mdl.nonconstant == [1, 2] and mdl.q[[d for d in range(8) if d & 1]].sum() == 0


def test_pipeline_and_evaluation():
    data = _data(seed=1, n=600); ev = _data(seed=2, n=400)
    req = Requirements([0.2, 0.35, 0.65], omega=0.2, alpha=0.08, beta=0.15, loss_miss=100, loss_false_block=10)
    rep = run_design(data, req, ("independence", "gaussian", "clayton"), "bic", evaluation=ev, n_boot=50, cross_check=True)
    d = rep.to_dict(); json.dumps(d, default=str)
    assert rep.dependence_aware.status == "optimal" and rep.dependence_aware.cross_check["agrees"]
    assert "dependence_aware" in rep.evaluation and "paired_bootstrap_dependence_aware_minus_independence" in rep.evaluation
    md = rep.to_markdown(); assert "Selected policies" in md
    e = evaluate_policy(LayeredPolicy((0, 1)), ev, req.costs, 0.2, 100, 10); assert e["K_identity_check"] < 1e-12


def test_cli_roundtrip(tmp_path):
    cal = tmp_path / "cal.csv"; out = tmp_path / "rep.json"
    assert main(["simulate", str(cal), "--n0", "300", "--n1", "300", "--p0", "0.03", "0.12", "0.05", "--p1", "0.8", "0.9", "0.85", "--groups", "40", "--seed", "3"]) == 0
    assert main(["design", str(cal), "--group", "group", "--costs", "0.2", "0.35", "0.65", "--alpha", "0.08", "--beta", "0.15", "--omega", "0.2", "--families", "independence", "gaussian", "--out", str(out), "--quiet"]) == 0
    rep = json.load(open(out)); assert rep["dependence_aware_design"]["status"] == "optimal"
