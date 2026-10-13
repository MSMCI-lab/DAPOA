"""End-to-end pipeline: calibration data + requirements -> fitted dependence models -> selected model ->
exact policy search (dependence-aware), with the independence-based and empirical-frequency comparators."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .copulas import FAMILIES, PatternModel, select_model
from .data import CalibrationData
from .evaluate import evaluate_policy, paired_bootstrap
from .optimize import DesignResult, Requirements, design
from .policies import policy_metrics


def _np(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if hasattr(o, "describe") and hasattr(o, "kind"):
        return str(o)
    return str(o)


@dataclass
class DesignReport:
    data_summary: dict
    requirements: dict
    selection: dict                                   # per class: criterion table and selected family
    models: dict                                      # per class: selected PatternModel (+ all fitted)
    dependence_aware: DesignResult
    independence: DesignResult
    empirical: Optional[DesignResult]
    comparisons: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    names: list = field(default_factory=list)

    # ------------------------------------------------------------------ rendering
    def _pol(self, res: DesignResult) -> dict:
        if res.policy is None:
            return {"status": res.status}
        return {"status": res.status, "policy": res.policy.describe(self.names), "definition": res.policy.to_dict(self.names), **{k: res.metrics[k] for k in ("M", "B", "K", "J")},
                "feasible_without_tolerance": res.feasible_without_tolerance, "n_unresolved_pairs": res.n_unresolved_pairs,
                "best_layered": (res.best_layered["policy"].describe(self.names) if res.best_layered else None),
                "best_verification": (res.best_verification["policy"].describe(self.names) if res.best_verification else None),
                "cross_check": ({k: _np(v) for k, v in res.cross_check.items()} if res.cross_check else None)}

    def to_dict(self) -> dict:
        return {"data": self.data_summary, "requirements": self.requirements, "model_selection": self.selection,
                "selected_models": {str(y): self.models[y]["selected"].to_dict() for y in (1, 0)},
                "all_models": {str(y): {f: m.to_dict() for f, m in self.models[y]["all"].items()} for y in (1, 0)},
                "dependence_aware_design": self._pol(self.dependence_aware), "independence_design": self._pol(self.independence),
                "empirical_frequency_design": (self._pol(self.empirical) if self.empirical else None), "comparisons": self.comparisons, "evaluation": self.evaluation}

    def to_json(self, path: Optional[str] = None, indent: int = 1) -> str:
        s = json.dumps(self.to_dict(), indent=indent, default=_np)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    def to_markdown(self) -> str:
        d = self.to_dict(); L = ["# DAPOA design report", ""]
        ds = d["data"]; L += [f"Calibration data: {ds['n_actions']} actions ({ds['n_unauthorized']} unauthorized, {ds['n_authorized']} authorized); components {', '.join(ds['components'])}.", ""]
        L += ["| component | calibration misses | calibration false blocks | cost |", "|---|---:|---:|---:|"]
        for c, cost in zip(ds["components"], d["requirements"]["costs"]):
            e = ds["error_counts"][c]; L.append(f"| {c} | {e['misses']} | {e['false_blocks']} | {cost:g} |")
        r = d["requirements"]; L += ["", f"Requirements: objective `{r['objective']}`, omega = {r['omega']}, alpha = {r['alpha']}, beta = {r['beta']}, budget = {r['budget']}, loss_miss = {r['loss_miss']}, loss_false_block = {r['loss_false_block']}.", ""]
        L += ["## Dependence model selection", ""]
        for y in ("1", "0"):
            s = d["model_selection"][y]; L += [f"Class y = {y} ({'unauthorized' if y == '1' else 'authorized'}): criterion `{s['criterion']}`, selected **{s['selected']}**. {s.get('note', '')}", "", "| family | log-likelihood | parameters | score |", "|---|---:|---:|---:|"]
            for f, t in s["table"].items():
                L.append(f"| {f} | {t['loglik']:.4f} | {t['n_params']} | {t['score']:.4f} |")
            L.append("")
        L += ["## Selected policies", "", "| search | status | policy | M | B | K | J |", "|---|---|---|---:|---:|---:|---:|"]
        for lab, key in (("dependence-aware", "dependence_aware_design"), ("independence", "independence_design"), ("empirical frequencies", "empirical_frequency_design")):
            p = d[key]
            if p is None:
                continue
            if p["status"] in ("infeasible", "unresolved"):
                L.append(f"| {lab} | {p['status']} | | | | | |"); continue
            L.append(f"| {lab} | {p['status']} | `{p['policy']}` | {p['M']:.4f} | {p['B']:.4f} | {p['K']:.4f} | {p['J']:.4f} |")
        if d["comparisons"]:
            L += ["", "## Comparisons under the selected dependence model", ""]
            for k, v in d["comparisons"].items():
                L.append(f"- {k}: " + ", ".join(f"{a} = {b:.4f}" if isinstance(b, float) else f"{a} = {b}" for a, b in v.items()))
        if d["evaluation"]:
            L += ["", "## Held-out evaluation", ""]
            for k, v in d["evaluation"].items():
                if isinstance(v, dict) and "M" in v and isinstance(v["M"], float):
                    L.append(f"- {k}: misses {int(v['n_miss'])}/{int(v['n1'])}, false blocks {int(v['n_false_block'])}/{int(v['n0'])}, K = {v['K']:.4f}, J = {v['J']:.4f}")
                elif isinstance(v, dict):
                    L.append(f"- {k}: " + "; ".join(f"d{q} = {v[q]['point']:+.4f} [{v[q]['ci_lo']:+.4f}, {v[q]['ci_hi']:+.4f}]" for q in ("M", "B", "K", "J") if q in v) + f"; discordant actions {v.get('n_discordant_actions')}")
        L += ["", "Fitted-model optimality refers to the fitted probabilities, not to the deployment distribution; fitted feasibility is not a statistical reliability certificate."]
        return "\n".join(L) + "\n"


def run_design(data: CalibrationData, req: Requirements, families: Sequence[str] = FAMILIES, criterion: str = "bic", seed: int = 0, method: str = "milp",
               cross_check: bool = False, evaluation: Optional[CalibrationData] = None, n_boot: int = 500, include_empirical_design: bool = True, **fit_kw) -> DesignReport:
    """Run the complete DAPOA procedure.

    1. fit the candidate copula families to each class (margins = empirical proportions) and select one per class;
    2. build the class-conditional pattern tables ``q1``, ``q0``;
    3. solve the exact design problem under the selected model, under independence (same margins) and, optionally,
       under the raw empirical frequencies;
    4. evaluate each selected policy under the selected model, and on held-out ``evaluation`` data if supplied.
    """
    names = list(data.components); m = data.m
    if len(req.costs) != m:
        raise ValueError("one cost per component is required")
    models, selection = {}, {}
    for y in (1, 0):
        sel, allm, tab = select_model(data, y, families, criterion, seed, **fit_kw); models[y] = {"selected": sel, "all": allm}; selection[str(y)] = tab
    q1, q0 = models[1]["selected"].q, models[0]["selected"].q
    q1_ind, q0_ind = models[1]["all"]["independence"].q, models[0]["all"]["independence"].q
    dep = design(q1, q0, req, method, cross_check); ind = design(q1_ind, q0_ind, req, method, cross_check)
    emp = None
    if include_empirical_design:
        emp = design(models[1]["all"]["empirical"].q, models[0]["all"]["empirical"].q, req, method, False)
    comparisons = {}
    if ind.policy is not None:
        under_dep = policy_metrics(ind.policy, q1, q0, req.costs, req.omega, req.loss_miss, req.loss_false_block)
        feas = (under_dep["M"] <= req.alpha + req.feas_tol and under_dep["B"] <= req.beta + req.feas_tol) if req.objective == "min_total_cost" else (under_dep["K"] <= req.budget + req.feas_tol and under_dep["B"] <= req.beta + req.feas_tol)
        comparisons["independence_policy_under_selected_model"] = {"policy": ind.policy.describe(names), **{k: under_dep[k] for k in ("M", "B", "K", "J")}, "feasible_under_selected_model": bool(feas)}
        if dep.policy is not None:
            comparisons["dependence_aware_minus_independence_under_selected_model"] = {k: dep.metrics[k] - under_dep[k] for k in ("M", "B", "K", "J")} | {"same_policy": dep.policy == ind.policy}
    if dep.best_layered is not None and dep.best_verification is not None:
        bl, bv = dep.best_layered["metrics"], dep.best_verification["metrics"]
        comparisons["best_verification_minus_best_layered"] = {"layered": dep.best_layered["policy"].describe(names), "verification": dep.best_verification["policy"].describe(names), **{f"d{k}": bv[k] - bl[k] for k in ("M", "B", "K", "J")}}
    evaluation_out = {}
    if evaluation is not None:
        for lab, res in (("dependence_aware", dep), ("independence", ind), ("empirical", emp)):
            if res is not None and res.policy is not None:
                e = evaluate_policy(res.policy, evaluation, req.costs, req.omega, req.loss_miss, req.loss_false_block); e.pop("decisions"); evaluation_out[lab] = e
        if dep.policy is not None and ind.policy is not None and n_boot > 0:
            evaluation_out["paired_bootstrap_dependence_aware_minus_independence"] = paired_bootstrap(dep.policy, ind.policy, evaluation, req.costs, req.omega, req.loss_miss, req.loss_false_block, n_boot, seed)
    req_d = {"costs": list(map(float, req.costs)), "omega": req.omega, "objective": req.objective, "alpha": req.alpha, "beta": req.beta, "budget": req.budget, "loss_miss": req.loss_miss, "loss_false_block": req.loss_false_block,
             "admissible": {k: (list(v) if isinstance(v, tuple) else v) for k, v in req.admissible.__dict__.items()}, "feas_tol": req.feas_tol, "tie_rel": req.tie_rel, "families": list(families), "criterion": criterion, "seed": seed, "solver": method}
    return DesignReport(data.summary(), req_d, selection, models, dep, ind, emp, comparisons, evaluation_out, names)
