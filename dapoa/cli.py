"""Command-line interface: ``dapoa design`` and ``dapoa simulate``."""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .copulas import FAMILIES
from .data import CalibrationData
from .optimize import Admissible, Requirements
from .pipeline import run_design


def _parse_costs(items, components):
    if len(items) == len(components) and all("=" not in s for s in items):
        return [float(s) for s in items]
    d = {}
    for s in items:
        k, v = s.split("="); d[k] = float(v)
    missing = [c for c in components if c not in d]
    if missing:
        raise SystemExit(f"missing costs for {missing}")
    return [d[c] for c in components]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dapoa", description="Dependence-Aware Policy Optimization Algorithms for AI-agent safeguard systems")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("design", help="fit dependence models, select one, and return the optimal safeguard configuration")
    d.add_argument("data", help="CSV with a label column (1 = unauthorized) and one 0/1 column per safeguard (1 = BLOCK)")
    d.add_argument("--components", nargs="+", help="safeguard columns (default: every column except label/group/id)")
    d.add_argument("--label", default="y"); d.add_argument("--group", default=None, help="group column (users/sessions) for grouped resampling and CV"); d.add_argument("--id", default=None)
    d.add_argument("--costs", nargs="+", required=True, help="invocation costs, positional or NAME=VALUE")
    d.add_argument("--objective", choices=["min_total_cost", "min_miss"], default="min_total_cost")
    d.add_argument("--alpha", type=float, default=0.05); d.add_argument("--beta", type=float, default=0.10); d.add_argument("--budget", type=float, default=None)
    d.add_argument("--omega", type=float, default=0.5); d.add_argument("--loss-miss", type=float, default=100.0); d.add_argument("--loss-false-block", type=float, default=10.0)
    d.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES)); d.add_argument("--criterion", choices=["bic", "aic", "cv"], default="bic")
    d.add_argument("--no-verification", action="store_true", help="layered screening only"); d.add_argument("--flag-to-review-only", action="store_true", help="restrict verification to fixed flag-to-review cascades")
    d.add_argument("--max-layered-size", type=int, default=None); d.add_argument("--allowed-reviewers", nargs="*", default=None); d.add_argument("--required", nargs="*", default=None)
    d.add_argument("--method", choices=["milp", "enumerate"], default="milp"); d.add_argument("--cross-check", action="store_true", help="verify by exhaustive enumeration (m <= 4)")
    d.add_argument("--evaluation", default=None, help="optional held-out CSV with the same columns"); d.add_argument("--n-boot", type=int, default=500); d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", default=None, help="write the JSON report here"); d.add_argument("--markdown", default=None, help="write the Markdown report here"); d.add_argument("--quiet", action="store_true")
    s = sub.add_parser("simulate", help="write a synthetic calibration CSV from a specified copula model")
    s.add_argument("out"); s.add_argument("--n0", type=int, default=500); s.add_argument("--n1", type=int, default=500); s.add_argument("--p0", nargs="+", type=float, default=[0.03, 0.12, 0.05, 0.025])
    s.add_argument("--p1", nargs="+", type=float, default=[0.84, 0.92, 0.90, 0.965]); s.add_argument("--family1", default="gaussian", choices=list(FAMILIES)); s.add_argument("--family0", default="gaussian", choices=list(FAMILIES))
    s.add_argument("--rho", type=float, default=0.45, help="equicorrelation (Gaussian / Student-t)"); s.add_argument("--nu", type=float, default=5.0); s.add_argument("--theta", type=float, default=1.5); s.add_argument("--groups", type=int, default=None); s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "simulate":
        from .simulate import simulate_decisions
        import pandas as pd
        m = len(a.p0); R = np.full((m, m), a.rho); np.fill_diagonal(R, 1.0)
        par = {"gaussian": {"correlation": R}, "student_t": {"correlation": R, "nu": a.nu}, "clayton": {"theta": a.theta}, "gumbel": {"theta": a.theta}, "independence": {}}
        data = simulate_decisions(a.n0, a.n1, a.p0, a.p1, a.family0, a.family1, par[a.family0], par[a.family1], n_groups=a.groups, seed=a.seed)
        df = pd.DataFrame(data.decisions, columns=data.components); df.insert(0, "y", data.y)
        if data.groups is not None:
            df.insert(1, "group", data.groups)
        df.to_csv(a.out, index=False); print(f"wrote {a.out}: {len(df)} actions, components {data.components}"); return 0
    data = CalibrationData.from_csv(a.data, a.components, a.label, a.group, a.id); names = data.components
    costs = _parse_costs(a.costs, names)
    adm = Admissible(allow_verification=not a.no_verification, flag_to_review_only=a.flag_to_review_only, max_layered_size=a.max_layered_size,
                     required_components=tuple(names.index(c) for c in (a.required or [])), allowed_reviewers=(tuple(names.index(c) for c in a.allowed_reviewers) if a.allowed_reviewers else None))
    req = Requirements(costs, a.omega, a.objective, a.alpha, a.beta, a.budget, a.loss_miss, a.loss_false_block, adm)
    ev = CalibrationData.from_csv(a.evaluation, names, a.label, a.group, a.id) if a.evaluation else None
    rep = run_design(data, req, a.families, a.criterion, a.seed, a.method, a.cross_check, ev, a.n_boot)
    if a.out:
        rep.to_json(a.out)
    md = rep.to_markdown()
    if a.markdown:
        with open(a.markdown, "w") as f:
            f.write(md)
    if not a.quiet:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
