"""Reproduce the heterogeneous four-safeguard pool of the paper (Section 4): independence S0 and the
equicorrelated Gaussian / Student-t / Clayton / Gumbel scenarios at latent Kendall association 0.30."""
import numpy as np

from dapoa import Requirements, Admissible, design, product_patterns, gaussian_patterns, student_t_patterns, clayton_patterns, gumbel_patterns, policy_metrics

NAMES = ["D1", "D2", "D3", "D4"]
MISS = np.array([0.160, 0.080, 0.100, 0.035]); FB = np.array([0.030, 0.120, 0.050, 0.025]); COST = [0.20, 0.35, 0.65, 2.00]
P1, P0 = 1 - MISS, FB
REQ = Requirements(COST, omega=0.10, objective="min_total_cost", alpha=0.05, beta=0.10, loss_miss=100, loss_false_block=10)
tau = 0.30; rho = np.sin(np.pi * tau / 2); R = np.full((4, 4), rho); np.fill_diagonal(R, 1)
scenarios = {
    "S0 independence": (product_patterns(P1), product_patterns(P0)),
    "S1 Gaussian": (gaussian_patterns(P1, R), gaussian_patterns(P0, R)),
    "S4 Student-t(5)": (student_t_patterns(P1, R, 5), student_t_patterns(P0, R, 5)),
    "S5 Clayton": (clayton_patterns(P1, 2 * tau / (1 - tau)), clayton_patterns(P0, 2 * tau / (1 - tau))),
    "S6 Gumbel": (gumbel_patterns(P1, 1 / (1 - tau)), gumbel_patterns(P0, 1 / (1 - tau))),
}
ind = design(*scenarios["S0 independence"], REQ).policy
print(f"{'scenario':18s} {'baseline (BASE)':28s} {'J_base':>7s}  {'selected (FULL)':34s} {'M':>7s} {'B':>7s} {'K':>7s} {'J':>7s}  independence pick M")
for name, (q1, q0) in scenarios.items():
    full = design(q1, q0, REQ); base = design(q1, q0, Requirements(COST, 0.10, "min_total_cost", 0.05, 0.10, None, 100, 10, Admissible(flag_to_review_only=True)))
    m = full.metrics; mi = policy_metrics(ind, q1, q0, COST, 0.10, 100, 10)
    print(f"{name:18s} {base.describe(NAMES):28s} {base.metrics['J']:7.4f}  {full.describe(NAMES):34s} {m['M']:7.4f} {m['B']:7.4f} {m['K']:7.4f} {m['J']:7.4f}  {mi['M']:.4f}{' (> alpha)' if mi['M'] > 0.05 else ''}")
print("\nDeployable definition of the S0 optimum (R1):", ind.to_dict(NAMES))
