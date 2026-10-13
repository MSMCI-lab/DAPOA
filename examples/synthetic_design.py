"""End-to-end example on simulated calibration and evaluation data: fit, select, design, evaluate."""
import numpy as np

from dapoa import Requirements, run_design, simulate_decisions

R1 = np.array([[1, .35, .85, .3], [.35, 1, .45, .2], [.85, .45, 1, .4], [.3, .2, .4, 1]])     # unauthorized-class latent correlation
R0 = np.full((4, 4), 0.2); np.fill_diagonal(R0, 1)
p0 = [0.03, 0.12, 0.05, 0.025]; p1 = [0.84, 0.92, 0.90, 0.965]
cal = simulate_decisions(600, 600, p0, p1, "gaussian", "gaussian", {"correlation": R0}, {"correlation": R1}, components=["Q", "G", "Mi", "H"], n_groups=150, seed=11)
ev = simulate_decisions(500, 500, p0, p1, "gaussian", "gaussian", {"correlation": R0}, {"correlation": R1}, components=["Q", "G", "Mi", "H"], n_groups=150, seed=12)
req = Requirements(costs=[0.10, 0.104, 0.102, 1.284], omega=0.5, objective="min_total_cost", alpha=0.05, beta=0.10, loss_miss=100, loss_false_block=10)
report = run_design(cal, req, criterion="bic", evaluation=ev, n_boot=200, cross_check=True, seed=0)
print(report.to_markdown())
report.to_json("synthetic_design_report.json")
