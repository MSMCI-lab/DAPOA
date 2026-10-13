"""DAPOA: Dependence-Aware Policy Optimization Algorithms for AI-agent safeguard systems.

Given calibration data (binary ALLOW/BLOCK decisions of m safeguards on labeled authorized and
unauthorized candidate actions) and operating requirements (invocation costs, error limits, loss
parameters, deployment mixture), DAPOA

1. estimates class-conditional joint decision probabilities with several copula families
   (independence, Gaussian, Student-t, Clayton, Gumbel) and the empirical frequencies,
2. selects the dependence model (BIC by default, AIC or grouped cross-validation optionally),
3. solves the exact reliability-constrained design problem over sequential layered screening and
   selective verification policies (dynamic programme + binary routing programmes), and
4. reports the selected configuration together with the independence-based comparator.

Reference: Y. Li, M. Xu and P. Zhao, "Dependence-Aware Reliability Assessment and Optimal Design of AI-Agent
Safeguard Systems" (2026).
"""
from .data import CalibrationData, pattern_matrix, pattern_index
from .copulas import (PatternModel, fit_family, fit_all, select_model, product_patterns, empirical_patterns,
                      gaussian_patterns, student_t_patterns, clayton_patterns, gumbel_patterns)
from .policies import LayeredPolicy, VerificationPolicy, PolicyClass, policy_metrics, flag_to_review
from .optimize import Requirements, Admissible, DesignResult, layered_opt, routing_opt, design
from .evaluate import evaluate_policy, paired_bootstrap
from .pipeline import run_design, DesignReport
from .simulate import simulate_decisions

__version__ = "0.1.0"
__all__ = [
    "CalibrationData", "pattern_matrix", "pattern_index",
    "PatternModel", "fit_family", "fit_all", "select_model", "product_patterns", "empirical_patterns",
    "gaussian_patterns", "student_t_patterns", "clayton_patterns", "gumbel_patterns",
    "LayeredPolicy", "VerificationPolicy", "PolicyClass", "policy_metrics", "flag_to_review",
    "Requirements", "Admissible", "DesignResult", "layered_opt", "routing_opt", "design",
    "evaluate_policy", "paired_bootstrap", "run_design", "DesignReport", "simulate_decisions",
]
