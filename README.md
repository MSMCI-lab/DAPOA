# DAPOA: Dependence-Aware Policy Optimization Algorithms

DAPOA selects and configures multi-component AI-agent safeguard systems when the safeguards' errors
are dependent. You supply **calibration data** (binary ALLOW/BLOCK decisions of each safeguard on
labeled authorized and unauthorized candidate actions) and **operating requirements** (invocation
costs, error limits, loss parameters, deployment mixture). DAPOA then

1. estimates the class-conditional joint decision probabilities with several dependence models
   (independence, Gaussian, Student-t, Clayton and Gumbel copulas, plus the empirical frequencies),
2. **selects the dependence model automatically** (BIC by default; AIC or grouped cross-validation optionally),
3. solves the **exact reliability-constrained design problem** over sequential layered screening and
   selective verification policies (subset dynamic programme + binary routing programmes), and
4. reports the optimal configuration (which safeguards, in what order, which reviewer, and the
   pattern-to-action routing map), the independence-based comparator, and held-out evaluation if provided.

It implements the framework of

> Y. Li, M. Xu and P. Zhao (2026). *Dependence-Aware Reliability Assessment and Optimal Design of AI-Agent Safeguard Systems.*

## Install

```bash
pip install git+https://github.com/MSMCI-lab/DAPOA.git
# or, from a clone
pip install -e ".[dev]" && pytest
```

Requirements: Python ≥ 3.10, numpy, scipy ≥ 1.15 (HiGHS mixed-integer solver and multivariate normal/t
rectangle probabilities), pandas.

## Quick start (command line)

```bash
# 1. a synthetic calibration set (or use your own CSV: label column y, one 0/1 column per safeguard, 1 = BLOCK)
dapoa simulate calibration.csv --n0 500 --n1 500 --family1 gaussian --rho 0.45 --groups 120 --seed 1

# 2. design: costs per safeguard, limits, losses and mixture
dapoa design calibration.csv --group group \
    --costs D1=0.20 D2=0.35 D3=0.65 D4=2.00 \
    --objective min_total_cost --alpha 0.05 --beta 0.10 --omega 0.10 \
    --loss-miss 100 --loss-false-block 10 \
    --criterion bic --cross-check --out report.json --markdown report.md
```

The Markdown report lists the fitted families with their log-likelihoods and BIC scores per class, the
selected model, and the optimal policy with its predicted miss probability `M`, false-block probability
`B`, operating cost `K` and total cost `J`, next to the independence-based selection evaluated under the
selected dependence model. Pass `--evaluation holdout.csv` to replay the selected policies on held-out
actions with paired group-bootstrap intervals.

## Quick start (Python)

```python
import numpy as np
from dapoa import CalibrationData, Requirements, run_design

data = CalibrationData.from_csv("calibration.csv", label="y", group="group")
req = Requirements(costs=[0.20, 0.35, 0.65, 2.00], omega=0.10, objective="min_total_cost",
                   alpha=0.05, beta=0.10, loss_miss=100, loss_false_block=10)
report = run_design(data, req, criterion="bic", cross_check=True)
print(report.to_markdown())
policy = report.dependence_aware.policy           # LayeredPolicy or VerificationPolicy
print(policy.to_dict(data.components))             # deployable definition (order, or F / reviewer / routing table)
```

Deployment of a returned policy only needs the safeguard decisions it actually invokes:

```python
final, invoked = policy.decide(lambda j: my_safeguards[j](candidate))   # 0 = ALLOW, 1 = BLOCK
```

### Lower-level API

| function | purpose |
|---|---|
| `fit_family(data, y, family)` / `select_model(data, y, families, criterion)` | pattern-likelihood estimation with fixed empirical margins; BIC/AIC/CV selection |
| `gaussian_patterns`, `student_t_patterns`, `clayton_patterns`, `gumbel_patterns`, `product_patterns` | pattern probabilities `q_y(d)` for specified parameters (no data needed) |
| `layered_opt(q1, q0, req)` | Algorithm 1 (LayeredOpt): exact subset selection and invocation ordering |
| `routing_opt(q1, q0, req, F, h)` | binary routing programme for one first-stage/reviewer pair (HiGHS or enumeration) |
| `design(q1, q0, req, cross_check=True)` | Algorithm 2: global optimum over both policy classes, optional enumeration check |
| `policy_metrics`, `evaluate_policy`, `paired_bootstrap` | model-based and held-out evaluation |
| `simulate_decisions` | synthetic decisions from specified class-conditional copulas |

`Requirements.admissible` (an `Admissible` object) encodes operational restrictions: layered-only or
verification-only classes, the fixed flag-to-review cascades only, a maximum layered size, required
components, allowed reviewers or first-stage sizes, a mandatory veto on the all-BLOCK pattern, and fixed
routing decisions. The budget problem (minimize the miss probability subject to an operating budget on
`K` and the false-block limit) is selected with `objective="min_miss", budget=b`.

Runtime: the exact search itself takes seconds for `m ≤ 5` safeguards. Copula estimation dominates: the
Archimedean families are instantaneous, while the Gaussian and Student-t fits evaluate latent rectangle
probabilities by deterministic quasi-Monte Carlo (fixed generator) inside the likelihood optimization, about
one to two minutes for four nonconstant safeguards per class on a laptop. Restrict `families` to skip the
Student-t profile if speed matters.

## What the numbers mean

* `M` = probability of allowing an unauthorized action, `B` = probability of blocking an authorized one,
  `K` = expected invocation cost per candidate under the mixture `omega` of unauthorized actions,
  `J = K + omega * loss_miss * M + (1 - omega) * loss_false_block * B`.
* Coordinates that are constant in a class (a safeguard that never errs in the calibration sample) are
  excluded from dependence estimation and restored as constants; zero observed errors do not establish
  zero population error.
* Fitted-model optimality is optimality under the fitted probabilities, and fitted feasibility is not a
  statistical reliability certificate. Evaluate on held-out data before deployment.
* Ties are broken by the fixed hierarchy (J, M, B, K, canonical index) or (M, K, B, canonical index) with a
  relative tolerance of 1e-10; a solver time limit is reported as `unresolved`, never as infeasible.

## Reproducing the paper's numerical pool

`examples/numerical_pool.py` runs the heterogeneous four-safeguard pool of Section 4 (Table 1) under the
independence and equicorrelated Gaussian scenarios and prints the returned verification maps. The tests
check the published values (e.g. under independence the optimum `R1` has `J = 1.086534`).

## Citation

```bibtex
@article{xu2026dapoa,
  title   = {Dependence-Aware Reliability Assessment and Optimal Design of AI-Agent Safeguard Systems},
  author  = {Li, Yijia and Xu, Maochao and Zhao, Peng},
  year    = {2026},
  note    = {Software: DAPOA, https://github.com/MSMCI-lab/DAPOA}
}
```

## License

MIT.
