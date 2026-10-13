import numpy as np
import pytest

from dapoa import pattern_matrix, product_patterns, gaussian_patterns, student_t_patterns, clayton_patterns, gumbel_patterns
from dapoa.copulas import partial_to_corr, corr_to_partial

P = np.array([0.84, 0.92, 0.90, 0.965])


def _check(q, p, tol=1e-6):
    D = pattern_matrix(len(p))
    assert abs(q.sum() - 1) < tol
    assert np.all(q >= -tol)
    assert np.max(np.abs(q @ D - p)) < tol


def test_product_identity():
    q = product_patterns(P); _check(q, P, 1e-12)
    assert q[0] == pytest.approx(np.prod(1 - P))


def test_gaussian_identity_correlation_equals_product():
    q = gaussian_patterns(P, np.eye(4)); _check(q, P)
    assert np.max(np.abs(q - product_patterns(P))) < 1e-6


def test_gaussian_margins_and_dependence():
    R = np.full((4, 4), 0.454); np.fill_diagonal(R, 1)
    q = gaussian_patterns(P, R); _check(q, P)
    assert q[0] > product_patterns(P)[0]          # positive dependence raises the joint-ALLOW probability


def test_student_t_margins():
    R = np.full((3, 3), 0.3); np.fill_diagonal(R, 1)
    q = student_t_patterns(P[:3], R, 5.0); _check(q, P[:3], 1e-5)


def test_archimedean_limits():
    assert np.max(np.abs(clayton_patterns(P, 1e-12) - product_patterns(P))) < 1e-12
    assert np.max(np.abs(gumbel_patterns(P, 1.0) - product_patterns(P))) < 1e-12
    for q in (clayton_patterns(P, 2.0), gumbel_patterns(P, 1.8)):
        _check(q, P, 1e-10)
        assert q[0] > product_patterns(P)[0]


def test_partial_correlation_roundtrip():
    R = np.array([[1, .5, .3, .1], [.5, 1, .2, .4], [.3, .2, 1, .6], [.1, .4, .6, 1.]])
    assert np.allclose(partial_to_corr(corr_to_partial(R)), R, atol=1e-10)
    rng = np.random.default_rng(0)
    for _ in range(20):
        Pm = np.zeros((4, 4)); Pm[np.triu_indices(4, 1)] = rng.uniform(-0.95, 0.95, 6)
        assert np.linalg.eigvalsh(partial_to_corr(Pm)).min() > 0
