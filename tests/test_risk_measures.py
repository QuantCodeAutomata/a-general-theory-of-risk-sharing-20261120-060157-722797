"""Unit tests for src/risk_measures.py on small synthetic fixtures."""

import numpy as np
import pytest

from src.risk_measures import (
    dirichlet_test_measure,
    entropic_conjugate,
    entropic_risk_measure,
    es_conjugate_is_zero,
    es_rockafellar_uryasev,
    esscher_tilted_measure,
    expected_shortfall,
    phi_inflation,
    relative_entropy,
    uniform_probability,
)


def test_uniform_probability_sums_to_one():
    P = uniform_probability(7)
    assert P.shape == (7,)
    assert np.isclose(P.sum(), 1.0)
    assert np.allclose(P, 1.0 / 7)


def test_relative_entropy_is_zero_for_identical_measures():
    P = uniform_probability(10)
    assert np.isclose(relative_entropy(P, P), 0.0, atol=1e-12)


def test_relative_entropy_is_nonnegative():
    rng = np.random.default_rng(1)
    P = uniform_probability(20)
    for _ in range(10):
        Q = rng.dirichlet(np.full(20, 2.0))
        assert relative_entropy(Q, P) >= -1e-10


def test_relative_entropy_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        relative_entropy(np.ones(3) / 3, np.ones(4) / 4)


def test_entropic_risk_measure_cash_additive():
    """Ent^gamma(X + c) = Ent^gamma(X) + c (paper, Definition 4, Axiom II, p.5)."""
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, 50)
    P = uniform_probability(50)
    gamma = 2.5
    c = 0.37
    assert np.isclose(entropic_risk_measure(X + c, gamma, P),
                       entropic_risk_measure(X, gamma, P) + c, atol=1e-10)


def test_entropic_risk_measure_monotone():
    """Ent^gamma is monotone: X >= Y pointwise implies Ent^gamma(X) >= Ent^gamma(Y)."""
    rng = np.random.default_rng(3)
    Y = rng.uniform(-1, 1, 40)
    X = Y + rng.uniform(0, 1, 40)  # X >= Y pointwise
    P = uniform_probability(40)
    assert entropic_risk_measure(X, 1.5, P) >= entropic_risk_measure(Y, 1.5, P)


def test_entropic_risk_measure_constant():
    """Ent^gamma(c) = c for a constant c (zero variance under exp/log)."""
    P = uniform_probability(30)
    X = np.full(30, 0.75)
    assert np.isclose(entropic_risk_measure(X, 3.0, P), 0.75, atol=1e-10)


def test_entropic_risk_measure_rejects_nonpositive_gamma():
    with pytest.raises(ValueError):
        entropic_risk_measure(np.ones(5), 0.0)
    with pytest.raises(ValueError):
        entropic_risk_measure(np.ones(5), -1.0)


def test_esscher_tilt_is_a_probability_measure():
    rng = np.random.default_rng(4)
    X = rng.uniform(-1, 1, 60)
    P = uniform_probability(60)
    Q = esscher_tilted_measure(X, 2.0, P)
    assert np.isclose(Q.sum(), 1.0, atol=1e-10)
    assert np.all(Q >= 0)


def test_entropic_conjugate_matches_gamma_times_entropy():
    rng = np.random.default_rng(5)
    P = uniform_probability(25)
    Q = rng.dirichlet(np.full(25, 1.0))
    gamma = 1.8
    assert np.isclose(entropic_conjugate(gamma, Q, P), gamma * relative_entropy(Q, P))


def test_expected_shortfall_matches_rockafellar_uryasev():
    """ES^alpha matches the Rockafellar-Uryasev representation (Proposition 4,
    Eq. (10), p.15-16) for several alpha levels."""
    rng = np.random.default_rng(6)
    X = rng.uniform(-2, 2, 300)
    P = uniform_probability(300)
    for alpha in (0.05, 0.1, 0.3, 0.5, 0.9, 1.0):
        dual_value = expected_shortfall(X, alpha, P)
        ru_value, _ = es_rockafellar_uryasev(X, alpha, P)
        assert np.isclose(dual_value, ru_value, atol=1e-6), f"mismatch at alpha={alpha}"


def test_expected_shortfall_alpha_one_equals_expectation():
    """ES^1(X) = E^P(X) (no inflation at alpha=1, Example 9)."""
    rng = np.random.default_rng(7)
    X = rng.uniform(-1, 1, 100)
    P = uniform_probability(100)
    assert np.isclose(expected_shortfall(X, 1.0, P), np.sum(P * X), atol=1e-10)


def test_expected_shortfall_monotone_decreasing_in_alpha():
    """Smaller alpha is more conservative: ES^alpha1(X) >= ES^alpha2(X) for alpha1 < alpha2."""
    rng = np.random.default_rng(8)
    X = rng.uniform(-1, 1, 200)
    P = uniform_probability(200)
    alphas = [0.05, 0.1, 0.2, 0.5, 0.9, 1.0]
    values = [expected_shortfall(X, a, P) for a in alphas]
    assert all(values[i] >= values[i + 1] - 1e-9 for i in range(len(values) - 1))


def test_expected_shortfall_rejects_invalid_alpha():
    X = np.ones(5)
    with pytest.raises(ValueError):
        expected_shortfall(X, 0.0)
    with pytest.raises(ValueError):
        expected_shortfall(X, 1.5)


def test_expected_shortfall_single_state():
    """Edge case: a single state reduces ES to that state's value for any alpha."""
    X = np.array([3.14])
    P = np.array([1.0])
    assert np.isclose(expected_shortfall(X, 0.37, P), 3.14)


def test_es_conjugate_is_zero_boundary():
    """ES^alpha's conjugate is 0 at Q=P (since dP/dP=1<=1/alpha for alpha<=1)."""
    P = uniform_probability(50)
    assert es_conjugate_is_zero(P, 0.3, P)
    assert es_conjugate_is_zero(P, 1.0, P)


def test_es_conjugate_is_infinite_outside_constraint_set():
    """A measure concentrated on one state violates dQ/dP <= 1/alpha for small alpha."""
    n = 50
    Q = np.zeros(n)
    Q[0] = 1.0
    P = uniform_probability(n)
    assert not es_conjugate_is_zero(Q, 0.5, P)  # dQ/dP = 50 >> 1/0.5 = 2


def test_phi_inflation_matches_expected_shortfall():
    rng = np.random.default_rng(9)
    X = rng.uniform(0, 1, 100)
    P = uniform_probability(100)
    gamma = 4.0
    assert np.isclose(phi_inflation(X, gamma, P), expected_shortfall(X, 1.0 / gamma, P))


def test_phi_inflation_rejects_gamma_below_one():
    with pytest.raises(ValueError):
        phi_inflation(np.ones(5), 0.5)


def test_phi_inflation_is_nondecreasing_and_bounded_by_essential_sup():
    """Proposition 5 (p.16-17): phi(gamma) = ES^{1/gamma}(X) is non-decreasing in
    gamma and converges to P-ess sup X as gamma -> infinity."""
    rng = np.random.default_rng(10)
    X = rng.uniform(0, 1, 500)
    P = uniform_probability(500)
    gammas = np.linspace(1.0, 100.0, 20)
    values = [phi_inflation(X, g, P) for g in gammas]
    assert all(values[i] <= values[i + 1] + 1e-9 for i in range(len(values) - 1))
    assert values[-1] <= np.max(X) + 1e-9
    assert values[-1] > values[0]  # genuinely increasing over the tested range


def test_dirichlet_test_measure_is_a_probability_vector():
    rng = np.random.default_rng(11)
    Q = dirichlet_test_measure(30, 1.0, rng)
    assert np.isclose(Q.sum(), 1.0)
    assert np.all(Q >= 0)
