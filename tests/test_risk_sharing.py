"""Unit tests for src/risk_sharing.py: Theorems 1-5 on small synthetic fixtures."""

import numpy as np
import pytest

from src.risk_measures import entropic_risk_measure, expected_shortfall, uniform_probability
from src.risk_sharing import (
    atom_optimal_allocation,
    discretize_agents,
    entropic_dual_check,
    entropic_risk_sharing,
    es_dual_check,
    es_risk_sharing,
    pareto_improve,
    weighted_total_risk,
)
from data.generate_synthetic import finite_probability_space, uniform_aggregate_endowment


@pytest.fixture
def small_state_space():
    n_omega = 200
    X = uniform_aggregate_endowment(n_omega, seed=0)
    P = finite_probability_space(n_omega)
    return X, P


def test_discretize_agents_midpoint_rule():
    a = discretize_agents(4, 0.0, 1.0)
    assert np.allclose(a, [0.125, 0.375, 0.625, 0.875])


def test_discretize_agents_rejects_nonpositive_count():
    with pytest.raises(ValueError):
        discretize_agents(0)


def test_entropic_risk_sharing_matches_theoretical_value(small_state_space):
    """Theorem 2 / Example 8 (p.11-13): brute-force value equals Ent^Gamma(X)."""
    X, P = small_state_space
    gammas = 1.0 + discretize_agents(20, 0.0, 1.0)  # gamma_a = 1 + a
    result = entropic_risk_sharing(X, gammas, P)
    assert result.solver_status == "optimal"
    assert np.isclose(result.Gamma, np.mean(gammas))
    assert np.isclose(result.brute_force_value, result.theoretical_value, rtol=1e-4)
    assert result.constraint_violation < 1e-6


def test_entropic_optimal_allocation_achieves_theoretical_value(small_state_space):
    """The formula X_a = gamma_a X / Gamma is feasible and achieves Ent^Gamma(X)
    (Theorem 2, Example 8, p.13)."""
    X, P = small_state_space
    gammas = 1.0 + discretize_agents(20, 0.0, 1.0)
    result = entropic_risk_sharing(X, gammas, P)
    Gamma = result.Gamma
    alloc = result.theoretical_allocation
    assert np.allclose(np.sum(alloc, axis=0) / alloc.shape[0], X, atol=1e-10)
    total_risk = np.mean([
        entropic_risk_measure(alloc[a], gammas[a], P) for a in range(gammas.shape[0])
    ])
    assert np.isclose(total_risk, entropic_risk_measure(X, Gamma, P), rtol=1e-6)


def test_entropic_risk_sharing_rejects_nonpositive_gamma():
    X = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        entropic_risk_sharing(X, np.array([1.0, -0.5]))


def test_es_risk_sharing_continuum_gap_is_strictly_positive(small_state_space):
    """Theorem 4(2) (p.18-19): when mu({a: beta_a = ess sup beta_a}) = 0 (a
    continuum with no atom at the essential supremum), the brute-force optimum
    strictly exceeds the theoretical value ES^{beta_star}(X)."""
    X, P = small_state_space
    betas = discretize_agents(30, 0.1, 0.5)  # continuum, no atom at 0.5
    result = es_risk_sharing(X, betas, P, beta_sup=0.5)
    assert np.isclose(result.atom_mass, 0.0, atol=1e-9)
    assert result.gap_to_theoretical > 0


def test_es_risk_sharing_gap_shrinks_with_finer_discretization(small_state_space):
    """As M grows, agents cluster closer to the true continuum supremum
    beta_sup = 0.5, so the gap to ES^{0.5}(X) shrinks (but by Theorem 4(2)
    never reaches zero, since no discrete agent ever equals 0.5 exactly)."""
    X, P = small_state_space
    gaps = []
    for m in (10, 40, 160):
        betas = discretize_agents(m, 0.1, 0.5)
        result = es_risk_sharing(X, betas, P, beta_sup=0.5)
        gaps.append(result.gap_to_theoretical)
    assert gaps[0] > gaps[1] > gaps[2] > 0


def test_es_risk_sharing_atom_achieves_theoretical_value_exactly(small_state_space):
    """Theorem 4(1) (p.17-18): with an atom of mu-mass > 0 at beta_star, giving
    all risk to that atom achieves ES^{beta_star}(X) exactly."""
    X, P = small_state_space
    m = 19
    cont_betas = discretize_agents(m, 0.1, 0.5)
    betas = np.append(cont_betas, 0.5)  # an explicit atom at beta_star = 0.5
    mu = np.append(np.full(m, 0.7 / m), 0.3)
    result = es_risk_sharing(X, betas, P, mu_weights=mu)
    assert result.atom_mass > 0
    assert abs(result.gap_to_theoretical) < 1e-6

    alloc = atom_optimal_allocation(X, betas, mu, result.beta_star)
    total_risk = weighted_total_risk(alloc, betas, mu, P)
    assert np.isclose(total_risk, result.theoretical_value, atol=1e-6)
    assert np.allclose(mu @ alloc, X, atol=1e-10)  # feasibility of A(X)


def test_es_risk_sharing_rejects_invalid_betas():
    X = np.array([0.1, 0.2])
    with pytest.raises(ValueError):
        es_risk_sharing(X, np.array([0.0, 1.2]))


def test_atom_optimal_allocation_requires_an_atom():
    betas = np.array([0.2, 0.3, 0.4])
    mu = np.full(3, 1 / 3)
    with pytest.raises(ValueError):
        atom_optimal_allocation(np.ones(5), betas, mu, beta_star=0.5)


def test_pareto_improve_strictly_reduces_total_risk(small_state_space):
    """Theorem 4(2) / Theorem 5 (p.18-22): any candidate allocation in a
    continuum without an atom at beta_star admits a Pareto (hence aggregate
    risk) improvement."""
    X, P = small_state_space
    m = 40
    betas = discretize_agents(m, 0.1, 0.5)
    mu = np.full(m, 1.0 / m)
    equal_split = np.tile(X / m, (m, 1))  # naive equal-split allocation
    new_alloc, risk_before, risk_after = pareto_improve(
        equal_split, betas, mu, top_quantile_fraction=0.1, shift_fraction=0.3, P=P,
    )
    assert risk_after < risk_before
    assert np.allclose(mu @ new_alloc, mu @ equal_split, atol=1e-8)


def test_entropic_dual_check_conjugates_match(small_state_space):
    """Theorem 1 (p.9): the conjugate of the value function equals the integral
    of individual conjugates, for entropic agents."""
    X, P = small_state_space
    rng = np.random.default_rng(1)
    gammas = 1.0 + 2.0 * discretize_agents(50, 0.0, 1.0)
    test_measures = [rng.dirichlet(np.full(X.shape[0], 1.0)) for _ in range(5)]
    result = entropic_dual_check(X, gammas, test_measures, P)
    assert result.max_abs_diff < 1e-8
    assert abs(result.sup_representation_argmax_gap) < 1e-6


def test_es_dual_check_conjugates_agree(small_state_space):
    """Theorem 1 for ES agents: the conjugate indicator of the value function
    agrees with that of the integral of individual conjugates."""
    X, P = small_state_space
    rng = np.random.default_rng(2)
    alphas = 0.1 + 0.3 * discretize_agents(30, 0.0, 1.0)
    test_measures = [rng.dirichlet(np.full(X.shape[0], 1.0)) for _ in range(5)]
    result = es_dual_check(alphas, test_measures, P)
    assert result["all_agree"]


def test_weighted_total_risk_with_single_agent_matches_expected_shortfall():
    X = np.array([0.1, 0.5, 0.9])
    P = uniform_probability(3)
    betas = np.array([0.3])
    mu = np.array([1.0])
    alloc = X.reshape(1, -1)
    assert np.isclose(weighted_total_risk(alloc, betas, mu, P), expected_shortfall(X, 0.3, P))
