"""Tests for experiment 4 (Theorem 1: dual representation of the integral
infimal convolution, and Proposition 3's first-order optimality condition).

Uses a small probability space and few agents/test measures so the tests run
quickly while exercising the exact functions the experiment script calls.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.generate_synthetic import bimodal_endowment
from exp import exp4
from src.risk_measures import dirichlet_test_measure, uniform_probability


@pytest.fixture
def small_setup():
    P = uniform_probability(150)
    X = bimodal_endowment(150, seed=11)
    rng = np.random.default_rng(11)
    test_measures = [dirichlet_test_measure(150, 1.0, rng) for _ in range(8)]
    return X, P, test_measures


def test_entropic_conjugate_integral_matches_value_function_conjugate(small_setup):
    """Theorem 1: (box Ent^{gamma_a} mu(da))^*(Q) = int_A gamma_a H(Q||P) mu(da)
    must hold for every tested Q, not only on average."""
    X, P, test_measures = small_setup
    result = exp4.run_entropic_case(X, P, test_measures)
    assert result["max_abs_diff_conjugates"] < 1e-8
    assert result["gap_sup_to_value_function"] < 1e-6


def test_dual_sup_equals_value_function_at_esscher_tilt(small_setup):
    X, P, test_measures = small_setup
    result = exp4.run_entropic_case(X, P, test_measures)
    assert abs(result["dual_sup_value_at_esscher_tilt"] - result["value_function_Ent_Gamma"]) < 1e-6


def test_first_order_optimality_common_subgradient(small_setup):
    """Proposition 3: all agents' optimal shares admit the same subgradient Q*;
    numerically this is the common Esscher tilt and the Fenchel-Young equality
    rho_a(X_a) = E^Q*[X_a] - rho_a^*(Q*) holds for every sampled agent."""
    X, P, _ = small_setup
    result = exp4.run_first_order_optimality(X, P)
    assert result["max_deviation_agent_subgradient_from_common_Q_star"] < 1e-8
    assert result["max_fenchel_young_gap"] < 1e-6


def test_es_conjugate_integral_agrees_with_value_function(small_setup):
    """Theorem 1 for expected shortfall: the conjugate-zero region of the
    integral of individual conjugates must coincide with that of ES^{alpha_star},
    alpha_star = ess sup alpha_a, for every tested Q."""
    X, P, test_measures = small_setup
    result = exp4.run_es_case(P, test_measures)
    assert result["all_agree"] is True


def test_measurability_of_agent_map_is_affine(small_setup):
    X, P, test_measures = small_setup
    result = exp4.run_measurability_check(P, test_measures)
    assert result["is_affine_in_a"] is True
