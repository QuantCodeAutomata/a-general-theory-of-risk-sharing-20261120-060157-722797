"""Tests for experiment 3 (Theorem 5: non-existence of Pareto efficient
allocations with expected-shortfall preferences and a continuum of agents).

Uses a small probability space and few agents so the tests run quickly while
still exercising the exact code path the experiment script calls.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.generate_synthetic import uniform_aggregate_endowment
from exp import exp3
from src.risk_measures import uniform_probability


@pytest.fixture
def small_state_space():
    P = uniform_probability(200)
    X = uniform_aggregate_endowment(200, seed=7)
    return X, P


def test_precondition_holds_for_uniform_endowment(small_state_space):
    X, P = small_state_space
    result = exp3.run_precondition_check(X, P)
    assert result["precondition_holds"] is True


def test_continuum_gap_is_positive_and_small(small_state_space):
    """max(betas) < beta_sup for any finite midpoint grid, so the finite-agent
    optimum strictly exceeds the continuum value -- the quantitative signature
    of Theorem 5's non-existence result."""
    X, P = small_state_space
    result = exp3.run_continuum_nonexistence_case(X, P)
    assert result["max_beta_in_sample"] < result["beta_sup"]
    assert result["atom_mass_at_sup"] == 0.0
    assert 0.0 < result["gap_to_theoretical"] < 0.05


def test_pareto_improvement_is_monotone_nonincreasing(small_state_space):
    X, P = small_state_space
    result = exp3.run_pareto_improvement_demo(X, P)
    assert result["is_monotone_nonincreasing"] is True
    # Improving steps must not increase risk above the starting allocation.
    assert result["risk_trace"][-1] <= result["risk_trace"][0]


def test_finite_agent_optimum_always_attained_exactly(small_state_space):
    """Theorem 4 part 1: a finite agent space always has an atom of positive
    mass at its maximal beta, so the finite-agent optimum matches the
    continuum value exactly (gap identically 0) and the attaining atom's mass
    shrinks as N grows -- the mechanism behind the continuum's non-existence."""
    X, P = small_state_space
    result = exp3.run_finite_agent_convergence(X, P)
    gaps = [row["gap_to_continuum_value"] for row in result["rows"]]
    masses = [row["optimizing_atom_mass"] for row in result["rows"]]
    assert all(abs(g) < 1e-6 for g in gaps)
    assert result["atom_mass_monotone_decreasing"] is True
    assert masses[0] > masses[-1] > 0
