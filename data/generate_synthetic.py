"""Synthetic data generation for the risk-sharing experiments.

The paper's theorems hold for a general finite measure space (Omega, F, P)
and an arbitrary essentially bounded random variable X in L^infty(P). The
paper does not specify a probability space or distribution for X to use in a
numerical illustration (see Example 8, p.13, and Example 10, p.22, which
leave these choices open); the generators below make that choice explicit so
every experiment uses the same aggregate-endowment convention.
"""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray


def finite_probability_space(n_omega: int) -> ArrayLike:
    """N_Omega equiprobable states approximating L^infty(P)."""
    if n_omega <= 0:
        raise ValueError("n_omega must be a positive integer")
    return np.full(n_omega, 1.0 / n_omega)


def uniform_aggregate_endowment(n_omega: int, seed: int, low: float = 0.0,
                                 high: float = 1.0) -> ArrayLike:
    """Aggregate endowment X ~ Uniform[low, high] on n_omega equiprobable states,
    represented as a sorted vector of order statistics (an exact, deterministic
    discretization of the uniform law rather than a single random draw, so that
    the finite probability space is genuinely non-atomic in the limit
    n_omega -> infinity, as required by Theorem 5's non-atomicity hypothesis)."""
    if n_omega <= 0:
        raise ValueError("n_omega must be a positive integer")
    rng = np.random.default_rng(seed)
    u = rng.uniform(low, high, size=n_omega)
    return np.sort(u)


def truncated_normal_endowment(n_omega: int, seed: int, mean: float = 0.0, std: float = 1.0,
                                low: float = -3.0, high: float = 3.0) -> ArrayLike:
    """Aggregate endowment X following a truncated normal law on n_omega
    equiprobable states, used as an alternative distribution for the discrete
    (small-N) comparison in experiment 1."""
    if n_omega <= 0:
        raise ValueError("n_omega must be a positive integer")
    rng = np.random.default_rng(seed)
    draws = []
    while len(draws) < n_omega:
        candidate = rng.normal(mean, std, size=n_omega)
        candidate = candidate[(candidate >= low) & (candidate <= high)]
        draws.extend(candidate.tolist())
    return np.sort(np.array(draws[:n_omega]))


def bimodal_endowment(n_omega: int, seed: int) -> ArrayLike:
    """Bimodal aggregate endowment (mixture of two normals), used in experiment 4
    to test the dual representation on a non-uniform, non-unimodal law."""
    if n_omega <= 0:
        raise ValueError("n_omega must be a positive integer")
    rng = np.random.default_rng(seed)
    n1 = n_omega // 2
    n2 = n_omega - n1
    comp1 = rng.normal(-1.0, 0.4, size=n1)
    comp2 = rng.normal(1.5, 0.5, size=n2)
    return np.sort(np.concatenate([comp1, comp2]))
