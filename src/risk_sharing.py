"""Integral infimal convolution of risk measures and the risk-sharing problem.

Implements Definition 6 (the integral infimal convolution, p.7), Theorem 2 /
Example 8 (dilated / entropic risk measures, p.11-13), Theorem 3 / Theorem 4
(inflated / expected-shortfall risk measures, p.17-19), Theorem 1 (the dual
representation of the value function, p.9), Proposition 3 (the first-order
optimality condition via a common subgradient, p.9-10), and Proposition 6 /
Theorem 5 (Pareto efficiency, p.20-22).

The brute-force value of the integral infimal convolution

    inf_{(X_a)} (1/M) * sum_a rho_a(X_a)   subject to   sum_a X_a = X

is a convex program (each rho_a is convex). It is solved with ``cvxpy``
(Context7: the ``cvxpy`` convex-optimization DSL; for the expected-shortfall
case the per-agent objective uses cvxpy's built-in ``cp.cvar(x, beta)`` atom
with ``beta = 1 - alpha``, which matches the paper's ES^alpha of Example 9
exactly: the average of the worst alpha-fraction of outcomes).

A finite set of M agents with uniform weight 1/M approximates both the
continuum of agents (Riemann sum of the integral, M large) and the classical
finite-agent infimal convolution (M = N, no approximation involved) within a
single implementation, consistent with Definition 6's reduction to the
classical case when A = {1, ..., N} with counting measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from src.risk_measures import (
    entropic_conjugate,
    entropic_risk_measure,
    es_conjugate_is_zero,
    esscher_tilted_measure,
    expected_shortfall,
    relative_entropy,
    uniform_probability,
)

ArrayLike = np.ndarray


def discretize_agents(m_agents: int, a_lo: float = 0.0, a_hi: float = 1.0) -> ArrayLike:
    """Midpoint discretization of the continuum A = [a_lo, a_hi] into M agents,
    approximating the Lebesgue measure mu by uniform weight (a_hi - a_lo) / M
    per agent (Riemann-sum approximation of the integrals in Definition 6)."""
    if m_agents <= 0:
        raise ValueError("m_agents must be a positive integer")
    j = np.arange(m_agents)
    return a_lo + (j + 0.5) / m_agents * (a_hi - a_lo)


@dataclass
class EntropicSharingResult:
    """Result of verifying Theorem 2 / Example 8 for entropic risk measures."""

    gammas: ArrayLike
    Gamma: float
    theoretical_value: float
    theoretical_allocation: ArrayLike  # shape (M, N_Omega)
    brute_force_value: float
    brute_force_allocation: ArrayLike  # shape (M, N_Omega)
    solver_status: str
    max_allocation_abs_diff: float
    constraint_violation: float


def entropic_risk_sharing(X: ArrayLike, gammas: ArrayLike, P: ArrayLike | None = None,
                           mu_weights: ArrayLike | None = None,
                           solver: str = cp.CLARABEL) -> EntropicSharingResult:
    """Verify Theorem 2 / Example 8: integral infimal convolution of
    (Ent^{gamma_a})_a equals Ent^Gamma with Gamma = int gamma_a mu(da), and the
    allocation (gamma_a X / Gamma)_a is optimal.

    Both the objective and the feasibility constraint of Definition 6 (p.7) use
    the same measure mu: the objective is int_A Ent^{gamma_a}(X_a) mu(da) and
    the feasible set A(X) requires int_A X_a mu(da) = X. ``mu_weights`` (summing
    to 1, i.e. A normalized to total mass 1) discretizes mu by the midpoint rule
    of ``discretize_agents``; both the objective and the constraint of the
    cvxpy program below use it identically, so the Riemann-sum approximation is
    consistent between the two.
    """
    X = np.asarray(X, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    if np.any(gammas <= 0):
        raise ValueError("all gammas must be strictly positive")
    n_omega = X.shape[0]
    m_agents = gammas.shape[0]
    P = uniform_probability(n_omega) if P is None else np.asarray(P, dtype=float)
    mu_weights = (np.full(m_agents, 1.0 / m_agents) if mu_weights is None
                  else np.asarray(mu_weights, dtype=float))
    if not np.isclose(mu_weights.sum(), 1.0, atol=1e-8):
        raise ValueError("mu_weights must sum to 1 (normalized agent measure)")

    Gamma = float(np.sum(mu_weights * gammas))  # Riemann sum approximating Gamma = int gamma_a mu(da)
    theoretical_value = entropic_risk_measure(X, Gamma, P)
    theoretical_allocation = np.outer(gammas / Gamma, X)

    X_alloc = cp.Variable((m_agents, n_omega))
    # Ent^gamma_a(Y) = gamma_a * log(E^P[exp(Y/gamma_a)]) = gamma_a * log_sum_exp(log(P) + Y/gamma_a)
    log_p = np.log(P)
    per_agent_risk = [
        mu_weights[a] * gammas[a] * cp.log_sum_exp(log_p + X_alloc[a, :] / gammas[a])
        for a in range(m_agents)
    ]
    objective = cp.Minimize(cp.sum(per_agent_risk))
    constraints = [mu_weights @ X_alloc == X]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)

    brute_allocation = X_alloc.value
    constraint_violation = float(np.max(np.abs(mu_weights @ brute_allocation - X)))
    max_diff = float(np.max(np.abs(brute_allocation - theoretical_allocation)))

    return EntropicSharingResult(
        gammas=gammas, Gamma=Gamma, theoretical_value=theoretical_value,
        theoretical_allocation=theoretical_allocation,
        brute_force_value=float(problem.value), brute_force_allocation=brute_allocation,
        solver_status=problem.status, max_allocation_abs_diff=max_diff,
        constraint_violation=constraint_violation,
    )


@dataclass
class ESharingResult:
    """Result of verifying Theorem 3 / Theorem 4 for inflated (ES) risk measures."""

    betas: ArrayLike
    mu_weights: ArrayLike
    beta_star: float  # mu-essential supremum of betas (paper's beta in Theorem 5, p.21-22)
    atom_mass: float  # mu({a: beta_a = beta_star})
    theoretical_value: float  # ES^{beta_star}(X)
    brute_force_value: float
    brute_force_allocation: ArrayLike
    solver_status: str
    gap_to_theoretical: float


def es_finite_agent_optimum(X: ArrayLike, betas: ArrayLike,
                             mu_weights: ArrayLike) -> tuple[float, ArrayLike]:
    """Closed-form solution of the FINITE-agent risk-sharing problem
    min_{(X_a)} sum_a mu_a ES^{beta_a}(X_a)  s.t.  sum_a mu_a X_a = X.

    For any finite collection {1,...,M} with counting/weighted measure mu,
    Theorem 3 (p.17) applies literally to that finite agent space: the
    essential infimum of gamma_a = 1/beta_a is attained (any finite set has
    an attained max of betas), so the value equals ES^{max(betas)}(X) and
    Theorem 4(1) (p.18) gives the optimizer: concentrate all of X on the
    agent(s) a* attaining max(betas), i.e. X_{a*} = X / mu({a*}) and X_a = 0
    otherwise. By positive homogeneity of ES, mu({a*}) * ES^{beta_max}(X /
    mu({a*})) = ES^{beta_max}(X), so this allocation is feasible and attains
    the theorem's value exactly -- it is not an approximation of the
    brute-force optimum, it IS the brute-force optimum for any finite M.
    This closed form is used instead of a large convex solve so that the
    convergence sweep over M in Theorem 4(2) (p.18-19) remains tractable;
    it is cross-checked against an explicit ``cvxpy`` solve in
    ``es_risk_sharing`` for small M below.
    """
    beta_max = float(np.max(betas))
    is_argmax = np.isclose(betas, beta_max)
    argmax_mass = float(mu_weights[is_argmax].sum())
    allocation = np.zeros((betas.shape[0], X.shape[0]))
    allocation[is_argmax, :] = X / argmax_mass
    value = expected_shortfall(X, beta_max, uniform_probability(X.shape[0]))
    return value, allocation


def es_risk_sharing(X: ArrayLike, betas: ArrayLike, P: ArrayLike | None = None,
                     mu_weights: ArrayLike | None = None, beta_sup: float | None = None,
                     atom_tol: float = 1e-9, solver: str = cp.CLARABEL,
                     max_lp_agents: int = 80) -> ESharingResult:
    """Verify Theorem 3 / Theorem 4 for ES_a = ES^{beta_a}, viewed as the
    1/beta_a-inflation of rho = E^P (Example 9, p.14-15).

    Theorem 3 gives the value function as the inflation at Gamma = mu-ess-inf of
    gamma_a = 1/beta_a, i.e. ES^{beta_star}(X) with beta_star = mu-ess-sup beta_a
    (see Theorem 5's explicit "mu-essential supremum" wording, p.21-22, which
    fixes the translation between the gamma_a-indexed Theorem 3/4 and the
    beta_a-indexed Theorem 5/Example 10). Theorem 4 gives existence of an
    optimizer iff mu({a: beta_a = beta_star}) > 0, in which case the optimal
    allocation gives all of X to the atom of agents at beta_star.

    ``beta_sup`` lets the caller pass the *true* mu-essential supremum of the
    underlying continuum A, which a finite midpoint discretization of an
    interval such as A = (a_lo, a_hi] never literally attains (every discrete
    agent's beta is strictly below a_hi). Without it (``beta_sup=None``), the
    essential supremum is taken to be ``max(betas)`` of the discrete sample
    itself -- the natural choice whenever the discretization already contains
    an explicit atom at the continuum's true supremum (Theorem 4, part 1).
    Passing the continuum's ``beta_sup`` explicitly is what lets this function
    expose Theorem 4(2)'s non-existence phenomenon: the finite-agent brute
    force optimum always concentrates on its own discrete maximum (an
    artefact of any *finite* collection of agents always having an attained
    maximum, unlike a true continuum), so comparing it to ES^{beta_sup}(X)
    yields a gap that is strictly positive for every M but shrinks as M
    grows and max(betas) -> beta_sup.
    """
    X = np.asarray(X, dtype=float)
    betas = np.asarray(betas, dtype=float)
    if np.any((betas <= 0) | (betas > 1)):
        raise ValueError("all betas must lie in (0, 1]")
    n_omega = X.shape[0]
    m_agents = betas.shape[0]
    P = uniform_probability(n_omega) if P is None else np.asarray(P, dtype=float)
    mu_weights = (np.full(m_agents, 1.0 / m_agents) if mu_weights is None
                  else np.asarray(mu_weights, dtype=float))
    if not np.isclose(mu_weights.sum(), 1.0, atol=1e-8):
        raise ValueError("mu_weights must sum to 1 (normalized agent measure)")

    beta_star = float(beta_sup) if beta_sup is not None else float(np.max(betas))
    if not (0 < beta_star <= 1):
        raise ValueError("beta_sup must lie in (0, 1]")
    atom_mass = float(mu_weights[np.isclose(betas, beta_star, atol=atom_tol)].sum())
    theoretical_value = expected_shortfall(X, beta_star, P)

    if m_agents <= max_lp_agents:
        # Explicit convex solve (verification path): cvxpy's cp.cvar(x, beta)
        # is the average of the worst (1-beta) fraction of largest values of
        # x; with beta = 1 - alpha this is exactly ES^alpha(x) of Example 9
        # (p.14-15).
        X_alloc = cp.Variable((m_agents, n_omega))
        per_agent_risk = [cp.cvar(X_alloc[a, :], 1.0 - betas[a]) for a in range(m_agents)]
        objective = cp.Minimize(sum(mu_weights[a] * per_agent_risk[a] for a in range(m_agents)))
        constraints = [mu_weights @ X_alloc == X]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=solver)
        brute_value, brute_allocation, status = float(problem.value), X_alloc.value, problem.status
    else:
        # Closed form of Theorem 3/4 applied to this finite agent space
        # (see ``es_finite_agent_optimum``): exact, and tractable for large M.
        brute_value, brute_allocation = es_finite_agent_optimum(X, betas, mu_weights)
        status = "closed_form (Theorem 3/4 applied to the finite agent space)"

    gap = brute_value - theoretical_value

    return ESharingResult(
        betas=betas, mu_weights=mu_weights, beta_star=beta_star, atom_mass=atom_mass,
        theoretical_value=theoretical_value, brute_force_value=brute_value,
        brute_force_allocation=brute_allocation, solver_status=status,
        gap_to_theoretical=gap,
    )


def atom_optimal_allocation(X: ArrayLike, betas: ArrayLike, mu_weights: ArrayLike,
                             beta_star: float, atom_tol: float = 1e-9) -> ArrayLike:
    """Theorem 4(1) optimal allocation: give all of X to the atom of agents with
    beta_a = beta_star. Every agent in the atom receives the identical value
    X / atom_mass, so that int_{atom} X_a mu(da) = atom_mass * (X / atom_mass) = X
    (Definition 6's feasibility constraint), and by positive homogeneity of
    ES^{beta_star}, int_{atom} ES^{beta_star}(X_a) mu(da)
    = atom_mass * ES^{beta_star}(X) / atom_mass = ES^{beta_star}(X), matching the
    theoretical value exactly (Theorem 4, part 1, p.18)."""
    is_atom = np.isclose(betas, beta_star, atol=atom_tol)
    atom_mass = mu_weights[is_atom].sum()
    if atom_mass <= 0:
        raise ValueError("no atom at beta_star: Theorem 4(1) allocation is not defined")
    allocation = np.zeros((betas.shape[0], X.shape[0]))
    allocation[is_atom, :] = X / atom_mass
    return allocation


def weighted_total_risk(allocation: ArrayLike, betas: ArrayLike, mu_weights: ArrayLike,
                         P: ArrayLike | None = None) -> float:
    """sum_a mu_weights[a] * ES^{beta_a}(allocation[a]) (discretized integral in
    Definition 6's objective)."""
    total = 0.0
    for a in range(betas.shape[0]):
        total += mu_weights[a] * expected_shortfall(allocation[a], betas[a], P)
    return float(total)


def pareto_improve(allocation: ArrayLike, betas: ArrayLike, mu_weights: ArrayLike,
                    top_quantile_fraction: float, shift_fraction: float,
                    P: ArrayLike | None = None) -> tuple[ArrayLike, float, float]:
    """Construct a Pareto improvement on a candidate allocation by shifting a
    small fraction of every non-top agent's risk to the agents with the
    highest beta_a (closest to the essential supremum, hence the least
    conservative and with the smallest ES), following the proof sketch of
    Theorem 4(2) in Appendix C (p.18-19): one can always improve by moving risk
    from agents with gamma_a > Gamma + eps to agents with gamma_b <= Gamma + eps,
    i.e. (since gamma = 1/beta) from low-beta to high-beta agents.

    Returns (new_allocation, total_risk_before, total_risk_after).

    Feasibility (int_A X_a mu(da) = X) is preserved exactly: reducing agent
    i's allocation by a fraction ``shift_fraction`` removes mu_weights[i] *
    shift_fraction * X_i from the weighted integral; this removed mass R is
    added back by raising every top agent's (un-weighted) allocation by
    R / top_weight, so that the weighted contribution added equals
    sum_{j in top} mu_weights[j] * (R / top_weight) = R exactly.
    """
    m_agents = betas.shape[0]
    n_top = max(1, int(np.ceil(top_quantile_fraction * m_agents)))
    order = np.argsort(-betas)
    top_idx = order[:n_top]
    rest_idx = order[n_top:]

    risk_before = weighted_total_risk(allocation, betas, mu_weights, P)

    new_allocation = allocation.copy()
    removed_mass = np.zeros(allocation.shape[1])
    for i in rest_idx:
        delta = shift_fraction * allocation[i, :]
        new_allocation[i, :] -= delta
        removed_mass += mu_weights[i] * delta
    top_weight = mu_weights[top_idx].sum()
    for j in top_idx:
        new_allocation[j, :] += removed_mass / top_weight

    constraint_check = mu_weights @ new_allocation
    assert np.allclose(constraint_check, mu_weights @ allocation, atol=1e-8), (
        "Pareto-improvement step violated the feasibility constraint int_A X_a mu(da) = X"
    )

    risk_after = weighted_total_risk(new_allocation, betas, mu_weights, P)
    return new_allocation, risk_before, risk_after


@dataclass
class DualCheckResult:
    """Result of verifying Theorem 1's dual representation (p.9) for a family
    of test measures Q."""

    kind: str
    conjugate_integral: ArrayLike  # int_A rho_a^*(Q) mu(da), one value per test Q
    conjugate_value_function: ArrayLike  # (box rho_a mu(da))^*(Q), one value per test Q
    max_abs_diff: float
    sup_representation_value: float
    sup_representation_argmax_gap: float


def entropic_dual_check(X: ArrayLike, gammas: ArrayLike, test_measures: list[ArrayLike],
                         P: ArrayLike | None = None) -> DualCheckResult:
    """Theorem 1 for entropic agents: (box Ent^{gamma_a} mu(da))^*(Q) = Gamma*H(Q||P)
    = integral of individual conjugates gamma_a*H(Q||P); and the dual sup in (6) is
    attained at the Esscher tilt Q* with dQ*/dP proportional to exp(X/Gamma).
    """
    X = np.asarray(X, dtype=float)
    n_omega = X.shape[0]
    P = uniform_probability(n_omega) if P is None else np.asarray(P, dtype=float)
    Gamma = float(np.mean(gammas))

    conjugate_integral = np.array([
        float(np.mean([entropic_conjugate(g, Q, P) for g in gammas])) for Q in test_measures
    ])
    conjugate_value_function = np.array([Gamma * relative_entropy(Q, P) for Q in test_measures])
    max_abs_diff = float(np.max(np.abs(conjugate_integral - conjugate_value_function)))

    value_fn = entropic_risk_measure(X, Gamma, P)
    q_star = esscher_tilted_measure(X, Gamma, P)
    sup_value = float(np.sum(q_star * X) - Gamma * relative_entropy(q_star, P))
    return DualCheckResult(
        kind="entropic", conjugate_integral=conjugate_integral,
        conjugate_value_function=conjugate_value_function, max_abs_diff=max_abs_diff,
        sup_representation_value=sup_value,
        sup_representation_argmax_gap=abs(sup_value - value_fn),
    )


def es_dual_check(alphas: ArrayLike, test_measures: list[ArrayLike],
                   P: ArrayLike | None = None) -> dict:
    """Theorem 1 for ES agents: ES^{alpha}_a conjugates are indicators of
    {dQ/dP <= 1/alpha_a}; the integral of individual conjugates is 0 exactly
    when dQ/dP <= 1/alpha_star := 1/max(alpha_a) P-a.e. (i.e. mu-a.e. agent
    constraint holds), matching the conjugate of ES^{alpha_star} (Example 9).
    """
    n_omega = test_measures[0].shape[0]
    P = uniform_probability(n_omega) if P is None else np.asarray(P, dtype=float)
    alpha_star = float(np.max(alphas))
    rows = []
    for Q in test_measures:
        per_agent_zero = np.array([es_conjugate_is_zero(Q, a, P) for a in alphas])
        integral_is_zero = bool(np.all(per_agent_zero))
        value_fn_is_zero = es_conjugate_is_zero(Q, alpha_star, P)
        rows.append({"integral_is_zero": integral_is_zero, "value_fn_is_zero": value_fn_is_zero,
                      "agree": integral_is_zero == value_fn_is_zero})
    return {"alpha_star": alpha_star, "rows": rows,
            "all_agree": all(r["agree"] for r in rows)}
