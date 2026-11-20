"""Risk measures on a finite probability space.

Custom implementations of the entropic risk measure (Definition/Example 7,
p.11), expected shortfall as a coherent inflation of the expectation
(Example 9, p.14-15; Rockafellar-Uryasev representation, Proposition 4
Eq. (10), p.15-16), and the relative entropy functional H(Q||P) that
parametrizes the convex conjugate of the entropic risk measure. These
formulas are specific to the paper's sign and normalization conventions
(risk measures of a random "loss/endowment" X, with ess sup rather than
"-X" convention), so they are implemented from scratch rather than taken
from a generic risk-analytics library (Context7 was consulted for
``quantstats``/``arch``-style CVaR utilities and confirmed none reproduce
this exact dual definition and sign convention).

For the convex programs in ``src/risk_sharing.py`` the per-agent CVaR is
instead computed with the ``cvxpy`` atom ``cp.cvar(x, beta)``, since its
documented convention -- the average of the worst (1-beta) fraction of
largest values of x -- is exactly ES^alpha(X) of Example 9 with
beta = 1 - alpha; that atom is used there and cited accordingly.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

ArrayLike = np.ndarray


def uniform_probability(n_omega: int) -> ArrayLike:
    """Equiprobable measure P on n_omega states (finite approximation of L^infty(P))."""
    if n_omega <= 0:
        raise ValueError("n_omega must be a positive integer")
    return np.full(n_omega, 1.0 / n_omega)


def relative_entropy(Q: ArrayLike, P: ArrayLike) -> float:
    """H(Q||P) = E^Q[log(dQ/dP)] = sum_i Q_i log(Q_i/P_i) (paper, Example 7, p.11).

    Uses the convention 0 * log(0/P_i) = 0 for states with Q_i = 0.
    """
    Q = np.asarray(Q, dtype=float)
    P = np.asarray(P, dtype=float)
    if Q.shape != P.shape:
        raise ValueError("Q and P must have the same shape")
    if not np.isclose(Q.sum(), 1.0, atol=1e-6) or not np.isclose(P.sum(), 1.0, atol=1e-6):
        raise ValueError("Q and P must each be probability vectors summing to 1")
    mask = Q > 0
    ratio = np.ones_like(Q)
    ratio[mask] = Q[mask] / P[mask]
    terms = np.zeros_like(Q)
    terms[mask] = Q[mask] * np.log(ratio[mask])
    return float(np.sum(terms))


def entropic_risk_measure(X: ArrayLike, gamma: float, P: ArrayLike | None = None) -> float:
    """Ent^gamma(X) = gamma * log(E^P[exp(X/gamma)]) (paper, Example 7, p.11).

    Computed in a numerically stable way via the log-sum-exp identity.
    """
    if gamma <= 0:
        raise ValueError("gamma must be strictly positive")
    X = np.asarray(X, dtype=float)
    P = uniform_probability(X.shape[0]) if P is None else np.asarray(P, dtype=float)
    z = X / gamma
    m = np.max(z)
    log_e_exp = m + np.log(np.sum(P * np.exp(z - m)))
    return float(gamma * log_e_exp)


def esscher_tilted_measure(X: ArrayLike, gamma: float, P: ArrayLike | None = None) -> ArrayLike:
    """Esscher tilt dQ*/dP proportional to exp(X/gamma): the subgradient of Ent^gamma
    at X (paper, Example 8, p.13, citing Barrieu-El Karoui / Esscher tilting)."""
    if gamma <= 0:
        raise ValueError("gamma must be strictly positive")
    X = np.asarray(X, dtype=float)
    P = uniform_probability(X.shape[0]) if P is None else np.asarray(P, dtype=float)
    z = X / gamma
    z = z - np.max(z)
    weights = P * np.exp(z)
    return weights / weights.sum()


def entropic_conjugate(gamma: float, Q: ArrayLike, P: ArrayLike | None = None) -> float:
    """Ent^gamma's convex conjugate: Ent^{gamma,*}(Q) = gamma * H(Q||P) (paper, p.11, Example 7)."""
    Q = np.asarray(Q, dtype=float)
    P = uniform_probability(Q.shape[0]) if P is None else np.asarray(P, dtype=float)
    return gamma * relative_entropy(Q, P)


def expected_shortfall(X: ArrayLike, alpha: float, P: ArrayLike | None = None) -> float:
    """ES^alpha(X) = sup_{Q: dQ/dP <= 1/alpha} E^Q(X) (paper, Example 9, p.14-15).

    Computed directly from the dual definition: the worst-case Q concentrates as
    much mass as the constraint dQ/dP <= 1/alpha allows on the states with the
    largest values of X. Equivalently, this is the mean of the top alpha-fraction
    of P-mass of X, the classical expected-shortfall-at-level-alpha formula.
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    X = np.asarray(X, dtype=float)
    P = uniform_probability(X.shape[0]) if P is None else np.asarray(P, dtype=float)
    order = np.argsort(-X)  # descending: largest losses first
    X_sorted = X[order]
    P_sorted = P[order]
    cum_p = np.cumsum(P_sorted)
    budget = alpha
    weights = np.zeros_like(P_sorted)
    for i, p in enumerate(P_sorted):
        if budget <= 0:
            break
        take = min(p, budget)
        weights[i] = take
        budget -= take
    return float(np.sum(weights * X_sorted) / alpha)


def es_rockafellar_uryasev(X: ArrayLike, alpha: float, P: ArrayLike | None = None) -> tuple[float, float]:
    """ES^alpha(X) = inf_x (x + (1/alpha) E^P[(X-x)^+]) (Rockafellar-Uryasev
    representation, paper Proposition 4 Eq. (10) with base risk measure rho = E^P
    and gamma = 1/alpha). Returns (optimal value, minimizing x*).
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    X = np.asarray(X, dtype=float)
    P = uniform_probability(X.shape[0]) if P is None else np.asarray(P, dtype=float)

    def objective(x: float) -> float:
        return x + (1.0 / alpha) * np.sum(P * np.maximum(X - x, 0.0))

    lo, hi = float(np.min(X)), float(np.max(X))
    result = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-10})
    return float(result.fun), float(result.x)


def es_conjugate_is_zero(Q: ArrayLike, alpha: float, P: ArrayLike | None = None, tol: float = 1e-6) -> bool:
    """Indicator that ES^alpha's convex conjugate at Q is 0 rather than +infinity:
    true iff dQ/dP <= 1/alpha P-a.e. (paper, Example 9, p.14-15, coherent risk
    measure conjugate supported on {dQ/dP <= 1/alpha})."""
    Q = np.asarray(Q, dtype=float)
    P = uniform_probability(Q.shape[0]) if P is None else np.asarray(P, dtype=float)
    density_ratio = np.divide(Q, P, out=np.zeros_like(Q), where=P > 0)
    return bool(np.max(density_ratio) <= 1.0 / alpha + tol)


def phi_inflation(X: ArrayLike, gamma: float, P: ArrayLike | None = None) -> float:
    """phi(gamma) = rho~_gamma(X) for the inflation of rho = E^P, i.e.
    phi(gamma) = ES^{1/gamma}(X) (paper, Proposition 5, p.16-17)."""
    if gamma < 1:
        raise ValueError("gamma must be >= 1 for an inflation parameter")
    alpha = 1.0 / gamma
    return expected_shortfall(X, alpha, P)


def dirichlet_test_measure(n_omega: int, concentration: float, rng: np.random.Generator) -> ArrayLike:
    """Draw a random probability vector Q absolutely continuous w.r.t. a uniform P,
    used as a generic test measure for dual-representation checks (Theorem 1)."""
    return rng.dirichlet(np.full(n_omega, concentration))
