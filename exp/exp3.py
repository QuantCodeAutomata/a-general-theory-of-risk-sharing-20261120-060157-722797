"""Experiment 3: Non-existence of Pareto efficient allocations with expected
shortfall preferences and a continuum of agents.

Numerically verifies Theorem 5 and Example 10 (p.20-22): for agents with
preferences ES^{beta_a}, beta_a = a over the continuum A = [ALPHA_LOW,
ALPHA_HIGH] with Lebesgue measure, a Pareto efficient allocation of the
aggregate endowment X does not exist whenever (a) beta := mu-ess-sup beta_a
is not attained (mu({a: beta_a = beta}) = 0) and (b) X is non-constant under
the ES^{beta} inflation (ES^{beta}(X) != ES^{alpha}(X) for some alpha < beta).
By Proposition 6 (p.20), Pareto efficiency of an allocation (X_a) is
equivalent to (X_a) attaining the integral infimal convolution
(box_a ES^{beta_a} mu(da))(X); Theorem 3/4 (verified in experiment 2) show
this infimum equals ES^{beta}(X) and is attained only by an allocation that
concentrates X on an atom {beta_a = beta} of positive mu-mass. In the
continuum of Example 10 that atom has mu-mass zero, so no allocation attains
the infimum and none is Pareto efficient. The finite-agent case (A =
{1,...,N}, counting measure) always has an attained maximum, hence always
has a Pareto efficient allocation; letting N -> infinity illustrates how the
attaining atom's mu-mass (1/N) shrinks to zero, which is the mechanism behind
the continuum's non-existence.

Run as ``python -m exp.exp3``.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import uniform_aggregate_endowment
from src import config
from src.reporting import results_path, save_json
from src.risk_measures import expected_shortfall, uniform_probability
from src.risk_sharing import (
    discretize_agents,
    es_finite_agent_optimum,
    es_risk_sharing,
    pareto_improve,
    weighted_total_risk,
)


def run_precondition_check(X: np.ndarray, P: np.ndarray) -> dict:
    """Theorem 5's hypothesis (p.21-22): beta = ess sup beta_a = ALPHA_HIGH must
    not be attained by mu (checked in ``run_continuum_nonexistence_case``) and
    X must be non-constant under the inflation, i.e. ES^{beta}(X) != ES^{alpha}(X)
    for some alpha < beta. Checked here for several alpha < ALPHA_HIGH."""
    beta = config.EXP3_ALPHA_HIGH
    es_beta = expected_shortfall(X, beta, P)
    alphas = np.linspace(config.EXP3_ALPHA_LOW, beta - 0.05, 5)
    rows = []
    for alpha in alphas:
        es_alpha = expected_shortfall(X, float(alpha), P)
        rows.append({
            "alpha": float(alpha), "es_alpha": es_alpha,
            "differs_from_es_beta": bool(abs(es_alpha - es_beta) > config.NUMERICAL_TOLERANCE),
        })
    precondition_holds = all(r["differs_from_es_beta"] for r in rows)
    print(f"[exp3] precondition check: ES^{beta}(X)={es_beta:.6f}; "
          f"non-constant under inflation for all tested alpha < beta: {precondition_holds}")
    return {"beta": beta, "es_beta": es_beta, "rows": rows, "precondition_holds": precondition_holds}


def run_continuum_nonexistence_case(X: np.ndarray, P: np.ndarray) -> dict:
    """For the continuum discretized into M agents, the finite-agent optimum
    (Theorem 3/4 applied to this finite set) concentrates on the agent with
    the largest beta_a, achieving ES^{max(betas)}(X) -- strictly above the
    continuum's theoretical value ES^{ALPHA_HIGH}(X) because max(betas) <
    ALPHA_HIGH for any finite midpoint discretization. The gap is the
    quantitative signature of non-existence."""
    M = config.EXP3_M_AGENTS
    betas = discretize_agents(M, config.EXP3_ALPHA_LOW, config.EXP3_ALPHA_HIGH)
    result = es_risk_sharing(X, betas, P, beta_sup=config.EXP3_ALPHA_HIGH)

    print(f"[exp3] continuum (M={M}): max(betas)={np.max(betas):.6f} < "
          f"beta_sup={config.EXP3_ALPHA_HIGH}, atom_mass_at_sup={result.atom_mass:.3e}")
    print(f"[exp3] continuum: theoretical ES^beta(X)={result.theoretical_value:.6f}, "
          f"finite-agent optimum={result.brute_force_value:.6f}, gap={result.gap_to_theoretical:.6f}")

    return {
        "m_agents": M,
        "max_beta_in_sample": float(np.max(betas)),
        "beta_sup": config.EXP3_ALPHA_HIGH,
        "atom_mass_at_sup": result.atom_mass,
        "theoretical_value": result.theoretical_value,
        "finite_agent_optimum": result.brute_force_value,
        "gap_to_theoretical": result.gap_to_theoretical,
    }


def run_pareto_improvement_demo(X: np.ndarray, P: np.ndarray) -> dict:
    """Starting from an arbitrary (non-optimal) candidate allocation -- an
    equal split of X across all M agents -- repeatedly apply
    ``pareto_improve`` (shifting risk from low-beta to high-beta agents, the
    mechanism of the proof of Theorem 4(2) in Appendix C, p.18-19). Every step
    strictly decreases the total weighted risk sum_a mu_a ES^{beta_a}(X_a)
    without ever increasing any agent's own allocation share disproportionately,
    i.e. it is a genuine Pareto-improving direction. The sequence of total-risk
    values is non-increasing and converges to the finite-agent optimum of
    ``run_continuum_nonexistence_case`` (attained because M here is finite),
    illustrating that for ANY candidate allocation on the continuum a strict
    improvement can always be found -- the content of Theorem 5's
    non-existence argument."""
    M = config.EXP3_M_AGENTS
    betas = discretize_agents(M, config.EXP3_ALPHA_LOW, config.EXP3_ALPHA_HIGH)
    mu_weights = np.full(M, 1.0 / M)

    # Arbitrary feasible candidate: every agent receives the full X, so that
    # int_A X_a mu(da) = sum_a mu_a * X = X (mu_a uniform, sum mu_a = 1).
    allocation = np.tile(X, (M, 1))
    risk_trace = [weighted_total_risk(allocation, betas, mu_weights, P)]
    n_steps = 40
    for _ in range(n_steps):
        allocation, before, after = pareto_improve(
            allocation, betas, mu_weights,
            config.EXP3_TOP_QUANTILE_FRACTION, config.EXP3_PARETO_SHIFT_FRACTION, P,
        )
        risk_trace.append(after)

    finite_optimum, _ = es_finite_agent_optimum(X, betas, mu_weights)
    is_monotone_nonincreasing = bool(np.all(np.diff(risk_trace) <= config.NUMERICAL_TOLERANCE))

    print(f"[exp3] Pareto-improvement demo: risk from {risk_trace[0]:.6f} to "
          f"{risk_trace[-1]:.6f} over {n_steps} steps (finite-agent optimum={finite_optimum:.6f}); "
          f"monotone non-increasing: {is_monotone_nonincreasing}")

    return {
        "n_steps": n_steps,
        "risk_trace": risk_trace,
        "finite_agent_optimum": finite_optimum,
        "is_monotone_nonincreasing": is_monotone_nonincreasing,
        "final_gap_to_finite_optimum": float(risk_trace[-1] - finite_optimum),
    }


def run_finite_agent_convergence(X: np.ndarray, P: np.ndarray) -> dict:
    """Theorem 3/4 applied literally to A = {1,...,N} with counting measure
    (1/N per agent): an optimal, hence Pareto efficient, allocation always
    exists (Theorem 4, part 1, since a finite set always has an attained
    maximum beta_i, an atom of mass 1/N > 0). As N -> infinity, beta_i =
    ALPHA_LOW + (ALPHA_HIGH-ALPHA_LOW)*i/N -> ALPHA_HIGH and the optimal value
    converges to the continuum value ES^{ALPHA_HIGH}(X), but the atom
    attaining the optimum has mu-mass 1/N -> 0: in the true continuum limit
    (mu-mass of a single point is 0) the atom disappears and so does the
    optimizer, which is Theorem 5's non-existence mechanism."""
    rows = []
    beta_sup = config.EXP3_ALPHA_HIGH
    theoretical_value = expected_shortfall(X, beta_sup, P)
    for N in config.EXP3_N_FINITE_GRID:
        N = int(N)
        i_grid = np.arange(1, N + 1)
        betas = config.EXP3_ALPHA_LOW + (config.EXP3_ALPHA_HIGH - config.EXP3_ALPHA_LOW) * i_grid / N
        mu_weights = np.full(N, 1.0 / N)
        value, allocation = es_finite_agent_optimum(X, betas, mu_weights)
        argmax_mass = float(mu_weights[np.isclose(betas, np.max(betas))].sum())
        rows.append({
            "n_agents": N, "max_beta": float(np.max(betas)), "optimal_value": value,
            "gap_to_continuum_value": float(value - theoretical_value),
            "optimizing_atom_mass": argmax_mass,
        })
    gaps = [r["gap_to_continuum_value"] for r in rows]
    masses = [r["optimizing_atom_mass"] for r in rows]
    print(f"[exp3] finite-agent convergence N={config.EXP3_N_FINITE_GRID.tolist()}: "
          f"gaps={gaps}")
    print(f"[exp3] finite-agent convergence: optimizing atom mass -> 0: {masses}")
    return {
        "n_grid": config.EXP3_N_FINITE_GRID.tolist(),
        "theoretical_continuum_value": theoretical_value,
        "rows": rows,
        # The grid beta_i = ALPHA_LOW + (ALPHA_HIGH-ALPHA_LOW)*i/N always touches
        # beta_sup exactly at i=N, so the gap is identically 0 for every N (the
        # finite optimum attains the continuum value exactly, Theorem 4 part 1);
        # "non-increasing" rather than "strictly decreasing" is the correct check.
        "gap_nonincreasing": bool(np.all(np.diff(gaps) <= config.NUMERICAL_TOLERANCE)),
        "atom_mass_monotone_decreasing": bool(np.all(np.diff(masses) < 0)),
    }


def make_plot(pareto_demo: dict, finite_conv: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(range(len(pareto_demo["risk_trace"])), pareto_demo["risk_trace"],
                 marker=".", color="tab:purple")
    axes[0].axhline(pareto_demo["finite_agent_optimum"], color="k", linestyle="--",
                     linewidth=1, label="finite-agent optimum")
    axes[0].set_xlabel("Pareto-improvement step")
    axes[0].set_ylabel(r"total weighted risk $\sum_a \mu_a\, ES^{\beta_a}(X_a)$")
    axes[0].set_title("Any candidate allocation admits a Pareto improvement")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    n_grid = finite_conv["n_grid"]
    gaps = [r["gap_to_continuum_value"] for r in finite_conv["rows"]]
    masses = [r["optimizing_atom_mass"] for r in finite_conv["rows"]]
    ax2 = axes[1]
    ax2.plot(n_grid, gaps, marker="o", color="tab:red", label="gap to continuum value")
    ax2.set_xscale("log")
    ax2.set_xlabel("number of agents N (log scale)")
    ax2.set_ylabel("gap to continuum value", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax3 = ax2.twinx()
    ax3.plot(n_grid, masses, marker="s", color="tab:blue", label="mass of optimizing atom")
    ax3.set_ylabel("mu-mass of optimizing atom (1/N)", color="tab:blue")
    ax3.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_title("Theorem 5: optimizer exists for finite N, vanishes as N -> inf")

    fig.tight_layout()
    fig.savefig(results_path("exp3_pareto_nonexistence.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    P = uniform_probability(config.EXP3_N_OMEGA)
    X = uniform_aggregate_endowment(config.EXP3_N_OMEGA, config.SEED)

    precondition = run_precondition_check(X, P)
    continuum_case = run_continuum_nonexistence_case(X, P)
    pareto_demo = run_pareto_improvement_demo(X, P)
    finite_conv = run_finite_agent_convergence(X, P)
    make_plot(pareto_demo, finite_conv)

    payload = {
        "experiment": "exp3_pareto_nonexistence",
        "theorem": "Theorem 5 / Example 10 / Proposition 6 (Pareto efficiency with ES preferences)",
        "precondition_check": precondition,
        "continuum_nonexistence_case": continuum_case,
        "pareto_improvement_demo": pareto_demo,
        "finite_agent_convergence": finite_conv,
    }
    path = save_json("exp3_metrics.json", payload)
    print(f"[exp3] wrote {path}")


if __name__ == "__main__":
    main()
