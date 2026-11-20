"""Experiment 1: Optimal risk sharing with dilated (entropic) risk measures
across a continuum of agents.

Numerically verifies Theorem 2 and Example 8 (p.11-13): the integral infimal
convolution of entropic risk measures (Ent^{gamma_a})_{a in A} equals
Ent^Gamma with Gamma = int_A gamma_a mu(da), and the allocation
X_a = gamma_a * X / Gamma achieves this value. Also verifies the dual
representation of Theorem 1 in the entropic case and compares the continuum
(A = [0,1], Lebesgue measure) against the classical finite-agent case
(A = {1,...,N}, counting measure), to which Definition 6 reduces when A is
finite.

Run as ``python -m exp.exp1``.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import truncated_normal_endowment, uniform_aggregate_endowment
from src import config
from src.reporting import results_path, save_json
from src.risk_measures import dirichlet_test_measure, entropic_risk_measure, uniform_probability
from src.risk_sharing import discretize_agents, entropic_dual_check, entropic_risk_sharing


def run_continuum_case() -> dict:
    """Continuum of agents A = [0,1], Lebesgue measure, gamma_a = 1 + a."""
    P = uniform_probability(config.EXP1_N_OMEGA)
    X = uniform_aggregate_endowment(config.EXP1_N_OMEGA, config.SEED)

    a_grid = discretize_agents(config.EXP1_M_AGENTS, 0.0, 1.0)
    gammas = config.EXP1_GAMMA_INTERCEPT + config.EXP1_GAMMA_SLOPE * a_grid

    result = entropic_risk_sharing(X, gammas, P)

    rng = np.random.default_rng(config.SEED)
    test_measures = [dirichlet_test_measure(config.EXP1_N_OMEGA, 1.0, rng) for _ in range(10)]
    dual = entropic_dual_check(X, gammas, test_measures, P)

    print(f"[exp1] continuum: M={config.EXP1_M_AGENTS}, Gamma={result.Gamma:.6f}")
    print(f"[exp1] continuum: theoretical Ent^Gamma(X) = {result.theoretical_value:.6f}, "
          f"brute-force value = {result.brute_force_value:.6f}")
    print(f"[exp1] continuum: max |allocation diff| = {result.max_allocation_abs_diff:.3e}, "
          f"constraint violation = {result.constraint_violation:.3e}")
    print(f"[exp1] continuum: dual conjugate max |diff| = {dual.max_abs_diff:.3e}, "
          f"sup-representation gap = {dual.sup_representation_argmax_gap:.3e}")

    return {
        "n_omega": config.EXP1_N_OMEGA,
        "m_agents": config.EXP1_M_AGENTS,
        "Gamma": result.Gamma,
        "theoretical_value": result.theoretical_value,
        "brute_force_value": result.brute_force_value,
        "value_abs_diff": abs(result.theoretical_value - result.brute_force_value),
        "max_allocation_abs_diff": result.max_allocation_abs_diff,
        "constraint_violation": result.constraint_violation,
        "solver_status": result.solver_status,
        "dual_max_abs_diff": dual.max_abs_diff,
        "dual_sup_representation_value": dual.sup_representation_value,
        "dual_sup_representation_argmax_gap": dual.sup_representation_argmax_gap,
    }, result, a_grid, gammas


def run_discrete_case() -> dict:
    """Classical finite-agent case: A = {1,...,N}, counting measure,
    gamma_i = 1 + i/N, X following a truncated normal law (data_requirements:
    'truncated normal distribution on a finite probability space')."""
    N = config.EXP1_N_DISCRETE_AGENTS
    n_omega = config.EXP1_N_OMEGA
    P = uniform_probability(n_omega)
    X = truncated_normal_endowment(n_omega, config.SEED + 1)

    i_grid = (np.arange(N) + 1) / N  # i/N for i = 1,...,N
    gammas = config.EXP1_GAMMA_INTERCEPT + config.EXP1_GAMMA_SLOPE * i_grid
    result = entropic_risk_sharing(X, gammas, P)

    print(f"[exp1] discrete (N={N}): Gamma={result.Gamma:.6f}, "
          f"theoretical={result.theoretical_value:.6f}, "
          f"brute-force={result.brute_force_value:.6f}, "
          f"max |allocation diff|={result.max_allocation_abs_diff:.3e}")

    return {
        "n_agents": N,
        "Gamma": result.Gamma,
        "theoretical_value": result.theoretical_value,
        "brute_force_value": result.brute_force_value,
        "value_abs_diff": abs(result.theoretical_value - result.brute_force_value),
        "max_allocation_abs_diff": result.max_allocation_abs_diff,
        "constraint_violation": result.constraint_violation,
        "solver_status": result.solver_status,
    }


def run_convergence_sweep() -> dict:
    """Sweep M to show the discretized integral infimal convolution matches
    Ent^Gamma(X) exactly regardless of M (the entropic formula of Theorem 2 is
    exact for every finite collection of agents, since Definition 6 with A
    finite and mu uniform already IS the classical infimal convolution --
    there is no discretization bias to shrink, unlike the inflated/ES case of
    experiment 2)."""
    P = uniform_probability(config.EXP1_N_OMEGA)
    X = uniform_aggregate_endowment(config.EXP1_N_OMEGA, config.SEED)
    m_grid = [5, 20, 100, config.EXP1_M_AGENTS]
    value_diffs = []
    allocation_diffs = []
    for m in m_grid:
        a_grid = discretize_agents(m, 0.0, 1.0)
        gammas = config.EXP1_GAMMA_INTERCEPT + config.EXP1_GAMMA_SLOPE * a_grid
        result = entropic_risk_sharing(X, gammas, P)
        value_diffs.append(abs(result.theoretical_value - result.brute_force_value))
        allocation_diffs.append(result.max_allocation_abs_diff)
    print(f"[exp1] convergence sweep M={m_grid}: value diffs={value_diffs}")
    return {"m_grid": m_grid, "value_abs_diffs": value_diffs, "allocation_abs_diffs": allocation_diffs}


def make_plot(a_grid: np.ndarray, gammas: np.ndarray, result) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(a_grid, gammas / result.Gamma, color="tab:blue")
    axes[0].set_xlabel("agent a")
    axes[0].set_ylabel(r"allocation share $\gamma_a / \Gamma$")
    axes[0].set_title("Theoretical optimal allocation share (Theorem 2)")
    axes[0].grid(alpha=0.3)

    sample_states = np.linspace(0, result.theoretical_allocation.shape[1] - 1, 200).astype(int)
    axes[1].scatter(result.theoretical_allocation[::25, sample_states].ravel(),
                     result.brute_force_allocation[::25, sample_states].ravel(),
                     s=4, alpha=0.4, color="tab:orange")
    lims = [result.theoretical_allocation.min(), result.theoretical_allocation.max()]
    axes[1].plot(lims, lims, "k--", linewidth=1, label="y = x")
    axes[1].set_xlabel(r"theoretical allocation $X_a = \gamma_a X/\Gamma$")
    axes[1].set_ylabel("brute-force (cvxpy) allocation")
    axes[1].set_title("Brute-force optimum vs. theoretical formula")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(results_path("exp1_entropic_allocation.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    continuum_metrics, result, a_grid, gammas = run_continuum_case()
    discrete_metrics = run_discrete_case()
    convergence_metrics = run_convergence_sweep()
    make_plot(a_grid, gammas, result)

    payload = {
        "experiment": "exp1_entropic_risk_sharing",
        "theorem": "Theorem 2 / Example 8 (dilated / entropic risk measures)",
        "continuum_case": continuum_metrics,
        "discrete_case": discrete_metrics,
        "convergence_sweep": convergence_metrics,
    }
    path = save_json("exp1_metrics.json", payload)
    print(f"[exp1] wrote {path}")


if __name__ == "__main__":
    main()
