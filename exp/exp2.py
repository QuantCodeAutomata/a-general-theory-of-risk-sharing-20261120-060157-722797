"""Experiment 2: Risk sharing with inflated (expected shortfall) risk measures
-- value function and optimizer (non-)existence.

Numerically verifies Theorem 3 and Theorem 4 (p.17-19) for the family
varrho_a = ES_P^{beta_a}, viewed as the inflation varrho_a = rho~_{gamma_a} of
rho = E^P with gamma_a = 1/beta_a (Example 9, p.14-15). Theorem 3 states that

    (box_{a in A} varrho_a mu(da)) = rho~_Gamma,   Gamma = mu-ess-inf gamma_a,

and since gamma_a = 1/beta_a is a decreasing function of beta_a, the
essential infimum of gamma_a corresponds to the essential SUPREMUM of beta_a:
Gamma = 1/(mu-ess-sup beta_a). The value function is therefore
ES_P^{beta}(X) with beta = mu-ess-sup beta_a -- NOT the essential infimum of
beta_a. (A naive reading directly in beta-space would instead compute
"beta = ess inf beta_a = 0.1"; this conflates beta_a with the inflation
parameter gamma_a and is corrected here against the paper's Theorem 3, which is stated
in gamma-space with gamma_a in [1, infinity) -- see the direct confirmation in
Theorem 5 and Example 10, p.21-22, which independently states the
*essential supremum* of beta_a as the relevant threshold for the same
family varrho_a = ES_P^{beta_a}.) This correction is recorded in
results/RESULTS.md under "Deviations from the paper".

Theorem 4 then gives existence of an optimal allocation iff
mu({a : beta_a = beta}) > 0 (an atom at the essential supremum), with
non-existence otherwise whenever X is non-constant under ES^beta. The
experiment also verifies the Rockafellar-Uryasev representation (Proposition
4, Eq. (10), p.15-16) and the monotonicity/concavity/limit of
phi(gamma) = ES^{1/gamma}(X) (Proposition 5, p.16-17).

Run as ``python -m exp.exp2``.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import uniform_aggregate_endowment
from src import config
from src.reporting import results_path, save_json
from src.risk_measures import (
    es_rockafellar_uryasev,
    expected_shortfall,
    phi_inflation,
    uniform_probability,
)
from src.risk_sharing import (
    atom_optimal_allocation,
    discretize_agents,
    es_risk_sharing,
    weighted_total_risk,
)


def run_continuum_gap_case(X: np.ndarray, P: np.ndarray) -> dict:
    """Theorem 4(2) (p.18-19): mu({a: beta_a = ess sup beta_a}) = 0 for a pure
    continuum discretization, so the brute-force optimum strictly exceeds the
    theoretical value ES^{beta}(X), beta = ess sup beta_a = ALPHA_HIGH."""
    betas = discretize_agents(config.EXP2_M_AGENTS, config.EXP2_ALPHA_LOW, config.EXP2_ALPHA_HIGH)
    result = es_risk_sharing(X, betas, P, beta_sup=config.EXP2_ALPHA_HIGH)

    print(f"[exp2] continuum gap: M={config.EXP2_M_AGENTS}, beta_star={result.beta_star}, "
          f"atom_mass={result.atom_mass:.3e}")
    print(f"[exp2] continuum gap: theoretical ES^beta(X)={result.theoretical_value:.6f}, "
          f"brute-force={result.brute_force_value:.6f}, gap={result.gap_to_theoretical:.6f}")

    return {
        "m_agents": config.EXP2_M_AGENTS,
        "beta_star": result.beta_star,
        "atom_mass": result.atom_mass,
        "theoretical_value": result.theoretical_value,
        "brute_force_value": result.brute_force_value,
        "gap_to_theoretical": result.gap_to_theoretical,
        "solver_status": result.solver_status,
    }


def run_convergence_sweep(X: np.ndarray, P: np.ndarray) -> dict:
    """Sweep M: the gap ES-brute-force minus ES^{beta}(X) shrinks towards zero
    as the discretization's finite maximum approaches ALPHA_HIGH, but by
    Theorem 4(2) never closes exactly for a pure continuum."""
    gaps = []
    max_betas = []
    for m in config.EXP2_M_GRID:
        betas = discretize_agents(int(m), config.EXP2_ALPHA_LOW, config.EXP2_ALPHA_HIGH)
        result = es_risk_sharing(X, betas, P, beta_sup=config.EXP2_ALPHA_HIGH)
        gaps.append(result.gap_to_theoretical)
        max_betas.append(float(np.max(betas)))
    print(f"[exp2] convergence sweep M={config.EXP2_M_GRID.tolist()}: gaps={gaps}")
    return {
        "m_grid": config.EXP2_M_GRID.tolist(),
        "max_beta_in_sample": max_betas,
        "gaps": gaps,
        "monotone_decreasing": bool(np.all(np.diff(gaps) < 0)),
    }


def run_atom_existence_case(X: np.ndarray, P: np.ndarray) -> dict:
    """Theorem 4(1) (p.17-18): place an explicit atom of mu-mass
    EXP2_ATOM_MASS at beta_a = ALPHA_HIGH = ess sup beta_a; giving all risk to
    that atom achieves ES^{ALPHA_HIGH}(X) exactly."""
    m = config.EXP2_M_AGENTS
    cont_betas = discretize_agents(m, config.EXP2_ALPHA_LOW, config.EXP2_ALPHA_HIGH)
    betas = np.append(cont_betas, config.EXP2_ALPHA_HIGH)
    atom_mass = config.EXP2_ATOM_MASS
    mu_weights = np.append(np.full(m, (1.0 - atom_mass) / m), atom_mass)

    result = es_risk_sharing(X, betas, P, mu_weights=mu_weights)
    alloc = atom_optimal_allocation(X, betas, mu_weights, result.beta_star)
    total_risk = weighted_total_risk(alloc, betas, mu_weights, P)
    feasibility_violation = float(np.max(np.abs(mu_weights @ alloc - X)))

    print(f"[exp2] atom existence: beta_star={result.beta_star}, atom_mass={result.atom_mass:.3f}")
    print(f"[exp2] atom existence: theoretical={result.theoretical_value:.6f}, "
          f"brute-force={result.brute_force_value:.6f}, gap={result.gap_to_theoretical:.3e}")
    print(f"[exp2] atom existence: closed-form allocation total risk={total_risk:.6f}, "
          f"feasibility violation={feasibility_violation:.3e}")

    return {
        "atom_mass_specified": atom_mass,
        "beta_star": result.beta_star,
        "atom_mass_detected": result.atom_mass,
        "theoretical_value": result.theoretical_value,
        "brute_force_value": result.brute_force_value,
        "gap_to_theoretical": result.gap_to_theoretical,
        "closed_form_total_risk": total_risk,
        "closed_form_feasibility_violation": feasibility_violation,
        "solver_status": result.solver_status,
    }


def run_rockafellar_uryasev_check(X: np.ndarray, P: np.ndarray) -> dict:
    """Proposition 4, Eq. (10) (p.15-16): ES^alpha(X) = inf_x (x + (1/alpha)
    E^P[(X-x)^+]) must match the dual definition of expected shortfall for a
    range of alpha."""
    alphas = np.linspace(0.05, 0.95, 10)
    rows = []
    for alpha in alphas:
        ru_value, x_star = es_rockafellar_uryasev(X, float(alpha), P)
        dual_value = expected_shortfall(X, float(alpha), P)
        rows.append({
            "alpha": float(alpha), "rockafellar_uryasev_value": ru_value,
            "dual_value": dual_value, "abs_diff": abs(ru_value - dual_value),
            "minimizer_x_star": x_star,
        })
    max_abs_diff = max(r["abs_diff"] for r in rows)
    print(f"[exp2] Rockafellar-Uryasev vs dual ES: max |diff| over {len(alphas)} alphas = {max_abs_diff:.3e}")
    return {"rows": rows, "max_abs_diff": max_abs_diff}


def run_phi_monotonicity_check(X: np.ndarray, P: np.ndarray) -> dict:
    """Proposition 5 (p.16-17): phi(gamma) = ES^{1/gamma}(X) is non-decreasing
    and concave in gamma, and converges to P-ess sup X as gamma -> infinity."""
    gammas = config.EXP2_PHI_GAMMA_GRID
    phi_vals = np.array([phi_inflation(X, float(g), P) for g in gammas])
    diffs = np.diff(phi_vals)
    second_diffs = np.diff(diffs)
    ess_sup_X = float(np.max(X))

    is_nondecreasing = bool(np.all(diffs >= -config.NUMERICAL_TOLERANCE))
    is_concave = bool(np.all(second_diffs <= config.NUMERICAL_TOLERANCE))
    gap_to_ess_sup = float(ess_sup_X - phi_vals[-1])

    print(f"[exp2] phi(gamma) monotonicity: non-decreasing={is_nondecreasing}, concave={is_concave}")
    print(f"[exp2] phi(gamma) limit: phi({gammas[-1]:.1f})={phi_vals[-1]:.6f}, "
          f"ess sup X={ess_sup_X:.6f}, gap={gap_to_ess_sup:.3e}")

    return {
        "gammas": gammas.tolist(),
        "phi_values": phi_vals.tolist(),
        "is_nondecreasing": is_nondecreasing,
        "is_concave": is_concave,
        "ess_sup_X": ess_sup_X,
        "gap_at_largest_gamma": gap_to_ess_sup,
    }


def make_plot(convergence: dict, phi_check: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(convergence["m_grid"], convergence["gaps"], marker="o", color="tab:red")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("number of agents M (log scale)")
    axes[0].set_ylabel(r"gap: brute-force value $-\ ES^{\beta}(X)$")
    axes[0].set_title("Theorem 4(2): gap shrinks but never closes")
    axes[0].grid(alpha=0.3)

    axes[1].plot(phi_check["gammas"], phi_check["phi_values"], color="tab:green")
    axes[1].axhline(phi_check["ess_sup_X"], color="k", linestyle="--", linewidth=1,
                     label=r"$P\mathrm{-ess\,sup}\,X$")
    axes[1].set_xlabel(r"inflation parameter $\gamma$")
    axes[1].set_ylabel(r"$\varphi(\gamma)=ES^{1/\gamma}(X)$")
    axes[1].set_title("Proposition 5: monotone, concave, limit at ess sup X")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(results_path("exp2_inflated_risk_sharing.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    P = uniform_probability(config.EXP2_N_OMEGA)
    X = uniform_aggregate_endowment(config.EXP2_N_OMEGA, config.SEED)

    continuum_gap = run_continuum_gap_case(X, P)
    convergence = run_convergence_sweep(X, P)
    atom_existence = run_atom_existence_case(X, P)
    rockafellar_uryasev = run_rockafellar_uryasev_check(X, P)
    phi_check = run_phi_monotonicity_check(X, P)
    make_plot(convergence, phi_check)

    payload = {
        "experiment": "exp2_inflated_risk_sharing",
        "theorem": "Theorem 3 / Theorem 4 (inflated / expected-shortfall risk measures)",
        "continuum_gap_case": continuum_gap,
        "convergence_sweep": convergence,
        "atom_existence_case": atom_existence,
        "rockafellar_uryasev_check": rockafellar_uryasev,
        "phi_monotonicity_check": phi_check,
    }
    path = save_json("exp2_metrics.json", payload)
    print(f"[exp2] wrote {path}")


if __name__ == "__main__":
    main()
