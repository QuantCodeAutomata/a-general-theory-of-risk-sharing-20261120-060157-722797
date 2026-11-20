"""Experiment 4: dual representation of the integral infimal convolution
(Theorem 1, p.9) and the first-order optimality condition (Proposition 3,
p.10).

Theorem 1 states that for a collection of convex risk measures (rho_a)_{a in
A}, the convex conjugate of the integral infimal convolution equals the
integral of the individual conjugates:

    (box_{a in A} rho_a mu(da))^*(Q) = int_A rho_a^*(Q) mu(da),

and consequently the value function has the dual representation

    (box_{a in A} rho_a mu(da))(X) = sup_{Q << P} ( E^Q[X] - int_A rho_a^*(Q) mu(da) ).

This is verified for two families: entropic risk measures (Example 7, p.6,
conjugate rho^*(Q) = gamma * H(Q||P)) and expected shortfall (Example 9,
p.14-15, conjugate is an indicator of {dQ/dP <= 1/alpha}). Proposition 3
(p.10) gives the first-order optimality condition: an allocation (X_a) is
optimal iff all agents share a common subgradient Q* of their own risk
measure at X_a; for the entropic family this is checked by showing the
Esscher tilt of agent a's optimal share X_a = gamma_a X / Gamma at gamma_a
equals the Esscher tilt of X at Gamma itself, independent of a.

Run as ``python -m exp.exp4``.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import bimodal_endowment
from src import config
from src.reporting import results_path, save_json
from src.risk_measures import (
    dirichlet_test_measure,
    entropic_conjugate,
    entropic_risk_measure,
    esscher_tilted_measure,
    relative_entropy,
    uniform_probability,
)
from src.risk_sharing import discretize_agents, entropic_dual_check, es_dual_check


def run_entropic_case(X: np.ndarray, P: np.ndarray, test_measures: list[np.ndarray]) -> dict:
    """Theorem 1 for the entropic family Ent^{gamma_a}, gamma_a = EXP4_GAMMA_INTERCEPT
    + EXP4_GAMMA_SLOPE * a over the discretized continuum A = [0,1]."""
    agents = discretize_agents(config.EXP4_M_AGENTS)
    gammas = config.EXP4_GAMMA_INTERCEPT + config.EXP4_GAMMA_SLOPE * agents
    Gamma = float(np.mean(gammas))

    check = entropic_dual_check(X, gammas, test_measures, P)
    value_fn = entropic_risk_measure(X, Gamma, P)

    print(f"[exp4] entropic: Gamma={Gamma:.6f}, Ent^Gamma(X)={value_fn:.6f}")
    print(f"[exp4] entropic: max|conjugate_integral - conjugate_value_fn| over "
          f"{len(test_measures)} random Q = {check.max_abs_diff:.3e}")
    print(f"[exp4] entropic: dual sup at Esscher tilt = {check.sup_representation_value:.6f}, "
          f"gap to value function = {check.sup_representation_argmax_gap:.3e}")

    return {
        "gamma_mean_Gamma": Gamma,
        "value_function_Ent_Gamma": value_fn,
        "conjugate_integral": check.conjugate_integral.tolist(),
        "conjugate_value_function": check.conjugate_value_function.tolist(),
        "max_abs_diff_conjugates": check.max_abs_diff,
        "dual_sup_value_at_esscher_tilt": check.sup_representation_value,
        "gap_sup_to_value_function": check.sup_representation_argmax_gap,
        "gammas": gammas.tolist(),
    }


def run_first_order_optimality(X: np.ndarray, P: np.ndarray) -> dict:
    """Proposition 3 (p.10): the optimal allocation X_a = gamma_a X / Gamma shares
    a single subgradient Q* across all agents. For entropic risk measures the
    subgradient of Ent^{gamma} at Y is the Esscher tilt of Y at parameter gamma;
    since X_a is a rescaling of X by gamma_a/Gamma, its Esscher tilt at gamma_a
    collapses algebraically to the Esscher tilt of X at Gamma, for every a."""
    agents = discretize_agents(config.EXP4_M_AGENTS)
    gammas = config.EXP4_GAMMA_INTERCEPT + config.EXP4_GAMMA_SLOPE * agents
    Gamma = float(np.mean(gammas))

    q_star = esscher_tilted_measure(X, Gamma, P)
    max_dev = 0.0
    fenchel_gaps = []
    for gamma_a in gammas[:: max(1, agents.shape[0] // 25)]:  # subsample agents for the printed check
        X_a = gamma_a * X / Gamma
        q_a = esscher_tilted_measure(X_a, float(gamma_a), P)
        max_dev = max(max_dev, float(np.max(np.abs(q_a - q_star))))
        # Fenchel-Young equality: rho_a(X_a) = E^{Q*}[X_a] - rho_a^*(Q*)
        lhs = entropic_risk_measure(X_a, float(gamma_a), P)
        rhs = float(np.sum(q_star * X_a)) - entropic_conjugate(float(gamma_a), q_star, P)
        fenchel_gaps.append(abs(lhs - rhs))

    max_fenchel_gap = float(np.max(fenchel_gaps))
    print(f"[exp4] Proposition 3: max |Q_a* - Q*| across sampled agents = {max_dev:.3e}")
    print(f"[exp4] Proposition 3: max Fenchel-Young gap rho_a(X_a) vs "
          f"E^Q*[X_a]-rho_a^*(Q*) = {max_fenchel_gap:.3e}")

    return {
        "max_deviation_agent_subgradient_from_common_Q_star": max_dev,
        "max_fenchel_young_gap": max_fenchel_gap,
    }


def run_es_case(P: np.ndarray, test_measures: list[np.ndarray]) -> dict:
    """Theorem 1 for the expected-shortfall family ES^{alpha_a}, alpha_a =
    EXP4_ALPHA_INTERCEPT + EXP4_ALPHA_SLOPE * a: the conjugate rho_a^*(Q) is 0 if
    dQ/dP <= 1/alpha_a and +infinity otherwise (Example 9, p.14-15), so the
    mu-integral of conjugates is 0 exactly when dQ/dP <= 1/alpha_star mu-a.e.,
    alpha_star = ess sup alpha_a -- matching the conjugate of the value function
    ES^{alpha_star} given by Theorem 3."""
    agents = discretize_agents(config.EXP4_M_AGENTS)
    alphas = config.EXP4_ALPHA_INTERCEPT + config.EXP4_ALPHA_SLOPE * agents
    result = es_dual_check(alphas, test_measures, P)
    print(f"[exp4] ES: alpha_star (ess sup) = {result['alpha_star']:.6f}; "
          f"conjugate-integral and value-function conjugate agree on all test Q: "
          f"{result['all_agree']}")
    return result


def run_measurability_check(P: np.ndarray, test_measures: list[np.ndarray]) -> dict:
    """a |-> rho_a^*(Q) measurability (p.5, Section 2.2): for fixed Q, plot the
    entropic conjugate gamma_a * H(Q||P) as a function of the agent index j; it
    is continuous (indeed affine) in a since gamma_a is affine in a, confirming
    a |-> rho_a^*(Q) is Lebesgue measurable (continuous) on A = [0,1]."""
    agents = discretize_agents(config.EXP4_M_AGENTS)
    gammas = config.EXP4_GAMMA_INTERCEPT + config.EXP4_GAMMA_SLOPE * agents
    Q = test_measures[0]
    H = relative_entropy(Q, P)
    conjugates = gammas * H
    is_affine_in_a = float(np.max(np.abs(np.diff(conjugates, 2)))) < 1e-8
    print(f"[exp4] measurability: a -> rho_a^*(Q) is affine in a (second difference "
          f"~ 0): {is_affine_in_a}")
    return {
        "agents": agents.tolist(),
        "conjugate_values": conjugates.tolist(),
        "relative_entropy_H_Q_P": H,
        "is_affine_in_a": is_affine_in_a,
    }


def make_plot(entropic: dict, measurability: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    n_test = len(entropic["conjugate_integral"])
    idx = np.arange(n_test)
    axes[0].plot(idx, entropic["conjugate_integral"], "o", label=r"$\int_A \varrho_a^*(Q)\,\mu(da)$",
                 color="tab:blue")
    axes[0].plot(idx, entropic["conjugate_value_function"], "x", label=r"$\Gamma\,H(Q\|P)$",
                 color="tab:red", markersize=9)
    axes[0].set_xlabel("test measure index")
    axes[0].set_ylabel("conjugate value")
    axes[0].set_title("Theorem 1: integral of conjugates = conjugate of value function")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(measurability["agents"], measurability["conjugate_values"], color="tab:green")
    axes[1].set_xlabel("agent a")
    axes[1].set_ylabel(r"$\varrho_a^*(Q) = \gamma_a H(Q\|P)$")
    axes[1].set_title(r"$a \mapsto \varrho_a^*(Q)$ is measurable (affine) in $a$")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(results_path("exp4_dual_representation.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    P = uniform_probability(config.EXP4_N_OMEGA)
    X = bimodal_endowment(config.EXP4_N_OMEGA, config.SEED)

    rng = np.random.default_rng(config.SEED)
    test_measures = [
        dirichlet_test_measure(config.EXP4_N_OMEGA, config.EXP4_DIRICHLET_CONCENTRATION, rng)
        for _ in range(config.EXP4_N_TEST_MEASURES)
    ]

    entropic = run_entropic_case(X, P, test_measures)
    proposition_3 = run_first_order_optimality(X, P)
    es_case = run_es_case(P, test_measures)
    measurability = run_measurability_check(P, test_measures)
    make_plot(entropic, measurability)

    payload = {
        "experiment": "exp4_dual_representation",
        "theorem": "Theorem 1 (dual representation, p.9) and Proposition 3 (first-order optimality, p.10)",
        "entropic_case": entropic,
        "first_order_optimality_proposition_3": proposition_3,
        "es_case": es_case,
        "measurability_check": measurability,
    }
    path = save_json("exp4_metrics.json", payload)
    print(f"[exp4] wrote {path}")


if __name__ == "__main__":
    main()
