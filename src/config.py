"""Parameters for the numerical experiments.

Every numerical choice below (number of states, number of agents, specific
risk-tolerance or quantile schedules, sample size) is not specified by the
paper "A General Theory of Risk Sharing": the paper states its theorems for a
general finite-measure agent space (Omega, F, P) and a general A-measurable
collection of risk preferences, without fixing a probability space, an agent
count, or a numerical example. Each constant below therefore carries a
comment noting the paper section whose formula it feeds and stating that its
numeric value is a modeling choice made for the numerical verification, not a
value drawn from the paper.
"""

import numpy as np

# Reproducibility seed used by every experiment's random draws.
SEED = 42  # not specified in the paper

# ---------------------------------------------------------------------------
# Experiment 1: Dilated (entropic) risk measures, Theorem 2 / Example 8 (p11-13)
# ---------------------------------------------------------------------------
EXP1_N_OMEGA = 500  # number of equiprobable states approximating L^infty(P); not specified in the paper
EXP1_M_AGENTS = 200  # number of agents discretizing the continuum A=[0,1]; not specified in the paper
EXP1_GAMMA_SLOPE = 1.0  # gamma_a = 1 + slope * a, a in [0,1] (Example 8, p13); slope not specified in the paper
EXP1_GAMMA_INTERCEPT = 1.0  # intercept of gamma_a; not specified in the paper
EXP1_N_DISCRETE_AGENTS = 5  # N for the classical (finite-agent) infimal convolution comparison; not specified in the paper

# ---------------------------------------------------------------------------
# Experiment 2: Inflated (expected shortfall) risk measures, Theorems 3-4 (p17-19)
# ---------------------------------------------------------------------------
EXP2_N_OMEGA = 1000  # equiprobable states; not specified in the paper
EXP2_M_AGENTS = 500  # agents discretizing the continuum A=[ALPHA_LOW, ALPHA_HIGH]; not specified in the paper
EXP2_ALPHA_LOW = 0.1  # lower end of the quantile-level agent space A (Example 10 setup, p22); not specified in the paper
EXP2_ALPHA_HIGH = 0.5  # upper end of A; beta_a = a ranges over (ALPHA_LOW, ALPHA_HIGH]; not specified in the paper
EXP2_ATOM_MASS = 0.3  # mu-mass placed on the atom {beta_a = ALPHA_HIGH} in the existence scenario of Theorem 4(1); not specified in the paper
EXP2_M_GRID = np.array([50, 100, 250, 500, 1000])  # discretization sizes M used to show the gap to ES^{ess sup} shrinking but never closing (Theorem 4(2)); not specified in the paper
EXP2_PHI_GAMMA_GRID = np.linspace(1.0, 20.0, 40)  # gamma grid for checking Proposition 5 monotonicity/concavity of phi(gamma)=ES^{1/gamma}(X); not specified in the paper

# ---------------------------------------------------------------------------
# Experiment 3: Non-existence of Pareto efficient allocations, Theorem 5 (p21-22)
# ---------------------------------------------------------------------------
EXP3_N_OMEGA = 1000  # equiprobable states; not specified in the paper
EXP3_M_AGENTS = 500  # agents discretizing the continuum A=[ALPHA_LOW, ALPHA_HIGH] (Example 10, p22); not specified in the paper
EXP3_ALPHA_LOW = 0.1  # equal to Example 10's "alpha-bar" lower bound; not specified in the paper
EXP3_ALPHA_HIGH = 0.5  # equal to Example 10's "alpha-bar" upper bound (essential supremum of beta_a); not specified in the paper
EXP3_PARETO_SHIFT_FRACTION = 0.02  # fraction of each agent's risk shifted toward the top-beta agents per Pareto-improvement step; not specified in the paper
EXP3_TOP_QUANTILE_FRACTION = 0.05  # fraction of agents (by beta) receiving the shifted risk; not specified in the paper
EXP3_N_FINITE_AGENTS = 10  # N for the finite-agent comparison, beta_i = ALPHA_LOW + (ALPHA_HIGH-ALPHA_LOW)*i/N (p22); not specified in the paper
EXP3_N_FINITE_GRID = np.array([5, 10, 20, 50, 100, 250])  # N -> infinity sequence showing concentration of the optimizer; not specified in the paper

# ---------------------------------------------------------------------------
# Experiment 4: Dual representation, Theorem 1 (p9) and Proposition 3 (p9-10)
# ---------------------------------------------------------------------------
EXP4_N_OMEGA = 500  # equiprobable states; not specified in the paper
EXP4_M_AGENTS = 200  # agents discretizing A=[0,1]; not specified in the paper
EXP4_GAMMA_SLOPE = 2.0  # entropic case: gamma_a = 1 + slope * a; not specified in the paper
EXP4_GAMMA_INTERCEPT = 1.0  # not specified in the paper
EXP4_ALPHA_INTERCEPT = 0.1  # ES case: alpha_a = intercept + slope * a; not specified in the paper
EXP4_ALPHA_SLOPE = 0.3  # not specified in the paper
EXP4_N_TEST_MEASURES = 25  # number of random Q << P (Dirichlet draws) used to test (5); not specified in the paper
EXP4_DIRICHLET_CONCENTRATION = 1.0  # concentration parameter of the Dirichlet distribution generating test measures; not specified in the paper

# Numerical tolerance for equality checks against theoretical formulas.
NUMERICAL_TOLERANCE = 1e-4  # solver/discretization tolerance; not specified in the paper
