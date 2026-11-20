# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A numerical replication of selected results from *A General Theory of Risk
Sharing*: the integral infimal convolution of convex risk measures over a
continuum of agents (Theorem 1, duality), specialized to dilated/entropic
risk measures (Section 4.1, Theorem 2) and inflated/expected-shortfall risk
measures (Section 4.2, Theorems 3-4), and the application to Pareto
efficiency with a continuum of agents (Section 5, Theorem 5). See
`README.md` for the full list of experiments and `results/RESULTS.md` for
the numbers produced by the last run and their comparison with the paper.

There is no market data anywhere in this repository: every experiment is a
numerical check of a theorem on a synthetic, finite approximation of the
paper's abstract probability space and agent space.

## Layout

```
src/            risk measures, risk-sharing / infimal-convolution machinery, config, reporting
exp/            one script per experiment (exp1..exp4), run as a module
data/           generate_synthetic.py builds the finite probability space and aggregate endowment X
tests/          pytest unit tests of src/ and of the experiment scripts
results/        JSON metrics, PNG plots, and RESULTS.md written by the experiments (overwritten on rerun)
```

`src/config.py` holds every numerical constant (state counts, agent counts,
risk-tolerance/quantile schedules) as module-level upper-case variables,
each with a comment naming the paper section it specializes and stating
whether the paper fixes that value or leaves it to the implementer.
`data/generate_synthetic.py` is the only place that builds the probability
space and the endowment `X`; no experiment script or `src/` module generates
its own input data.

## Commands

Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tests:

```bash
python -m pytest -m "not market_data"
```

(no test in this repository is marked `market_data`, since no experiment
loads market data; the flag is kept for consistency with the general
convention.)

Experiments (each overwrites its artifacts under `results/`):

```bash
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
python -m exp.exp4
```

All four at once, after the test suite:

```bash
./reproduce.sh
```

Every script accepts no paper-full/reduced flag: the configuration in
`src/config.py` already is the configuration used for the paper
comparison (finite, equiprobable discretizations are themselves the method
being checked, not a reduction of a larger run).

## Conventions

- Parameters live only in `src/config.py`, as module-level `UPPER_CASE`
  variables with a one-line comment on meaning, unit and the paper section
  (or "not specified by the paper") that motivates the value.
- Data preparation lives only in `data/generate_synthetic.py`; `src/` and
  `exp/` import it rather than generating ad hoc arrays.
- Run experiments as `python -m exp.<name>` and tests as `python -m pytest`
  from the repository root; do not edit `sys.path` in code.
- After rerunning an experiment, the files it writes under `results/` are
  overwritten; `results/RESULTS.md`'s numbers and verdicts are not
  regenerated automatically and must be checked against the new files and
  updated by hand.
- No API keys are used anywhere in this repository (no market data); do not
  add a `.env` file or network calls to the experiment scripts.
