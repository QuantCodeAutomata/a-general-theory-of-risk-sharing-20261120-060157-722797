#!/usr/bin/env bash
# Reproduce every experiment of "A General Theory of Risk Sharing" from a
# clean checkout: install dependencies, run the test suite, then run all
# four experiments in sequence, overwriting the artifacts in results/.
set -eu

python -m pip install -r requirements.txt

python -m pytest -m "not market_data"

python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
python -m exp.exp4

echo "Done. See results/RESULTS.md for the comparison against the paper."
