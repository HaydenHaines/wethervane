#!/bin/sh
set -eu
PK=/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/m01-remaining-fixed-parent-sol-v1
TMP=$(mktemp -d /tmp/luna-six-review-XXXXXX)
cp "$PK/common.py" "$PK/selection.py" "$PK/compare.py" "$PK/baseline_compare.py" "$PK/evaluator.py" "$PK/contract.json" "$TMP/"
cp "$(dirname "$0")/reproduce.py" "$TMP/"
cd "$TMP"
PYTHONDONTWRITEBYTECODE=1 nice -n 15 /home/hayden/projects/ballotmancy-improvement-lab/.improvement-ledger/runtime/calibration-v1-env/bin/python -B reproduce.py
