#!/bin/sh
set -eu
PK=/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/m01-six-parent-witness-sol-v1
TMP=$(mktemp -d /tmp/luna-six-tol-XXXXXX)
cp -a "$PK" "$TMP/packet"
PYTHONDONTWRITEBYTECODE=1 nice -n 15 /home/hayden/projects/ballotmancy-improvement-lab/.improvement-ledger/runtime/calibration-v1-env/bin/python -B "$(dirname "$0")/probe.py" "$TMP/packet"
