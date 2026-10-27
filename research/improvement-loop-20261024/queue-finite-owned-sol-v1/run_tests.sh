#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec nice -n 15 python3 -B run_tests.py
