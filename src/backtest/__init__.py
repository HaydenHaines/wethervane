# src/backtest/__init__.py
"""
Backtest harness for WetherVane.

Runs the model against historical information states and compares outputs
to known election results, producing error artifacts and markdown reports.

Usage:
    uv run python -m src.backtest run --year 2024 --race-types senate presidential
    uv run python -m src.backtest report --cutoff 2024-10-31
"""
from src.backtest.inputs import HistoricalInputs, build_historical_inputs
from src.backtest.runner import BacktestRun, run_backtest
from src.backtest.actuals import Actuals, load_actuals
from src.backtest.errors import ErrorArtifact, compute_errors
from src.backtest.catalog import BacktestCatalog
from src.backtest.report import generate_report

__all__ = [
    "HistoricalInputs",
    "build_historical_inputs",
    "BacktestRun",
    "run_backtest",
    "Actuals",
    "load_actuals",
    "ErrorArtifact",
    "compute_errors",
    "BacktestCatalog",
    "generate_report",
]
