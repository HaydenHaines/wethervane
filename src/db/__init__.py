"""Database layer for the WetherVane pipeline.

The source of truth for all model outputs is data/wethervane.duckdb.
Parquets in data/shifts/, data/communities/, data/predictions/ remain as
intermediate pipeline artifacts; DuckDB is built from them by build_database.py.

FastAPI reads from DuckDB directly via SQL queries.
"""
