# api/tests/test_correlated_types.py
"""Tests for GET /types/{type_id}/correlated."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient


class TestCorrelatedTypes:
    def test_returns_correlated_list(self, client: TestClient):
        resp = client.get("/api/v1/types/0/correlated")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        # 4 types total → at most 3 correlated (n=5 capped to J-1=3)
        assert len(data) <= 3

    def test_response_has_required_fields(self, client: TestClient):
        resp = client.get("/api/v1/types/0/correlated")
        assert resp.status_code == 200
        for item in resp.json():
            assert "type_id" in item
            assert "display_name" in item
            assert "super_type_id" in item
            assert "n_counties" in item
            assert "correlation" in item
            assert isinstance(item["correlation"], float)

    def test_self_excluded(self, client: TestClient):
        """The query type_id must not appear in its own correlated list."""
        resp = client.get("/api/v1/types/0/correlated")
        assert resp.status_code == 200
        ids = [item["type_id"] for item in resp.json()]
        assert 0 not in ids

    def test_correlation_values_bounded(self, client: TestClient):
        """Correlations from a symmetric positive-semidefinite matrix are in [-1, 1]."""
        resp = client.get("/api/v1/types/0/correlated")
        assert resp.status_code == 200
        for item in resp.json():
            assert -1.0 <= item["correlation"] <= 1.0

    def test_results_sorted_descending(self, client: TestClient):
        """Most correlated type comes first."""
        resp = client.get("/api/v1/types/0/correlated")
        assert resp.status_code == 200
        data = resp.json()
        if len(data) > 1:
            correlations = [item["correlation"] for item in data]
            assert correlations == sorted(correlations, reverse=True)

    def test_n_param_respected(self, client: TestClient):
        """?n=1 should return at most 1 result."""
        resp = client.get("/api/v1/types/0/correlated?n=1")
        assert resp.status_code == 200
        assert len(resp.json()) <= 1

    def test_404_for_out_of_range(self, client: TestClient):
        """type_id beyond matrix size returns 404."""
        resp = client.get("/api/v1/types/999/correlated")
        assert resp.status_code == 404

    def test_503_when_correlation_not_loaded(self, client: TestClient):
        """Returns 503 when type_correlation is None (matrix not loaded at startup)."""
        # Temporarily unset the matrix
        original = client.app.state.type_correlation
        client.app.state.type_correlation = None
        try:
            resp = client.get("/api/v1/types/0/correlated")
            assert resp.status_code == 503
        finally:
            client.app.state.type_correlation = original
