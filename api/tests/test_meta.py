# api/tests/test_meta.py
def test_health_returns_ok(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["db"] == "connected"


def test_model_version_returns_fields(client):
    resp = client.get("/api/v1/model/version")
    assert resp.status_code == 200
    data = resp.json()
    assert "version_id" in data
    assert "k" in data
    assert data["k"] == 3  # TEST_K
    assert "holdout_r" in data
    assert "created_at" in data
