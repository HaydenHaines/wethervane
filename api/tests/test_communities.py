# api/tests/test_communities.py
def test_list_communities_returns_k_items(client):
    resp = client.get("/api/v1/communities")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3  # TEST_K communities

def test_list_community_has_required_fields(client):
    resp = client.get("/api/v1/communities")
    item = resp.json()[0]
    assert "community_id" in item
    assert "display_name" in item
    assert "n_counties" in item
    assert "states" in item
    assert isinstance(item["states"], list)
    assert "dominant_type_id" in item

def test_community_detail_returns_profile(client):
    resp = client.get("/api/v1/communities/0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["community_id"] == 0
    assert "counties" in data
    assert len(data["counties"]) > 0
    assert "shift_profile" in data
    assert isinstance(data["shift_profile"], dict)

def test_community_detail_404_unknown(client):
    resp = client.get("/api/v1/communities/999")
    assert resp.status_code == 404
