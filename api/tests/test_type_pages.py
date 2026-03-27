# api/tests/test_type_pages.py
"""Tests for type detail page API requirements.

Covers /api/v1/types/{id} response structure, 404 handling, county data,
demographics, and sitemap type page inclusion.
"""


class TestTypeDetailStructure:
    def test_type_detail_has_all_required_fields(self, client):
        resp = client.get("/api/v1/types/0")
        assert resp.status_code == 200
        data = resp.json()
        required_fields = {
            "type_id",
            "super_type_id",
            "display_name",
            "n_counties",
            "counties",
            "demographics",
        }
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_type_detail_counties_include_fips_name_state(self, client):
        resp = client.get("/api/v1/types/0")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["counties"]) > 0
        for county in data["counties"]:
            assert "county_fips" in county
            assert "county_name" in county
            assert "state_abbr" in county
            assert len(county["county_fips"]) == 5

    def test_type_detail_county_count_matches_list(self, client):
        resp = client.get("/api/v1/types/0")
        assert resp.status_code == 200
        data = resp.json()
        # n_counties must equal actual length of counties list
        assert data["n_counties"] == len(data["counties"])

    def test_type_detail_demographics_is_dict(self, client):
        resp = client.get("/api/v1/types/0")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["demographics"], dict)
        # At least some demographic fields should be present
        assert len(data["demographics"]) > 0
        # All values must be numeric
        for key, val in data["demographics"].items():
            assert isinstance(val, (int, float)), f"Non-numeric demographic value for {key}: {val}"

    def test_type_detail_narrative_field_present(self, client):
        """narrative field must be present (may be null or a string)."""
        resp = client.get("/api/v1/types/0")
        assert resp.status_code == 200
        data = resp.json()
        assert "narrative" in data
        # In the test fixture, narrative is set to a string
        assert data["narrative"] is None or isinstance(data["narrative"], str)

    def test_type_detail_mean_pred_dem_share_present(self, client):
        """mean_pred_dem_share should be present (numeric or null)."""
        resp = client.get("/api/v1/types/0")
        assert resp.status_code == 200
        data = resp.json()
        assert "mean_pred_dem_share" in data
        if data["mean_pred_dem_share"] is not None:
            assert isinstance(data["mean_pred_dem_share"], float)

    def test_type_detail_super_type_id_is_integer(self, client):
        resp = client.get("/api/v1/types/0")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["super_type_id"], int)


class TestTypeDetailNotFound:
    def test_invalid_type_returns_404(self, client):
        resp = client.get("/api/v1/types/999")
        assert resp.status_code == 404

    def test_negative_type_returns_404_or_422(self, client):
        # FastAPI may return 422 for path param validation or 404 from endpoint
        resp = client.get("/api/v1/types/-1")
        assert resp.status_code in (404, 422)

    def test_all_fixture_types_accessible(self, client):
        """Every type in the test fixture should return 200."""
        for type_id in range(4):  # 4 types in test fixture (IDs 0-3)
            resp = client.get(f"/api/v1/types/{type_id}")
            assert resp.status_code == 200, f"Type {type_id} returned {resp.status_code}"


class TestTypeDetailCountyData:
    def test_type_counties_linked_to_correct_type(self, client):
        """Counties returned for type 3 should only include counties assigned to type 3."""
        resp = client.get("/api/v1/types/3")
        assert resp.status_code == 200
        data = resp.json()
        # In test fixture, only 12001 is assigned to type 3
        assert data["n_counties"] == 1
        assert data["counties"][0]["county_fips"] == "12001"

    def test_type_counties_state_abbr_is_valid(self, client):
        resp = client.get("/api/v1/types/0")
        assert resp.status_code == 200
        data = resp.json()
        for county in data["counties"]:
            assert len(county["state_abbr"]) == 2
            assert county["state_abbr"].isupper()

    def test_different_types_have_different_counties(self, client):
        """No county should appear in two different types' county lists."""
        all_fips: set[str] = set()
        for type_id in range(4):
            resp = client.get(f"/api/v1/types/{type_id}")
            assert resp.status_code == 200
            for county in resp.json()["counties"]:
                fips = county["county_fips"]
                assert fips not in all_fips, f"County {fips} appears in multiple types"
                all_fips.add(fips)


class TestSuperTypesForTypePage:
    def test_super_types_endpoint_returns_display_names(self, client):
        """The type page fetches super-types to display the super-type name."""
        resp = client.get("/api/v1/super-types")
        assert resp.status_code == 200
        data = resp.json()
        for st in data:
            assert "super_type_id" in st
            assert "display_name" in st
            assert isinstance(st["display_name"], str)
            assert len(st["display_name"]) > 0

    def test_super_type_covers_all_type_ids(self, client):
        """Every type's super_type_id must appear in the super-types list."""
        types_resp = client.get("/api/v1/types")
        assert types_resp.status_code == 200
        super_resp = client.get("/api/v1/super-types")
        assert super_resp.status_code == 200

        super_ids = {st["super_type_id"] for st in super_resp.json()}
        for t in types_resp.json():
            assert t["super_type_id"] in super_ids, (
                f"Type {t['type_id']} references unknown super_type_id {t['super_type_id']}"
            )
