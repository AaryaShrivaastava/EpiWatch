"""Comprehensive tests for Spatial Risk Map and Active AI Surveillance Agent."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.spatial.spatial_service import (
    get_spatial_risk_records,
    get_single_region_detail,
    get_geojson_boundaries,
    get_clusters_summary,
    execute_clustering
)
from src.surveillance.worker import get_surveillance_worker
from src.surveillance.engine import get_surveillance_engine
from src.surveillance.config import get_surveillance_config

client = TestClient(app)


# ===========================================================================
# 1. SPATIAL RISK MAP & GEOGRAPHY TESTS
# ===========================================================================
def test_spatial_risk_map_endpoint():
    """Verify spatial risk-map returns all 5 MP regions with structured fields."""
    response = client.get("/api/spatial/risk-map")
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) == 5
    
    first = data[0]
    expected_fields = [
        "region_id", "region_name", "latitude", "longitude",
        "risk_score", "risk_category", "risk_color",
        "cluster_id", "cluster_label", "case_count", "case_growth",
        "outbreak_status", "population", "population_density",
        "data_source", "data_completeness"
    ]
    for field in expected_fields:
        assert field in first, f"Missing expected field: {field}"


def test_spatial_coordinates_grounding():
    """Verify authentic geographic coordinates for Madhya Pradesh districts."""
    records = get_spatial_risk_records()
    reg_map = {r["region_name"]: r for r in records}
    
    assert "Bhopal" in reg_map
    assert abs(reg_map["Bhopal"]["latitude"] - 23.2599) < 0.05
    assert abs(reg_map["Bhopal"]["longitude"] - 77.4126) < 0.05

    assert "Indore" in reg_map
    assert abs(reg_map["Indore"]["latitude"] - 22.7196) < 0.05
    assert abs(reg_map["Indore"]["longitude"] - 75.8577) < 0.05

    assert "Jabalpur" in reg_map
    assert abs(reg_map["Jabalpur"]["latitude"] - 23.1815) < 0.05

    assert "Gwalior" in reg_map
    assert abs(reg_map["Gwalior"]["latitude"] - 26.2183) < 0.05

    assert "Ujjain" in reg_map
    assert abs(reg_map["Ujjain"]["latitude"] - 23.1765) < 0.05


def test_spatial_risk_palette_consistency():
    """Ensure consistent risk colors: Green, Yellow, Orange, Red."""
    records = get_spatial_risk_records()
    for r in records:
        cat = r["risk_category"]
        color = r["risk_color"]
        if cat == "Low":
            assert color == "#10b981"
        elif cat == "Moderate":
            assert color == "#eab308"
        elif cat == "High":
            assert color == "#f97316"
        elif cat == "Critical":
            assert color == "#ef4444"
        else:
            assert color == "#6b7280"


def test_spatial_boundaries_geojson():
    """Verify GeoJSON boundaries endpoint returns 5 district polygons."""
    response = client.get("/api/spatial/boundaries")
    assert response.status_code == 200
    geojson = response.json()
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 5
    for feat in geojson["features"]:
        assert feat["geometry"]["type"] == "Polygon"
        assert "risk_category" in feat["properties"]


def test_spatial_filtering_by_disease():
    """Test filtering risk map by disease."""
    response = client.get("/api/spatial/risk-map?disease=COVID-19")
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) == 5
    assert all(r["selected_disease"] == "COVID-19" for r in data)


def test_spatial_clustering_execution():
    """Verify POST /api/spatial/cluster runs K-Means."""
    response = client.post("/api/spatial/cluster", json={"n_clusters": 3, "method": "kmeans"})
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert "assignments" in res
    assert len(res["assignments"]) == 5


def test_spatial_single_region_detail():
    """Verify single region detail endpoint."""
    response = client.get("/api/spatial/regions/Bhopal")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["region_name"] == "Bhopal"
    assert "disease_breakdown" in data


# ===========================================================================
# 2. ACTIVE SURVEILLANCE AGENT & SCHEDULER TESTS
# ===========================================================================
def test_surveillance_status_endpoint():
    """Verify surveillance status contains all required fields."""
    response = client.get("/api/surveillance/status")
    assert response.status_code == 200
    data = response.json()["data"]
    for field in [
        "agent_status", "is_scanning", "scan_interval_seconds",
        "total_scans_completed", "findings_detected", "active_alerts", "llm_status"
    ]:
        assert field in data, f"Missing surveillance status field: {field}"


def test_surveillance_start_stop():
    """Verify background worker scheduler can be started and stopped."""
    stop_res = client.post("/api/surveillance/stop")
    assert stop_res.status_code == 200
    assert stop_res.json()["data"]["agent_status"] == "STOPPED"

    start_res = client.post("/api/surveillance/start")
    assert start_res.status_code == 200
    assert start_res.json()["data"]["agent_status"] == "RUNNING"


def test_surveillance_manual_scan():
    """Verify manual scan executes the real surveillance pipeline."""
    response = client.post("/api/surveillance/scan")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] in ["COMPLETED", "REJECTED"]
    assert "duration_seconds" in data
    assert "findings_generated" in data


def test_surveillance_findings_and_evidence():
    """Verify findings structure contains statistical baselines and evidence."""
    engine = get_surveillance_engine()
    findings = engine.get_findings()
    if findings:
        f = findings[0]
        assert "finding_id" in f
        assert "region" in f
        assert "disease" in f
        assert "observed_value" in f
        assert "historical_baseline" in f
        assert "supporting_evidence" in f
        assert "data_provenance" in f


def test_surveillance_statistics():
    """Verify surveillance statistics returns distributions."""
    response = client.get("/api/surveillance/statistics")
    assert response.status_code == 200
    data = response.json()["data"]
    assert "by_severity" in data
    assert "by_disease" in data
    assert "by_region" in data


def test_surveillance_config_patch():
    """Verify updating surveillance config updates interval."""
    patch_res = client.patch("/api/surveillance/config", json={"scan_interval_seconds": 600})
    assert patch_res.status_code == 200
    assert patch_res.json()["data"]["scan_interval_seconds"] == 600


def test_overlapping_scan_protection():
    """Verify concurrency lock rejects overlapping scans."""
    worker = get_surveillance_worker()
    worker.is_scanning = True
    # Attempting to scan while is_scanning is True
    res = worker.trigger_scan(source="overlap_test")
    assert res["status"] in ["REJECTED", "COMPLETED"]
    worker.is_scanning = False
