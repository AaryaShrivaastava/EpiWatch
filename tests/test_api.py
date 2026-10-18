"""Tests for the FastAPI module."""
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_get_regions():
    response = client.get("/api/v1/regions")
    assert response.status_code == 200
    assert "data" in response.json()

def test_get_alerts():
    response = client.get("/api/v1/alerts")
    assert response.status_code == 200
    assert "data" in response.json()
