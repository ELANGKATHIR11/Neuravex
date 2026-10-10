"""
Integration tests for the decoupled Neuravex FastAPI REST API layer.
"""

import pytest
from fastapi.testclient import TestClient

from neuravex.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.3.0"
    assert "device" in data
    assert data["ready"] is True


def test_api_ready_endpoint(client):
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"


def test_api_capabilities_endpoint(client):
    response = client.get("/api/v1/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "cuda_available" in data
    assert "supported_models" in data
    assert "nano" in data["supported_models"]
    assert "omni" in data["supported_models"]
    assert "tasks" in data


def test_tracking_session_lifecycle(client):
    # 1. Create session
    create_resp = client.post("/api/v1/tracking/sessions", json={"session_id": "test_sess_01", "model_size": "pico"})
    assert create_resp.status_code == 200
    data = create_resp.json()
    assert data["session_id"] == "test_sess_01"

    # 2. Duplicate session creation should conflict
    dup_resp = client.post("/api/v1/tracking/sessions", json={"session_id": "test_sess_01"})
    assert dup_resp.status_code == 409

    # 3. Delete session
    del_resp = client.delete("/api/v1/tracking/sessions/test_sess_01")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # 4. Deleting non-existent session should 404
    del_missing = client.delete("/api/v1/tracking/sessions/non_existent_123")
    assert del_missing.status_code == 404
