"""
Tests for API health endpoints.
"""
import pytest
from fastapi.testclient import TestClient

from api.app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/healthz")
    
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_missing_file(client):
    """Test predict endpoint without file."""
    response = client.post("/predict")
    
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_invalid_file(client):
    """Test predict endpoint with invalid file."""
    # Create a dummy text file
    files = {"file": ("test.txt", "not an audio file", "text/plain")}
    response = client.post("/predict", files=files)
    
    # Should return 400 due to audio processing error
    assert response.status_code == 400
    assert "error" in response.json()
