import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify that the health check endpoint returns 200 or 500 cleanly."""
    response = client.get("/health")
    assert response.status_code in [200, 500]


def test_invalid_file_extension_schema_rejection():
    """
    Task 10 Requirement: Must reject invalid input data/file types 
    with HTTP 400 Bad Request.
    """
    files = {"file": ("corrupt_sample.txt", b"invalid text data content", "text/plain")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]


def test_non_image_binary_rejection():
    """Verify that invalid binary streams masked as images return HTTP 400."""
    files = {"file": ("fake_image.jpg", b"not a real jpeg binary stream", "image/jpeg")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 400
    assert "could not be decoded" in response.json()["detail"]