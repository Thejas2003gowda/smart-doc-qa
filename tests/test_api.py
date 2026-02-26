from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_ask_returns_valid_response():
    response = client.post("/ask", json={"question": "test?"})
    if response.status_code == 200:
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "model" in data
    else:
        assert response.status_code == 400