import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_valid():
    response = client.post("/query", json={
        "question": "Do mitochondria play a role in programmed cell death?",
        "n_results": 3
    })
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "low_confidence" in data
    assert "latency_ms" in data
    assert len(data["sources"]) > 0
    assert isinstance(data["low_confidence"], bool)
    assert data["latency_ms"] > 0


def test_query_too_short():
    response = client.post("/query", json={
        "question": "hi",
        "n_results": 3
    })
    assert response.status_code == 422


def test_query_too_many_results():
    response = client.post("/query", json={
        "question": "Do mitochondria play a role in programmed cell death?",
        "n_results": 99
    })
    assert response.status_code == 422


def test_query_missing_question():
    response = client.post("/query", json={
        "n_results": 3
    })
    assert response.status_code == 422


def test_low_confidence_flag():
    response = client.post("/query", json={
        "question": "What is the best pizza topping in the world?",
        "n_results": 3
    })
    assert response.status_code == 200
    data = response.json()
    assert data["low_confidence"] is True