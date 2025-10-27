import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app  
from fastapi.testclient import TestClient

client = TestClient(app)

def test_happy_path():
    response = client.post("/predict", json={"level": "medium", "text": "I really enjoy this amazing day"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == 0

def test_foul_example():
    response = client.post("/predict", json={"level": "high", "text": "You are an idiot"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in [0,1]

def test_invalid_input():
    response = client.post("/predict", json={"text_only": "Missing 'level' field"})
    assert response.status_code == 422
