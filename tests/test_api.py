"""
Test suite for the Emotion Classifier API.

Run with:
    pytest -v

Note: these tests load the real Keras model via the app's lifespan context,
so the first test run will take a few seconds while TensorFlow initializes.
"""
import pytest
from fastapi.testclient import TestClient

from main import app

EMOTION_LABELS = {"sadness", "joy", "love", "anger", "fear", "surprise"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"]
    assert body["model_loaded"] is True


def test_predict_returns_valid_emotion(client):
    res = client.post("/predict", json={"text": "I am so happy and excited today!"})
    assert res.status_code == 200
    body = res.json()

    assert body["predicted_emotion"] in EMOTION_LABELS
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["all_probabilites"].keys()) == EMOTION_LABELS

    # probabilities should sum to ~1
    total = sum(body["all_probabilites"].values())
    assert abs(total - 1.0) < 1e-3


def test_predict_rejects_empty_text(client):
    res = client.post("/predict", json={"text": ""})
    assert res.status_code == 422  # pydantic min_length validation


def test_predict_rejects_missing_field(client):
    res = client.post("/predict", json={})
    assert res.status_code == 422


def test_predict_rejects_overlong_text(client):
    res = client.post("/predict", json={"text": "a" * 3000})
    assert res.status_code == 422  # exceeds max_length=2000


@pytest.mark.parametrize(
    "text,expected_emotion",
    [
        ("I am so happy right now, everything is wonderful", "joy"),
        ("I feel so alone and hopeless today", "sadness"),
        ("I am furious that they cancelled the trip", "anger"),
    ],
)
def test_predict_sanity_cases(client, text, expected_emotion):
    """
    Loose sanity check on a few unambiguous examples. Not a strict
    accuracy test — just guards against the model/tokenizer becoming
    completely disconnected from the API's preprocessing.
    """
    res = client.post("/predict", json={"text": text})
    assert res.status_code == 200
    assert res.json()["predicted_emotion"] == expected_emotion
