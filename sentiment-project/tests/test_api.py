import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api import app


@pytest.fixture
def client():
    """Синхронный клиент для тестирования FastAPI."""
    yield TestClient(app)


@pytest.fixture
def mock_predictor_loaded():
    """Мок predictor, где модель загружена."""
    mock = MagicMock()
    mock.is_loaded = True
    mock.model_name = "sentiment_model_v1"
    mock.model_path = "models/model.pkl"
    mock.load_error = None

    def predict_side_effect(texts):
        results = []
        for t in texts:
            # Простая эвристика для мока: если есть "good"/"fantastic" -> positive, иначе negative
            sentiment = "positive" if any(w in t.lower() for w in ["good", "fantastic", "great", "excellent"]) else "negative"
            results.append({
                "text": t,
                "sentiment": sentiment,
                "confidence": 0.97
            })
        return results

    mock.predict = MagicMock(side_effect=predict_side_effect)
    return mock


@pytest.fixture
def mock_predictor_not_loaded():
    """Мок predictor, где модель НЕ загружена."""
    mock = MagicMock()
    mock.is_loaded = False
    mock.model_name = None
    mock.model_path = None
    mock.load_error = "Model not found"
    mock.predict.side_effect = RuntimeError("Model not loaded")
    return mock


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Sentiment Classification API"
    assert "/health" in data["endpoints"]
    assert "/predict" in data["endpoints"]


def test_health_ok(client, mock_predictor_loaded):
    with patch("src.api.predictor", mock_predictor_loaded):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["model_name"] == "sentiment_model_v1"


def test_health_error(client, mock_predictor_not_loaded):
    with patch("src.api.predictor", mock_predictor_not_loaded):
        response = client.get("/health")
        assert response.status_code == 200  # health всегда 200, даже если модель не загружена
        data = response.json()
        assert data["status"] == "error"
        assert data["model_loaded"] is False
        assert "detail" in data


def test_predict_ok(client, mock_predictor_loaded):
    payload = {"text": "This movie was absolutely fantastic and touching!"}

    with patch("src.api.predictor", mock_predictor_loaded):
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    prediction = data["prediction"]
    assert prediction["text"] == payload["text"]
    assert prediction["sentiment"] in ("positive", "negative")
    assert "confidence" in prediction


def test_predict_validation_error_empty_text(client, mock_predictor_loaded):
    payload = {"text": ""}  # нарушает min_length=1 в Pydantic

    with patch("src.api.predictor", mock_predictor_loaded):
        response = client.post("/predict", json=payload)

    assert response.status_code == 422  # Pydantic validation error


def test_predict_service_unavailable_when_model_not_loaded(client, mock_predictor_not_loaded):
    payload = {"text": "Good movie"}

    with patch("src.api.predictor", mock_predictor_not_loaded):
        response = client.post("/predict", json=payload)

    assert response.status_code == 503  # HTTP_503_SERVICE_UNAVAILABLE


def test_predict_batch_ok(client, mock_predictor_loaded):
    payload = {
        "texts": [
            "One of the best films I have ever seen.",
            "Terrible plot and boring acting, a waste of time."
        ]
    }

    with patch("src.api.predictor", mock_predictor_loaded):
        response = client.post("/predict/batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
    assert data["count"] == 2


def test_predict_batch_validation_error_too_many_texts(client, mock_predictor_loaded):
    # Pydantic max_length=1000 на списке texts
    payload = {"texts": ["Good"] * 1001}

    with patch("src.api.predictor", mock_predictor_loaded):
        response = client.post("/predict/batch", json=payload)

    assert response.status_code == 422


def test_predict_batch_service_unavailable_when_model_not_loaded(client, mock_predictor_not_loaded):
    payload = {"texts": ["Good movie"]}

    with patch("src.api.predictor", mock_predictor_not_loaded):
        response = client.post("/predict/batch", json=payload)

    assert response.status_code == 503
