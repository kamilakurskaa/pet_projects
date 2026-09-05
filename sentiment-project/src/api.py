"""
FastAPI-сервис для инференса модели классификации тональности отзывов.

Эндпоинты:
  GET  /         — краткая информация о сервисе
  GET  /health   — статус сервиса и модели
  POST /predict  — предсказание для одного отзыва
  POST /predict/batch — пакетное предсказание (до 1000 текстов)

"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import predictor
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)

# Pydantic-схемы

class PredictRequest(BaseModel):
    """Запрос на предсказание для одного отзыва."""

    text: str = Field(
        ..., min_length=1,
        description="Текст отзыва для классификации",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"text": "This movie was absolutely fantastic and touching!"}
        }
    }


class PredictionItem(BaseModel):
    """Результат предсказания для одного текста."""

    text:       str            = Field(..., description="Исходный текст отзыва")
    sentiment:  str            = Field(..., description="Тональность: positive / negative")
    confidence: Optional[float] = Field(
        None, description="Уверенность модели (0–1), если доступна"
    )


class PredictResponse(BaseModel):
    """Ответ эндпоинта /predict."""

    prediction: PredictionItem


class BatchPredictRequest(BaseModel):
    """Запрос на пакетное предсказание."""

    texts: List[str] = Field(
        ..., min_length=1, max_length=1000,
        description="Список текстов отзывов (не более 1000)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "texts": [
                    "One of the best films I have ever seen.",
                    "Terrible plot and boring acting, a waste of time.",
                ]
            }
        }
    }


class BatchPredictResponse(BaseModel):
    """Ответ эндпоинта /predict/batch."""

    predictions: List[PredictionItem]
    count:       int = Field(..., description="Количество обработанных текстов")


class HealthResponse(BaseModel):
    """Ответ эндпоинта /health."""

    status:       str            = Field(..., description="ok / error")
    model_loaded: bool           = Field(..., description="Загружена ли модель")
    model_name:   Optional[str]  = Field(None, description="Название модели")
    model_path:   Optional[str]  = Field(None, description="Путь к файлу модели")
    detail:       Optional[str]  = Field(None, description="Дополнительное сообщение")

# Lifespan и приложение

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружает модель один раз при старте сервиса."""
    logger.info("Загрузка модели при старте...")

    if not predictor.is_loaded:
        logger.info("Загрузка модели при старте...")
        predictor.load()

    if predictor.is_loaded:
        logger.info("Модель успешно загружена: %s", predictor.model_name)
    else:
        logger.warning("Модель не загружена: %s", predictor.load_error)
    yield


config     = load_config()
api_config = config.get("api", {})
setup_logging(config.get("logging", {}))

app = FastAPI(
    title=api_config.get("title", "Sentiment Classification API"),
    description=(
        "REST-обёртка над ML-моделью бинарной классификации тональности "
        "текстовых отзывов (positive / negative)."
    ),
    version=api_config.get("version", "1.0.0"),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Эндпоинты

@app.get("/", tags=["service"])
def root() -> dict:
    """Краткая информация о сервисе и доступных эндпоинтах."""
    return {
        "service": "Sentiment Classification API",
        "endpoints": ["/health", "/predict", "/predict/batch", "/docs"],
    }


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Проверка состояния сервиса и статуса загруженной модели."""
    if predictor.is_loaded:
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=predictor.model_name,
            model_path=predictor.model_path,
        )
    return HealthResponse(
        status="error",
        model_loaded=False,
        detail=predictor.load_error,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest) -> PredictResponse:
    """
    Предсказать тональность одного текстового отзыва.

    Возвращает класс (positive / negative) и уверенность модели.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=predictor.load_error or "Модель не загружена",
        )
    try:
        result = predictor.predict([request.text])[0]
        logger.info("predict: sentiment=%s confidence=%s", result["sentiment"], result.get("confidence"))
        return PredictResponse(prediction=PredictionItem(**result))
    except Exception as exc:
        logger.error("Ошибка предсказания: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """
    Пакетное предсказание тональности для набора текстов (до 1000).
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=predictor.load_error or "Модель не загружена",
        )
    try:
        results = predictor.predict(request.texts)
        logger.info("predict_batch: %d текстов обработано", len(results))
        return BatchPredictResponse(
            predictions=[PredictionItem(**r) for r in results],
            count=len(results),
        )
    except Exception as exc:
        logger.error("Ошибка пакетного предсказания: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
