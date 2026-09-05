"""
Загрузка модели и выполнение инференса (классификация тональности).

Модель хранится в формате pickle-словаря:
    {"model_name": str, "vectorizer": TfidfVectorizer, "model": <sklearn estimator>}

Пример использования:
    predictor = Predictor()
    predictor.load()
    result = predictor.predict(["This movie was great!"])
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.utils import clean, load_config

logger = logging.getLogger(__name__)


# Разрешение пути к модели
_CANDIDATE_PATHS = [
    os.getenv("MODEL_PATH"),       # переменная окружения (приоритет)
    "models/model.pkl",            # целевая структура репозитория
    "models/baseline_model.pkl",   # fallback (старое имя)
]


def _resolve_model_path() -> Path | None:
    """Возвращает первый существующий путь к файлу модели или None."""
    for candidate in _CANDIDATE_PATHS:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            return path
    return None

# Класс предиктора

class Predictor:
    """
    Обёртка над TF-IDF векторизатором и sklearn-классификатором.

    Поддерживает одиночный и пакетный инференс, возвращает
    тональность (positive / negative) и уверенность модели.

    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        config = load_config(config_path)
        self._model_path  = Path(config["paths"]["model"])
        self._features_path = Path(config["paths"]["features"])

        self.model_name: str | None = None
        self.model_path: str | None = None
        self.is_loaded:  bool = False
        self.load_error: str | None = None

        self._vectorizer = None
        self._model      = None
        self._features:  list[str] = []

    def load(self) -> None:
        """
        Загружает модель и векторизатор из pickle-файла.

        Сначала проверяется путь из config.yaml, затем кандидаты
        из переменной окружения MODEL_PATH и стандартные расположения.
        """
        # Ищем файл: сначала из конфига, затем по кандидатам
        path = self._model_path if self._model_path.is_file() else _resolve_model_path()

        if path is None:
            self.load_error = (
                "Файл модели не найден. Задайте MODEL_PATH или "
                "положите pkl в models/model.pkl"
            )
            self.is_loaded = False
            logger.error(self.load_error)
            return

        try:
            with open(path, "rb") as f:
                data: dict[str, Any] = pickle.load(f)

            self._vectorizer = data["vectorizer"]
            self._model      = data["model"]
            self.model_name  = data.get("model_name", type(self._model).__name__)
            self.model_path  = str(path)
            self.is_loaded   = True
            self.load_error  = None

            # Загружаем список признаков, если файл существует
            if self._features_path.is_file():
                with open(self._features_path, "rb") as f:
                    self._features = pickle.load(f)

            logger.info(
                "Модель загружена: %s (%d признаков)",
                self.model_name,
                len(self._features),
            )

        except Exception as exc:
            self.is_loaded  = False
            self.load_error = f"Ошибка загрузки модели: {exc}"
            logger.error(self.load_error)

    @property
    def features(self) -> list[str]:
        """Список TF-IDF признаков."""
        return list(self._features)

    def _confidence(self, features_matrix) -> np.ndarray | None:
        """
        Уверенность модели в предсказании.

        Для моделей с predict_proba — берём max класса.
        Для LinearSVC — sigmoid от расстояния до гиперплоскости.
        """
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(features_matrix).max(axis=1)
        if hasattr(self._model, "decision_function"):
            scores = np.atleast_1d(self._model.decision_function(features_matrix))
            return 1.0 / (1.0 + np.exp(-np.abs(scores)))
        return None

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Предсказывает тональность для списка текстов.

        Параметры
        ----------
        texts : list[str]
            Список текстов отзывов.

        Возвращает
        ----------
        list[dict]
            Список словарей с ключами text, sentiment, confidence (если доступна).

        Raises
        ------
        RuntimeError
            Если модель не загружена.
        """
        if not self.is_loaded:
            raise RuntimeError(self.load_error or "Модель не загружена")

        cleaned      = [clean(t) for t in texts]
        features_mat = self._vectorizer.transform(cleaned)
        labels       = self._model.predict(features_mat)
        confidences  = self._confidence(features_mat)

        results: list[dict[str, Any]] = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            item: dict[str, Any] = {"text": text, "sentiment": str(label)}
            if confidences is not None:
                item["confidence"] = round(float(confidences[i]), 4)
            results.append(item)

        return results

# Глобальный экземпляр для использования в API
predictor = Predictor()
