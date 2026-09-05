"""
Обучение модели бинарной классификации тональности отзывов.

"""

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from utils import clean, load_config, setup_logging

logger = logging.getLogger(__name__)

# Вспомогательные функции

def load_data(data_path: str) -> pd.DataFrame:
    """
    Загружает датасет отзывов.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл данных не найден: {data_path}")

    df = pd.read_csv(path)
    logger.info("Загружено %d строк, столбцы: %s", len(df), list(df.columns))

    for col in ("review", "sentiment"):
        if col not in df.columns:
            raise ValueError(f"Столбец '{col}' отсутствует в файле данных")

    before = len(df)
    df = df.dropna(subset=["review", "sentiment"])
    if len(df) < before:
        logger.warning("Удалено %d строк с пропусками", before - len(df))

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Применяет функцию очистки текста к столбцу review."""
    logger.info("Предобработка текстов...")
    df = df.copy()
    df["clean"] = df["review"].apply(clean)
    empty = (df["clean"].str.strip() == "").sum()
    if empty:
        logger.warning("%d отзывов стали пустыми после очистки", empty)
    logger.info("Предобработка завершена")
    return df

# Основной пайплайн обучения

def train(config: dict) -> None:
    # --- Параметры из конфига ---
    data_path     = config["data"]["path"]
    test_size     = config["data"]["test_size"]
    random_state  = config["data"]["random_state"]
    tfidf_params  = config["tfidf"]
    model_type    = config["model"]["type"]
    model_params  = config["model"].get("params", {})
    model_path    = Path(config["paths"]["model"])
    features_path = Path(config["paths"]["features"])
    cv_folds      = config.get("training", {}).get("cv_folds", 5)

    # 1. Загрузка и предобработка данных
    logger.info("=" * 60)
    logger.info("ЭТАП 1: Загрузка данных")
    logger.info("=" * 60)
    df = load_data(data_path)
    logger.info(
        "Распределение классов:\n%s",
        df["sentiment"].value_counts().to_string(),
    )

    df = preprocess(df)

    # 2. Разделение на train / test
    logger.info("=" * 60)
    logger.info("ЭТАП 2: Train/test split (test_size=%.2f)", test_size)
    logger.info("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"],
        df["sentiment"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["sentiment"],
    )
    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

    # 3. TF-IDF векторизация
    logger.info("=" * 60)
    logger.info("ЭТАП 3: TF-IDF векторизация")
    logger.info("=" * 60)
    tfidf_params = config["tfidf"].copy()
    if "ngram_range" in tfidf_params:
        tfidf_params["ngram_range"] = tuple(tfidf_params["ngram_range"])
    vectorizer = TfidfVectorizer(**tfidf_params)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)
    feature_names = vectorizer.get_feature_names_out().tolist()
    logger.info(
        "Признаков: %d, Матрица: %s",
        len(feature_names),
        X_train_tfidf.shape,
    )

    # 4. Выбор и обучение модели
    logger.info("=" * 60)
    logger.info("ЭТАП 4: Обучение модели — %s", model_type)
    logger.info("=" * 60)
    _model_map = {
        "logistic_regression": LogisticRegression,
        "linear_svc": LinearSVC,
        "multinomial_nb": MultinomialNB,
    }
    if model_type not in _model_map:
        raise ValueError(
            f"Неизвестный тип модели: '{model_type}'. "
            f"Доступные: {list(_model_map)}"
        )

    ModelClass = _model_map[model_type]
    model = ModelClass(**model_params)
    model.fit(X_train_tfidf, y_train)
    logger.info("Модель обучена")

    # 5. Кросс-валидация
    logger.info("=" * 60)
    logger.info("ЭТАП 5: Кросс-валидация (%d-fold)", cv_folds)
    logger.info("=" * 60)
    cv_scores = cross_val_score(
        model, X_train_tfidf, y_train,
        cv=cv_folds, scoring="f1_weighted",
    )
    logger.info(
        "CV F1 (weighted): %.4f ± %.4f",
        cv_scores.mean(),
        cv_scores.std(),
    )

    # 6. Оценка на тесте
    logger.info("=" * 60)
    logger.info("ЭТАП 6: Оценка на тестовой выборке")
    logger.info("=" * 60)
    y_pred = model.predict(X_test_tfidf)

    f1        = f1_score(y_test, y_pred, pos_label="positive")
    precision = precision_score(y_test, y_pred, pos_label="positive")
    recall    = recall_score(y_test, y_pred, pos_label="positive")

    logger.info("F1-Score:  %.4f  (порог > 0.75: %s)", f1, "✓" if f1 > 0.75 else "✗")
    logger.info("Precision: %.4f  (порог > 0.70: %s)", precision, "✓" if precision > 0.7 else "✗")
    logger.info("Recall:    %.4f  (порог > 0.70: %s)", recall, "✓" if recall > 0.7 else "✗")
    logger.info("\n%s", classification_report(y_test, y_pred))

    # 7. Сохранение артефактов
    logger.info("=" * 60)
    logger.info("ЭТАП 7: Сохранение модели")
    logger.info("=" * 60)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Сохраняем словарь совместимый с SentimentModel.load()
    bundle = {
        "model_name": model_type,
        "vectorizer": vectorizer,
        "model": model,
    }
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("Модель сохранена: %s", model_path)

    with open(features_path, "wb") as f:
        pickle.dump(feature_names, f)
    logger.info("Признаки сохранены: %s", features_path)

    logger.info("=" * 60)
    logger.info("ГОТОВО")
    logger.info("=" * 60)

def main() -> None:
    """Точка входа для CLI."""
    parser = argparse.ArgumentParser(
        description="Обучение модели классификации тональности"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Путь к файлу конфигурации (default: config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("logging", {}))
    train(config)


if __name__ == "__main__":
    main()
