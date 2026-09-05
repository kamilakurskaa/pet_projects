"""
Вспомогательные функции проекта классификации тональности отзывов.

"""

import logging
import re
import string
from pathlib import Path

import nltk
import yaml

logger = logging.getLogger(__name__)


def setup_logging(logging_config: dict | None = None) -> None:
    """
    Настраивает базовое логирование.
    """
    if logging_config is None:
        logging_config = {}

    level = getattr(
        logging,
        logging_config.get("level", "INFO").upper(),
        logging.INFO,
    )
    fmt = logging_config.get(
        "format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logging.basicConfig(level=level, format=fmt)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Загружает конфигурацию из YAML-файла.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

def _load_stopwords() -> set:
    """Загружает стоп-слова NLTK, при необходимости скачивая их."""
    try:
        from nltk.corpus import stopwords
        base = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        base = set(stopwords.words("english"))
    return base


# Кастомные стоп-слова: часто встречаются в обоих классах и не несут тональности
CUSTOM_STOPWORDS: set[str] = {
    "film", "movie", "one", "would", "could",
    "really", "even", "get", "make", "see",
    "also", "first", "people", "much", "like",
}

STOP_WORDS: set[str] = _load_stopwords().union(CUSTOM_STOPWORDS)

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean(text: str) -> str:
    """
    Очищает текст
    """
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)       # HTML-теги
    text = re.sub(r"http\S+", " ", text)     # URL
    text = text.translate(_PUNCT_TABLE)       # пунктуация
    text = " ".join(w for w in text.split() if w not in STOP_WORDS)
    return text
