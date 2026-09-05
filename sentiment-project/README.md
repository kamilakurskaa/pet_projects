# Классификация тональности отзывов

Полный цикл ML-проекта: от исследовательского анализа данных до развёртывания модели в виде REST API.

**Задача:** бинарная классификация текстовых отзывов по тональности (`positive` / `negative`).

## Датасет

- Источник: `data/texts/review.csv`
- Целевая переменная: `sentiment` — тональность отзыва (`positive` / `negative`)
- Объём: 60 000 отзывов (сбалансированный: 30 000 + 30 000)

## Структура репозитория

```text
├── README.md                      # Описание проекта
├── requirements.txt               # Зависимости Python
├── Dockerfile                     # Docker-образ для API
├── .gitignore                     # Игнорируемые файлы
├── config.yaml                    # Конфигурация (пути, гиперпараметры)
│
├── src/                           # Исходный код
│   ├── __init__.py
│   ├── train.py                   # Обучение модели
│   ├── predict.py                 # Инференс (загрузка + предсказание)
│   ├── api.py                     # FastAPI-сервис
│   └── utils.py                   # Предобработка текста и утилиты
│
├── notebooks/                     # Jupyter ноутбуки
│   └── Project_Kurskaya.ipynb      # EDA и пайплайн обучения и оценки модели
│
├── models/                        # Сериализованные модели
│   ├── model.pkl                  # Словарь {model_name, vectorizer, model}
│   └── features.pkl               # Список TF-IDF признаков
│
├── tests/                         # Тесты
│   ├── __init__.py
│   └── test_api.py                # Pytest-тесты для API
│
└── scripts/                       # Скрипты для развёртывания
    ├── deploy.sh                  # Сборка и запуск Docker
    └── test_request.sh            # Тестовые запросы к API
```

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Обучение модели

Запустите ноутбук для EDA и первичного обучения:
```bash
jupyter notebook notebooks/Project_Kurskaya.ipynb
```


### 3. Запуск API

```bash
# Через uvicorn (для разработки)
uvicorn src.api:app --reload --port 8000

# Или через Docker 
bash scripts/deploy.sh
```

### 4. Тестирование API

```bash
# Все эндпоинты сразу
bash scripts/test_request.sh

# Только health-check
bash scripts/test_request.sh --health

# Одиночное предсказание
bash scripts/test_request.sh --predict

# Пакетное предсказание
bash scripts/test_request.sh --batch
```

Или через pytest:
```bash
pytest tests/test_api.py -v
```

## Эндпоинты API

| Метод | Путь            | Описание                              |
|-------|-----------------|---------------------------------------|
| GET   | `/`             | Информация о сервисе                  |
| GET   | `/health`       | Статус сервиса и модели               |
| POST  | `/predict`      | Предсказание для одного отзыва        |
| POST  | `/predict/batch`| Пакетное предсказание (до 1000 текстов)|

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

### Пример запроса

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was absolutely fantastic!"}'
```

```json
{
  "prediction": {
    "text": "This movie was absolutely fantastic!",
    "sentiment": "positive",
    "confidence": 0.9731
  }
}
```

## Модель

- **Алгоритм:** LogisticRegression (scikit-learn), настраивается в `config.yaml`
- **Векторизация:** TF-IDF (max\_features=30 000, ngram\_range=(1, 2))
- **Метрики на тесте:**
  - F1-Score: 0.953
  - Precision: 0.956
  - Recall: 0.95

## Конфигурация

Все параметры задаются в `config.yaml`:

```yaml
model:
  type: logistic_regression  # или linear_svc / multinomial_nb
  params:
    max_iter: 1000
    random_state: 42

tfidf:
  max_features: 30000
  ngram_range: [1, 2]
```
