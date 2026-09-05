FROM python:3.11-slim

# Отключаем .pyc и буферизацию вывода; задаём путь к NLTK-данным и модели
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/local/nltk_data \
    MODEL_PATH=/app/models/model.pkl

WORKDIR /app

# Устанавливаем зависимости отдельным слоем для кэширования
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скачиваем стоп-слова NLTK заранее — контейнеру не нужна сеть в рантайме
RUN python -m nltk.downloader -d /usr/local/nltk_data stopwords

# Копируем исходный код, конфиг и сериализованную модель
COPY src/       ./src/
COPY models/    ./models/
COPY tests/      ./tests/
COPY config.yaml .

EXPOSE 8000

# Проверка здоровья контейнера через эндпоинт /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c \
    "import urllib.request, sys; \
     sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
