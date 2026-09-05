#!/usr/bin/env bash
# deploy.sh — сборка и запуск Docker-образа сервиса классификации тональности

set -euo pipefail

IMAGE_NAME="sentiment-api"
PORT="${PORT:-8000}"

build_image() {
    echo "Сборка Docker-образа: ${IMAGE_NAME}..."
    docker build -t "${IMAGE_NAME}" .
    echo "Образ собран: ${IMAGE_NAME}"
}

run_container() {
    echo "Запуск контейнера на порту ${PORT}..."
    docker run -d \
        --name "${IMAGE_NAME}" \
        -p "${PORT}:8000" \
        --restart unless-stopped \
        "${IMAGE_NAME}"
    echo "Контейнер запущен: http://localhost:${PORT}"
    echo "Swagger UI:        http://localhost:${PORT}/docs"
}

stop_container() {
    echo "Остановка контейнера ${IMAGE_NAME}..."
    docker stop "${IMAGE_NAME}" 2>/dev/null || true
    docker rm   "${IMAGE_NAME}" 2>/dev/null || true
    echo "Контейнер остановлен"
}

case "${1:-}" in
    --build-only) build_image ;;
    --run-only)   run_container ;;
    --stop)       stop_container ;;
    --restart)    stop_container; build_image; run_container ;;
    *)            build_image; run_container ;;
esac
