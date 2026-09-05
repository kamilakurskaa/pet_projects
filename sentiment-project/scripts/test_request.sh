#!/usr/bin/env bash
# test_request.sh — тестовые запросы к API классификации тональности


set -euo pipefail

BASE_URL="${API_URL:-http://localhost:8000}"

POSITIVE_TEXT="This movie was absolutely fantastic and touching!"
NEGATIVE_TEXT="Terrible plot and boring acting, a complete waste of time."

# Функция для красивого вывода
print_section() {
  echo "────────────────────────────────────────────────────────────"
  echo "$1"
  echo "────────────────────────────────────────────────────────────"
}

# Проверка health
check_health() {
  print_section "🔍 GET /health"
  local response
  response=$(curl -s -w "%{http_code}" -o /tmp/health_resp.json "${BASE_URL}/health")
  local status_code=$response
  local body=$(cat /tmp/health_resp.json)

  if [[ "$status_code" != "200" ]]; then
    echo "HTTP $status_code — сервис не готов"
    echo "$body"
    return 1
  fi

  python -m json.tool <<< "$body" || echo "$body"
  echo
}

# Одиночное предсказание
predict_single() {
  print_section "POST /predict (positive sample)"
  local payload
  # Используем одинарные кавычки и подстановку переменных — это надёжно в bash
  payload="{\"text\":\"${POSITIVE_TEXT}\"}"

  curl -s -X POST "${BASE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d "$payload" \
    | python -m json.tool || echo "Ошибка ответа (не JSON)"
  echo
}

# Пакетное предсказание
predict_batch() {
  print_section "POST /predict/batch"
  local payload
  payload="{\"texts\":[\"${POSITIVE_TEXT}\",\"${NEGATIVE_TEXT}\"]}"

  curl -s -X POST "${BASE_URL}/predict/batch" \
    -H "Content-Type: application/json" \
    -d "$payload" \
    | python -m json.tool || echo "Ошибка ответа (не JSON)"
  echo
}

# --- main ---

case "${1:-}" in
  --health)
    check_health
    ;;
  --predict)
    predict_single
    ;;
  --batch)
    predict_batch
    ;;
  *)
    print_section "🚀 Тестирование Sentiment Classification API"
    echo "URL: ${BASE_URL}"
    echo
    check_health
    predict_single
    predict_batch
    echo "Все тесты выполнены"
    ;;
esac
