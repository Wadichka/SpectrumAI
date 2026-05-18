#!/bin/sh
# Production entrypoint: накатывает миграции БД и запускает gunicorn.
# Должен быть исполняемым (chmod +x). Для Windows-хоста git-флаг
# проставляется через `git update-index --chmod=+x backend/docker-entrypoint.sh`.

set -e

echo "[entrypoint] Running alembic upgrade head..."
alembic upgrade head

echo "[entrypoint] Starting gunicorn (workers=${GUNICORN_WORKERS:-2})..."
exec gunicorn app.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers "${GUNICORN_WORKERS:-2}" \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
