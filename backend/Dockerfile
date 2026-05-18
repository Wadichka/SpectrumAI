# syntax=docker/dockerfile:1.6
# Образ бэкенда SpectrumAI (Этап 15, глава 4.6).
# Multi-stage: builder ставит зависимости + пакет ``app`` в /opt/venv,
# runtime берёт venv + миграции + пакет pipelines из ml/. Запуск — через
# gunicorn с UvicornWorker (ASGI); миграции накатываются entrypoint-скриптом.
#
# Build context — корень репозитория (чтобы Dockerfile видел и backend/,
# и ml/pipelines/). docker-compose.yml задаёт context: .

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Системные библиотеки для сборки rdkit/faiss/scipy.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libxrender1 \
        libxext6 \
        libsm6 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

COPY backend/pyproject.toml ./
COPY backend/app ./app

# CPU-only torch ставим первым с официального CPU-индекса PyTorch,
# иначе pip потянет ~2 ГБ CUDA-toolkit (torch ≥2.8 объявляет cuda-toolkit
# обязательной зависимостью на linux/amd64, что не нужно для CPU-инференса).
# Cache mount сохраняет pip-кэш между билдами на хост-машине.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install .


# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    SPECTRUMAI_ML_ROOT=/opt/ml

# Runtime-зависимости для rdkit/faiss/scipy + curl для healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 \
        libxext6 \
        libsm6 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

COPY --chown=app:app backend/alembic ./alembic
COPY --chown=app:app backend/alembic.ini ./alembic.ini
COPY --chown=app:app backend/docker-entrypoint.sh ./docker-entrypoint.sh
# Pipelines нужны бэкенду в рантайме (см. backend/app/ml/_ml_path.py).
COPY --chown=app:app ml/pipelines /opt/ml/pipelines
RUN chmod +x ./docker-entrypoint.sh

USER app

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
