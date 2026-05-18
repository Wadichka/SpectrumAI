# Установка и запуск SpectrumAI

Инструкция по развёртыванию системы распознавания органических соединений по
ИК-спектрам в standalone-режиме (всё на одной машине через Docker; согласно
главе 4.6 пояснительной записки).

---

## 1. Требования

- **ОС:** Windows 10/11, macOS 12+, Linux (Ubuntu 22.04+ / любой дистрибутив с поддержкой Docker).
- **Docker:** Docker Engine 24+ или Docker Desktop 4.30+. Включает `docker compose v2`.
- **Память:** не менее **4 ГБ RAM** свободно (PostgreSQL + 2 gunicorn-воркера с PyTorch-моделями).
- **Диск:** ~6 ГБ под образы и тома (PyTorch + RDKit + transformers занимают основную часть).
- **CPU:** x86_64 / arm64 с поддержкой AVX2 желателен для FAISS.
- Опционально: GPU NVIDIA с CUDA 11.8+ для ускорения инференса (см. раздел «GPU»).

Сеть наружу нужна только при первой сборке образа (загрузка пакетов pip/npm и docker-base-образов).

---

## 2. Подготовка

```bash
git clone <repo-url> SpectrumAI
cd SpectrumAI

cp .env.example .env
# Обязательно поменяйте POSTGRES_PASSWORD на production-значение, если поднимаете в сети.
```

В каталоге `models/` должны лежать веса моделей. На фазе 1 (синтетика) там уже есть заглушки. На фазе 2 (реальные данные) скачайте релизные веса согласно `models/MANIFEST.json`.

---

## 3. Запуск

```bash
docker compose up --build -d
```

Что произойдёт:
1. Соберутся образы `backend` (multi-stage с gunicorn) и `frontend` (Vite build → nginx).
2. Поднимутся четыре сервиса: PostgreSQL, Redis, бэкенд, фронтенд (nginx).
3. Entrypoint бэкенда автоматически выполнит `alembic upgrade head` — БД получит все таблицы.
4. Gunicorn стартует с двумя воркерами `UvicornWorker` на порту 8000.

Проверка состояния:

```bash
docker compose ps
```

Все четыре контейнера должны быть в статусе `running (healthy)`.

---

## 4. Точки доступа

| URL | Что |
|---|---|
| http://localhost | Фронтенд (SPA). React Router работает корректно (try_files в nginx). |
| http://localhost/api/v1/... | REST API через nginx-прокси. |
| http://localhost:8000/docs | Swagger UI (FastAPI) — для прямого исследования API. |
| http://localhost:8000/openapi.json | OpenAPI-спека. |
| http://localhost:8000/health | Health-эндпоинт (используется в Docker healthcheck). |

PostgreSQL (5432) и Redis (6379) **не открыты** наружу: только внутри docker-сети.

---

## 5. Локальная разработка с hot-reload

Docker-стек собирает образ один раз и не подхватывает изменения в коде на лету. Для разработки с hot-reload запускайте бэкенд и фронт **из `.venv` и npm** локально, рядом с поднятыми Postgres/Redis:

```bash
# Поднять только инфраструктуру:
docker compose up -d postgres redis

# Backend (отдельный терминал):
cd backend
..\.venv\Scripts\Activate.ps1   # Windows PowerShell; на Linux/macOS: source ../.venv/bin/activate
pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --reload

# Frontend (другой терминал):
cd frontend
npm install
npm run dev
```

При таком сценарии фронтенд доступен на http://localhost:5173, бэкенд — на http://localhost:8000. В `frontend/src/api/client.ts` baseURL подхватит `VITE_API_URL=http://localhost:8000` (можно положить в `frontend/.env.local`).

---

## 6. Логи и диагностика

```bash
# Все сервисы:
docker compose logs -f

# Только бэкенд:
docker compose logs -f backend

# Логи миграций:
docker compose logs backend | grep -i alembic
```

В логах gunicorn видны структурированные логи приложения (`structlog`) и access-логи.

---

## 7. Бэкап и восстановление БД

Бэкап:

```bash
docker compose exec postgres pg_dump -U spectrumai spectrumai > backup-$(date +%F).sql
```

Восстановление:

```bash
docker compose exec -T postgres psql -U spectrumai spectrumai < backup-2026-05-18.sql
```

---

## 8. Обновление до новой версии

```bash
git pull
docker compose up --build -d
```

Миграции применяются автоматически (entrypoint). Старые контейнеры пересоздаются с новыми образами; тома (БД, Redis) сохраняются.

---

## 9. Остановка

```bash
# Остановить, сохранив тома:
docker compose down

# Остановить и удалить тома (полная очистка):
docker compose down -v
```

---

## 10. GPU-ускорение (опционально)

Если на машине есть NVIDIA GPU и установлен `nvidia-container-toolkit`:

1. В `.env` указать `ML_DEVICE=cuda`.
2. В `docker-compose.yml` добавить в сервис `backend`:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

3. Заменить базовый образ в `backend/Dockerfile` на `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime` (потребует адаптации pip-команд: убрать `--index-url https://download.pytorch.org/whl/cpu`).

Для базовой дипломной демонстрации CPU-вариант полностью соответствует требованию NFR времени отклика < 2 секунд (глава 4.10).

---

## 11. HTTPS (опционально, multi-user сценарий)

Standalone-режим работает по HTTP localhost — это соответствует главе 4.11 (требований к шифрованию для локального запуска нет). Для серверного развёртывания в локальной сети поставить перед nginx-фронтендом обратный прокси (Caddy / Traefik / nginx) с автоматическим получением сертификата Let's Encrypt либо самоподписанным сертификатом.

---

## 12. Troubleshooting

**Бэкенд падает с `connection refused` к postgres** — `depends_on` с `condition: service_healthy` уже настроен, но если БД долго инициализируется на холодной машине, повторно запустите `docker compose up -d backend` через 10–15 секунд. Можно увеличить `start_period` в healthcheck postgres.

**Фронтенд показывает «Network Error» при загрузке спектра** — проверьте, что бэкенд healthy: `docker compose ps`. И что в `frontend/nginx.conf` `proxy_pass` указывает на `http://backend:8000` (имя сервиса из compose).

**Ошибка `client intended to send too large body` от nginx** — поднимите `client_max_body_size` в `frontend/nginx.conf` (текущее значение 500M соответствует пределу батча из главы 4.11).

**Модели не загружаются** — проверьте, что `./models/` содержит файлы по путям из `ML_*` env-переменных (см. `models/MANIFEST.json`). Том монтируется как read-only.

**Долгая первая сборка** — pip скачивает torch (~190 МБ CPU-wheel), rdkit и transformers — это нормально. Кэш Docker сохраняется между сборками (BuildKit cache mount); обычная пересборка занимает 1–3 минуты.

---

## 13. Структура развёртывания

```
┌──────────────────┐         ┌────────────────┐         ┌──────────────┐
│  Браузер         │  :80    │  nginx-front   │  /api/* │  backend     │
│  http://localhost│ ──────▶ │  (SPA + proxy) │ ──────▶ │  gunicorn    │
└──────────────────┘         └────────────────┘         │  +2 workers  │
                                                        └──────┬───────┘
                                                               │
                                              ┌────────────────┼───────────────┐
                                              ▼                ▼               ▼
                                       ┌────────────┐   ┌────────────┐  ┌────────────┐
                                       │ postgres   │   │   redis    │  │  models/   │
                                       │ (volume)   │   │  (volume)  │  │  (ro vol)  │
                                       └────────────┘   └────────────┘  └────────────┘
```

Соответствует UML-диаграмме развёртывания (рисунок 4.5 пояснительной записки).
