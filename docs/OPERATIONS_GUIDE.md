# Руководство по эксплуатации SpectrumAI

Документ для администратора / оператора SpectrumAI: ежедневные операции,
обслуживание, обновление, аварийное восстановление (NFR-09, глава 11).
Базовая инструкция по установке — `docs/INSTALL.md`.

---

## 1. Состояние стека

Стек состоит из четырёх контейнеров, описанных в `docker-compose.yml`:

| Контейнер | Образ | Назначение | Порт наружу |
|-----------|-------|------------|-------------|
| `spectrumai-postgres` | postgres:16-alpine | Хранилище соединений, спектров, истории | — (internal) |
| `spectrumai-redis` | redis:7-alpine | Кэш | — (internal) |
| `spectrumai-backend` | spectrumai-backend (multi-stage) | FastAPI + gunicorn (UvicornWorker × 2) | 8000 |
| `spectrumai-frontend` | spectrumai-frontend (multi-stage) | nginx раздаёт SPA + proxy `/api/*` | 80 |

Проверка статуса:
```bash
docker compose ps
```

Все четыре должны быть в состоянии **`running (healthy)`**. Если хотя бы один
unhealthy — см. §6 «Troubleshooting».

---

## 2. Резервное копирование БД

PostgreSQL — единственное персистентное хранилище. Redis-кэш можно потерять
без последствий.

### 2.1. Создание дампа

```bash
docker compose exec postgres pg_dump -U spectrumai spectrumai \
    > backup-$(date +%F).sql
```

Для бинарного формата (быстрее, компактнее):
```bash
docker compose exec postgres pg_dump -U spectrumai -F c -f /tmp/backup.dump spectrumai
docker compose cp postgres:/tmp/backup.dump ./backup-$(date +%F).dump
```

Рекомендуемая частота: **еженедельно** при активной эксплуатации;
**перед каждым обновлением версии** — обязательно.

### 2.2. Восстановление

Перед восстановлением остановите бэкенд:
```bash
docker compose stop backend
```

Текстовый дамп:
```bash
docker compose exec -T postgres psql -U spectrumai spectrumai < backup-2026-05-18.sql
```

Бинарный дамп:
```bash
docker compose cp ./backup-2026-05-18.dump postgres:/tmp/backup.dump
docker compose exec postgres pg_restore -U spectrumai -d spectrumai \
    --clean --if-exists /tmp/backup.dump
```

Запустите бэкенд обратно:
```bash
docker compose start backend
docker compose ps    # ждём healthy
```

### 2.3. Расписание

В Linux/Mac добавьте в `cron`:
```cron
0 3 * * 0  cd /path/to/SpectrumAI && \
    docker compose exec -T postgres pg_dump -U spectrumai spectrumai \
    | gzip > /var/backups/spectrumai-$(date +\%F).sql.gz
```

В Windows — задание планировщика с эквивалентной командой PowerShell.

---

## 3. Обновление ML-моделей

ML-чекпойнты хранятся в каталоге `models/`, который монтируется в
backend-контейнер как `/models:ro`. Структура (см. `models/README.md`):

```
models/
├── MANIFEST.json                          ← версии и метрики
├── ircnn-contrastive-0.2.0/
│   └── best.pt                            ← двухбашенная модель + thresholds
├── ircnn-multilabel-0.1.0/
│   └── best.pt                            ← одиночный CNN (fallback)
└── faiss/
    └── ircnn-contrastive-0.2.0/
        ├── index.faiss
        └── mapping.json
```

### 3.1. Подмена модели

1. Поместите новый чекпойнт в `models/<name>-<version>/best.pt`.
2. Обновите `models/MANIFEST.json`:
   ```json
   {
     "active": "ircnn-contrastive-0.3.0",
     "models": [
       {
         "name": "ircnn-contrastive",
         "version": "0.3.0",
         "data_source": "real",
         "phase": 2,
         "metrics": { ... },
         "trained_at": "2026-09-15"
       }
     ]
   }
   ```
3. Обновите env-переменные в `.env`:
   ```ini
   ML_CONTRASTIVE_CHECKPOINT=/models/ircnn-contrastive-0.3.0/best.pt
   ML_FAISS_ROOT=/models/faiss/ircnn-contrastive-0.3.0
   ```
4. Перезапустите backend:
   ```bash
   docker compose restart backend
   ```
5. Проверьте, что новая модель загружена:
   ```bash
   curl http://localhost:8000/api/v1/health
   docker compose logs backend | grep -i model
   ```

---

## 4. Пересчёт FAISS-индекса

После обучения новой контрастной модели — индекс старой модели становится
несовместим. Пересчёт через скрипт `ml/scripts/build_faiss_index.py`:

```bash
.venv/Scripts/python ml/scripts/build_faiss_index.py \
    --checkpoint models/ircnn-contrastive-0.3.0/best.pt \
    --compounds-parquet ml/data/processed/spectra_normalized.parquet \
    --output models/faiss/ircnn-contrastive-0.3.0
```

Для очень больших баз (>100 тыс. соединений) выполняйте на машине с GPU
(аргумент `--device cuda`). Получившиеся `index.faiss` и `mapping.json`
копируйте в `models/faiss/.../`.

---

## 5. Мониторинг логов

Бэкенд пишет структурированные логи через `structlog` в stdout
контейнера. Формат — JSON-line; ключевые поля:

| Поле | Содержание |
|------|-----------|
| `timestamp` | ISO-8601 UTC |
| `level` | `info`/`warning`/`error` |
| `event` | имя события (`api_parsing_error`, `inference_complete`, …) |
| `reason` | расшифровка ошибки |
| `request_id` | при ошибках API |

Просмотр:
```bash
# Хвост всех сервисов
docker compose logs -f

# Только бэкенд
docker compose logs -f backend

# Только ошибки за последний час
docker compose logs --since 1h backend | grep -i '"level": "error"'
```

Access-логи gunicorn пишутся в тот же stdout; формат — `client - method
path HTTP/version status`.

Для production-сценария рекомендуется направлять stdout в Loki/Elastic
через docker logging driver — за рамками фазы 1.

---

## 6. Обновление до новой версии

```bash
cd /path/to/SpectrumAI
git fetch --all
git checkout v1.0-prototype     # или другой тег
docker compose up --build -d
```

Что произойдёт:
1. Образы пересоберутся (BuildKit-кэш ускорит пересборку Python-зависимостей).
2. Старые контейнеры будут заменены новыми.
3. Тома (БД, Redis) сохранятся; миграции `alembic upgrade head` накатятся
   автоматически в entrypoint бэкенда.

Откат при проблеме:
```bash
git checkout <previous-tag>
docker compose up --build -d
```

Если миграция оказалась несовместимой, восстановите БД из бэкапа (§2.2).

---

## 7. Аварийное восстановление

### 7.1. PostgreSQL не стартует

```bash
docker compose logs postgres | tail -30
```

Типичные причины:
- **«database is corrupted»** — восстановите из бэкапа (§2.2) на пустом томе:
  ```bash
  docker compose down
  docker volume rm spectrumai_postgres-data
  docker compose up -d postgres
  # ждём healthy, потом восстанавливаем
  ```
- **«could not allocate disk space»** — освободите место на хосте.
- **«auth failed»** — проверьте `.env` (POSTGRES_PASSWORD).

### 7.2. Backend unhealthy

```bash
docker compose logs backend | tail -50
```

Частые причины:
- **«Connection refused» к postgres** — БД ещё не поднялась. Подождите 30 с
  и перезапустите backend: `docker compose restart backend`.
- **«ModuleNotFoundError»** — пересборка образа после изменения зависимостей:
  `docker compose build backend && docker compose up -d backend`.
- **«FileNotFoundError: checkpoint»** — проверьте, что путь
  `ML_CONTRASTIVE_CHECKPOINT` указывает на существующий файл в `models/`.

### 7.3. Redis недоступен

Redis — только кэш; данные при потере не критичны. Перезапуск:
```bash
docker compose restart redis
```

Если кэш «отравлен» (некорректный результат застрял в кэше) — очистите:
```bash
docker compose exec redis redis-cli FLUSHALL
```

### 7.4. Frontend не открывается

Проверьте контейнер:
```bash
docker compose ps frontend
docker compose logs frontend
```

Если nginx падает на `client intended to send too large body` — поднимите
`client_max_body_size` в `frontend/nginx.conf` (текущее 500 МБ).

---

## 8. Производительность

Ожидаемые показатели на CPU (см. `docs/test-report/benchmark/`):
- Инференс одного спектра: ~2 мс (медиана);
- Полный цикл `/identify`: ~40 мс (медиана), ~140 мс p95 под нагрузкой
  10 пользователей;
- Throughput `/identify/batch`: ~27 спектров/сек.

Если показатели хуже — посмотрите:
- Загрузка CPU (`docker stats`);
- Количество gunicorn-воркеров (`GUNICORN_WORKERS` в `.env`, по умолчанию 2);
- Не запущено ли что-то ещё на хосте (антивирус, индексация);
- Не залогирован ли в Redis огромный кэш (`redis-cli INFO memory`).

Для GPU-ускорения см. `docs/INSTALL.md` §10.

---

## 9. Удалить всё (clean slate)

```bash
docker compose down -v        # контейнеры + тома (БД, Redis)
docker image rm spectrumai-backend spectrumai-frontend
rm -rf models/*/best.pt       # только если хочешь сбросить и модели
```

После этого `docker compose up --build -d` — поднимет систему с нуля.

---

## 10. Куда обращаться

- Установка с нуля: `docs/INSTALL.md`.
- Архитектура и компоненты: `CLAUDE.md` + глава 4 пояснительной записки.
- Тесты и отчёты: `docs/test-report/`.
- План развития: `DEVELOPMENT_PLAN.md`.
