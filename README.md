# SpectrumAI

Дипломный проект: система распознавания органических соединений по их
инфракрасным спектрам.

1D-CNN multi-label классификация функциональных групп + контрастное
обучение (двухбашенная схема) + FAISS-ретривал → REST API + React SPA.
Полный стек поднимается одной командой `docker compose up`.

---

## О проекте

- **Целевая аудитория:** студенты и сотрудники химических лабораторий,
  работающие с ИК-спектрами органических соединений.
- **Цель:** автоматизировать идентификацию соединения по загруженному
  спектру — определить функциональные группы и предложить top-K
  похожих соединений из базы знаний.
- **Фазы разработки** (CLAUDE.md §11, `DEVELOPMENT_PLAN.md`):
  - **Фаза 1** (тег `v1.0-prototype`, ✅ выполнена) — синтетический
    датасет, отладка полного пайплайна.
  - **Фаза 2** (тег `v1.5-predefense`, в процессе) — компактный
    предзащитный прототип на 2000 реальных NIST-спектрах.
    Метрики **после Kaggle-обучения**: см. `models/MANIFEST.json` и
    `ml/experiments/predefense/metrics.json`. Ожидаемо F1 macro
    0.4–0.6, top-5 retrieval 0.35–0.55.
  - **Фаза 3** (тег `v2.0-final`, после предзащиты) — полный датасет
    NIST + SDBS + Coblentz с hyperparameter sweep; цель — достичь
    метрик главы 6.4.4.

Архитектура и обоснование решений — глава 4 пояснительной записки
(C4-модель, UML-диаграммы, ADR-1..ADR-8) и `CLAUDE.md` §3–4.

---

## Быстрый старт

Требуется Docker Engine 24+ или Docker Desktop 4.30+:

```bash
cp .env.example .env
docker compose up --build -d
```

Через минуту-другую система готова:

- http://localhost — фронтенд (nginx раздаёт SPA, проксирует `/api/*` на бэк).
- http://localhost:8000/docs — Swagger UI (OpenAPI-документация API).
- http://localhost:8000/openapi.json — машинно-читаемая OpenAPI-схема.

Остановка: `docker compose down` (тома сохраняются) или `docker compose
down -v` (полная очистка).

Подробная инструкция (требования, troubleshooting, GPU, бэкап БД) —
`docs/INSTALL.md`.

---

## Документация

| Документ | Аудитория | Описание |
|---|---|---|
| `docs/USER_GUIDE.md` | Пользователи-химики | Как загружать спектры, читать результаты, пользоваться поиском, экспортировать PDF. |
| `docs/INSTALL.md` | DevOps / разработчики | Установка, запуск, обновление, бэкап, GPU, HTTPS, troubleshooting. |
| `docs/OPERATIONS_GUIDE.md` | Администраторы | Операции, мониторинг, обновление моделей, аварийное восстановление. |
| `docs/DEMO_SCRIPT.md` | Защита ВКР | Пошаговый сценарий демонстрации на 10 минут. |
| `docs/KAGGLE_TRAINING.md` | ML-разработчики | Инструкция по запуску предзащитного обучения в Kaggle (фаза 2). |
| `docs/test-report/` | Тестировщики, комиссия | Сводный отчёт по главе 11: матрица требований, coverage 94%, benchmark, e2e, security. |
| `CLAUDE.md` | Разработчики | Постоянный контекст: стек, соглашения, структура каталогов. |
| `DEVELOPMENT_PLAN.md` | Разработчики | План этапов 0–24 с критериями приёмки и коммитами (три фазы). |
| `demo/predefense_spectra/README.md` | ML-разработчики, защита | Шаблон подбора 5–10 демо-спектров для предзащиты. |
| `ml/README.md` | ML-разработчики | Структура каталога `ml/` (датасеты, ноутбуки, скрипты). |
| `models/README.md` | ML-разработчики | Формат `MANIFEST.json`, версионирование чекпойнтов. |

---

## Локальная разработка с hot-reload

Docker-стек собирает образ один раз и не подхватывает изменения кода на
лету. Для разработки запускайте бэкенд и фронт **локально** из `.venv`
и npm, оставив в Docker только инфраструктуру:

```bash
docker compose up -d postgres redis

# Backend (отдельный терминал):
cd backend
..\.venv\Scripts\Activate.ps1    # Windows; на Linux/macOS: source ../.venv/bin/activate
pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --reload

# Frontend (другой терминал):
cd frontend
npm install
npm run dev
```

Фронтенд: http://localhost:5173, бэкенд: http://localhost:8000.

См. `CLAUDE.md` §6 «Соглашения по коду».

---

## Технологический стек

**Backend** (Python 3.11): FastAPI, SQLAlchemy 2.x, Alembic, PostgreSQL 16,
Redis 7, PyTorch (CPU-only из официального индекса PyTorch), RDKit,
faiss-cpu, structlog, reportlab + matplotlib (PDF-отчёты).

**Frontend**: React 18 + TypeScript + Vite, Tailwind CSS, Zustand,
react-router-dom, axios, plotly.js, i18next.

**Тестирование**: pytest, pytest-benchmark, hypothesis, locust, bandit,
vitest, @testing-library/react, Playwright.

**Инфраструктура**: Docker + docker-compose (multi-stage, BuildKit
cache mount), gunicorn (UvicornWorker × 2), nginx (раздача SPA +
proxy `/api/*`), GitHub Actions CI (ruff, mypy, pytest, bandit).

Детали — `CLAUDE.md` §4 и глава 8 пояснительной записки.

---

## Состояние тестирования (v1.0-prototype)

- **291 автоматизированный тест** зелёный.
- **Покрытие бэкенда: 94%** (целевой NFR-05 = 80%).
- **NFR-01** (отклик < 2 с): p95 = 140 мс под нагрузкой Locust.
- **0 active findings** от Bandit (security).
- **17/19 требований ТЗ** покрыты автотестами (PASS). 2 SKIP — NFR-07
  (модерируемый SUS-тест, проводится перед защитой) и NFR-09 (приёмка
  комплекта документации комиссией).

Подробная сводка — `docs/test-report/README.md`.

---

## Phase 2 (предзащитный прототип, тег `v1.5-predefense`)

Чекпойнты обучены в Kaggle T4 (одна сессия, ~12 часов), интегрированы
в стек. Метрики извлечены из `ml/experiments/predefense/metrics.json`
и `models/MANIFEST.json`.

| Метрика | Достигнуто | Целевой (фаза 2) | Pass | Источник |
|---------|-----------|------------------|------|----------|
| Macro F1 | 0.388 | 0.40–0.60 | ✗ | contrastive checkpoint |
| Micro F1 | 0.499 | 0.45–0.65 | ✓ | contrastive checkpoint |
| Macro AP | 0.333 | 0.30–0.50 | ✓ | contrastive checkpoint |
| Top-1 retrieval | 0.049 | 0.20–0.35 | ✗ | FAISS eval |
| Top-5 retrieval | 0.143 | 0.35–0.55 | ✗ | FAISS eval |
| Top-10 retrieval | 0.254 | — | — | FAISS eval |
| MRR | 0.116 | — | — | FAISS eval |
| Hamming loss | 0.262 | < 0.30 | ✓ | contrastive checkpoint |
| Размер датасета | 1919 NIST | 2000–3000 | ✗ | `predefense_stats.json` |

**Метрики ниже целевых §6.4.4** — это **ожидаемое и приемлемое
поведение для фазы 2** (см. CLAUDE.md §11 «Стратегия данных и
обучения»). Цель фазы 2 — закрыть end-to-end интеграцию (бэкенд +
фронт + Docker + БД + FAISS + UX) на реальных NIST-данных, а не
достичь финального качества модели. Причины ниже-целевых метрик:

- скрейпинг NIST WebBook остановлен на 35% (5700/16494 соединений) из-за
  более жёсткого rate-limit, чем ожидалось → подвыборка 1919 вместо
  целевых 2500–3000;
- ряд функциональных групп с малым покрытием (amide_primary=7,
  sulfoxide_sulfone=26, amide_secondary=29, alkyne=55) — модель почти
  не учится на них;
- один прогон без HP sweep (по плану §19 — sweep это фаза 3).

**Фаза 3** (этапы 21–24 DEVELOPMENT_PLAN.md) расширяет датасет до
3000–10000 спектров (NIST + SDBS + Coblentz) с hyperparameter sweep —
там целевые метрики §6.4.4 ожидаемо достигаются. Тег фазы 3 —
`v2.0-final`.

---

## Лицензия

MIT — см. `LICENSE`. Проект разработан в рамках выпускной
квалификационной работы; пояснительная записка — в каталоге автора.
