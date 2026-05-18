# Отчёт о тестировании SpectrumAI

Сводный отчёт по плану из главы 11 пояснительной записки. Соответствует
Этапу 16 `DEVELOPMENT_PLAN.md`. Все измерения получены на **фазе 1
(синтетические данные)** — целевые ML-метрики таблицы 11.2 будут
пересняты после Этапа 20 (обучение на реальных данных NIST/SDBS/Coblentz).

Дата прогона: 2026-05-18. Локальная конфигурация:
- ОС: Windows 11 Home 26200
- CPU: x86_64
- Python 3.11.9 (venv); Node 20; Docker Desktop
- Стек: `docker compose up -d` (PostgreSQL 16, Redis 7, FastAPI+gunicorn, nginx)

## Сводка по требованиям

| Категория | Требований | PASS | FAIL | SKIP | Источник |
|---|---|---|---|---|---|
| Функциональные (FR-01..FR-10) | 10 | 10 | 0 | 0 | [requirements-matrix.md](requirements-matrix.md) |
| Нефункциональные (NFR-01..NFR-09) | 9 | 7 | 0 | 2 | [requirements-matrix.md](requirements-matrix.md) |
| **Всего** | **19** | **17** | **0** | **2** | |

SKIP — требования, оцениваемые вне автоматизированного контура (NFR-07
SUS-индекс — модерируемое исследование с 3-5 пользователями; NFR-09 — комплект
документации, приёмка комиссией перед защитой). FR-09 закрыт в Этапе 17:
сервер генерирует PDF (`POST /api/v1/reports/identification`), фронт
скачивает через `ResultsActions` (UC-05).

## Покрытие кода (NFR-05)

**TOTAL: 94%** (целевой 80%, приёмочный 70%). Полный HTML-отчёт:
[`coverage/index.html`](coverage/index.html).

Топ-5 модулей с наименьшим покрытием:

| Модуль | Покрытие | Причина |
|---|---|---|
| `app/services/identification.py` | 81% | Оркестрация, частично проверена интеграцией |
| `app/services/history.py` | 86% | Пагинация и фильтры — покрыты, но не все ветки |

Остальные модули — выше 90%, бо́льшая часть на 100%.

## Производительность (NFR-01)

NFR-01: среднее время отклика **<2 секунд**. Измерения pytest-benchmark
([`benchmark/results.json`](benchmark/results.json)) — на CPU без GPU:

| Метрика | Целевой | Приёмочный | Достигнуто |
|---|---|---|---|
| Инференс одного спектра (median) | ≤150 мс | ≤500 мс | **1.9 мс** ✓ |
| Полный цикл /identify (median) | ≤1.0 с | ≤2.0 с | **39 мс** ✓ |
| Полный цикл /identify (max) | — | ≤2.0 с | **45 мс** ✓ |

Нагрузочный тест Locust (`-u 5 -r 1 -t 30s` против локального стека) —
58 запросов, 0 отказов: `/identify` p95 = 140 мс, throughput batch ≈ 27
спектров/сек. Полный отчёт: [`locust/report.html`](locust/report.html).

## Категории тестов

| № | Категория | Инструмент | Файл/каталог | Результат |
|---|---|---|---|---|
| 1 | Unit | pytest | `backend/tests/unit/`, `ml/tests/` | 132 + 83 passed |
| 2 | Integration | pytest+httpx | `backend/tests/integration/` | внутри 132 |
| 3 | Robustness | pytest+hypothesis | `backend/tests/robustness/` | 33 passed |
| 4 | Performance | pytest-benchmark | `backend/tests/perf/` | 3 passed (NFR-01 ✓) |
| 5 | Load | Locust | `backend/tests/load/` | 0 failures (см. выше) |
| 6 | Security | Bandit | `[tool.bandit]` в `pyproject.toml` | 0 active findings ([security/bandit.txt](security/bandit.txt)) |
| 7 | Frontend unit | vitest | `frontend/tests/` | 36 passed |
| 8 | E2E | Playwright | `frontend/tests/e2e/` | 4 passed (UC-01/04/06/08) |
| 9 | Usability | SUS (manual) | [`manual/sus-questionnaire.md`](manual/sus-questionnaire.md) | Запланировано перед защитой |

## Юзабилити и SUS

Глава 11 §11.6.6 предписывает модерируемое тестирование с 3–5
пользователями целевой аудитории (студенты химических специальностей).
Это исследование проводится автором вручную перед защитой по протоколу
[`manual/usability-protocol.md`](manual/usability-protocol.md) и
опроснику [`manual/sus-questionnaire.md`](manual/sus-questionnaire.md).
Целевой SUS ≥ 70.

## Протокол испытаний (ГОСТ 19.301-79)

[`gost-19.301-79.md`](gost-19.301-79.md) — шаблон формального протокола
для приложения к пояснительной записке. Поля «дата» и «комиссия»
заполняются автором/комиссией перед защитой.

## Воспроизведение измерений

См. `docs/INSTALL.md` (стек) и:
- `backend/tests/load/README.md` (нагрузочный тест);
- `backend/tests/perf/test_*_benchmark.py` (NFR-01 бенчмарки);
- `frontend/playwright.config.ts` + `frontend/tests/e2e/` (UC-сценарии);
- `backend/pyproject.toml` секция `[tool.bandit]` (security).

Полная команда coverage:
```bash
cd backend
../.venv/Scripts/python -m pytest --cov=app --cov-report=html:../docs/test-report/coverage
```

## Ограничения фазы 1

- ML-метрики таблицы 11.2 (Macro-F1, Top-1, MRR и т. п.) **не** проверены,
  потому что обучение модели запланировано на Этап 20 (фаза 2) на реальных
  данных. На синтетике эти метрики не имеют ценности.
- Конфигурации тестирования (Win 10 / Ubuntu 22.04, Chrome/Firefox/Edge)
  проверены частично: e2e-сценарии запускались только на chromium локально;
  расширение matrix-проверок — следующий шаг.
- Полная локализация интерфейса (NFR-04) проверена визуально, не
  автоматизированными скриншот-тестами.
