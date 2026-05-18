# SpectrumAI

Bachelor's degree project — recognition system of organic compounds by their IR spectra.

Дипломный проект: распознавание органических соединений по инфракрасным спектрам. 1D-CNN + контрастное обучение (двухбашенная схема) + FAISS-ретривал + REST API + React SPA.

## Документация

- **`CLAUDE.md`** — постоянный контекст проекта (стек, соглашения, структура).
- **`DEVELOPMENT_PLAN.md`** — пошаговый план разработки (этапы 0–21).
- **`docs/INSTALL.md`** — подробная инструкция по установке, диагностике, бэкапу БД и GPU-ускорению.

## Быстрый старт

```bash
cp .env.example .env
docker compose up --build -d
```

После старта:

- http://localhost — фронтенд (nginx раздаёт собранный SPA, проксирует `/api/*` на бэк).
- http://localhost:8000/docs — Swagger UI.

Остановка: `docker compose down` (тома сохраняются) или `docker compose down -v` (полная очистка).

## Лицензия

MIT — см. `LICENSE`.
