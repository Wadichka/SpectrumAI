# Бенчмарки производительности (NFR-01)

Замеры pytest-benchmark для оценки требования NFR-01 (среднее время
отклика ≤ 2 секунд). Конфигурация: 10 раундов, 2 warmup-раунда,
1 итерация на раунд (см. `backend/tests/perf/`).

## Файлы

| Файл | Содержание |
|------|------------|
| `results.json` | Сырые данные последнего прогона (`pytest --benchmark-json`) |

## Воспроизведение

```bash
cd backend
../.venv/Scripts/python -m pytest tests/perf/ --benchmark-only \
    --benchmark-columns=median,mean,max,stddev \
    --benchmark-json=../docs/test-report/benchmark/results.json
```

## Результаты последнего прогона

| Тест | Median | Mean | Max | Цель (приёмочный) |
|------|--------|------|-----|-------------------|
| `test_inference_median_meets_nfr` | 1.89 мс | 2.01 мс | 2.66 мс | ≤ 500 мс |
| `test_inference_max_within_acceptance` | 1.86 мс | 1.88 мс | 1.99 мс | ≤ 500 мс |
| `test_full_identify_pipeline_meets_nfr` | 38.8 мс | 39.8 мс | 44.6 мс | ≤ 2000 мс |

Все три бенчмарка укладываются в NFR-01 с большим запасом (фактор 25-1000x).

## Замечания

- Тесты используют production-конфиг 1D-CNN из `ml/configs/cnn1d.yaml`
  (5 свёрточных блоков, 256 каналов в максимуме), CPU-only.
- FAISS-ретривер отключён (`spectrum_tower=None`); полный pipeline с
  ретривалом будет дольше — оценка появится после этапа 21 (FAISS на
  полной базе).
- Бенчмарки запускаются исключительно с `--benchmark-only`. В обычных
  прогонах `pytest` они скипаются благодаря `addopts` в `pyproject.toml`.
