"""Бенчмарк инференса CNN на ``ProcessedSpectrum`` (NFR-01, §11.6.5).

Эти тесты измеряют только ML-инференс (без парсинга/предобработки/HTTP),
чтобы валидировать пороги из таблицы 11.4:

- target inference latency  ≤ 150 мс median (целевой)
- acceptance latency        ≤ 500 мс median (приёмочный)

Запуск:
    pytest tests/perf/test_inference_benchmark.py --benchmark-only \\
        --benchmark-json=../docs/test-report/benchmark/inference.json
"""

from __future__ import annotations

from app.ml.inference_service import InferenceService
from app.preprocessing.config import ProcessedSpectrum

_NFR_LIMIT_S = 2.0  # NFR-01: полная идентификация ≤ 2 секунды.


def test_inference_median_meets_nfr(
    benchmark,  # type: ignore[no-untyped-def]
    inference_service: InferenceService,
    processed_spectrum: ProcessedSpectrum,
) -> None:
    """Медиана одиночного инференса должна укладываться в NFR-01."""
    result = benchmark.pedantic(
        inference_service._predict_sync,
        args=(processed_spectrum, 10),
        rounds=10,
        warmup_rounds=2,
        iterations=1,
    )
    assert result.spectrum_length == 3601
    # Жёсткий потолок NFR-01 — даже отдельный CNN-инференс должен укладываться
    # в общий бюджет (≤ 2 секунды на всю идентификацию).
    assert benchmark.stats.stats.median < _NFR_LIMIT_S


def test_inference_max_within_acceptance(
    benchmark,  # type: ignore[no-untyped-def]
    inference_service: InferenceService,
    processed_spectrum: ProcessedSpectrum,
) -> None:
    """Худший прогон тоже не должен пробивать приёмочный порог."""
    benchmark.pedantic(
        inference_service._predict_sync,
        args=(processed_spectrum, 10),
        rounds=10,
        warmup_rounds=2,
        iterations=1,
    )
    assert benchmark.stats.stats.max < _NFR_LIMIT_S
