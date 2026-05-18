"""Unit-тесты для ``app.reports.render_identification_report``."""

from __future__ import annotations

from datetime import UTC, datetime

from app.domain.dto import (
    CompoundCandidate,
    FunctionalGroupPrediction,
    GradCamPayload,
    IdentificationResult,
)
from app.reports import render_identification_report


def _make_minimal_result(
    *, with_spectrum: bool = True, with_gradcam: bool = False
) -> IdentificationResult:
    predictions = [
        FunctionalGroupPrediction(
            code=f"FG{i:02d}",
            name=f"group_{i}",
            probability=0.9 if i == 1 else 0.1,
            threshold=0.5,
            predicted=(i == 1),
        )
        for i in range(1, 6)
    ]
    candidates = [
        CompoundCandidate(
            rank=1,
            compound_id=42,
            smiles="CCO",
            name="ethanol",
            formula="C2H6O",
            cas_number="64-17-5",
            score=0.91,
            consistent=True,
            jaccard=0.83,
            matched_groups=("group_1",),
        )
    ]
    spectrum = [0.1 + 0.01 * (i % 7) for i in range(3601)] if with_spectrum else None
    gradcam = (
        GradCamPayload(group_code="FG01", group_name="group_1", values=[0.0] * 3601)
        if with_gradcam
        else None
    )
    return IdentificationResult(
        predictions=predictions,
        candidates=candidates,
        spectrum_length=3601,
        model_versions={"mode": "test", "checkpoint": "fake"},
        threshold_mode="fixed",
        processing_time_ms=42,
        timestamp=datetime.now(UTC),
        spectrum=spectrum,
        gradcam=gradcam,
    )


def test_renders_valid_pdf_bytes() -> None:
    result = _make_minimal_result()
    pdf = render_identification_report(result, request_id=1)
    assert pdf[:5] == b"%PDF-"
    assert len(pdf) > 2_000  # есть содержимое + графика


def test_renders_without_spectrum_and_gradcam() -> None:
    result = _make_minimal_result(with_spectrum=False, with_gradcam=False)
    pdf = render_identification_report(result)
    assert pdf.startswith(b"%PDF-")


def test_renders_with_gradcam_section() -> None:
    result = _make_minimal_result(with_gradcam=True)
    pdf = render_identification_report(result, request_id=99)
    # Grad-CAM добавляет вторую страницу — длина растёт.
    short_pdf = render_identification_report(_make_minimal_result(with_gradcam=False))
    assert len(pdf) > len(short_pdf)


def test_renders_empty_candidates() -> None:
    result = _make_minimal_result()
    empty = result.model_copy(update={"candidates": []})
    pdf = render_identification_report(empty)
    assert pdf.startswith(b"%PDF-")
