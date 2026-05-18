"""REST-эндпоинт генерации PDF-отчётов (UC-05 / FR-09).

Stateless: фронт передаёт ``IdentificationResponse`` в теле, бэк
рендерит PDF и отдаёт его потоком. В историю не сохраняем — на
phase 1 БД хранит только метаданные, а полный результат у фронта
в `useIdentificationStore.lastResponse`.
"""

from __future__ import annotations

import io

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.v1.schemas import IdentificationResponse
from app.domain.dto import IdentificationResult
from app.reports import render_identification_report

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post(
    "/identification",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "PDF-отчёт об идентификации",
        }
    },
)
async def export_identification_report(payload: IdentificationResponse) -> StreamingResponse:
    """Принимает полный ответ /identify и возвращает PDF.

    Возвращает поток ``application/pdf`` с заголовком ``Content-Disposition``
    для скачивания. Имя файла включает ``request_id`` если он есть.
    """
    result = IdentificationResult(
        predictions=list(payload.predictions),
        candidates=list(payload.candidates),
        spectrum_length=payload.spectrum_length,
        model_versions=dict(payload.model_versions),
        threshold_mode=payload.threshold_mode,
        processing_time_ms=payload.processing_time_ms,
        timestamp=payload.timestamp,
        gradcam=payload.gradcam,
        spectrum=list(payload.spectrum) if payload.spectrum is not None else None,
    )
    pdf_bytes = render_identification_report(result, request_id=payload.request_id)

    suffix = payload.request_id if payload.request_id is not None else "unsaved"
    filename = f"identification-{suffix}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
