"""Эндпоинты идентификации (UC-01 и UC-06)."""

from __future__ import annotations

from typing import Literal, cast

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.v1.dependencies import get_identification_service
from app.api.v1.schemas import (
    ApiError,
    BatchIdentificationItemResponse,
    BatchIdentificationResponse,
    IdentificationResponse,
)
from app.services.identification import IdentificationService

_BATCH_FILES_LIMIT = 20
_BATCH_TOTAL_BYTES_LIMIT = 50 * 1024 * 1024  # 50 МБ

router = APIRouter(tags=["identify"])


def _to_response(result: object, *, request_id: int | None = None) -> IdentificationResponse:
    """Маппит доменный DTO ``IdentificationResult`` в API-схему."""
    return IdentificationResponse.model_validate(
        {**result.model_dump(), "request_id": request_id}  # type: ignore[attr-defined]
    )


@router.post(
    "/identify",
    response_model=IdentificationResponse,
    summary="Идентификация спектра (UC-01)",
)
async def identify(
    file: UploadFile = File(..., description="JCAMP-DX или CSV"),
    include_gradcam: bool = Form(True, description="Включить Grad-CAM для top-1 группы"),
    top_k: int = Form(10, ge=1, le=50),
    service: IdentificationService = Depends(get_identification_service),
) -> IdentificationResponse:
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ApiError(code="EMPTY_FILE", message="Файл пустой").model_dump(exclude_none=True),
        )
    result = await service.identify_one(
        file_bytes=file_bytes,
        filename=file.filename or "spectrum",
        include_gradcam=include_gradcam,
        top_k=top_k,
    )
    return _to_response(result)


@router.post(
    "/identify/batch",
    response_model=BatchIdentificationResponse,
    summary="Пакетная идентификация (UC-06)",
)
async def identify_batch(
    files: list[UploadFile] = File(..., description="До 20 файлов JCAMP-DX/CSV"),
    include_gradcam: bool = Form(False),
    top_k: int = Form(10, ge=1, le=50),
    service: IdentificationService = Depends(get_identification_service),
) -> BatchIdentificationResponse:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ApiError(code="NO_FILES", message="Не передано ни одного файла").model_dump(
                exclude_none=True
            ),
        )
    if len(files) > _BATCH_FILES_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ApiError(
                code="TOO_MANY_FILES",
                message=f"Максимум {_BATCH_FILES_LIMIT} файлов за раз",
            ).model_dump(exclude_none=True),
        )

    payloads: list[tuple[bytes, str]] = []
    total_bytes = 0
    for upload in files:
        data = await upload.read()
        total_bytes += len(data)
        if total_bytes > _BATCH_TOTAL_BYTES_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ApiError(
                    code="PAYLOAD_TOO_LARGE",
                    message="Суммарный размер файлов превышает 50 МБ",
                ).model_dump(exclude_none=True),
            )
        payloads.append((data, upload.filename or "spectrum"))

    batch = await service.identify_batch(payloads, include_gradcam=include_gradcam, top_k=top_k)
    items: list[BatchIdentificationItemResponse] = []
    for item in batch.items:
        result_dto = _to_response(item.result) if item.result is not None else None
        error_dto = (
            ApiError(code=item.error_code or "ERROR", message=item.error_message or "")
            if item.status == "error"
            else None
        )
        item_status = cast(Literal["success", "error"], item.status)
        items.append(
            BatchIdentificationItemResponse(
                filename=item.filename,
                status=item_status,
                result=result_dto,
                error=error_dto,
            )
        )
    return BatchIdentificationResponse(
        items=items,
        total_processing_time_ms=batch.total_processing_time_ms,
    )
