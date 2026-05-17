"""Добавление спектра в БД (UC-03)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.v1.dependencies import get_spectrum_service
from app.api.v1.schemas import ApiError, SpectrumCreatedResponse
from app.services.spectrum import SpectrumService

router = APIRouter(tags=["spectra"])


@router.post(
    "/spectra",
    response_model=SpectrumCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Добавить спектр в локальную базу (UC-03)",
)
async def add_spectrum(
    file: UploadFile = File(..., description="JCAMP-DX/CSV-спектр"),
    smiles: str = Form(..., description="SMILES-строка соединения"),
    name: str | None = Form(default=None),
    source: str = Form(default="manual"),
    service: SpectrumService = Depends(get_spectrum_service),
) -> SpectrumCreatedResponse:
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ApiError(code="EMPTY_FILE", message="Файл пустой").model_dump(exclude_none=True),
        )
    created = service.add_spectrum(
        file_bytes=file_bytes,
        filename=file.filename or "manual.jdx",
        smiles=smiles,
        name=name,
        source=source,
    )
    return SpectrumCreatedResponse(
        spectrum_id=created.spectrum_id,
        compound_id=created.compound_id,
        status="created",
    )
