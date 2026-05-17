"""Единый обработчик доменных исключений → HTTP-ответы (§4.4.7)."""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api.v1.schemas import ApiError
from app.domain.errors import (
    DomainError,
    EntityNotFoundError,
    ParsingError,
    SpectrumValidationError,
)

log = structlog.get_logger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Регистрирует обработчики иерархии ``DomainError`` в FastAPI."""

    @app.exception_handler(EntityNotFoundError)
    async def _entity_not_found(_: Request, exc: EntityNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": _error_payload("ENTITY_NOT_FOUND", str(exc))},
        )

    @app.exception_handler(ParsingError)
    async def _parsing(_: Request, exc: ParsingError) -> JSONResponse:
        log.warning("api_parsing_error", reason=str(exc))
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": _error_payload("PARSING_ERROR", str(exc))},
        )

    @app.exception_handler(SpectrumValidationError)
    async def _spectrum_validation(_: Request, exc: SpectrumValidationError) -> JSONResponse:
        log.warning("api_validation_error", reason=str(exc))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": _error_payload("VALIDATION_ERROR", str(exc))},
        )

    @app.exception_handler(DomainError)
    async def _domain(_: Request, exc: DomainError) -> JSONResponse:
        log.error("api_domain_error", reason=str(exc), exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": _error_payload("DOMAIN_ERROR", str(exc))},
        )


def _error_payload(code: str, message: str) -> dict[str, str]:
    return ApiError(code=code, message=message).model_dump(exclude_none=True)
