"""FastAPI-зависимости для роутеров v1 (§4.4.7)."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.ml.components import get_ml_components
from app.ml.inference_service import InferenceService
from app.services.compound import CompoundService
from app.services.history import HistoryService
from app.services.identification import IdentificationService
from app.services.spectrum import SpectrumService


def _session_factory_from(session: Session) -> Iterator[Session]:
    yield session


def get_inference_service(session: Session = Depends(get_db)) -> InferenceService:
    """Создаёт InferenceService на запрос (singleton MLComponents + per-request session)."""
    components = get_ml_components()
    return InferenceService(
        components,
        db_session_factory=lambda: _session_factory_from(session),
    )


def get_identification_service(
    session: Session = Depends(get_db),
    inference: InferenceService = Depends(get_inference_service),
) -> IdentificationService:
    return IdentificationService(inference=inference, session=session)


def get_compound_service(session: Session = Depends(get_db)) -> CompoundService:
    return CompoundService(session)


def get_history_service(session: Session = Depends(get_db)) -> HistoryService:
    return HistoryService(session)


def get_spectrum_service(session: Session = Depends(get_db)) -> SpectrumService:
    return SpectrumService(session)


__all__ = [
    "get_compound_service",
    "get_history_service",
    "get_identification_service",
    "get_inference_service",
    "get_spectrum_service",
]
