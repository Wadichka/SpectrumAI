"""Сервис-оркестратор идентификации (Этап 9, §4.4.7 главы 4).

Объединяет существующие слои в один сценарий:
парсинг → предобработка → ML-инференс → Grad-CAM для top-1 группы →
запись истории (IdentificationRequest + PredictedFunctionalGroup).

Бизнес-логика *не* живёт в роутерах FastAPI — только в этом сервисе.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import structlog
import torch
from sqlalchemy.orm import Session

from app.db.models.identification_request import IdentificationRequest
from app.db.models.predicted_functional_group import PredictedFunctionalGroup
from app.db.repositories.functional_group import FunctionalGroupRepository
from app.domain.dto import GradCamPayload, IdentificationResult
from app.domain.errors import ParsingError, SpectrumValidationError
from app.interpretation.gradcam import compute_gradcam, default_target_layer
from app.ml.inference_service import InferenceService
from app.parsing import parse_spectrum
from app.preprocessing import preprocess

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BatchIdentificationItem:
    """Результат идентификации одного файла в пакете."""

    filename: str
    status: str  # "success" | "error"
    result: IdentificationResult | None
    error_code: str | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class BatchIdentificationResult:
    items: list[BatchIdentificationItem]
    total_processing_time_ms: int


class IdentificationService:
    """Оркестратор сценария UC-01 (single) и UC-06 (batch)."""

    def __init__(
        self,
        inference: InferenceService,
        session: Session,
    ) -> None:
        self._inference = inference
        self._session = session

    async def identify_one(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        include_gradcam: bool = True,
        top_k: int | None = None,
    ) -> IdentificationResult:
        raw = parse_spectrum(file_bytes)
        processed = preprocess(raw)
        base_result = await self._inference.predict(processed, top_k=top_k)
        gradcam = (
            self._compute_top_gradcam(base_result, processed_intensities=processed.intensities)
            if include_gradcam
            else None
        )
        if gradcam is not None:
            result = base_result.model_copy(update={"gradcam": gradcam})
        else:
            result = base_result
        self._persist_history(result, filename=filename)
        return result

    async def identify_batch(
        self,
        files: Sequence[tuple[bytes, str]],
        *,
        include_gradcam: bool = False,
        top_k: int | None = None,
    ) -> BatchIdentificationResult:
        started = time.perf_counter()
        items: list[BatchIdentificationItem] = []
        for file_bytes, filename in files:
            try:
                result = await self.identify_one(
                    file_bytes=file_bytes,
                    filename=filename,
                    include_gradcam=include_gradcam,
                    top_k=top_k,
                )
                items.append(
                    BatchIdentificationItem(
                        filename=filename,
                        status="success",
                        result=result,
                        error_code=None,
                        error_message=None,
                    )
                )
            except (ParsingError, SpectrumValidationError) as exc:
                items.append(
                    BatchIdentificationItem(
                        filename=filename,
                        status="error",
                        result=None,
                        error_code=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                log.warning(
                    "batch_item_failed",
                    filename=filename,
                    code=type(exc).__name__,
                    reason=str(exc),
                )
        total_ms = int((time.perf_counter() - started) * 1000)
        return BatchIdentificationResult(items=items, total_processing_time_ms=total_ms)

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------
    def _compute_top_gradcam(
        self,
        result: IdentificationResult,
        *,
        processed_intensities: object,
    ) -> GradCamPayload | None:
        positives = [p for p in result.predictions if p.predicted]
        candidates = positives or list(result.predictions)
        if not candidates:
            return None
        top = max(candidates, key=lambda p: p.probability)
        components = self._inference.components
        cnn = components.cnn
        target_layer = default_target_layer(cnn)
        tensor = torch.tensor(processed_intensities, dtype=torch.float32)
        cam = compute_gradcam(
            cnn,
            target_layer,
            tensor,
            class_indices=[components.class_names.index(top.name)],
            target_length=result.spectrum_length,
        )
        values = cam[components.class_names.index(top.name)]
        return GradCamPayload(
            group_code=top.code,
            group_name=top.name,
            values=[float(v) for v in values.tolist()],
        )

    def _persist_history(self, result: IdentificationResult, *, filename: str) -> None:
        request = IdentificationRequest(
            input_spectrum_path=filename,
            processing_time_ms=int(result.processing_time_ms),
            status="success",
        )
        self._session.add(request)
        self._session.flush()
        fg_repo = FunctionalGroupRepository(self._session)
        groups = {g.code: g for g in fg_repo.list_all()}
        for prediction in result.predictions:
            if not prediction.predicted:
                continue
            fg = groups.get(prediction.code)
            if fg is None:
                continue
            self._session.add(
                PredictedFunctionalGroup(
                    request_id=request.id,
                    functional_group_id=fg.id,
                    probability=float(prediction.probability),
                )
            )
        self._session.commit()


def make_identification_service(
    inference: InferenceService,
    session: Session,
) -> IdentificationService:
    return IdentificationService(inference=inference, session=session)


def session_factory(session: Session) -> Iterator[Session]:
    """Утилита для совместимости с db_session_factory в InferenceService."""
    yield session


__all__ = [
    "BatchIdentificationItem",
    "BatchIdentificationResult",
    "IdentificationService",
    "make_identification_service",
]
