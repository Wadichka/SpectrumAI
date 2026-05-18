"""Асинхронный сервис ML-инференса (§4.4.3 главы 4).

На вход — ``ProcessedSpectrum`` (Этап 3, ``app.preprocessing.config``).
На выход — ``IdentificationResult`` (DTO, ``app.domain.dto``):

1. Прогон спектра через CNN → multi-label вероятности + бинарные предсказания
   по per-class порогам (§6.4, §6.9.3).
2. Если есть FAISS-ретривер — спектр → проекция через ``SpectrumTower`` →
   top-K кандидатов с cosine-similarity.
3. Для каждого кандидата — cross-validation предсказанных групп с реальными
   группами кандидата по SMARTS (§4.4.5, Jaccard ≥ threshold).
4. По возможности подтянуть метаданные кандидата из БД через
   ``CompoundRepository``; иначе оставить ``None``.

forward тяжёлый, поэтому весь синхронный путь оборачивается в
``asyncio.to_thread``, что укладывается в NFR < 2 секунд (§4.8).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import structlog
import torch

from app.core.config import get_settings
from app.domain.dto import (
    CompoundCandidate,
    FunctionalGroupPrediction,
    IdentificationResult,
)
from app.ml import _ml_path  # noqa: F401 — sys.path
from app.ml.components import MLComponents
from app.ml.cross_validation import (
    candidate_groups_from_smiles,
    compute_consistency,
)
from app.preprocessing.config import ProcessedSpectrum
from pipelines.retrieval import CompoundCandidate as RetrievalCandidate

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

log = structlog.get_logger(__name__)


class InferenceService:
    """Высокоуровневая обёртка над :class:`MLComponents` для API."""

    def __init__(
        self,
        components: MLComponents,
        *,
        db_session_factory: Callable[[], Iterator[Session]] | None = None,
        default_top_k: int | None = None,
        consistency_threshold: float | None = None,
    ) -> None:
        self._components = components
        self._db_session_factory = db_session_factory
        settings = get_settings()
        self._default_top_k = int(default_top_k or settings.ml_top_k_default)
        self._consistency_threshold = float(
            consistency_threshold
            if consistency_threshold is not None
            else settings.ml_consistency_threshold
        )

    @property
    def components(self) -> MLComponents:
        return self._components

    async def predict(
        self,
        processed: ProcessedSpectrum,
        *,
        top_k: int | None = None,
    ) -> IdentificationResult:
        """Асинхронная обёртка: CPU-bound forward выполняется в потоке."""
        effective_top_k = int(top_k or self._default_top_k)
        return await asyncio.to_thread(self._predict_sync, processed, effective_top_k)

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------
    def _predict_sync(self, processed: ProcessedSpectrum, top_k: int) -> IdentificationResult:
        started = time.perf_counter()
        components = self._components
        device = components.device

        intensities = np.asarray(processed.intensities, dtype=np.float32)
        tensor = torch.from_numpy(intensities).unsqueeze(0).unsqueeze(0).to(device)
        # Форма: (1, 1, L).

        with torch.no_grad():
            embedding = components.cnn.forward_embedding(tensor)
            logits = components.cnn.classifier(embedding)
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        predictions = self._build_predictions(probabilities)
        predicted_names = [p.name for p in predictions if p.predicted]

        candidates: list[CompoundCandidate] = []
        if components.spectrum_tower is not None and components.retriever is not None:
            with torch.no_grad():
                projection = components.spectrum_tower.project(embedding)
            raw_candidates = components.retriever.find_top_k(projection.squeeze(0), k=top_k)
            candidates = self._tag_candidates(raw_candidates, predicted_names)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        threshold_mode = "per_class" if not _all_equal(components.thresholds) else "fixed"

        return IdentificationResult(
            predictions=predictions,
            candidates=candidates,
            spectrum_length=int(intensities.size),
            model_versions=dict(components.model_versions),
            threshold_mode=threshold_mode,
            processing_time_ms=elapsed_ms,
            timestamp=datetime.now(UTC),
            spectrum=intensities.astype(float).tolist(),
        )

    def _build_predictions(self, probabilities: np.ndarray) -> list[FunctionalGroupPrediction]:
        components = self._components
        items: list[FunctionalGroupPrediction] = []
        for index, name in enumerate(components.class_names):
            prob = float(probabilities[index])
            threshold = float(components.thresholds[index])
            items.append(
                FunctionalGroupPrediction(
                    code=components.class_codes[index],
                    name=name,
                    probability=prob,
                    threshold=threshold,
                    predicted=prob >= threshold,
                )
            )
        return items

    def _tag_candidates(
        self,
        raw_candidates: list[RetrievalCandidate],
        predicted_names: list[str],
    ) -> list[CompoundCandidate]:
        results: list[CompoundCandidate] = []
        for raw in raw_candidates:
            try:
                actual_groups = candidate_groups_from_smiles(raw.smiles)
            except ValueError as exc:
                log.warning(
                    "candidate_groups_failed",
                    smiles=raw.smiles,
                    reason=str(exc),
                )
                actual_groups = ()
            consistency = compute_consistency(
                predicted_names, actual_groups, threshold=self._consistency_threshold
            )
            metadata = self._lookup_compound_metadata(raw.smiles)
            results.append(
                CompoundCandidate(
                    rank=raw.rank,
                    compound_id=raw.compound_id,
                    smiles=raw.smiles,
                    name=metadata.get("name"),
                    formula=metadata.get("formula"),
                    cas_number=metadata.get("cas_number"),
                    score=float(raw.score),
                    consistent=consistency.consistent,
                    jaccard=float(consistency.jaccard),
                    matched_groups=consistency.matched,
                    missing_groups=consistency.missing,
                    extra_groups=consistency.extra,
                )
            )
        return results

    def _lookup_compound_metadata(self, smiles: str) -> dict[str, str | None]:
        """Подтягивает поля Compound из БД, если фабрика сессии задана.

        На phase 1 БД обычно пустая (синтетика не сидируется), и метаданные
        остаются ``None`` — фронт показывает только SMILES и formula по
        SMILES (если потребуется — отдельная задача Этапа 9/13).
        """
        if self._db_session_factory is None:
            return {"name": None, "formula": None, "cas_number": None}
        try:
            session_gen = self._db_session_factory()
            session = next(session_gen)
        except StopIteration:
            return {"name": None, "formula": None, "cas_number": None}
        try:
            from sqlalchemy import select

            from app.db.models.compound import Compound

            stmt = select(Compound).where(Compound.smiles_canonical == smiles)
            compound = session.scalars(stmt).one_or_none()
        finally:
            with contextlib.suppress(StopIteration):
                next(session_gen, None)
        if compound is None:
            return {"name": None, "formula": None, "cas_number": None}
        return {
            "name": compound.name,
            "formula": compound.molecular_formula,
            "cas_number": compound.cas_number,
        }


def _all_equal(arr: np.ndarray) -> bool:
    """True, если все элементы массива равны (используется для threshold_mode)."""
    return bool(np.all(arr == arr.flat[0]))


__all__ = ["InferenceService"]
