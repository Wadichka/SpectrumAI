"""Репозиторий сущности ``SpectrumEmbedding``."""

from __future__ import annotations

from app.db.models.spectrum_embedding import SpectrumEmbedding
from app.db.repositories.base import BaseRepository


class SpectrumEmbeddingRepository(BaseRepository[SpectrumEmbedding]):
    """Доступ к векторным представлениям спектров (1:1 со Spectrum)."""

    model = SpectrumEmbedding

    def get_by_spectrum_id(self, spectrum_id: int) -> SpectrumEmbedding | None:
        """Эмбеддинг по идентификатору спектра."""
        return self.session.get(SpectrumEmbedding, spectrum_id)

    def upsert(
        self,
        *,
        spectrum_id: int,
        embedding_vector: bytes,
        model_version: str,
    ) -> SpectrumEmbedding:
        """Создаёт или обновляет эмбеддинг для спектра.

        Связь 1:1 — если запись уже есть, перезаписываются вектор и версия
        энкодера; смена ``model_version`` для существующего эмбеддинга означает
        пересчёт после смены модели (см. §5.9.2).
        """
        existing = self.get_by_spectrum_id(spectrum_id)
        if existing is None:
            instance = SpectrumEmbedding(
                spectrum_id=spectrum_id,
                embedding_vector=embedding_vector,
                model_version=model_version,
            )
            self.session.add(instance)
            self.session.flush()
            return instance
        existing.embedding_vector = embedding_vector
        existing.model_version = model_version
        self.session.flush()
        return existing
