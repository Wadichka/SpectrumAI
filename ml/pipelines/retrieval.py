"""FAISS-ретривал кандидатов соединений (§6.9 главы 6).

Инкапсулирует операции с индексом FAISS, построенным по молекулярным
эмбеддингам референсной базы (см. ``ml/scripts/build_faiss_index.py``):
загрузка индекса с диска, поиск top-k ближайших соседей по cosine similarity
(inner product на L2-нормированных векторах), маппинг внутренних индексов
FAISS в записи соединений (``compound_id`` + ``smiles``).

Формат каталога индекса:

    <root>/
        index.faiss           # бинарный индекс (faiss.write_index)
        mapping.json          # [{"id": int, "compound_id": int, "smiles": str}, ...]
        meta.json             # размерность, тип индекса, чекпойнт-источник

Production-ретривал (с подтягиванием полей из БД) — Этап 8/9; здесь только
библиотечная функция.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
from torch import Tensor


@dataclass(frozen=True, slots=True)
class CompoundCandidate:
    """Один кандидат-результат retrieval."""

    rank: int
    compound_id: int
    smiles: str
    score: float  # cosine similarity (inner product на L2-нормированных)


class FaissRetriever:
    """Загрузчик и обёртка над FAISS-индексом для top-k поиска.

    Args:
        index: загруженный FAISS-индекс.
        mapping: список словарей с полями ``compound_id`` и ``smiles``;
            индекс в списке равен внутреннему index_id FAISS.
        meta: служебные метаданные (размерность, имя чекпойнта); опционально.
    """

    def __init__(
        self,
        index: faiss.Index,
        mapping: list[dict[str, Any]],
        *,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if index.ntotal != len(mapping):
            raise ValueError(
                f"размер индекса ({index.ntotal}) не совпадает с длиной mapping ({len(mapping)})"
            )
        self.index = index
        self.mapping = mapping
        self.meta = dict(meta) if meta else {}

    @classmethod
    def load(cls, root: str | Path) -> FaissRetriever:
        """Загружает индекс и mapping из каталога."""
        root_path = Path(root)
        index_path = root_path / "index.faiss"
        mapping_path = root_path / "mapping.json"
        meta_path = root_path / "meta.json"
        if not index_path.exists():
            raise FileNotFoundError(f"индекс не найден: {index_path}")
        if not mapping_path.exists():
            raise FileNotFoundError(f"mapping.json не найден: {mapping_path}")
        index = faiss.read_index(str(index_path))
        mapping: list[dict[str, Any]] = json.loads(mapping_path.read_text(encoding="utf-8"))
        meta: dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(index, mapping, meta=meta)

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    @property
    def dim(self) -> int:
        return int(self.index.d)

    def _to_query(self, query: Tensor | npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        if isinstance(query, Tensor):
            arr = query.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            arr = np.asarray(query, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(
                f"ожидался query формы (D,) или (B, D) с D={self.dim}, получено {arr.shape}"
            )
        return np.ascontiguousarray(arr)

    def find_top_k(self, query: Tensor | npt.NDArray[Any], k: int = 10) -> list[CompoundCandidate]:
        """Top-k кандидатов для одиночного query (берётся 0-я строка)."""
        if k <= 0:
            raise ValueError("k должен быть положительным")
        arr = self._to_query(query)
        k_eff = min(k, self.size)
        scores, indices = self.index.search(arr[:1], k_eff)
        return [
            CompoundCandidate(
                rank=rank + 1,
                compound_id=int(self.mapping[int(idx)]["compound_id"]),
                smiles=str(self.mapping[int(idx)]["smiles"]),
                score=float(score),
            )
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0], strict=True))
        ]

    def find_top_k_batch(
        self, queries: Tensor | npt.NDArray[Any], k: int = 10
    ) -> list[list[CompoundCandidate]]:
        """Top-k для батча query-эмбеддингов."""
        arr = self._to_query(queries)
        k_eff = min(k, self.size)
        scores, indices = self.index.search(arr, k_eff)
        results: list[list[CompoundCandidate]] = []
        for row_scores, row_indices in zip(scores, indices, strict=True):
            results.append(
                [
                    CompoundCandidate(
                        rank=rank + 1,
                        compound_id=int(self.mapping[int(idx)]["compound_id"]),
                        smiles=str(self.mapping[int(idx)]["smiles"]),
                        score=float(score),
                    )
                    for rank, (idx, score) in enumerate(zip(row_indices, row_scores, strict=True))
                ]
            )
        return results


def build_index_flat_ip(
    embeddings: npt.NDArray[np.floating[Any]],
) -> faiss.IndexFlatIP:
    """Создаёт `IndexFlatIP` и заполняет его L2-нормированными векторами.

    Эмбеддинги должны быть L2-нормированы заранее — inner product тогда даёт
    cosine similarity (§6.6.4).
    """
    if embeddings.ndim != 2:
        raise ValueError(f"ожидалась матрица (N, D), получена форма {embeddings.shape}")
    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(embeddings.astype(np.float32)))
    return index


__all__ = ["CompoundCandidate", "FaissRetriever", "build_index_flat_ip"]
