"""Тесты FaissRetriever (§6.9)."""

from __future__ import annotations

import json
from pathlib import Path

import faiss  # type: ignore[import-untyped]
import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from pipelines.retrieval import (
    CompoundCandidate,
    FaissRetriever,
    build_index_flat_ip,
)


@pytest.fixture()
def mini_index(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((5, 16)).astype(np.float32)
    z = torch.from_numpy(raw)
    z = F.normalize(z, p=2, dim=-1)
    index = build_index_flat_ip(z.numpy())
    root = tmp_path / "mini"
    root.mkdir()
    faiss.write_index(index, str(root / "index.faiss"))
    mapping = [{"id": i, "compound_id": i + 100, "smiles": f"C{i}"} for i in range(5)]
    (root / "mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    (root / "meta.json").write_text(json.dumps({"dim": 16, "n_vectors": 5}), encoding="utf-8")
    return root


def test_load_and_size(mini_index: Path) -> None:
    retr = FaissRetriever.load(mini_index)
    assert retr.size == 5
    assert retr.dim == 16


def test_find_top_k_returns_candidates(mini_index: Path) -> None:
    retr = FaissRetriever.load(mini_index)
    query = np.zeros((16,), dtype=np.float32)
    query[0] = 1.0  # любой единичный вектор
    candidates = retr.find_top_k(query, k=3)
    assert len(candidates) == 3
    assert all(isinstance(c, CompoundCandidate) for c in candidates)
    assert [c.rank for c in candidates] == [1, 2, 3]
    # Score должны идти по убыванию (cosine similarity на нормированных).
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_find_top_k_uses_internal_index_mapping(mini_index: Path) -> None:
    retr = FaissRetriever.load(mini_index)
    # Берём один из векторов из самого индекса в качестве запроса — гарантированно
    # вернёт его же первым (score ≈ 1).
    stored = retr.index.reconstruct(2)
    candidates = retr.find_top_k(stored, k=1)
    assert candidates[0].compound_id == 102  # 2 + 100
    assert candidates[0].smiles == "C2"
    assert candidates[0].score == pytest.approx(1.0, abs=1e-4)


def test_batch_search(mini_index: Path) -> None:
    retr = FaissRetriever.load(mini_index)
    queries = np.stack([retr.index.reconstruct(i) for i in range(3)], axis=0)
    results = retr.find_top_k_batch(queries, k=2)
    assert len(results) == 3
    for i, row in enumerate(results):
        assert row[0].compound_id == i + 100
        assert row[0].score == pytest.approx(1.0, abs=1e-4)


def test_invalid_query_shape_raises(mini_index: Path) -> None:
    retr = FaissRetriever.load(mini_index)
    with pytest.raises(ValueError):
        retr.find_top_k(np.zeros((8,), dtype=np.float32), k=1)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FaissRetriever.load(tmp_path / "nope")
