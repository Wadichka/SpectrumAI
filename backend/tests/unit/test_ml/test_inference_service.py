"""Интеграционные unit-тесты InferenceService (Этап 8)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import faiss  # type: ignore[import-untyped]
import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from app.domain.dto import IdentificationResult
from app.ml.components import MLComponents
from app.ml.inference_service import InferenceService
from app.preprocessing.config import ProcessedSpectrum
from pipelines.labeling import FUNCTIONAL_GROUPS, GROUP_NAMES, N_GROUPS
from pipelines.models.cnn1d import FunctionalGroupsCNN
from pipelines.models.spectrum_tower import SpectrumTower
from pipelines.retrieval import FaissRetriever

_TINY_BLOCKS: list[dict[str, int | float]] = [
    {"in_channels": 1, "out_channels": 8, "kernel_size": 5, "padding": 2, "dropout": 0.0},
    {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "padding": 1, "dropout": 0.0},
]


def _tiny_cnn() -> FunctionalGroupsCNN:
    """Минимальный CNN с 25 классами для скоростного теста."""
    return FunctionalGroupsCNN(_TINY_BLOCKS, embedding_dim=16, head_dropout=0.0, n_classes=N_GROUPS)


def _tiny_tower(cnn: FunctionalGroupsCNN) -> SpectrumTower:
    return SpectrumTower(cnn, projection_dim=8, hidden_dim=12, dropout=0.0)


def _make_faiss_index(tmp_path: Path, *, n: int = 4, dim: int = 8) -> Path:
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors = F.normalize(torch.from_numpy(vectors), p=2, dim=-1).numpy()
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(vectors))
    root = tmp_path / "faiss"
    root.mkdir()
    faiss.write_index(index, str(root / "index.faiss"))
    smiles_pool = ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]
    mapping = [{"id": i, "compound_id": 1000 + i, "smiles": smiles_pool[i]} for i in range(n)]
    (root / "mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    return root


@pytest.fixture()
def processed_spectrum() -> ProcessedSpectrum:
    rng = np.random.default_rng(0)
    intensities = rng.random(256).astype(np.float64)
    wavenumbers = np.linspace(400, 4000, intensities.size, dtype=np.float64)
    return ProcessedSpectrum(wavenumbers=wavenumbers, intensities=intensities, metadata={})


def _build_components(
    *, retriever: FaissRetriever | None = None, with_tower: bool = True
) -> MLComponents:
    cnn = _tiny_cnn()
    cnn.eval()
    tower = _tiny_tower(cnn) if with_tower else None
    if tower is not None:
        tower.eval()
    code_lookup = {g.name: g.code for g in FUNCTIONAL_GROUPS}
    class_codes = tuple(code_lookup[name] for name in GROUP_NAMES)
    return MLComponents(
        cnn=cnn,
        spectrum_tower=tower,
        retriever=retriever,
        class_names=GROUP_NAMES,
        class_codes=class_codes,
        thresholds=np.full(N_GROUPS, 0.5),
        default_threshold=0.5,
        device=torch.device("cpu"),
        model_versions={"mode": "test", "checkpoint": "tiny"},
    )


def test_predict_returns_identification_result(
    processed_spectrum: ProcessedSpectrum,
) -> None:
    components = _build_components(retriever=None, with_tower=False)
    service = InferenceService(components)
    result = asyncio.run(service.predict(processed_spectrum))
    assert isinstance(result, IdentificationResult)
    assert len(result.predictions) == N_GROUPS
    assert result.candidates == []
    assert result.spectrum_length == 256
    assert result.model_versions["mode"] == "test"
    assert result.threshold_mode == "fixed"


def test_predict_with_retriever_returns_candidates(
    tmp_path: Path, processed_spectrum: ProcessedSpectrum
) -> None:
    faiss_root = _make_faiss_index(tmp_path, dim=8)
    retriever = FaissRetriever.load(faiss_root)
    components = _build_components(retriever=retriever)
    service = InferenceService(components)
    result = asyncio.run(service.predict(processed_spectrum, top_k=3))
    assert len(result.candidates) == 3
    assert all(0.0 <= c.jaccard <= 1.0 for c in result.candidates)
    assert all(isinstance(c.consistent, bool) for c in result.candidates)
    # Ранги — 1..K.
    assert [c.rank for c in result.candidates] == [1, 2, 3]


def test_predict_per_class_threshold_mode(
    processed_spectrum: ProcessedSpectrum,
) -> None:
    components = _build_components(retriever=None, with_tower=False)
    varied_thresholds = np.linspace(0.1, 0.9, N_GROUPS)
    components = MLComponents(
        cnn=components.cnn,
        spectrum_tower=components.spectrum_tower,
        retriever=components.retriever,
        class_names=components.class_names,
        class_codes=components.class_codes,
        thresholds=varied_thresholds,
        default_threshold=components.default_threshold,
        device=components.device,
        model_versions=components.model_versions,
    )
    service = InferenceService(components)
    result = asyncio.run(service.predict(processed_spectrum))
    assert result.threshold_mode == "per_class"
    # Каждое предсказание использует свой threshold.
    for i, pred in enumerate(result.predictions):
        assert pred.threshold == pytest.approx(varied_thresholds[i])


def test_predict_predicted_flag_consistent_with_threshold(
    processed_spectrum: ProcessedSpectrum,
) -> None:
    """Бит ``predicted`` = (probability >= threshold)."""
    components = _build_components(retriever=None, with_tower=False)
    service = InferenceService(components)
    result = asyncio.run(service.predict(processed_spectrum))
    for pred in result.predictions:
        assert pred.predicted == (pred.probability >= pred.threshold)
