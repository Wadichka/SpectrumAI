"""Unit-тесты загрузки MLComponents (singleton, fallback, no-checkpoint)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from app.core.config import Settings
from app.ml.components import MLComponents, get_ml_components, reset_ml_components


@pytest.fixture(autouse=True)
def _reset_components() -> None:
    """Каждый тест получает свежий singleton."""
    reset_ml_components()


def _make_settings(
    *,
    contrastive: Path | None = None,
    cnn: Path | None = None,
    faiss_root: Path | None = None,
) -> Settings:
    """Создаёт Settings с явными путями (несуществующими, если None)."""
    missing = Path("/__definitely__/not__here__")
    return Settings(
        ml_contrastive_checkpoint=contrastive or missing,
        ml_cnn_checkpoint=cnn or missing,
        ml_faiss_root=faiss_root or missing,
    )


def test_no_checkpoint_raises_runtime_error(tmp_path: Path) -> None:
    settings = _make_settings()
    with pytest.raises(RuntimeError, match="ML-чекпойнт"):
        get_ml_components(settings)


def _save_cnn_state(path: Path) -> None:
    from app.ml.components import _build_default_cnn

    cnn = _build_default_cnn()
    torch.save({"state_dict": cnn.state_dict(), "thresholds": [0.4] * 25}, path)


def test_cnn_only_fallback(tmp_path: Path) -> None:
    cnn_path = tmp_path / "cnn.pt"
    _save_cnn_state(cnn_path)
    settings = _make_settings(cnn=cnn_path)
    components = get_ml_components(settings)
    assert isinstance(components, MLComponents)
    assert components.spectrum_tower is None
    assert components.retriever is None
    assert components.model_versions["mode"] == "cnn-only"
    assert components.thresholds.shape == (25,)
    assert float(components.thresholds[0]) == pytest.approx(0.4)


def test_singleton_returns_same_instance(tmp_path: Path) -> None:
    cnn_path = tmp_path / "cnn.pt"
    _save_cnn_state(cnn_path)
    settings = _make_settings(cnn=cnn_path)
    a = get_ml_components(settings)
    b = get_ml_components(settings)
    assert a is b


def test_reset_invalidates_singleton(tmp_path: Path) -> None:
    cnn_path = tmp_path / "cnn.pt"
    _save_cnn_state(cnn_path)
    settings = _make_settings(cnn=cnn_path)
    a = get_ml_components(settings)
    reset_ml_components()
    b = get_ml_components(settings)
    assert a is not b


def test_contrastive_checkpoint_loaded_when_present(tmp_path: Path) -> None:
    """Сохраняем валидный contrastive-чекпойнт и проверяем загрузку."""
    from app.ml.components import _build_default_cnn, _build_default_spectrum_tower

    cnn = _build_default_cnn()
    tower = _build_default_spectrum_tower(cnn)
    contrastive_path = tmp_path / "contrastive.pt"
    torch.save(
        {
            "spectrum_tower_state_dict": tower.state_dict(),
            "thresholds": [0.5] * 25,
            "class_names": list(components_class_names()),
        },
        contrastive_path,
    )
    settings = _make_settings(contrastive=contrastive_path)
    components = get_ml_components(settings)
    assert components.spectrum_tower is not None
    assert components.model_versions["mode"] == "contrastive"


def components_class_names() -> tuple[str, ...]:
    """Импортируется отдельно, чтобы не путать с фикстурой."""
    from pipelines.labeling import GROUP_NAMES

    return GROUP_NAMES
