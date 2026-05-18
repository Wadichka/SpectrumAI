"""Тесты MoleculeTower — используем tiny-mock из HF-internal-testing,
чтобы не тащить 150 МБ ChemBERTa в CI/локальный pytest."""

from __future__ import annotations

import os

import pytest
import torch

from pipelines.models.molecule_tower import (
    DEFAULT_MODEL_NAME,
    TINY_MODEL_NAME,
    MoleculeTower,
    resolve_model_name,
)


@pytest.fixture(scope="module")
def tiny_tower() -> MoleculeTower:
    return MoleculeTower(TINY_MODEL_NAME, projection_dim=8, hidden_dim=16)


def test_resolve_model_name_default_when_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPECTRUMAI_USE_TINY_BERT", raising=False)
    monkeypatch.delenv("SPECTRUMAI_CHEMBERTA_NAME", raising=False)
    assert resolve_model_name() == DEFAULT_MODEL_NAME


def test_resolve_model_name_tiny_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPECTRUMAI_USE_TINY_BERT", "1")
    monkeypatch.delenv("SPECTRUMAI_CHEMBERTA_NAME", raising=False)
    assert resolve_model_name() == TINY_MODEL_NAME


def test_resolve_model_name_explicit_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPECTRUMAI_USE_TINY_BERT", "1")
    assert resolve_model_name("custom/model") == "custom/model"


def test_canonicalize_ethanol() -> None:
    canonical = MoleculeTower.canonicalize("OCC")
    assert canonical in {"CCO", "OCC"}  # RDKit обычно отдаёт CCO


def test_canonicalize_invalid_raises() -> None:
    with pytest.raises(ValueError):
        MoleculeTower.canonicalize("not_a_real_smiles")


def test_forward_shape_and_l2_norm(tiny_tower: MoleculeTower) -> None:
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CN"]
    tiny_tower.eval()
    with torch.no_grad():
        z = tiny_tower(smiles)
    assert tuple(z.shape) == (4, tiny_tower.projection_dim)
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_frozen_encoder_does_not_train(tiny_tower: MoleculeTower) -> None:
    for p in tiny_tower.encoder.parameters():
        assert not p.requires_grad


def test_train_mode_keeps_frozen_encoder_in_eval(tiny_tower: MoleculeTower) -> None:
    tiny_tower.train()
    assert not tiny_tower.encoder.training
    # Проекция при этом должна быть в train.
    assert tiny_tower.projection.training
    tiny_tower.eval()


def test_env_chemberta_name_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPECTRUMAI_CHEMBERTA_NAME", TINY_MODEL_NAME)
    monkeypatch.delenv("SPECTRUMAI_USE_TINY_BERT", raising=False)
    assert resolve_model_name() == TINY_MODEL_NAME
    # Защищаемся от случайной заразы окружения теста.
    assert os.environ["SPECTRUMAI_CHEMBERTA_NAME"] == TINY_MODEL_NAME
