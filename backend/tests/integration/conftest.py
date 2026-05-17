"""Общие фикстуры для интеграционных тестов REST API.

Override-ы зависимостей:
- ``get_db`` → SQLite in-memory с засеянными функциональными группами.
- ``get_inference_service`` → fake InferenceService без реальных чекпойнтов
  (быстрый и детерминированный).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime

import numpy as np
import pytest
import pytest_asyncio
import torch
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.v1.dependencies import get_inference_service
from app.db.models import Base, FunctionalGroup
from app.db.session import get_db
from app.domain.dto import (
    CompoundCandidate,
    FunctionalGroupPrediction,
    IdentificationResult,
)
from app.main import app
from app.ml.components import MLComponents
from app.ml.inference_service import InferenceService

# Полный набор функциональных групп — повторяет sed из миграции 0001_initial.
_FG_SEED = [
    ("FG01", "alcohol_OH", "[OX2H][CX4]"),
    ("FG02", "phenol_OH", "[OX2H][c]"),
    ("FG03", "carbonyl", "[CX3]=[OX1]"),
    ("FG04", "aldehyde", "[CX3H1](=O)[#6]"),
    ("FG05", "ketone", "[#6][CX3](=O)[#6]"),
    ("FG06", "carboxylic_acid", "[CX3](=O)[OX2H]"),
    ("FG07", "ester", "[#6][CX3](=O)[OX2][#6]"),
    ("FG08", "amide_primary", "[CX3](=O)[NX3H2]"),
    ("FG09", "amide_secondary", "[CX3](=O)[NX3H1]"),
    ("FG10", "amide_tertiary", "[CX3](=O)[NX3H0]"),
    ("FG11", "amine_primary", "[NX3;H2;!$(NC=O)]"),
    ("FG12", "amine_secondary", "[NX3;H1;!$(NC=O)]"),
    ("FG13", "amine_tertiary", "[NX3;H0;!$(NC=O)]"),
    ("FG14", "nitrile", "[CX2]#[NX1]"),
    ("FG15", "nitro", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("FG16", "ether", "[OD2]([#6])[#6]"),
    ("FG17", "alkene", "[CX3]=[CX3]"),
    ("FG18", "alkyne", "[CX2]#[CX2]"),
    ("FG19", "aromatic_ring", "c1ccccc1"),
    ("FG20", "ch2_group", "[CH2]"),
    ("FG21", "ch3_group", "[CH3]"),
    ("FG22", "c_f_bond", "[CX4]F"),
    ("FG23", "c_cl_bond", "[CX4]Cl"),
    ("FG24", "sulfoxide_sulfone", "[#6][SX3](=O)[#6],[#6][SX4](=O)(=O)[#6]"),
    ("FG25", "thiol_thioether", "[SX2H],[SX2]([#6])[#6]"),
]


class _FakeCNN(torch.nn.Module):
    """Минимальный двухслойный 1D-CNN для тестов (совместим с GradCAM1D)."""

    def __init__(self, n_classes: int = 25, in_length: int = 3601) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(1, 4, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(4, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
        )
        self.embedding_dim = 4
        self.classifier = torch.nn.Linear(4, n_classes)
        self._in_length = in_length

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 2:
            x = x.unsqueeze(1)
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_embedding(x))


class _FakeInferenceService:
    """Stub-сервис, возвращающий детерминированный IdentificationResult."""

    def __init__(self) -> None:
        self._cnn = _FakeCNN()
        self._components = MLComponents(
            cnn=self._cnn,
            spectrum_tower=None,
            retriever=None,
            class_names=tuple(name for _, name, _ in _FG_SEED),
            class_codes=tuple(code for code, _, _ in _FG_SEED),
            thresholds=np.full(25, 0.5),
            default_threshold=0.5,
            device=torch.device("cpu"),
            model_versions={"mode": "test", "checkpoint": "fake"},
        )

    @property
    def components(self) -> MLComponents:
        return self._components

    async def predict(
        self,
        processed,  # type: ignore[no-untyped-def]
        *,
        top_k: int | None = None,
    ) -> IdentificationResult:
        predictions = [
            FunctionalGroupPrediction(
                code=self._components.class_codes[i],
                name=self._components.class_names[i],
                probability=0.9 if i == 0 else 0.1,
                threshold=0.5,
                predicted=(i == 0),
            )
            for i in range(25)
        ]
        candidates = [
            CompoundCandidate(
                rank=1,
                compound_id=1,
                smiles="CCO",
                name=None,
                formula=None,
                cas_number=None,
                score=0.95,
                consistent=True,
                jaccard=1.0,
                matched_groups=("alcohol_OH",),
                missing_groups=(),
                extra_groups=(),
            )
        ]
        return IdentificationResult(
            predictions=predictions,
            candidates=candidates,
            spectrum_length=3601,
            model_versions=dict(self._components.model_versions),
            threshold_mode="fixed",
            processing_time_ms=12,
            timestamp=datetime.now(UTC),
        )


@pytest.fixture()
def integration_db_session() -> Iterator[Session]:
    """SQLite in-memory с засеянными функциональными группами.

    Используется `StaticPool`, чтобы все open()-вызовы FastAPI-зависимости
    `get_db` шарили один in-memory database.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _enable_fk(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    seed_session = factory()
    seed_session.add_all(
        [
            FunctionalGroup(code=code, name=name, smarts_pattern=smarts)
            for code, name, smarts in _FG_SEED
        ]
    )
    seed_session.commit()
    seed_session.close()

    yield factory()
    engine.dispose()


@pytest_asyncio.fixture()
async def async_client(integration_db_session: Session) -> AsyncIterator[AsyncClient]:
    factory = sessionmaker(
        bind=integration_db_session.bind,
        autoflush=False,
        expire_on_commit=False,
    )

    def _override_get_db() -> Iterator[Session]:
        session = factory()
        try:
            yield session
        finally:
            session.close()

    def _override_get_inference_service() -> InferenceService:
        return _FakeInferenceService()  # type: ignore[return-value]

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_inference_service] = _override_get_inference_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
