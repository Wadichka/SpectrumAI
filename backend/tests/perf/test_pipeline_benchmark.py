"""Полный pipeline идентификации через HTTP (NFR-01, §11.6.5).

Замеряет цепочку parser → preprocess → CNN-inference → cross-validation
из главы 4 §4.4 и сопоставляет с порогами таблицы 11.4:

- target end-to-end latency  ≤ 1.0 с median (целевой)
- acceptance latency         ≤ 2.0 с (NFR-01 — приёмочный)

В качестве клиента — синхронный ``fastapi.testclient.TestClient`` (бенчмарки
работают только в синхронном режиме). DB и ML-сервис подменены на лёгкие
in-memory варианты, без обращения к внешним PostgreSQL/Redis — это даёт
устойчивые измерения именно CPU-bound пути.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.v1.dependencies import get_inference_service
from app.db.models import Base, FunctionalGroup
from app.db.session import get_db
from app.main import app
from app.ml.components import MLComponents
from app.ml.inference_service import InferenceService
from pipelines.models.cnn1d import FunctionalGroupsCNN

_NFR_LIMIT_S = 2.0
_JCAMP_SAMPLE = (
    b"##TITLE=perf-sample\n##JCAMP-DX=4.24\n##DATA TYPE=INFRARED SPECTRUM\n"
    b"##XUNITS=1/CM\n##YUNITS=ABSORBANCE\n##FIRSTX=400\n##LASTX=405\n"
    b"##NPOINTS=6\n##XFACTOR=1\n##YFACTOR=1\n##XYDATA=(X++(Y..Y))\n"
    b"400 0.10 0.22 0.34 0.41 0.28 0.13\n##END=\n"
)

_FG_NAMES = tuple(f"FG{i:02d}" for i in range(1, 26))


_BLOCKS = [
    {"in_channels": 1, "out_channels": 32, "kernel_size": 11, "padding": 5, "dropout": 0.10},
    {"in_channels": 32, "out_channels": 64, "kernel_size": 9, "padding": 4, "dropout": 0.15},
    {"in_channels": 64, "out_channels": 128, "kernel_size": 7, "padding": 3, "dropout": 0.20},
    {"in_channels": 128, "out_channels": 256, "kernel_size": 5, "padding": 2, "dropout": 0.25},
    {"in_channels": 256, "out_channels": 256, "kernel_size": 3, "padding": 1, "dropout": 0.0},
]


def _make_components() -> MLComponents:
    cnn = FunctionalGroupsCNN(blocks=_BLOCKS, embedding_dim=128, n_classes=25)
    cnn.eval()
    return MLComponents(
        cnn=cnn,
        spectrum_tower=None,
        retriever=None,
        class_names=_FG_NAMES,
        class_codes=_FG_NAMES,
        thresholds=np.full(25, 0.5, dtype=np.float32),
        default_threshold=0.5,
        device=torch.device("cpu"),
        model_versions={"mode": "perf-benchmark"},
    )


@pytest.fixture()
def http_client() -> Iterator[TestClient]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    # Минимальный посев FG, чтобы /identify не падал на пустой БД.
    with factory() as seed:
        seed.add_all(
            FunctionalGroup(code=name, name=name, smarts_pattern="C") for name in _FG_NAMES
        )
        seed.commit()

    components = _make_components()
    service = InferenceService(components)

    def _override_get_db() -> Iterator[Session]:
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_inference_service] = lambda: service

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
    engine.dispose()


def _post_identify(client: TestClient) -> None:
    response = client.post(
        "/api/v1/identify",
        files={"file": ("perf.jdx", _JCAMP_SAMPLE, "application/octet-stream")},
    )
    assert response.status_code == 200, response.text


def test_full_identify_pipeline_meets_nfr(
    benchmark,  # type: ignore[no-untyped-def]
    http_client: TestClient,
) -> None:
    """Полный цикл /identify не должен пробивать 2-секундный бюджет (NFR-01)."""
    benchmark.pedantic(
        _post_identify,
        args=(http_client,),
        rounds=10,
        warmup_rounds=2,
        iterations=1,
    )
    assert benchmark.stats.stats.median < _NFR_LIMIT_S
    assert benchmark.stats.stats.max < _NFR_LIMIT_S
