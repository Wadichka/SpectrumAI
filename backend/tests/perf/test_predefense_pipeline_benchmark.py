"""SLA-бенчмарк на реальных предзащитных чекпойнтах (фаза 2, NFR-01).

Отличается от ``test_pipeline_benchmark.py`` тем, что загружает РЕАЛЬНЫЕ
``cnn1d-predefense-0.5.0`` + ``contrastive-predefense-0.5.0`` + FAISS.
Цель — убедиться, что end-to-end ``/api/v1/identify`` на CPU укладывается
в 2-секундный бюджет (NFR-01 главы 3, таблица 11.4) с фактическими весами,
а не стабами.

Тест помечен ``@pytest.mark.slow`` — пропускается без явного
``-m slow``, чтобы не тянуть 5+ ГБ моделей в обычном CI. Запуск:

    pytest backend/tests/perf/test_predefense_pipeline_benchmark.py -m slow -v
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKEND_ROOT = _REPO_ROOT / "backend"
_ML_ROOT = _REPO_ROOT / "ml"
for path in (_BACKEND_ROOT, _ML_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.api.v1.dependencies import get_inference_service  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.db.models import Base, FunctionalGroup  # noqa: E402
from app.db.session import get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.ml.components import _build_components, reset_ml_components  # noqa: E402
from app.ml.inference_service import InferenceService  # noqa: E402
from pipelines.labeling import FUNCTIONAL_GROUPS  # noqa: E402

_NFR_LIMIT_S = 2.0
_TARGET_MEDIAN_S = 1.0  # «целевой» бюджет таблицы 11.4 (информативно).

_CNN_CKPT = _REPO_ROOT / "models" / "checkpoints" / "cnn1d-predefense-0.5.0" / "best.pt"
_CONTRASTIVE_CKPT = (
    _REPO_ROOT / "models" / "checkpoints" / "contrastive-predefense-0.5.0" / "best.pt"
)
_FAISS_ROOT = _REPO_ROOT / "models" / "faiss" / "contrastive-predefense-0.5.0"
_DEMO_JDX = _REPO_ROOT / "demo" / "predefense_spectra" / "C1194021.jdx"


pytestmark = pytest.mark.slow


def _ensure_artifacts() -> None:
    """Гарантирует, что предзащитные артефакты на месте; иначе skip."""
    missing = [
        str(p)
        for p in (_CNN_CKPT, _CONTRASTIVE_CKPT, _FAISS_ROOT / "index.faiss", _DEMO_JDX)
        if not p.exists()
    ]
    if missing:
        pytest.skip(f"predefense artifacts missing: {missing}")


@pytest.fixture(scope="module")
def predefense_components():  # type: ignore[no-untyped-def]
    """Загружает реальные CNN + SpectrumTower + FAISS retriever."""
    _ensure_artifacts()
    reset_ml_components()
    settings = Settings(
        ml_contrastive_checkpoint=_CONTRASTIVE_CKPT,
        ml_cnn_checkpoint=_CNN_CKPT,
        ml_faiss_root=_FAISS_ROOT,
        ml_device="cpu",
    )
    return _build_components(settings)


@pytest.fixture()
def http_client(predefense_components) -> Iterator[TestClient]:  # type: ignore[no-untyped-def]
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    # Засеиваем реальные FG (codes/names), чтобы /identify умел их разрешить.
    with factory() as seed:
        seed.add_all(
            FunctionalGroup(code=g.code, name=g.name, smarts_pattern=g.smarts)
            for g in FUNCTIONAL_GROUPS
        )
        seed.commit()

    service = InferenceService(predefense_components)

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


def _post_identify(client: TestClient, payload: bytes) -> None:
    response = client.post(
        "/api/v1/identify",
        files={"file": ("demo.jdx", payload, "application/octet-stream")},
    )
    assert response.status_code == 200, response.text


def test_predefense_identify_meets_nfr(  # type: ignore[no-untyped-def]
    benchmark,
    http_client: TestClient,
) -> None:
    """`/api/v1/identify` с предзащитными весами не пробивает 2-секундный бюджет."""
    payload = _DEMO_JDX.read_bytes()
    benchmark.pedantic(
        _post_identify,
        args=(http_client, payload),
        rounds=10,
        warmup_rounds=2,
        iterations=1,
    )
    assert benchmark.stats.stats.median < _NFR_LIMIT_S, (
        f"median {benchmark.stats.stats.median:.3f}s превышает NFR-01 ({_NFR_LIMIT_S}s)"
    )
    assert benchmark.stats.stats.max < _NFR_LIMIT_S, (
        f"max {benchmark.stats.stats.max:.3f}s превышает NFR-01 ({_NFR_LIMIT_S}s)"
    )
