"""Graceful degradation API при внутренних ошибках (глава 11 §11.6.4).

Когда ML-инференс или БД неожиданно падают, API должен отдать
структурированный JSON-ответ с осмысленным кодом ошибки и без утечки
стектрейсов. Тесты — self-contained: используют свой in-memory SQLite
и подменяют ML через ``app.dependency_overrides``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.v1.dependencies import get_inference_service
from app.db.models import Base
from app.db.session import get_db
from app.domain.errors import DomainError
from app.main import app

_JCAMP_SAMPLE = (
    b"##TITLE=tiny\n##JCAMP-DX=4.24\n##DATA TYPE=INFRARED SPECTRUM\n"
    b"##XUNITS=1/CM\n##YUNITS=ABSORBANCE\n##FIRSTX=400\n##LASTX=405\n"
    b"##NPOINTS=6\n##XFACTOR=1\n##YFACTOR=1\n##XYDATA=(X++(Y..Y))\n"
    b"400 0.10 0.22 0.34 0.41 0.28 0.13\n##END=\n"
)


class _FailingInferenceService:
    """ML-сервис, всегда падающий с DomainError-ошибкой."""

    async def predict(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise DomainError("ML-инференс недоступен (синтетическая ошибка для теста)")


def _make_in_memory_db() -> sessionmaker[Session]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@pytest_asyncio.fixture()
async def failing_ml_client() -> AsyncIterator[AsyncClient]:
    factory = _make_in_memory_db()

    def _override_get_db() -> Iterator[Session]:
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_inference_service] = lambda: _FailingInferenceService()  # type: ignore[return-value]

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_inference_failure_returns_structured_500(
    failing_ml_client: AsyncClient,
) -> None:
    """При DomainError в ML-сервисе API возвращает 500 с JSON-телом."""
    response = await failing_ml_client.post(
        "/api/v1/identify",
        files={"file": ("tiny.jdx", _JCAMP_SAMPLE, "application/octet-stream")},
    )

    assert response.status_code == 500
    body = response.json()
    assert body["detail"]["code"] == "DOMAIN_ERROR"
    # Тело должно быть JSON, без HTML-стектрейса.
    assert "Traceback" not in response.text


@pytest.mark.asyncio
async def test_invalid_file_returns_422_not_500(
    failing_ml_client: AsyncClient,
) -> None:
    """Битый JCAMP должен вернуть 422 PARSING_ERROR до обращения к ML."""
    response = await failing_ml_client.post(
        "/api/v1/identify",
        files={"file": ("bad.jdx", b"\x00\x01garbage", "application/octet-stream")},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["detail"]["code"] == "PARSING_ERROR"


@pytest.mark.asyncio
async def test_empty_file_returns_validation_error(
    failing_ml_client: AsyncClient,
) -> None:
    """Пустой файл должен вернуть 400/422 с осмысленным сообщением."""
    response = await failing_ml_client.post(
        "/api/v1/identify",
        files={"file": ("empty.jdx", b"", "application/octet-stream")},
    )

    assert response.status_code in (400, 422)
    body = response.json()
    assert "detail" in body
