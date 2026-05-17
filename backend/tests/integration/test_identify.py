"""Интеграционные тесты POST /api/v1/identify (UC-01)."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import AsyncClient

_FIXTURE = (
    Path(__file__).resolve().parents[1] / "unit" / "parsing" / "fixtures" / "valid_ethanol.jdx"
)


@pytest.mark.asyncio
async def test_identify_happy_path(async_client: AsyncClient) -> None:
    file_bytes = _FIXTURE.read_bytes()
    response = await async_client.post(
        "/api/v1/identify",
        files={"file": ("ethanol.jdx", file_bytes, "chemical/x-jcamp-dx")},
        data={"include_gradcam": "false", "top_k": "5"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["predictions"]) == 25
    assert isinstance(body["candidates"], list)
    assert body["gradcam"] is None
    assert body["spectrum_length"] == 3601
    assert body["processing_time_ms"] >= 0
    assert "mode" in body["model_versions"]


@pytest.mark.asyncio
async def test_identify_rejects_empty_file(async_client: AsyncClient) -> None:
    response = await async_client.post(
        "/api/v1/identify",
        files={"file": ("empty.jdx", b"", "chemical/x-jcamp-dx")},
        data={"include_gradcam": "false"},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "EMPTY_FILE"


@pytest.mark.asyncio
async def test_identify_invalid_format_returns_422(async_client: AsyncClient) -> None:
    response = await async_client.post(
        "/api/v1/identify",
        files={"file": ("garbage.jdx", b"this is not a spectrum", "text/plain")},
        data={"include_gradcam": "false"},
    )
    assert response.status_code in (400, 422)
    detail = response.json()["detail"]
    assert "code" in detail
