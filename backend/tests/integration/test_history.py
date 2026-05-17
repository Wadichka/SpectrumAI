"""Интеграционные тесты GET /api/v1/history (UC-08)."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import AsyncClient

_FIXTURE = (
    Path(__file__).resolve().parents[1] / "unit" / "parsing" / "fixtures" / "valid_ethanol.jdx"
)


@pytest.mark.asyncio
async def test_history_empty(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/history?page=1&size=20")
    assert response.status_code == 200
    body = response.json()
    assert body["data"] == []
    assert body["total"] == 0


@pytest.mark.asyncio
async def test_history_after_identify(async_client: AsyncClient) -> None:
    file_bytes = _FIXTURE.read_bytes()
    identify_response = await async_client.post(
        "/api/v1/identify",
        files={"file": ("ethanol.jdx", file_bytes, "chemical/x-jcamp-dx")},
        data={"include_gradcam": "false"},
    )
    assert identify_response.status_code == 200, identify_response.text

    history_response = await async_client.get("/api/v1/history")
    assert history_response.status_code == 200
    body = history_response.json()
    assert body["total"] >= 1
    entry = body["data"][0]
    assert entry["input_filename"] == "ethanol.jdx"
    assert entry["status"] == "success"
