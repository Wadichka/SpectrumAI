"""Интеграционные тесты POST /api/v1/identify/batch (UC-06)."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import AsyncClient

_FIXTURE = (
    Path(__file__).resolve().parents[1] / "unit" / "parsing" / "fixtures" / "valid_ethanol.jdx"
)


@pytest.mark.asyncio
async def test_batch_two_files(async_client: AsyncClient) -> None:
    data = _FIXTURE.read_bytes()
    response = await async_client.post(
        "/api/v1/identify/batch",
        files=[
            ("files", ("a.jdx", data, "chemical/x-jcamp-dx")),
            ("files", ("b.jdx", data, "chemical/x-jcamp-dx")),
        ],
        data={"include_gradcam": "false"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["items"]) == 2
    assert all(item["status"] == "success" for item in body["items"])
    assert body["total_processing_time_ms"] >= 0


@pytest.mark.asyncio
async def test_batch_mixed_success_and_error(async_client: AsyncClient) -> None:
    data = _FIXTURE.read_bytes()
    response = await async_client.post(
        "/api/v1/identify/batch",
        files=[
            ("files", ("good.jdx", data, "chemical/x-jcamp-dx")),
            ("files", ("bad.jdx", b"not a spectrum", "text/plain")),
        ],
        data={"include_gradcam": "false"},
    )
    assert response.status_code == 200
    items = response.json()["items"]
    assert {i["filename"]: i["status"] for i in items} == {
        "good.jdx": "success",
        "bad.jdx": "error",
    }
