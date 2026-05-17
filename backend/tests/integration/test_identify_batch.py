"""Интеграционные тесты POST /api/v1/identify/batch (UC-06)."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import AsyncClient

_FIXTURE = (
    Path(__file__).resolve().parents[1] / "unit" / "parsing" / "fixtures" / "valid_ethanol.jdx"
)


@pytest.mark.asyncio
async def test_batch_happy_path_3_files(async_client: AsyncClient) -> None:
    payload = _FIXTURE.read_bytes()
    files = [
        ("files", (f"sample_{i}.jdx", payload, "chemical/x-jcamp-dx")) for i in range(3)
    ]
    response = await async_client.post(
        "/api/v1/identify/batch",
        files=files,
        data={"include_gradcam": "false", "top_k": "5"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["items"]) == 3
    assert {item["status"] for item in body["items"]} == {"success"}
    assert body["total_processing_time_ms"] >= 0
    for item in body["items"]:
        assert item["error"] is None
        assert item["result"] is not None
        assert len(item["result"]["predictions"]) == 25


@pytest.mark.asyncio
async def test_batch_partial_failure(async_client: AsyncClient) -> None:
    valid = _FIXTURE.read_bytes()
    files = [
        ("files", ("good.jdx", valid, "chemical/x-jcamp-dx")),
        ("files", ("garbage.jdx", b"this is not a spectrum", "text/plain")),
    ]
    response = await async_client.post(
        "/api/v1/identify/batch",
        files=files,
        data={"include_gradcam": "false"},
    )
    assert response.status_code == 200, response.text
    items = response.json()["items"]
    assert len(items) == 2
    statuses = {item["filename"]: item["status"] for item in items}
    assert statuses["good.jdx"] == "success"
    assert statuses["garbage.jdx"] == "error"
    bad = next(item for item in items if item["filename"] == "garbage.jdx")
    assert bad["result"] is None
    assert bad["error"] is not None
    assert "code" in bad["error"]


@pytest.mark.asyncio
async def test_batch_rejects_too_many_files(async_client: AsyncClient) -> None:
    payload = _FIXTURE.read_bytes()
    files = [
        ("files", (f"sample_{i}.jdx", payload, "chemical/x-jcamp-dx")) for i in range(21)
    ]
    response = await async_client.post("/api/v1/identify/batch", files=files)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "TOO_MANY_FILES"


@pytest.mark.asyncio
async def test_batch_rejects_too_large_payload(async_client: AsyncClient) -> None:
    big_chunk = b"x" * (30 * 1024 * 1024)  # 30 МБ × 2 = 60 МБ > 50 МБ
    files = [
        ("files", ("first.csv", big_chunk, "text/csv")),
        ("files", ("second.csv", big_chunk, "text/csv")),
    ]
    response = await async_client.post("/api/v1/identify/batch", files=files)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "PAYLOAD_TOO_LARGE"
