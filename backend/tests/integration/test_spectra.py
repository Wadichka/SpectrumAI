"""Интеграционные тесты POST /api/v1/spectra (UC-03)."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import AsyncClient

_FIXTURE = (
    Path(__file__).resolve().parents[1] / "unit" / "parsing" / "fixtures" / "valid_ethanol.jdx"
)


@pytest.mark.asyncio
async def test_add_spectrum_creates_compound_and_spectrum(
    async_client: AsyncClient,
) -> None:
    response = await async_client.post(
        "/api/v1/spectra",
        files={"file": ("ethanol.jdx", _FIXTURE.read_bytes(), "chemical/x-jcamp-dx")},
        data={"smiles": "CCO", "name": "ethanol", "source": "manual"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "created"
    assert body["spectrum_id"] > 0
    assert body["compound_id"] > 0


@pytest.mark.asyncio
async def test_add_spectrum_invalid_smiles(async_client: AsyncClient) -> None:
    response = await async_client.post(
        "/api/v1/spectra",
        files={"file": ("ethanol.jdx", _FIXTURE.read_bytes(), "chemical/x-jcamp-dx")},
        data={"smiles": "definitely-not-a-smiles", "source": "manual"},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "VALIDATION_ERROR"
