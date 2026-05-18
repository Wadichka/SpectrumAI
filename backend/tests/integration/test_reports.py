"""Integration-тест эндпоинта POST /api/v1/reports/identification."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from httpx import AsyncClient


def _build_payload() -> dict[str, object]:
    return {
        "request_id": 7,
        "predictions": [
            {
                "code": "FG01",
                "name": "alcohol_OH",
                "probability": 0.92,
                "threshold": 0.5,
                "predicted": True,
            }
        ],
        "candidates": [
            {
                "rank": 1,
                "compound_id": 42,
                "smiles": "CCO",
                "name": "ethanol",
                "formula": "C2H6O",
                "cas_number": "64-17-5",
                "score": 0.95,
                "consistent": True,
                "jaccard": 0.8,
                "matched_groups": ["alcohol_OH"],
                "missing_groups": [],
                "extra_groups": [],
            }
        ],
        "gradcam": None,
        "spectrum": [0.1] * 3601,
        "spectrum_length": 3601,
        "model_versions": {"mode": "test"},
        "threshold_mode": "fixed",
        "processing_time_ms": 12,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@pytest.mark.asyncio
async def test_report_endpoint_returns_pdf(async_client: AsyncClient) -> None:
    response = await async_client.post(
        "/api/v1/reports/identification",
        json=_build_payload(),
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content[:5] == b"%PDF-"
    disposition = response.headers.get("content-disposition", "")
    assert "identification-7.pdf" in disposition


@pytest.mark.asyncio
async def test_report_endpoint_without_request_id(async_client: AsyncClient) -> None:
    payload = _build_payload()
    payload["request_id"] = None
    response = await async_client.post("/api/v1/reports/identification", json=payload)
    assert response.status_code == 200
    assert "identification-unsaved.pdf" in response.headers.get("content-disposition", "")


@pytest.mark.asyncio
async def test_report_endpoint_validation_error(async_client: AsyncClient) -> None:
    response = await async_client.post(
        "/api/v1/reports/identification",
        json={"predictions": []},  # неполный payload
    )
    assert response.status_code in (400, 422)
