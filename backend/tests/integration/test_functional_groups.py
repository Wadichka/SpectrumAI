"""Интеграционные тесты /api/v1/functional-groups."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_functional_groups_returns_25(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/functional-groups")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 25
    codes = {item["code"] for item in data}
    assert "FG01" in codes
    assert "FG25" in codes
    first = data[0]
    assert {"code", "name"}.issubset(first.keys())
