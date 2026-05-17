"""Интеграционные тесты /api/v1/compounds (UC-02, UC-04)."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.orm import Session

from app.db.models.compound import Compound


@pytest.mark.asyncio
async def test_search_empty_database(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/compounds?page=1&size=10")
    assert response.status_code == 200
    body = response.json()
    assert body["data"] == []
    assert body["total"] == 0


@pytest.mark.asyncio
async def test_search_finds_seeded_compound(
    async_client: AsyncClient, integration_db_session: Session
) -> None:
    compound = Compound(
        name="ethanol",
        smiles_canonical="CCO",
        inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
        inchi_key="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        molecular_formula="C2H6O",
        molecular_weight=46.07,
    )
    integration_db_session.add(compound)
    integration_db_session.commit()

    response = await async_client.get("/api/v1/compounds?q=etha")
    assert response.status_code == 200
    data = response.json()["data"]
    assert any(item["smiles"] == "CCO" for item in data)


@pytest.mark.asyncio
async def test_compound_detail_returns_404_when_missing(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/compounds/999")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["code"] == "ENTITY_NOT_FOUND"


@pytest.mark.asyncio
async def test_structure_svg_returns_svg(
    async_client: AsyncClient, integration_db_session: Session
) -> None:
    compound = Compound(
        name="ethanol",
        smiles_canonical="CCO",
        inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
        inchi_key="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        molecular_formula="C2H6O",
        molecular_weight=46.07,
    )
    integration_db_session.add(compound)
    integration_db_session.commit()

    response = await async_client.get(
        f"/api/v1/compounds/{compound.id}/structure.svg?width=200&height=120"
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert "<svg" in response.text
