"""Тесты ``ModelVersionRepository``."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.db.models.model_version import ModelVersion
from app.db.repositories.model_version import ModelVersionRepository


def test_latest_by_type_returns_most_recent(db_session: Session) -> None:
    repo = ModelVersionRepository(db_session)
    repo.add(
        ModelVersion(
            model_type="classifier",
            version="0.1.0",
            file_path="/models/cls-0.1.0.pt",
            trained_at=datetime(2026, 1, 10),
        )
    )
    repo.add(
        ModelVersion(
            model_type="classifier",
            version="0.2.0",
            file_path="/models/cls-0.2.0.pt",
            trained_at=datetime(2026, 4, 1),
        )
    )
    repo.add(
        ModelVersion(
            model_type="embedder",
            version="0.1.0",
            file_path="/models/emb-0.1.0.pt",
            trained_at=datetime(2026, 5, 1),
        )
    )
    db_session.flush()

    latest_cls = repo.latest_by_type("classifier")
    assert latest_cls is not None
    assert latest_cls.version == "0.2.0"

    latest_emb = repo.latest_by_type("embedder")
    assert latest_emb is not None
    assert latest_emb.version == "0.1.0"


def test_latest_by_type_returns_none_when_absent(db_session: Session) -> None:
    repo = ModelVersionRepository(db_session)
    assert repo.latest_by_type("unknown") is None
