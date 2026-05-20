"""identification_request.result_payload JSONB

Добавляет nullable-колонку `result_payload` в `identification_request`,
куда сохраняется полный сериализованный ответ /identify
(см. §20 предзащиты). Это нужно, чтобы открытие исторической записи
из UI отдавало те же predictions/candidates/gradcam, что и при свежей
идентификации, а не текущий live-state из in-memory store фронта.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "identification_request",
        sa.Column(
            "result_payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("identification_request", "result_payload")
