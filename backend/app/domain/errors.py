"""Иерархия доменных исключений SpectrumAI.

API-слой маппит эти исключения в соответствующие HTTP-статусы (CLAUDE.md §6).
Доменный код и сервисы должны кидать только наследников ``DomainError``.
"""

from __future__ import annotations


class DomainError(Exception):
    """Базовый класс всех доменных исключений приложения."""


class EntityNotFoundError(DomainError):
    """Запрашиваемая сущность не найдена в хранилище."""

    def __init__(self, entity: str, identifier: object) -> None:
        super().__init__(f"{entity} с идентификатором {identifier!r} не найдена")
        self.entity = entity
        self.identifier = identifier
