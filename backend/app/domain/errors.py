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


class ParsingError(DomainError):
    """Ошибка разбора файла спектра.

    Возникает при повреждённом или нераспознанном формате, отсутствии
    обязательного заголовочного поля, неподдерживаемых единицах измерения.
    Сообщение содержит ``format_name`` (например, ``"JCAMP-DX"``, ``"CSV"``)
    и ``position`` (имя поля заголовка или номер строки), чтобы пользователь
    мог локализовать проблему (см. главу 3, требование к информативным
    сообщениям об ошибках).
    """

    def __init__(
        self,
        message: str,
        *,
        format_name: str | None = None,
        position: str | int | None = None,
    ) -> None:
        parts: list[str] = []
        if format_name is not None:
            parts.append(f"[{format_name}]")
        if position is not None:
            parts.append(f"@{position}")
        parts.append(message)
        super().__init__(" ".join(parts))
        self.format_name = format_name
        self.position = position


class SpectrumValidationError(DomainError):
    """Семантическая ошибка содержимого спектра.

    Возникает при наличии NaN/inf, выходе за допустимый диапазон волновых
    чисел, нарушении монотонности, несоответствии длины массива заявленному
    ``NPOINTS`` и пр. (см. главу 4 §4.4.1 — перечень проверок валидатора).
    Поле ``field`` указывает на компонент данных, где обнаружена проблема
    (``"wavenumbers"``, ``"intensities"``, ``"npoints"``).
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        full_message = f"[{field}] {message}" if field else message
        super().__init__(full_message)
        self.field = field
