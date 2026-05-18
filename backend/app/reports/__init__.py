"""Серверная генерация PDF-отчётов (UC-05, FR-09, глава 4 §4.4.7).

Reportlab + matplotlib: одна функция ``render_identification_report``
принимает ``IdentificationResult`` и возвращает ``bytes`` PDF-документа.
"""

from app.reports.pdf_generator import render_identification_report

__all__ = ["render_identification_report"]
