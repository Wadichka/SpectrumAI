"""PDF-генерация отчёта об идентификации (UC-05 / FR-09).

Использует reportlab для разметки и matplotlib (Agg-backend) для
встраиваемого PNG-графика спектра. Реализация stateless: на вход —
``IdentificationResult``, на выход — ``bytes`` готового PDF.

Layout (по плану защиты):

1. Шапка: «Отчёт об идентификации», дата, время обработки, версии моделей.
2. График спектра 400–4000 см⁻¹ (matplotlib → PNG → ReportLab Image).
3. Таблица предсказанных функциональных групп (top-N).
4. Таблица кандидатов-соединений (top-K) с score/Jaccard.
5. (Опц.) График Grad-CAM, если payload содержит карту.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.domain.dto import IdentificationResult

# DejaVu Sans (Regular + Bold) даёт корректную кириллицу в Paragraph/Table.
# В Docker backend пакет fonts-dejavu установлен в Dockerfile (runtime stage).
# Локально под Windows пользуем системный Arial; в остальных случаях fallback
# на Helvetica — без кириллицы, но без падения генерации.
_REGULAR_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
)
_BOLD_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


def _register_unicode_font() -> tuple[str, str]:
    """Регистрирует TTF-шрифты для кириллицы.

    Возвращает (regular_name, bold_name). При невозможности зарегистрировать
    Regular — оба значения = ``"Helvetica"`` (кириллица не отобразится, но
    PDF соберётся). Если есть Regular, но нет Bold — bold подставляем Regular.
    """
    regular = next((p for p in _REGULAR_CANDIDATES if Path(p).exists()), None)
    if regular is None:
        return "Helvetica", "Helvetica"
    pdfmetrics.registerFont(TTFont("ReportFont", regular))
    bold = next((p for p in _BOLD_CANDIDATES if Path(p).exists()), None)
    if bold is not None:
        pdfmetrics.registerFont(TTFont("ReportFont-Bold", bold))
        pdfmetrics.registerFontFamily(
            "ReportFont",
            normal="ReportFont",
            bold="ReportFont-Bold",
            italic="ReportFont",
            boldItalic="ReportFont-Bold",
        )
        return "ReportFont", "ReportFont-Bold"
    return "ReportFont", "ReportFont"


_UNICODE_FONT, _UNICODE_FONT_BOLD = _register_unicode_font()

# matplotlib рисует график спектра/Grad-CAM с подписями на русском.
# DejaVu Sans поставляется с matplotlib, но явно фиксируем семейство —
# страховка на случай нестандартных rc-файлов.
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def _render_spectrum_png(intensities: list[float], length: int) -> bytes:
    """Рисует спектр matplotlib'ом, возвращает PNG-байты для встраивания в PDF."""
    grid = np.linspace(400.0, 4000.0, length)
    fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=120)
    ax.plot(grid, intensities, color="#1f77b4", linewidth=0.9)
    ax.set_xlabel("Волновое число, см⁻¹")
    ax.set_ylabel("Поглощение")
    ax.set_xlim(400, 4000)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # ИК-спектры традиционно с убывающим x
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def _render_gradcam_png(values: list[float], group_name: str) -> bytes:
    """Рисует Grad-CAM активацию по сетке 400–4000 см⁻¹."""
    grid = np.linspace(400.0, 4000.0, len(values))
    fig, ax = plt.subplots(figsize=(7.5, 2.4), dpi=120)
    ax.fill_between(grid, 0, values, alpha=0.6, color="#d62728")
    ax.set_xlabel("Волновое число, см⁻¹")
    ax.set_ylabel(f"Активация ({group_name})")
    ax.set_xlim(400, 4000)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def render_identification_report(
    result: IdentificationResult,
    *,
    request_id: int | None = None,
    top_predictions: int = 10,
) -> bytes:
    """Собирает PDF-отчёт и возвращает байты.

    Args:
        result: полный DTO результата идентификации.
        request_id: опциональный идентификатор запроса для шапки/имени файла.
        top_predictions: сколько строк включить в таблицу функциональных групп.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title=f"SpectrumAI — отчёт идентификации {request_id or ''}".strip(),
    )

    styles = getSampleStyleSheet()
    styles["Normal"].fontName = _UNICODE_FONT
    styles["Heading1"].fontName = _UNICODE_FONT_BOLD
    styles["Heading2"].fontName = _UNICODE_FONT_BOLD

    story: list[object] = []

    # Шапка.
    story.append(Paragraph("Отчёт об идентификации соединения", styles["Heading1"]))
    timestamp = result.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
    meta_lines = [
        f"Дата формирования: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Запрос: {request_id if request_id is not None else 'без номера'}",
        f"Время обработки: {result.processing_time_ms} мс",
        f"Метка времени: {timestamp}",
        f"Режим порогов: {result.threshold_mode}",
    ]
    for line in meta_lines:
        story.append(Paragraph(line, styles["Normal"]))
    versions = ", ".join(f"{k}: {v}" for k, v in result.model_versions.items())
    story.append(Paragraph(f"Версии моделей: {versions}", styles["Normal"]))
    story.append(Spacer(1, 0.4 * cm))

    # Спектр.
    if result.spectrum is not None:
        story.append(Paragraph("ИК-спектр (предобработанный)", styles["Heading2"]))
        png = _render_spectrum_png(result.spectrum, result.spectrum_length)
        story.append(Image(io.BytesIO(png), width=17 * cm, height=7.2 * cm))
        story.append(Spacer(1, 0.4 * cm))

    # Таблица предсказанных функциональных групп.
    story.append(Paragraph("Функциональные группы", styles["Heading2"]))
    sorted_preds = sorted(result.predictions, key=lambda p: p.probability, reverse=True)[
        :top_predictions
    ]
    fg_table_data = [["Код", "Группа", "Вероятность", "Порог", "Предсказана"]]
    for pred in sorted_preds:
        fg_table_data.append(
            [
                pred.code,
                pred.name,
                f"{pred.probability:.3f}",
                f"{pred.threshold:.2f}",
                "да" if pred.predicted else "нет",
            ]
        )
    fg_table = Table(fg_table_data, hAlign="LEFT")
    fg_table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), _UNICODE_FONT, 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9ca3af")),
                ("ALIGN", (2, 1), (4, -1), "CENTER"),
            ]
        )
    )
    story.append(fg_table)
    story.append(Spacer(1, 0.4 * cm))

    # Таблица кандидатов.
    story.append(Paragraph("Кандидаты-соединения", styles["Heading2"]))
    if not result.candidates:
        story.append(
            Paragraph(
                "Список кандидатов пуст (FAISS-ретривер не загружен).",
                styles["Normal"],
            )
        )
    else:
        cand_data = [["Ранг", "Название", "Формула", "CAS", "Score", "Jaccard", "Согл."]]
        for c in result.candidates:
            cand_data.append(
                [
                    str(c.rank),
                    c.name or c.smiles[:24],
                    c.formula or "—",
                    c.cas_number or "—",
                    f"{c.score:.3f}",
                    f"{c.jaccard:.2f}",
                    "✓" if c.consistent else "—",
                ]
            )
        cand_table = Table(cand_data, hAlign="LEFT")
        cand_table.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (-1, -1), _UNICODE_FONT, 9),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9ca3af")),
                    ("ALIGN", (0, 1), (0, -1), "CENTER"),
                    ("ALIGN", (4, 1), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(cand_table)
    story.append(Spacer(1, 0.4 * cm))

    # Grad-CAM (если есть).
    if result.gradcam is not None and result.gradcam.values:
        story.append(PageBreak())
        story.append(
            Paragraph(
                f"Активация Grad-CAM ({result.gradcam.group_name})",
                styles["Heading2"],
            )
        )
        png = _render_gradcam_png(result.gradcam.values, result.gradcam.group_name)
        story.append(Image(io.BytesIO(png), width=17 * cm, height=5.4 * cm))

    doc.build(story)
    return buffer.getvalue()
