from __future__ import annotations

import importlib
import os
from typing import Dict, List, Optional, Tuple

from rag_finance.utils.io_utils import ensure_dir
from rag_finance.utils.tabular_format import (
    build_finance_rows,
    build_stock_monthly_rows,
    build_stock_summary_rows,
)


def _ensure_base_font() -> str:
    try:
        pdfmetrics = importlib.import_module("reportlab.pdfbase.pdfmetrics")
        ttfonts = importlib.import_module("reportlab.pdfbase.ttfonts")
    except ImportError:  # pragma: no cover - optional dependency
        return "Helvetica"

    TTFont = getattr(ttfonts, "TTFont")

    registered = set(pdfmetrics.getRegisteredFontNames())
    preferred: List[tuple[str, str]] = [
        ("MalgunGothic", os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "malgun.ttf")),
        ("NanumGothic", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        ("AppleSDGothicNeo", "/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    ]

    for name, path in preferred:
        if name in registered:
            return name
        if os.path.isfile(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                return name
            except Exception:  # pragma: no cover - font issues
                continue

    return "Helvetica"


def _build_table(data: List[List[str]], font_name: str):
    colors = importlib.import_module("reportlab.lib.colors")
    platypus = importlib.import_module("reportlab.platypus")
    Table = getattr(platypus, "Table")
    TableStyle = getattr(platypus, "TableStyle")

    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("TOPPADDING", (0, 1), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
            ]
        )
    )
    return table


def export_report_pdf(
    output_path: str,
    sections: Dict[str, str],
    tabular_payload: Optional[Dict[str, object]] = None,
) -> None:
    try:
        colors = importlib.import_module("reportlab.lib.colors")
        pagesizes = importlib.import_module("reportlab.lib.pagesizes")
        styles_mod = importlib.import_module("reportlab.lib.styles")
        platypus = importlib.import_module("reportlab.platypus")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("reportlab 패키지를 설치해야 PDF 출력이 가능합니다.") from exc

    ensure_dir(os.path.dirname(output_path) or ".")
    base_font = _ensure_base_font()

    getSampleStyleSheet = getattr(styles_mod, "getSampleStyleSheet")
    ParagraphStyle = getattr(styles_mod, "ParagraphStyle")
    Paragraph = getattr(platypus, "Paragraph")
    SimpleDocTemplate = getattr(platypus, "SimpleDocTemplate")
    Spacer = getattr(platypus, "Spacer")
    A4 = getattr(pagesizes, "A4")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName=base_font,
        fontSize=18,
        leading=22,
        spaceAfter=12,
    )
    section_title_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontName=base_font,
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#303030"),
        spaceBefore=10,
        spaceAfter=6,
    )
    subheading_style = ParagraphStyle(
        "SubHeading",
        parent=styles["Heading3"],
        fontName=base_font,
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#404040"),
        spaceBefore=6,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName=base_font,
        fontSize=10,
        leading=14,
        spaceAfter=4,
    )
    bullet_style = ParagraphStyle(
        "ReportBullet",
        parent=body_style,
        leftIndent=14,
        bulletIndent=6,
    )

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=36,
        bottomMargin=36,
        leftMargin=42,
        rightMargin=42,
    )

    story: List[object] = []

    title_text = sections.get("Title", "").strip()
    if title_text:
        story.append(Paragraph(title_text, title_style))

    if tabular_payload:
        finance_rows = build_finance_rows(tabular_payload.get("finance"))
        summary_rows = build_stock_summary_rows(tabular_payload.get("stock"))
        monthly_rows = build_stock_monthly_rows(tabular_payload.get("stock"))

        if finance_rows or summary_rows or monthly_rows:
            if finance_rows:
                story.append(Paragraph("재무 요약", subheading_style))
                story.append(_build_table(finance_rows, base_font))
                story.append(Spacer(1, 10))
            if summary_rows:
                story.append(Paragraph("주요 주가 통계", subheading_style))
                story.append(_build_table(summary_rows, base_font))
                story.append(Spacer(1, 10))
            if monthly_rows:
                story.append(Paragraph("최근 월별 종가", subheading_style))
                story.append(_build_table(monthly_rows, base_font))
                story.append(Spacer(1, 14))

    section_order = [
        ("Summary", "요약"),
        ("Analysis", "분석"),
        ("Opinion", "투자의견"),
        ("Table", "테이블 요약"),
    ]

    for key, label in section_order:
        content = sections.get(key, "").strip()
        if not content:
            continue
        story.append(Paragraph(label, section_title_style))
        if key == "Table":
            table_rows, other_lines = _parse_markdown_table(content)
            if table_rows:
                story.append(_build_table(table_rows, base_font))
            for text_line in other_lines:
                story.append(Paragraph(text_line, body_style))
            story.append(Spacer(1, 10))
            continue

        for line in content.splitlines():
            text = line.strip()
            if not text:
                continue
            if text.startswith("- "):
                story.append(Paragraph(text[2:].strip(), bullet_style, bulletText="•"))
            else:
                story.append(Paragraph(text, body_style))
        story.append(Spacer(1, 10))

    if not story:
        story.append(Paragraph("생성된 리포트 내용이 없습니다.", body_style))

    doc.build(story)


def _parse_markdown_table(section_text: str) -> Tuple[List[List[str]], List[str]]:
    table_lines: List[str] = []
    other_lines: List[str] = []
    for raw in section_text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("|") and stripped.count("|") >= 2:
            table_lines.append(stripped)
        else:
            other_lines.append(stripped)

    if not table_lines:
        return [], other_lines

    rows: List[List[str]] = []
    for idx, line in enumerate(table_lines):
        trimmed = line.strip().strip("|")
        cells = [cell.strip() for cell in trimmed.split("|")]
        if idx == 1 and all(set(cell.replace(":", "")) <= {"-"} for cell in cells):
            continue
        rows.append(cells)

    if not rows:
        return [], other_lines

    max_len = max(len(row) for row in rows)
    normalized: List[List[str]] = []
    for row in rows:
        if len(row) < max_len:
            row = row + [""] * (max_len - len(row))
        normalized.append(row)

    return normalized, other_lines
