"""Export the reviewed Markdown report as PDF and editable Word documents."""

import html
import re
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as PDFImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/MasabTahir732/assignment-03-mlops-agent/blob/main/"


def font_path(bold=False):
    candidates = (
        [
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ]
        if bold
        else [
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    )
    return next((p for p in candidates if p.exists()), None)


def diagram():
    image = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(image)
    path = font_path()
    font = ImageFont.truetype(str(path), 22) if path else ImageFont.load_default()
    title_font = ImageFont.truetype(str(path), 30) if path else font
    draw.text(
        (45, 20),
        "MLOps agent: planning, real tools, evidence and deployment",
        font=title_font,
        fill="#142c4e",
    )

    def arrow(start, end):
        draw.line([start, end], fill="#526880", width=4)
        x, y = end
        if y > start[1]:
            draw.polygon([(x, y), (x - 9, y - 13), (x + 9, y - 13)], fill="#526880")

    def box(rect, text, fill="#eef4fb"):
        draw.rounded_rectangle(rect, radius=14, fill=fill, outline="#557da6", width=3)
        x1, y1, x2, y2 = rect
        lines = text.split("\n")
        for i, line in enumerate(lines):
            width = draw.textlength(line, font=font)
            draw.text(
                ((x1 + x2 - width) / 2, (y1 + y2) / 2 - len(lines) * 14 + i * 29),
                line,
                font=font,
                fill="#142c4e",
            )

    box((475, 80, 1025, 140), "Natural-language CLI request")
    arrow((750, 140), (750, 170))
    box((400, 170, 1100, 240), "Ollama / Gemini: structured tool-call plan")
    arrow((750, 240), (750, 270))
    box((350, 270, 1150, 345), "Executor: schema validation, allowlist, explicit consent")
    nodes = [
        (40, 405, 370, 495),
        (400, 405, 730, 495),
        (760, 405, 1090, 495),
        (1120, 405, 1450, 495),
    ]
    labels = [
        "DVC + Git\nversions / provenance",
        "MLflow client\nruns / metrics / registry",
        "Airflow 3 JWT API\nDAGs / tasks / logs",
        "Scoped kubectl\nhealth / events / logs",
    ]
    for rect, label in zip(nodes, labels):
        center = (rect[0] + rect[2]) / 2
        arrow((750, 345), (center, 405))
        box(rect, label)
        arrow((center, 495), (750, 550))
    box(
        (325, 550, 1175, 625),
        "Actual tool evidence → facts, uncertainties, recommendations",
        "#eaf5ef",
    )
    box(
        (60, 705, 1440, 780),
        "Airflow: validate → preprocess → train → evaluate/MLflow → quality gate",
        "#fff5df",
    )
    arrow((750, 780), (750, 810))
    box(
        (140, 810, 1360, 880),
        "Approved bundle → operator-built image → FastAPI / 2 replicas / ClusterIP / probes",
    )
    draw.line(
        [(925, 495), (925, 525), (1475, 525), (1475, 670), (750, 670)], fill="#9b814a", width=3
    )
    arrow((750, 670), (750, 705))
    target = ROOT / "screenshots/architecture.png"
    image.save(target)
    return target


def inline(text):
    value = html.escape(text)
    value = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f'<a href="{m[2] if m[2].startswith("https://") else REPO + m[2]}">{m[1]}</a>',
        value,
    )
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)
    return value


def plain(text):
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    return text.replace("**", "").replace("`", "")


def blocks(text):
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("```mermaid"):
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                index += 1
            yield "diagram", None
            index += 1
        elif line.startswith("#"):
            yield "heading", (len(line) - len(line.lstrip("#")), line.lstrip("# "))
            index += 1
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                row = [x.strip() for x in lines[index].strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", cell) for cell in row):
                    rows.append(row)
                index += 1
            yield "table", rows
        elif line.startswith("- "):
            item = [line[2:]]
            index += 1
            while index < len(lines) and lines[index].startswith("  "):
                item.append(lines[index].strip())
                index += 1
            yield "bullet", " ".join(item)
        else:
            paragraph = []
            while (
                index < len(lines)
                and lines[index].strip()
                and not lines[index].startswith(("#", "|", "```"))
            ):
                paragraph.append(lines[index])
                index += 1
            yield "paragraph", " ".join(paragraph)


def export():
    normal, bold = font_path(), font_path(True)
    family = "Helvetica"
    if normal and bold:
        pdfmetrics.registerFont(TTFont("ReportSans", str(normal)))
        pdfmetrics.registerFont(TTFont("ReportSans-Bold", str(bold)))
        pdfmetrics.registerFontFamily("ReportSans", normal="ReportSans", bold="ReportSans-Bold")
        family = "ReportSans"
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyReport",
            fontName=family,
            fontSize=9.2,
            leading=13,
            spaceAfter=7,
            alignment=TA_LEFT,
            splitLongWords=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CellReport", parent=styles["BodyReport"], fontSize=7.2, leading=10, spaceAfter=2
        )
    )
    for name in ("Title", "Heading1", "Heading2"):
        styles[name].fontName = family + "-Bold" if family == "ReportSans" else "Helvetica-Bold"
        styles[name].textColor = colors.HexColor("#142c4e")
    story = []
    doc = Document()
    doc.core_properties.author = "Masab Tahir"
    doc.core_properties.title = "Assignment 03 — MLOps AI Agent Technical Report"
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Arial"
    normal_style.font.size = Pt(10)
    doc.sections[0].header.paragraphs[0].text = "MLOps & Production AI Engineering | Assignment 03"
    diagram_path = diagram()
    for kind, value in blocks((ROOT / "TECHNICAL_REPORT.md").read_text(encoding="utf-8")):
        if kind == "heading":
            level, title = value
            story.append(Paragraph(inline(title), styles["Title" if level == 1 else "Heading1"]))
            doc.add_heading(title, level=0 if level == 1 else 1)
        elif kind == "paragraph":
            story.append(Paragraph(inline(value), styles["BodyReport"]))
            doc.add_paragraph(plain(value))
        elif kind == "bullet":
            story.append(Paragraph(inline(value), styles["BodyReport"], bulletText="•"))
            doc.add_paragraph(plain(value), style="List Bullet")
        elif kind == "diagram":
            story.append(PDFImage(str(diagram_path), width=178 * mm, height=106.8 * mm))
            story.append(Spacer(1, 5 * mm))
            doc.add_picture(str(diagram_path), width=Inches(6.3))
        elif kind == "table":
            cols = len(value[0])
            pdf_rows = [
                [Paragraph(inline(cell), styles["CellReport"]) for cell in row] for row in value
            ]
            widths = [37, 64, 77] if cols == 3 else [178 / cols] * cols
            table = Table(pdf_rows, colWidths=[w * mm for w in widths], repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eff7")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c5d0dc")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.extend([table, Spacer(1, 4 * mm)])
            word_table = doc.add_table(rows=1, cols=cols)
            word_table.style = "Light Shading Accent 1"
            for cell, text in zip(word_table.rows[0].cells, value[0]):
                cell.text = plain(text)
            for row in value[1:]:
                for cell, text in zip(word_table.add_row().cells, row):
                    cell.text = plain(text)
            doc.add_paragraph()

    def footer(canvas, document):
        canvas.setFont(family, 8)
        canvas.setFillColor(colors.HexColor("#526880"))
        canvas.drawString(16 * mm, 10 * mm, "Masab Tahir | MLOps Assignment 03")
        canvas.drawRightString(A4[0] - 16 * mm, 10 * mm, str(document.page))

    pdf = ROOT / "TECHNICAL_REPORT.pdf"
    SimpleDocTemplate(
        str(pdf),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=17 * mm,
        title=doc.core_properties.title,
        author="Masab Tahir",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    word = ROOT / "TECHNICAL_REPORT.docx"
    doc.save(word)
    print(
        f"Exported {pdf.name} ({pdf.stat().st_size:,} bytes) and {word.name} ({word.stat().st_size:,} bytes)"
    )


if __name__ == "__main__":
    export()
