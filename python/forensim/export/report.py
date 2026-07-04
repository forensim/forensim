"""Generate a forensic PDF report from reconstruction, simulation, and inference results."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reportlab.lib import colors  # type: ignore[import-untyped]
from reportlab.lib.pagesizes import letter  # type: ignore[import-untyped]
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # type: ignore[import-untyped]
from reportlab.lib.units import inch  # type: ignore[import-untyped]
from reportlab.platypus import (  # type: ignore[import-untyped]
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """Typed container for a report section."""

    title: str
    rows: list[tuple[str, str]]


@dataclass
class ReportInputs:
    """Inputs for a ForenSim PDF report."""

    case_title: str
    examiner: str
    notes: str
    reconstruction: dict[str, Any] | None
    simulation: dict[str, Any] | None
    inference: dict[str, Any] | None
    screenshot_bytes: list[bytes] | None = None


def _format_dict(data: dict[str, Any] | None, max_len: int = 120) -> list[tuple[str, str]]:
    if not data:
        return [("—", "No data available")]
    rows: list[tuple[str, str]] = []
    for key, value in data.items():
        if key in {"results", "hypotheses"}:
            continue
        text = str(value)
        if len(text) > max_len:
            text = text[: max_len - 3] + "..."
        rows.append((key.replace("_", " ").title(), text))
    return rows


def _hypothesis_table(hypotheses: list[dict[str, Any]]) -> list[list[str]]:
    header = ["Rank", "Description", "Posterior", "Bayes Factor", "Events"]
    rows = [header]
    for h in hypotheses:
        rows.append(
            [
                str(h.get("rank", "")),
                str(h.get("description", "")),
                f"{h.get('posterior', 0.0):.4f}",
                f"{h.get('bayes_factor', 0.0):.4f}" if h.get("bayes_factor") is not None else "—",
                ", ".join(str(e) for e in h.get("events", [])),
            ]
        )
    return rows


def _scenario_table(results: list[dict[str, Any]]) -> list[list[str]]:
    header = ["Object", "Velocity", "Trajectory Length", "Final Position"]
    rows = [header]
    for r in results:
        scenario = r.get("scenario", {})
        final = r.get("final_positions", {})
        final_str = "; ".join(f"{k}: {v}" for k, v in final.items())[:80]
        rows.append(
            [
                str(scenario.get("object_name", "—")),
                str(scenario.get("velocity", [])),
                str(r.get("trajectory_length", 0)),
                final_str,
            ]
        )
    return rows


def generate_pdf_report(inputs: ReportInputs, output_path: Path) -> Path:
    """Render a forensic report PDF to ``output_path``.

    Args:
        inputs: Case metadata and pipeline results.
        output_path: Destination PDF path.

    Returns:
        Path to the written PDF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ForenSimTitle",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor("#1a1a1a"),
        spaceAfter=14,
    )
    heading_style = ParagraphStyle(
        "ForenSimHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#b45309"),
        spaceAfter=8,
        spaceBefore=12,
    )
    body_style = styles["BodyText"]
    body_style.fontSize = 10

    story: list[Any] = []
    story.append(Paragraph("ForenSim Forensic Report", title_style))
    story.append(Paragraph(f"<b>Case:</b> {inputs.case_title}", body_style))
    story.append(Paragraph(f"<b>Examiner:</b> {inputs.examiner}", body_style))
    story.append(Spacer(1, 0.2 * inch))

    if inputs.notes:
        story.append(Paragraph("<b>Notes</b>", heading_style))
        story.append(Paragraph(inputs.notes, body_style))
        story.append(Spacer(1, 0.1 * inch))

    # Reconstruction section
    story.append(Paragraph("1. Reconstruction", heading_style))
    rec_rows = _format_dict(inputs.reconstruction)
    if rec_rows:
        table = Table(rec_rows, colWidths=[1.8 * inch, 4.7 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(table)
    story.append(Spacer(1, 0.1 * inch))

    # Simulation section
    if inputs.simulation:
        story.append(Paragraph("2. Physics Simulation", heading_style))
        sim_summary = _format_dict(inputs.simulation)
        if sim_summary:
            table = Table(sim_summary, colWidths=[1.8 * inch, 4.7 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 0.1 * inch))

        sim_results = inputs.simulation.get("results", [])
        if sim_results:
            story.append(Paragraph("Scenarios", body_style))
            rows = _scenario_table(sim_results)
            table = Table(rows, colWidths=[1.2 * inch, 1.5 * inch, 1.2 * inch, 2.6 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 0.1 * inch))

    # Inference section
    if inputs.inference:
        story.append(PageBreak())
        story.append(Paragraph("3. Probabilistic Inference", heading_style))
        inf_summary = _format_dict(inputs.inference)
        if inf_summary:
            table = Table(inf_summary, colWidths=[1.8 * inch, 4.7 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 0.1 * inch))

        hypotheses = inputs.inference.get("hypotheses", [])
        if hypotheses:
            story.append(Paragraph("Ranked Hypotheses", body_style))
            rows = _hypothesis_table(hypotheses)
            table = Table(
                rows,
                colWidths=[0.5 * inch, 2.6 * inch, 0.9 * inch, 1.0 * inch, 1.5 * inch],
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(table)

    # Screenshots
    if inputs.screenshot_bytes:
        story.append(PageBreak())
        story.append(Paragraph("4. Screenshots", heading_style))
        for i, data in enumerate(inputs.screenshot_bytes, start=1):
            try:
                img = Image(io.BytesIO(data), width=6.0 * inch, height=3.4 * inch)
                story.append(Paragraph(f"Figure {i}", body_style))
                story.append(img)
                story.append(Spacer(1, 0.1 * inch))
            except Exception as exc:
                logger.warning("Failed to embed screenshot %d: %s", i, exc)
                story.append(Paragraph(f"Figure {i} (could not embed)", body_style))

    doc.build(story)
    logger.info("PDF report written to %s", output_path)
    return output_path
