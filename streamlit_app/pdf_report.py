import io
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _safe_text(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    return text if text else "-"


def _np_to_rl_image(np_img: np.ndarray, width_mm: float = 78) -> RLImage:
    """
    Convert numpy image (H, W, C) or grayscale (H, W) to a ReportLab Image.
    """
    if np_img.ndim == 2:
        pil_img = PILImage.fromarray(np_img.astype(np.uint8), mode="L")
    else:
        pil_img = PILImage.fromarray(np_img.astype(np.uint8))

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    width_px, height_px = pil_img.size
    target_width = width_mm * mm
    target_height = (height_px / max(width_px, 1)) * target_width

    return RLImage(buf, width=target_width, height=target_height)


def _build_table(rows: List[List[str]], col_widths, header_bg="#dbeafe", header_fg="#12344d"):
    table = Table(rows, colWidths=col_widths)
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(header_fg)),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("LEADING", (0, 0), (-1, -1), 12),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )
    return table


def build_pdf_report(
    *,
    computed: Dict[str, Dict[str, Any]],
    available_eyes: List[str],
    analysis_input_mode: str,
    class_names: List[str],
    selected_view_mode: str,
    options: Dict[str, Any],
) -> bytes:
    """
    Build a PDF report from current analysis results and selected export options.
    """
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=_safe_text(options.get("report_title", "DR Analysis Report")),
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0b3d5c"),
        alignment=TA_LEFT,
        spaceAfter=8,
    )

    sub_style = ParagraphStyle(
        "ReportSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4b5563"),
        spaceAfter=10,
    )

    section_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#12344d"),
        spaceBefore=8,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=colors.black,
    )

    small_style = ParagraphStyle(
        "SmallText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#5b6470"),
    )

    story = []

    report_title = _safe_text(options.get("report_title", "DR Analysis Report"))
    generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    story.append(Paragraph(report_title, title_style))
    story.append(
        Paragraph(
            f"Generated on: {generated_time}<br/>Analysis mode: {analysis_input_mode.title()}",
            sub_style,
        )
    )
    story.append(Spacer(1, 4))

    # ---------------------------
    # Patient / report details
    # ---------------------------
    if options.get("include_report_details", True):
        story.append(Paragraph("Report Details", section_style))

        detail_rows = [["Field", "Value"]]
        detail_rows.append(["Patient name", _safe_text(options.get("patient_name"))])
        detail_rows.append(["Patient ID", _safe_text(options.get("patient_id"))])
        detail_rows.append(["Clinician / Examiner", _safe_text(options.get("clinician_name"))])
        detail_rows.append(["Institution", _safe_text(options.get("institution_name"))])
        detail_rows.append(["Notes", _safe_text(options.get("notes"))])
        detail_rows.append(["Available eyes", ", ".join([eye.title() for eye in available_eyes])])
        detail_rows.append(["Current analysis view", _safe_text(selected_view_mode)])

        story.append(
            _build_table(
                detail_rows,
                col_widths=[50 * mm, 120 * mm],
                header_bg="#dbeafe",
                header_fg="#12344d",
            )
        )
        story.append(Spacer(1, 10))

    # ---------------------------
    # Eye sections
    # ---------------------------
    for eye in available_eyes:
        c = computed[eye]
        conf = float(c["probs"][c["pred_idx"]])

        story.append(Paragraph(f"{eye.title()} Eye", section_style))

        if options.get("include_prediction_summary", True):
            info_rows = [["Item", "Result"]]
            info_rows.append(["Predicted grade", _safe_text(c["pred_name"])])
            info_rows.append(["Predicted class index", str(c["pred_idx"])])
            info_rows.append(["Confidence score", f"{conf * 100:.1f}%"])
            info_rows.append(["CAM method", _safe_text(c["cam_method_ui"])])
            info_rows.append(["Target class for CAM", _safe_text(class_names[c["target_class"]])])
            info_rows.append(["Target layer", _safe_text(c["layer_name"])])

            story.append(
                _build_table(
                    info_rows,
                    col_widths=[55 * mm, 115 * mm],
                    header_bg="#e0f2fe",
                    header_fg="#12344d",
                )
            )
            story.append(Spacer(1, 8))

        if options.get("include_probabilities", True):
            prob_rows = [["Class", "Probability"]]
            order = np.argsort(-c["probs"])
            for i in order:
                prob_rows.append([class_names[i], f"{float(c['probs'][i]) * 100:.1f}%"])

            story.append(Paragraph("Class Probabilities", body_style))
            story.append(Spacer(1, 4))
            story.append(
                _build_table(
                    prob_rows,
                    col_widths=[95 * mm, 75 * mm],
                    header_bg="#dcfce7",
                    header_fg="#14532d",
                )
            )
            story.append(Spacer(1, 8))

        # ---------------------------
        # Images
        # ---------------------------
        image_tables = []

        if options.get("include_original_image", False):
            image_tables.append(
                ("Original Image", _np_to_rl_image(c["raw_rgb"]))
            )

        if options.get("include_gradcam_heatmap", False):
            image_tables.append(
                ("Grad-CAM Heatmap", _np_to_rl_image(c["heatmap_rgb"]))
            )

        if options.get("include_gradcam_overlay", False):
            image_tables.append(
                ("Grad-CAM Overlay", _np_to_rl_image(c["cam_overlay_rgb"]))
            )

        if options.get("include_exudates_mask", False):
            ex_mask_img = (c["seg"]["ex_mask"] * 255).astype(np.uint8)
            image_tables.append(
                ("Exudates Mask", _np_to_rl_image(ex_mask_img))
            )

        if options.get("include_exudates_overlay", False):
            image_tables.append(
                ("Exudates Overlay", _np_to_rl_image(c["seg"]["ex_overlay"]))
            )

        if options.get("include_haemorrhages_mask", False):
            he_mask_img = (c["seg"]["he_mask"] * 255).astype(np.uint8)
            image_tables.append(
                ("Haemorrhages Mask", _np_to_rl_image(he_mask_img))
            )

        if options.get("include_haemorrhages_overlay", False):
            image_tables.append(
                ("Haemorrhages Overlay", _np_to_rl_image(c["seg"]["he_overlay"]))
            )

        if image_tables:
            story.append(Paragraph("Selected Images", body_style))
            story.append(Spacer(1, 4))

            # Place images two per row where possible
            row = []
            for idx, (label, rl_img) in enumerate(image_tables, start=1):
                cell = Table(
                    [[Paragraph(f"<b>{label}</b>", small_style)], [rl_img]],
                    colWidths=[82 * mm],
                )
                cell.setStyle(
                    TableStyle([
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ])
                )
                row.append(cell)

                if len(row) == 2:
                    story.append(Table([row], colWidths=[84 * mm, 84 * mm]))
                    story.append(Spacer(1, 6))
                    row = []

            if row:
                if len(row) == 1:
                    row.append("")
                story.append(Table([row], colWidths=[84 * mm, 84 * mm]))
                story.append(Spacer(1, 6))

        story.append(Spacer(1, 8))

    if options.get("include_disclaimer", True):
        story.append(Paragraph("Disclaimer", section_style))
        story.append(
            Paragraph(
                "This report is generated by a research/demo system for diabetic retinopathy analysis. "
                "It is not a medical diagnosis and should not be used as a substitute for clinical judgment "
                "or professional ophthalmic evaluation.",
                small_style,
            )
        )

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes