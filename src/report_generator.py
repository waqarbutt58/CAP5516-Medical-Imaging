import os
from datetime import datetime

# ── Student / Project Info ────────────────────────────────────────────────────
STUDENT_INFO = {
    "name":       "Waqar Rauf Butt",
    "roll_no":    "PHDAIF25M003",
    "supervisor": "Dr. Muhammad Farooq",
    "course":     "Medical Image Computing",
    "project":    "Multi-task Medical Image Analysis with Diffusion-Based Synthetic Augmentation",
    "dataset":    "Breast Ultrasound Images (BUSI)",
}


def generate_report(metrics: dict, save_path: str = "results/clinical_report.pdf"):
    """Generate a clinical-style PDF performance report with full student details."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        print("reportlab not installed. Run: pip install reportlab")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    doc = SimpleDocTemplate(
        save_path, pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "MainTitle", parent=styles["Title"],
        fontSize=18, spaceAfter=6, alignment=TA_CENTER,
        textColor=colors.HexColor("#1a237e")
    )
    subtitle_style = ParagraphStyle(
        "SubTitle", parent=styles["Normal"],
        fontSize=11, spaceAfter=4, alignment=TA_CENTER,
        textColor=colors.HexColor("#37474f")
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=12, spaceAfter=6, spaceBefore=12,
        textColor=colors.HexColor("#1565c0")
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["BodyText"],
        fontSize=10, spaceAfter=4
    )
    label_style = ParagraphStyle(
        "Label", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#37474f")
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph(STUDENT_INFO["project"], title_style))
    story.append(Paragraph("Clinical Performance Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor("#1a237e"), spaceAfter=8))

    # ── Student Info Table ────────────────────────────────────────────────────
    info_data = [
        ["Student Name",  STUDENT_INFO["name"],
         "Roll No",       STUDENT_INFO["roll_no"]],
        ["Supervisor",    STUDENT_INFO["supervisor"],
         "Course",        STUDENT_INFO["course"]],
        ["Dataset",       STUDENT_INFO["dataset"],
         "Report Date",   datetime.now().strftime("%Y-%m-%d %H:%M")],
    ]
    info_table = Table(info_data, colWidths=[3.5*cm, 5.5*cm, 3.5*cm, 5.5*cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), colors.HexColor("#e3f2fd")),
        ("BACKGROUND",  (2, 0), (2, -1), colors.HexColor("#e3f2fd")),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#b0bec5")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [colors.white, colors.HexColor("#f9f9f9"), colors.white]),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING",     (0, 0), (-1, -1), 5),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#b0bec5"), spaceAfter=4))

    # ── 1. Segmentation Results ───────────────────────────────────────────────
    story.append(Paragraph("1. Segmentation Results (U-Net)", h2_style))

    def fmt(val):
        return f"{val:.4f}" if isinstance(val, float) else "—"

    seg_data = [
        ["Metric", "Baseline (Real Only)", "Augmented (Best Ratio)", "Target"],
        ["Dice Score",  fmt(metrics.get("baseline_dice")),
                        fmt(metrics.get("augmented_dice")),  "> 0.75"],
        ["IoU",         fmt(metrics.get("baseline_iou")),
                        fmt(metrics.get("augmented_iou")),   "> 0.60"],
        ["Sensitivity", fmt(metrics.get("baseline_sensitivity")),
                        fmt(metrics.get("augmented_sensitivity")), "> 0.80"],
        ["Specificity", fmt(metrics.get("baseline_specificity")),
                        fmt(metrics.get("augmented_specificity")), "> 0.85"],
    ]
    seg_table = Table(seg_data, colWidths=[4*cm, 4.5*cm, 4.5*cm, 3*cm])
    seg_table.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#1565c0")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#b0bec5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f5f5f5")]),
        ("ALIGN",          (1, 0), (-1, -1), "CENTER"),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING",        (0, 0), (-1, -1), 5),
    ]))
    story.append(seg_table)
    story.append(Spacer(1, 0.4*cm))

    # ── 2. Classification Results ─────────────────────────────────────────────
    story.append(Paragraph("2. Classification Results (ResNet-50)", h2_style))

    cls_data = [
        ["Metric", "Baseline (Real Only)", "Augmented (Best Ratio)", "Target"],
        ["AUC (OvR)",    fmt(metrics.get("baseline_auc")),
                         fmt(metrics.get("augmented_auc")),  "> 0.85"],
        ["Accuracy",     fmt(metrics.get("baseline_acc")),
                         fmt(metrics.get("augmented_acc")),  "> 0.80"],
        ["F1 – Normal",  fmt(metrics.get("f1_normal")),   "—", "> 0.78"],
        ["F1 – Benign",  fmt(metrics.get("f1_benign")),   "—", "> 0.78"],
        ["F1 – Malignant", fmt(metrics.get("f1_malignant")), "—", "> 0.78"],
    ]
    cls_table = Table(cls_data, colWidths=[4*cm, 4.5*cm, 4.5*cm, 3*cm])
    cls_table.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#2e7d32")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#b0bec5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f5f5f5")]),
        ("ALIGN",          (1, 0), (-1, -1), "CENTER"),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING",        (0, 0), (-1, -1), 5),
    ]))
    story.append(cls_table)
    story.append(Spacer(1, 0.4*cm))

    # ── 3. Synthetic Image Quality ────────────────────────────────────────────
    story.append(Paragraph("3. Synthetic Image Quality (Diffusion Model)", h2_style))
    fid_data = [
        ["Metric", "Value", "Interpretation"],
        ["FID Score",  fmt(metrics.get("fid_score")),  "Lower is better (real ↔ synthetic similarity)"],
        ["Images Generated", str(metrics.get("n_synthetic", "—")), "Total synthetic images across all classes"],
    ]
    fid_table = Table(fid_data, colWidths=[4*cm, 3.5*cm, 8.5*cm])
    fid_table.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#6a1b9a")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#b0bec5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f5f5f5")]),
        ("ALIGN",          (1, 0), (1, -1), "CENTER"),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING",        (0, 0), (-1, -1), 5),
    ]))
    story.append(fid_table)
    story.append(Spacer(1, 0.4*cm))

    # ── 4. Ablation Study ─────────────────────────────────────────────────────
    if "ablation" in metrics:
        story.append(Paragraph("4. Ablation Study — Synthetic Data Ratio vs Performance", h2_style))
        abl = metrics["ablation"]
        abl_data = [["Synthetic Ratio", "Val AUC", "Val Accuracy", "Improvement vs Baseline"]]
        baseline_auc = abl.get(0.0, {}).get("best_val_auc", 0)
        for ratio in [0.0, 0.25, 0.50, 0.75, 1.0]:
            if ratio in abl:
                vals = abl[ratio]
                delta = vals["best_val_auc"] - baseline_auc
                delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
                abl_data.append([
                    f"{ratio:.0%}",
                    f"{vals['best_val_auc']:.4f}",
                    f"{vals['best_val_acc']:.4f}",
                    delta_str if ratio > 0 else "—",
                ])
        abl_table = Table(abl_data, colWidths=[4*cm, 4*cm, 4*cm, 4*cm])
        abl_table.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#e65100")),
            ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
            ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1, -1), 9),
            ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#b0bec5")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#fff3e0")]),
            ("ALIGN",          (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING",        (0, 0), (-1, -1), 5),
        ]))
        story.append(abl_table)
        story.append(Spacer(1, 0.4*cm))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#b0bec5"), spaceAfter=6))
    footer_data = [[
        f"Student: {STUDENT_INFO['name']}  |  Roll No: {STUDENT_INFO['roll_no']}  |  "
        f"Supervisor: {STUDENT_INFO['supervisor']}  |  Course: {STUDENT_INFO['course']}"
    ]]
    footer_table = Table(footer_data, colWidths=[16*cm])
    footer_table.setStyle(TableStyle([
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("TEXTCOLOR",  (0, 0), (-1, -1), colors.HexColor("#78909c")),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(footer_table)

    doc.build(story)
    print(f"Clinical report saved: {save_path}")
