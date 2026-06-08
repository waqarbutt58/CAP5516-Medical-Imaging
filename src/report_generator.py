import os
from datetime import datetime


def generate_report(metrics: dict, save_path: str = "results/clinical_report.pdf"):
    """Generate a clinical-style PDF performance report."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, Image as RLImage)
        from reportlab.lib.enums import TA_CENTER
    except ImportError:
        print("reportlab not installed. Run: pip install reportlab")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    doc = SimpleDocTemplate(save_path, pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"],
                                  fontSize=16, spaceAfter=12, alignment=TA_CENTER)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceAfter=6)
    body_style = styles["BodyText"]

    story = []

    # Title
    story.append(Paragraph("Clinical Performance Report", title_style))
    story.append(Paragraph("CAP5516 – Multi-task Medical Image Analysis", styles["Heading2"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
    story.append(Spacer(1, 0.5*cm))

    # Segmentation results
    story.append(Paragraph("1. Segmentation Results (U-Net)", h2_style))
    seg_data = [
        ["Metric", "Baseline (Real Only)", "Augmented (Best Ratio)"],
        ["Dice Score",
         f"{metrics.get('baseline_dice', 'N/A'):.4f}" if isinstance(metrics.get('baseline_dice'), float) else "N/A",
         f"{metrics.get('augmented_dice', 'N/A'):.4f}" if isinstance(metrics.get('augmented_dice'), float) else "N/A"],
        ["IoU",
         f"{metrics.get('baseline_iou', 'N/A'):.4f}" if isinstance(metrics.get('baseline_iou'), float) else "N/A",
         f"{metrics.get('augmented_iou', 'N/A'):.4f}" if isinstance(metrics.get('augmented_iou'), float) else "N/A"],
    ]
    t = Table(seg_data, colWidths=[5*cm, 5*cm, 5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2196f3")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # Classification results
    story.append(Paragraph("2. Classification Results (ResNet-50)", h2_style))
    cls_data = [
        ["Metric", "Baseline (Real Only)", "Augmented (Best Ratio)"],
        ["AUC (OvR)",
         f"{metrics.get('baseline_auc', 'N/A'):.4f}" if isinstance(metrics.get('baseline_auc'), float) else "N/A",
         f"{metrics.get('augmented_auc', 'N/A'):.4f}" if isinstance(metrics.get('augmented_auc'), float) else "N/A"],
        ["Accuracy",
         f"{metrics.get('baseline_acc', 'N/A'):.4f}" if isinstance(metrics.get('baseline_acc'), float) else "N/A",
         f"{metrics.get('augmented_acc', 'N/A'):.4f}" if isinstance(metrics.get('augmented_acc'), float) else "N/A"],
    ]
    t2 = Table(cls_data, colWidths=[5*cm, 5*cm, 5*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4caf50")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.5*cm))

    # Ablation summary
    if "ablation" in metrics:
        story.append(Paragraph("3. Ablation Study Summary", h2_style))
        abl = metrics["ablation"]
        abl_data = [["Synthetic Ratio", "Val AUC", "Val Accuracy"]]
        for ratio, vals in abl.items():
            abl_data.append([
                f"{ratio:.0%}",
                f"{vals['best_val_auc']:.4f}",
                f"{vals['best_val_acc']:.4f}",
            ])
        t3 = Table(abl_data, colWidths=[5*cm, 5*cm, 5*cm])
        t3.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#9c27b0")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(t3)

    doc.build(story)
    print(f"Clinical report saved: {save_path}")
