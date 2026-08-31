import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)

def create_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=15
    )

    heading2_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=14,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )

    code_style = ParagraphStyle(
        'CodeBlock',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#0F172A'),
        backColor=colors.HexColor('#F1F5F9'),
        borderColor=colors.HexColor('#CBD5E1'),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=body_style,
        leftIndent=12,
        bulletIndent=4,
        spaceAfter=4
    )

    story = []

    # Title & Header
    story.append(Paragraph("UltimateHybridDetector (hybrid_effnet_dinov2)", title_style))
    story.append(Paragraph("Comprehensive Technical Specification & Architecture Implementation", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=12))

    # 1. Executive Summary
    story.append(Paragraph("1. High-Level Architectural Summary", heading2_style))
    story.append(Paragraph(
        "The <b>UltimateHybridDetector</b> is a state-of-the-art dual-backbone, domain-adversarial deep learning architecture designed for robust binary detection of AI-generated vs. real images. It combines a local convolutional stream (<b>EfficientNet-B0</b>) and a global self-supervised Vision Transformer stream (<b>DINOv2 ViT-S/14</b>) coupled with a Deep Projection Head and a Domain Adversarial Neural Network (<b>DANN</b>) head.",
        body_style
    ))

    # Architecture Overview Box
    diagram_text = (
        "<b>[Input Image: B x 3 x 224 x 224]</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;├──&gt; <b>EfficientNet-B0 (CNN Stream)</b> ──&gt; Features + AvgPool ──&gt; <b>[B x 1280]</b> (Local Textures & Artifacts)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;└──&gt; <b>DINOv2 ViT-S/14 (ViT Stream)</b> ──&gt; CLS Token Embedding ──&gt; <b>[B x 384]</b> (Global Semantics & Context)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&gt; <b>Feature Concatenation Layer</b> ─────────────────────────────&gt; <b>[B x 1664]</b> Fused Vector<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&gt; <b>Classifier Head</b> (LayerNorm → Linear(1664→512) → Linear(512→128) → Linear(128→1)) ──&gt; <b>AI Logit [B x 1]</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&gt; <b>DANN Domain Head</b> (GRL(-α) → Linear(1664→256) → Linear(256→2)) ──────────&gt; <b>Domain Logit [B x 2]</b>"
    )
    story.append(Paragraph(diagram_text, code_style))

    # 2. Detailed Component Breakdown
    story.append(Paragraph("2. Component-by-Component Implementation", heading2_style))
    
    story.append(Paragraph("<b>A. Dual Backbone Feature Extractor</b>", body_style))
    story.append(Paragraph("• <b>EfficientNet-B0 (CNN Stream):</b> Extracts local convolutional features, spatial artifacts, edge anomalies, and high-frequency noise typical of GANs and Diffusion models. Outputs a 1280-dimensional pooled vector.", bullet_style))
    story.append(Paragraph("• <b>DINOv2 ViT-S/14 (ViT Stream):</b> Self-supervised Vision Transformer pre-trained on 142M curated images (LVD-142M). Uses 14x14 patch embeddings to capture global semantic coherence, lighting consistency, and non-local spatial relationships. Outputs a 384-dimensional CLS embedding.", bullet_style))
    story.append(Paragraph("• <b>Dual Stream Fusion:</b> Concatenates both vectors into a unified <b>1,664-dimensional feature representation</b> (<i>1280 + 384 = 1664</i>).", bullet_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>B. Deep Binary AI Classifier Projection Head</b>", body_style))
    story.append(Paragraph("• <b>Layer Normalization:</b> <code>nn.LayerNorm(1664)</code> normalizes feature magnitude scales across the CNN and Transformer streams.", bullet_style))
    story.append(Paragraph("• <b>Multi-Layer Dense Projection:</b> Linear(1664 → 512) → GELU → Dropout(0.3) → Linear(512 → 128) → GELU → Dropout(0.2) → Linear(128 → 1).", bullet_style))
    story.append(Paragraph("• <b>Output Logit:</b> Single raw logit scalar per image. Positive values indicate higher AI probability; output probability is computed via sigmoid: <i>P(AI) = 1 / (1 + exp(-logit))</i>.", bullet_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>C. Domain Adversarial Neural Network (DANN) Head</b>", body_style))
    story.append(Paragraph("• <b>Gradient Reversal Layer (GRL):</b> In forward pass, returns input unchanged; in backward pass, multiplies gradients by <i>-α</i>. Adversarially removes domain-specific/generator signatures.", bullet_style))
    story.append(Paragraph("• <b>Domain Classifier Head:</b> <code>nn.Linear(1664, 256) → GELU → Dropout(0.2) → Linear(256, 2)</code>. Classifies dataset source/generator, encouraging the backbone to learn domain-invariant representations.", bullet_style))

    # 3. Layer Summary Table
    story.append(Spacer(1, 4))
    story.append(Paragraph("3. Model Layer & Tensor Shape Summary", heading2_style))

    table_data = [
        ["Submodule / Layer", "Input Shape", "Output Shape", "Activation / Regularization"],
        ["EfficientNet-B0 Stream", "[B, 3, 224, 224]", "[B, 1280]", "Swish / AdaptiveAvgPool"],
        ["DINOv2 ViT-S/14 Stream", "[B, 3, 224, 224]", "[B, 384]", "LayerNorm / Self-Attention"],
        ["Feature Concatenation", "[B, 1280] & [B, 384]", "[B, 1664]", "torch.cat(..., dim=1)"],
        ["Layer Normalization", "[B, 1664]", "[B, 1664]", "nn.LayerNorm(1664)"],
        ["Classifier Dense 1", "[B, 1664]", "[B, 512]", "GELU, Dropout(p=0.3)"],
        ["Classifier Dense 2", "[B, 512]", "[B, 128]", "GELU, Dropout(p=0.2)"],
        ["AI Binary Logit Output", "[B, 128]", "[B, 1]", "Linear Logit (Raw)"],
        ["DANN Domain Head", "[B, 1664]", "[B, 2]", "GRL (-α), GELU, Dropout(0.2)"]
    ]

    t = Table(table_data, colWidths=[130, 105, 95, 170])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#FFFFFF')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8.5),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
        ('TOPPADDING', (0, 0), (-1, 0), 5),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8FAFC')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8FAFC'), colors.HexColor('#FFFFFF')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#334155')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t)

    # 4. Code Interface Signatures
    story.append(Spacer(1, 4))
    story.append(Paragraph("4. Python Code Interface & Usage", heading2_style))
    code_example = (
        "from src.models import build_model<br/><br/>"
        "# Instantiate Ultimate Hybrid Detector<br/>"
        "model = build_model(pretrained=True, architecture='hybrid_effnet_dinov2')<br/><br/>"
        "# Standard Inference Mode (Returns AI logit [B, 1]):<br/>"
        "ai_logits = model(images)  # shape: [batch_size, 1]<br/>"
        "probs = torch.sigmoid(ai_logits)<br/><br/>"
        "# Multi-Task DANN Training Mode (Returns AI logit + Domain logit):<br/>"
        "ai_logits, domain_logits = model(images, alpha=0.5, return_domain=True)"
    )
    story.append(Paragraph(code_example, code_style))

    doc.build(story)
    print("PDF generated successfully:", filename)

if __name__ == "__main__":
    output_pdf = "/Users/keyi/Documents/Megamind/UltimateHybridDetector_Implementation_Details.pdf"
    create_pdf(output_pdf)
