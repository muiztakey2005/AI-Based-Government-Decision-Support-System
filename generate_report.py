"""
generate_report.py — Professional weekly PDF report using ReportLab.
"""
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

# ── Colours ────────────────────────────────────────────────────────────
NAVY = HexColor('#0d1b2a')
BLUE = HexColor('#1b263b')
TEAL = HexColor('#00b4d8')
LIGHT = HexColor('#e0e1dd')
WHITE = HexColor('#ffffff')
RED = HexColor('#9b2226')
GREEN = HexColor('#2d6a4f')
ORANGE = HexColor('#e76f51')


def _clean_text(text):
    """Remove emojis and non-Latin1 characters for ReportLab's Helvetica."""
    if not isinstance(text, str):
        text = str(text)
    # Remove emojis and any character outside the Latin-1 range
    text = re.sub(r'[^\x00-\xff]', '', text)
    # Clean up leftover spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else "N/A"


def _cover_page(canvas, doc):
    """Draw a styled cover page."""
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, letter[1] - 200, letter[0], 200, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 32)
    canvas.drawCentredString(letter[0] / 2, letter[1] - 90,
                             "GovIntelligence Platform")
    canvas.setFont("Helvetica", 18)
    canvas.drawCentredString(letter[0] / 2, letter[1] - 130,
                             "Weekly Decision Intelligence Report")
    canvas.setFont("Helvetica", 14)
    canvas.setFillColor(LIGHT)
    canvas.drawCentredString(letter[0] / 2, letter[1] - 160,
                             datetime.now().strftime("%B %d, %Y"))
    canvas.setFont("Helvetica", 11)
    canvas.drawCentredString(letter[0] / 2, 120,
                             "AI-Driven Government Decision Intelligence")
    canvas.drawCentredString(letter[0] / 2, 100,
                             "New York City 311 Complaint Analysis")
    canvas.restoreState()


def _later_pages(canvas, doc):
    """Header/footer for subsequent pages."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(HexColor('#778da9'))
    canvas.drawString(72, letter[1] - 40,
                      "GovIntelligence — Weekly Report")
    canvas.drawRightString(letter[0] - 72, letter[1] - 40,
                           datetime.now().strftime("%Y-%m-%d"))
    canvas.drawCentredString(letter[0] / 2, 30,
                             f"Page {doc.page}")
    canvas.restoreState()


def generate_weekly_report(df: pd.DataFrame, output_dir: str = 'reports') -> str:
    """Generate a professional PDF report from the complaint DataFrame."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"weekly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(output_dir, filename)

    doc = SimpleDocTemplate(filepath, pagesize=letter,
                            topMargin=72, bottomMargin=60,
                            leftMargin=54, rightMargin=54)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CoverTitle', fontName='Helvetica-Bold',
                              fontSize=24, textColor=TEAL, alignment=TA_CENTER,
                              spaceAfter=20))
    styles.add(ParagraphStyle(name='SectionHead', fontName='Helvetica-Bold',
                              fontSize=16, textColor=NAVY, spaceAfter=12,
                              spaceBefore=18))
    styles.add(ParagraphStyle(name='SubHead', fontName='Helvetica-Bold',
                              fontSize=12, textColor=BLUE, spaceAfter=8))
    styles.add(ParagraphStyle(name='Body', fontName='Helvetica', fontSize=10,
                              textColor=black, spaceAfter=6))
    styles.add(ParagraphStyle(name='SmallBody', fontName='Helvetica',
                              fontSize=9, textColor=HexColor('#555555')))

    story = []

    # ── Cover page placeholder ──────────────────────────────────────
    story.append(Spacer(1, 500))
    story.append(PageBreak())

    # ── 1. Executive Summary / KPIs ─────────────────────────────────
    story.append(Paragraph("1. Executive Summary", styles['SectionHead']))
    total = len(df)
    open_cnt = len(df[df['status'].isin(['Open', 'In Progress'])]) \
        if 'status' in df.columns else 0
    closed_cnt = len(df[df['status'] == 'Closed']) \
        if 'status' in df.columns else 0
    boroughs = df['borough'].nunique() if 'borough' in df.columns else 0
    agencies = df['agency'].nunique() if 'agency' in df.columns else 0

    kpi_data = [
        ['Metric', 'Value'],
        ['Total Complaints', f'{total:,}'],
        ['Open / In Progress', f'{open_cnt:,}'],
        ['Closed', f'{closed_cnt:,}'],
        ['Boroughs Covered', f'{boroughs}'],
        ['Agencies Involved', f'{agencies}'],
        ['Report Date', datetime.now().strftime('%Y-%m-%d %H:%M')],
    ]
    kpi_table = Table(kpi_data, colWidths=[2.5 * inch, 2.5 * inch])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, HexColor('#f0f4f8')]),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 20))

    # ── 2. Borough Breakdown ────────────────────────────────────────
    story.append(Paragraph("2. Complaints by Borough", styles['SectionHead']))
    if 'borough' in df.columns:
        b_counts = df['borough'].value_counts().reset_index()
        b_counts.columns = ['Borough', 'Count']
        b_counts['Percentage'] = (b_counts['Count'] / total * 100).round(1)
        b_rows = [['Borough', 'Count', 'Percentage']] + \
                 b_counts.values.tolist()
        b_table = Table(b_rows, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        b_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), BLUE),
            ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [WHITE, HexColor('#f0f4f8')]),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(b_table)
    story.append(Spacer(1, 20))

    # ── 3. Agency Breakdown ─────────────────────────────────────────
    story.append(Paragraph("3. Complaints by Agency", styles['SectionHead']))
    if 'agency' in df.columns:
        a_counts = df['agency'].value_counts().reset_index()
        a_counts.columns = ['Agency', 'Count']
        a_counts['Percentage'] = (a_counts['Count'] / total * 100).round(1)
        a_rows = [['Agency', 'Count', 'Percentage']] + \
                 a_counts.values.tolist()
        a_table = Table(a_rows, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        a_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), BLUE),
            ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [WHITE, HexColor('#f0f4f8')]),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(a_table)
    story.append(Spacer(1, 20))

    # ── 4. Top Complaint Types ──────────────────────────────────────
    story.append(Paragraph("4. Top Complaint Types", styles['SectionHead']))
    if 'complaint_type' in df.columns:
        ct = df['complaint_type'].value_counts().head(10).reset_index()
        ct.columns = ['Complaint Type', 'Count']
        ct_rows = [['#', 'Complaint Type', 'Count']] + \
                  [[i+1, row['Complaint Type'], row['Count']]
                   for i, row in ct.iterrows()]
        ct_table = Table(ct_rows, colWidths=[0.5*inch, 3.5*inch, 1.5*inch])
        ct_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), NAVY),
            ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (2, 0), (2, -1), 'CENTER'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [WHITE, HexColor('#f0f4f8')]),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(ct_table)
    story.append(Spacer(1, 20))

    # ── 5. AI Recommendations ───────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("5. AI-Generated Recommendations", styles['SectionHead']))
    try:
        from modules.decision_engine import DecisionEngine
        engine = DecisionEngine(df)
        recs = engine.generate_recommendations(top_n=10)
    except Exception:
        recs = []
        
    if recs:
        rec_rows = [['Priority', 'ID', 'Type', 'Borough', 'Agency',
                      'Score', 'Action']]
        for r in recs:
            rec_rows.append([
                _clean_text(r['priority_label'])[:20], 
                _clean_text(r['unique_key'])[:18],
                _clean_text(r['complaint_type'])[:20], 
                _clean_text(r['borough'])[:12],
                _clean_text(r['agency']), 
                str(r['priority_score']),
                _clean_text(r['action'])[:60]
            ])
        rec_table = Table(rec_rows,
                          colWidths=[0.7*inch, 1*inch, 1*inch, 0.8*inch,
                                     0.6*inch, 0.5*inch, 2.2*inch])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), RED),
            ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [WHITE, HexColor('#fff0f0')]),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(rec_table)
    else:
        story.append(Paragraph("No recommendations available.", styles['Body']))
    story.append(Spacer(1, 20))

    # ── 6. Resource Allocation Table ────────────────────────────────
    story.append(Paragraph("6. Resource Allocation", styles['SectionHead']))
    try:
        from modules.optimizer import ResourceOptimizer
        opt = ResourceOptimizer(df)
        alloc = opt.optimize_allocation()
    except Exception:
        alloc = {}
        
    if alloc:
        alloc_rows = [['Agency', 'Resources', 'Open Complaints',
                       'Complaints/Resource', 'Share %']]
        for agency, info in alloc.items():
            alloc_rows.append([
                agency, str(info['resources']),
                str(info['open_complaints']),
                str(info['ratio']), f"{info['share_pct']}%"
            ])
        alloc_table = Table(alloc_rows,
                            colWidths=[1.2*inch, 1*inch, 1.3*inch,
                                       1.3*inch, 1*inch])
        alloc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), BLUE),
            ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [WHITE, HexColor('#f0f4f8')]),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(alloc_table)

        # Reallocation suggestions
        story.append(Spacer(1, 12))
        try:
            suggestions = opt.suggest_reallocation()
        except Exception:
            suggestions = []
        if suggestions:
            story.append(Paragraph("Reallocation Suggestions:", styles['SubHead']))
            for s in suggestions:
                story.append(Paragraph(f"• {_clean_text(s)}", styles['SmallBody']))
    story.append(Spacer(1, 20))

    # ── 7. Crisis Alerts ────────────────────────────────────────────
    story.append(Paragraph("7. Crisis Early Warning", styles['SectionHead']))
    try:
        from modules.analytics_engine import AnalyticsEngine
        analytics = AnalyticsEngine(df)
        alerts = analytics.detect_crisis()
    except Exception:
        alerts = []
        
    if alerts:
        for a in alerts:
            story.append(Paragraph(
                f"WARNING: {_clean_text(a['date'])} - {a['count']} complaints "
                f"(expected ~{a['expected']}, "
                f"deviation: +{a['deviation']} sigma)",
                styles['Body']))
    else:
        story.append(Paragraph(
            "No crisis-level spikes detected.", styles['Body']))

    # ── Build PDF ───────────────────────────────────────────────────
    doc.build(story, onFirstPage=_cover_page, onLaterPages=_later_pages)
    return filepath


if __name__ == '__main__':
    from modules.data_processor import DataProcessor
    dp = DataProcessor()
    df = dp.get_combined()
    path = generate_weekly_report(df)
    print(f"✅ Report generated: {path}")