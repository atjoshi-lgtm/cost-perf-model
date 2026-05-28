"""
Generate problem_formulation.pdf from problem_formulation.md.
Uses matplotlib mathtext for equations and reportlab for layout.
"""
from __future__ import annotations
import io, re, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Image as RLImage, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT     = Path(__file__).resolve().parent
MD_PATH  = ROOT / "problem_formulation.md"
PDF_PATH = ROOT / "problem_formulation.pdf"
PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm


def build_styles():
    s = {}
    s["body"] = ParagraphStyle("body", fontName="Times-Roman", fontSize=11, leading=16,
                                alignment=TA_JUSTIFY, spaceAfter=6)
    s["h1"] = ParagraphStyle("h1", fontName="Times-Bold", fontSize=18, leading=22,
                              spaceBefore=4, spaceAfter=8, textColor=colors.HexColor("#111111"))
    s["h2"] = ParagraphStyle("h2", fontName="Times-Bold", fontSize=14, leading=18,
                              spaceBefore=14, spaceAfter=4, textColor=colors.HexColor("#222222"))
    s["h3"] = ParagraphStyle("h3", fontName="Times-Bold", fontSize=12, leading=16,
                              spaceBefore=10, spaceAfter=3, textColor=colors.HexColor("#333333"))
    s["bullet"] = ParagraphStyle("bullet", fontName="Times-Roman", fontSize=11, leading=15,
                                  leftIndent=18, bulletIndent=6, spaceBefore=1, spaceAfter=1)
    s["th"] = ParagraphStyle("th", fontName="Times-Bold", fontSize=10, leading=13)
    s["td"] = ParagraphStyle("td", fontName="Times-Roman", fontSize=10, leading=13)
    return s


def render_math_image(latex: str, display: bool, max_w: float) -> RLImage:
    expr = f"${latex}$"
    fs = 13 if display else 11
    fig = plt.figure(figsize=(0.01, 0.01))
    r   = fig.canvas.get_renderer()
    t   = fig.text(0, 0, expr, fontsize=fs, usetex=False)
    bb  = t.get_window_extent(renderer=r)
    plt.close(fig)
    pad, dpi = 10, 150
    w_px = max(bb.width + 2*pad, 40)
    h_px = max(bb.height + 2*pad, 20)
    fig = plt.figure(figsize=(w_px/dpi, h_px/dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    fig.text(pad/w_px, pad/h_px, expr, fontsize=fs, usetex=False, verticalalignment="bottom")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = RLImage(buf)
    if display and img.drawWidth > max_w:
        sc = max_w / img.drawWidth
        img.drawWidth  *= sc
        img.drawHeight *= sc
    return img


def xe(t): return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def ifmt(text):
    text = re.sub(r'\*\*(.+?)\*\*', lambda m: f'<b>{m.group(1)}</b>', text)
    text = re.sub(r'\*(.+?)\*',     lambda m: f'<i>{m.group(1)}</i>', text)
    text = re.sub(r'`([^`]+)`', lambda m: f'<font face="Courier" size="9">{xe(m.group(1))}</font>', text)
    return text

def inline_math_to_text(text):
    return re.sub(r'\$([^\$\n]+?)\$',
                  lambda m: f'<i><font face="Courier" size="9">{m.group(1)}</font></i>', text)


def parse_md(md_text, styles, max_w):
    flowables = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # Display math $$...$$
        if line.strip().startswith("$$"):
            math_lines = []
            inner = line.strip()[2:]
            if inner.endswith("$$"):
                math_lines.append(inner[:-2].strip()); i += 1
            else:
                if inner: math_lines.append(inner)
                i += 1
                while i < len(lines) and not lines[i].strip().endswith("$$"):
                    math_lines.append(lines[i]); i += 1
                if i < len(lines):
                    math_lines.append(lines[i].strip()[:-2]); i += 1
            latex = "\n".join(math_lines).strip()
            try:
                img = render_math_image(latex, display=True, max_w=max_w)
                tbl = Table([[img]], colWidths=[max_w])
                tbl.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER")]))
                flowables += [Spacer(1,4), tbl, Spacer(1,4)]
            except Exception as exc:
                print(f"  [warn] {exc}", file=sys.stderr)
                flowables.append(Paragraph(f'<font face="Courier" size="9">{xe(latex)}</font>', styles["body"]))
            continue

        # Headings
        m = re.match(r'^(#{1,3})\s+(.*)', line)
        if m:
            lv = len(m.group(1))
            txt = inline_math_to_text(ifmt(xe(m.group(2))))
            key = {1:"h1",2:"h2",3:"h3"}[lv]
            flowables.append(Paragraph(txt, styles[key]))
            if lv == 1:
                flowables.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#333333")))
            elif lv == 2:
                flowables.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aaaaaa")))
            i += 1; continue

        # Table
        if line.strip().startswith("|"):
            tlines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tlines.append(lines[i]); i += 1
            rows = []
            for tl in tlines:
                if re.match(r'^\|[-| :]+\|?\s*$', tl): continue
                cells = [c.strip() for c in tl.strip().strip("|").split("|")]
                rows.append(cells)
            if rows:
                nc = max(len(r) for r in rows)
                cw = max_w / nc
                data, is_hdr = [], True
                for row in rows:
                    while len(row) < nc: row.append("")
                    sk = "th" if is_hdr else "td"
                    data.append([Paragraph(inline_math_to_text(ifmt(xe(c))), styles[sk]) for c in row])
                    is_hdr = False
                tbl = Table(data, colWidths=[cw]*nc, repeatRows=1)
                tbl.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e8e8e8")),
                    ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#aaaaaa")),
                    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f8f8f8")]),
                    ("VALIGN",(0,0),(-1,-1),"TOP"),
                    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
                    ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
                ]))
                flowables += [Spacer(1,4), tbl, Spacer(1,6)]
            continue

        # Bullet list
        if re.match(r'^[-*]\s+', line):
            while i < len(lines) and re.match(r'^[-*]\s+', lines[i]):
                txt = inline_math_to_text(ifmt(xe(lines[i][2:].strip())))
                flowables.append(Paragraph(f"• {txt}", styles["bullet"]))
                i += 1
            flowables.append(Spacer(1,3)); continue

        # Blank
        if not line.strip():
            flowables.append(Spacer(1,5)); i += 1; continue

        # Paragraph
        para = [line]; i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r'^(#{1,3}|[-*]|\||\$\$)', lines[i]):
            para.append(lines[i]); i += 1
        txt = inline_math_to_text(ifmt(xe(" ".join(para))))
        flowables.append(Paragraph(txt, styles["body"]))

    return flowables


def main():
    md_text = MD_PATH.read_text(encoding="utf-8")
    usable  = PAGE_W - 2 * MARGIN
    styles  = build_styles()
    doc     = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="Joint Cost–Performance Optimization of CDN Disk Capacity",
    )
    print("Parsing markdown …")
    flowables = parse_md(md_text, styles, max_w=usable)
    print("Building PDF …")
    doc.build(flowables)
    print(f"Saved: {PDF_PATH}")

if __name__ == "__main__":
    main()
