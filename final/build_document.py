"""Build the short technical PDF and standalone schematic figures."""

from html import escape
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Table, TableStyle

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
NAVY = "#15314b"
BLUE = "#167b9e"
ORANGE = "#bd6639"


def save(fig, name):
    for suffix in ("png", "svg"):
        path = FIGURES / f"{name}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(11.5, 3.7))
    ax.set(xlim=(0, 11.5), ylim=(0, 3.7))
    ax.axis("off")

    def box(x, y, w, h, title, body, color=BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                   facecolor="#f2f6f8", edgecolor=color, linewidth=1.3))
        ax.text(x+w/2, y+h*.69, title, ha="center", va="center", fontsize=10, weight="bold", color=color)
        ax.text(x+w/2, y+h*.31, body, ha="center", va="center", fontsize=8.5, color=NAVY)

    def arrow(a, b):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.4))

    box(.15, 1.25, 1.65, 1.05, "Official inputs", "Tracking + teams\nAutomatic kickoff tags")
    box(2.3, 2.2, 2.2, 1.05, "Stoppage TCN", "113 features · 10 Hz\nPhysics + kickoff decoding")
    box(2.3, .45, 2.2, 1.05, "Spatial TCN", "148 features · 5 Hz\nConditional H/A evidence")
    box(5.05, 2.2, 2.0, 1.05, "Fixed D / X", "Preserved exactly\nthrough final selection", ORANGE)
    box(5.05, .45, 2.0, 1.05, "15 decoder paths", "Candidate H/A spells\n22 features per segment")
    box(7.65, .9, 2.0, 1.65, "Learned scorer", "Boosted segment quality\nComplete alternating path\nPenalty = 0.4")
    box(10.2, 1.25, 1.05, 1.05, "CSV", "199,319\nframes", ORANGE)
    arrow((1.8, 1.95), (2.3, 2.7)); arrow((1.8, 1.6), (2.3, .97))
    arrow((4.5, 2.72), (5.05, 2.72)); arrow((4.5, .97), (5.05, .97))
    arrow((7.05, 2.72), (7.65, 2.22)); arrow((7.05, .97), (7.65, 1.25))
    arrow((9.65, 1.77), (10.2, 1.77))
    ax.text(.15, 3.5, "FROZEN INFERENCE PIPELINE", fontsize=11, color=NAVY, weight="bold")
    save(fig, "architecture")


def spell_path():
    fig, ax = plt.subplots(figsize=(11.5, 3.1))
    ax.set(xlim=(0, 12), ylim=(-.2, 4.2))
    ax.set_yticks([3, 2, 1, 0], ["Proposal path 1", "Proposal path 2", "Proposal path 3", "Selected path"])
    ax.set_xticks([0, 3, 6, 9, 12], ["Start", "", "", "", "End"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    rows = [[(0, 3, "H"), (3, 6, "A"), (6, 9, "H"), (9, 12, "D")],
            [(0, 4, "H"), (4, 6, "A"), (6, 9, "H"), (9, 12, "D")],
            [(0, 3, "H"), (3, 7, "A"), (7, 9, "H"), (9, 12, "D")],
            [(0, 3, "H"), (3, 6, "A"), (6, 9, "H"), (9, 12, "D")]]
    palette = {"H": BLUE, "A": ORANGE, "D": "#8796a1"}
    for row, spans in zip([3, 2, 1, 0], rows):
        for a, b, side in spans:
            ax.barh(row, b-a, left=a, height=.62, color=palette[side], alpha=1 if row==0 else .55)
            ax.text((a+b)/2, row, side, color="white", ha="center", va="center", fontsize=11, weight="bold")
    ax.axvline(9, color=NAVY, lw=1, ls="--", alpha=.5)
    ax.text(10.5, 3.75, "D stays fixed", ha="center", fontsize=10, color=NAVY)
    ax.set_title("Score whole segments; join only exact endpoints with alternating teams",
                 loc="left", fontsize=12, color=NAVY, pad=18, weight="bold")
    ax.set_xlabel("Illustrative timeline — schematic, not match predictions", fontsize=9, color=NAVY)
    fig.tight_layout()
    save(fig, "spell_selection")


def markup(text):
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="DejaVuMono" size="8">\1</font>', text)
    return text


def main():
    FIGURES.mkdir(exist_ok=True)
    architecture()
    spell_path()
    fontroot = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, file in [("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
                       ("DejaVu-Oblique", "DejaVuSans-Oblique.ttf"),
                       ("DejaVu-BoldOblique", "DejaVuSans-BoldOblique.ttf"),
                       ("DejaVuMono", "DejaVuSansMono.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(fontroot/file)))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                                  italic="DejaVu-Oblique", boldItalic="DejaVu-BoldOblique")
    body = ParagraphStyle("BodyFinal", fontName="DejaVu", fontSize=9, leading=12,
                          textColor=colors.HexColor(NAVY), spaceAfter=7, alignment=TA_LEFT)
    heading = ParagraphStyle("HeadingFinal", parent=body, fontName="DejaVu-Bold", fontSize=12,
                             leading=16, spaceBefore=12, spaceAfter=7, keepWithNext=True)
    subheading = ParagraphStyle("SubheadingFinal", parent=heading, fontSize=10.5, leading=14, spaceBefore=10)
    title = ParagraphStyle("TitleFinal", parent=heading, fontSize=21, leading=26, spaceBefore=0, spaceAfter=12)
    caption = ParagraphStyle("CaptionFinal", parent=body, fontSize=8, leading=11, textColor=colors.HexColor("#536778"))
    story = []
    lines = (ROOT/"technical_document.md").read_text().splitlines()
    paragraph = []
    table_rows = []

    def flush():
        if paragraph:
            story.append(Paragraph(markup(" ".join(paragraph)), body))
            paragraph.clear()

    def flush_table():
        if not table_rows:
            return
        table_style = ParagraphStyle("TableFinal", parent=body, fontSize=7.5, leading=10, spaceAfter=0)
        cells = [[Paragraph(markup(value), table_style) for value in row] for row in table_rows]
        header = table_rows[0]
        if len(header) == 3:
            widths = [120, 45, 335]
        elif header[0] == "Training source":
            widths = [110, 45, 75, 130, 140]
        elif header[0] == "Method":
            widths = [155, 70, 75, 65, 65, 70]
        else:
            widths = [180, 45, 90, 90, 95]
        table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f0f5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, 0), .7, colors.HexColor(NAVY)),
            ("LINEBELOW", (0, 1), (-1, -1), .3, colors.HexColor("#d7e1e8")),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("SPACEAFTER", (0, 0), (-1, -1), 9),
        ]))
        story.append(table)
        table_rows.clear()

    for line in lines:
        if line.startswith("|"):
            flush()
            row = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", cell) for cell in row):
                table_rows.append(row)
            continue
        flush_table()
        if line == "![Inference architecture](figures/architecture.png)":
            flush()
            story.extend([Image(str(FIGURES/"architecture.png"), width=500, height=162),
                          Paragraph("Figure 1. Frozen inference architecture for frame-level possession recognition.", caption)])
        elif line == "![Segment selection](figures/spell_selection.png)":
            flush()
            story.extend([Image(str(FIGURES/"spell_selection.png"), width=500, height=137),
                          Paragraph("Figure 2. Schematic candidate paths and complete-coverage segment selection; no match data are shown.", caption)])
        elif line.startswith("### "):
            flush()
            story.append(Paragraph(markup(line[4:]), subheading))
        elif line.startswith("# "):
            flush(); story.append(Paragraph(markup(line[2:]), title))
        elif line.startswith("## "):
            flush()
            story.append(Paragraph(markup(line[3:]), heading))
        elif not line.strip():
            flush()
        elif line.startswith("- "):
            flush(); paragraph.append("• "+line[2:])
        else:
            paragraph.append(line)
    flush()
    flush_table()
    doc = SimpleDocTemplate(str(ROOT/"Absolute Donut.pdf"), pagesize=A4,
                            rightMargin=46, leftMargin=46, topMargin=43, bottomMargin=43,
                            title="Absolute Donut — Learned spell scoring",
                            author="Eduardus Tjitrahardja; Louis Jefferson Zhang; Stuart Davis Tioniwar; Joel Van Doli Marpaung")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("DejaVu", 8)
        canvas.setFillColor(colors.HexColor("#536778"))
        canvas.drawString(46, 25, "Absolute Donut  ·  Learned spell scoring")
        canvas.drawRightString(A4[0]-46, 25, str(doc.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("Wrote Absolute Donut.pdf and four figure files")


if __name__ == "__main__":
    main()
