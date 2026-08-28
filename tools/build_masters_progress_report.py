from __future__ import annotations

import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_ROW_HEIGHT_RULE, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports"
ASSET_DIR = OUT_DIR / "_progress_report_assets"
OUT_PATH = OUT_DIR / "LLMSTU_Masters_Thesis_Progress_Report_2026-08-10.docx"

NAVY = "0B2545"
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
INK = "20252B"
MUTED = "59636E"
LIGHT = "F2F4F7"
MID = "D8DEE6"
PALE_BLUE = "E8EEF5"
PALE_GREEN = "EAF4ED"
GREEN = "2E6F40"
PALE_GOLD = "FFF4D6"
GOLD = "8A6500"
PALE_RED = "FBECEC"
RED = "9B1C1C"
WHITE = "FFFFFF"


def rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


def set_run_font(run, name="Calibri", size=11, color=INK, bold=False, italic=False):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    run.font.color.rgb = rgb(color)
    run.bold = bold
    run.italic = italic


def set_cell_shading(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_border(cell, **edges):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_borders = tc_pr.first_child_found_in("w:tcBorders")
    if tc_borders is None:
        tc_borders = OxmlElement("w:tcBorders")
        tc_pr.append(tc_borders)
    for edge_name, edge_data in edges.items():
        tag = f"w:{edge_name}"
        edge = tc_borders.find(qn(tag))
        if edge is None:
            edge = OxmlElement(tag)
            tc_borders.append(edge)
        for key, value in edge_data.items():
            edge.set(qn(f"w:{key}"), str(value))


def set_table_geometry(table, widths_dxa, indent_dxa=120):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")

    total = sum(widths_dxa)
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(total))
    tbl_w.set(qn("w:type"), "dxa")

    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(indent_dxa))
    tbl_ind.set(qn("w:type"), "dxa")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row in table.rows:
        row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
        for idx, cell in enumerate(row.cells):
            width = widths_dxa[idx]
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
            cell.width = Inches(width / 1440)
            set_cell_margins(cell)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def prevent_row_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def set_paragraph_border(paragraph, color=BLUE, size=12, space=6):
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = p_pr.find(qn("w:pBdr"))
    if p_bdr is None:
        p_bdr = OxmlElement("w:pBdr")
        p_pr.append(p_bdr)
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(size))
    bottom.set(qn("w:space"), str(space))
    bottom.set(qn("w:color"), color)
    p_bdr.append(bottom)


def add_page_field(paragraph):
    run = paragraph.add_run()
    fld_char_1 = OxmlElement("w:fldChar")
    fld_char_1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_char_2 = OxmlElement("w:fldChar")
    fld_char_2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char_1)
    run._r.append(instr)
    run._r.append(fld_char_2)


def add_total_pages_field(paragraph):
    run = paragraph.add_run()
    fld_char_1 = OxmlElement("w:fldChar")
    fld_char_1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " NUMPAGES "
    fld_char_2 = OxmlElement("w:fldChar")
    fld_char_2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char_1)
    run._r.append(instr)
    run._r.append(fld_char_2)


def add_num_definition(doc, fmt="bullet"):
    numbering = doc.part.numbering_part.element
    abstract_ids = [
        int(x.get(qn("w:abstractNumId")))
        for x in numbering.findall(qn("w:abstractNum"))
        if x.get(qn("w:abstractNumId"))
    ]
    num_ids = [
        int(x.get(qn("w:numId")))
        for x in numbering.findall(qn("w:num"))
        if x.get(qn("w:numId"))
    ]
    abstract_id = max(abstract_ids + [0]) + 1
    num_id = max(num_ids + [0]) + 1

    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(abstract_id))
    multi = OxmlElement("w:multiLevelType")
    multi.set(qn("w:val"), "singleLevel")
    abstract.append(multi)
    lvl = OxmlElement("w:lvl")
    lvl.set(qn("w:ilvl"), "0")
    start = OxmlElement("w:start")
    start.set(qn("w:val"), "1")
    lvl.append(start)
    num_fmt = OxmlElement("w:numFmt")
    num_fmt.set(qn("w:val"), "bullet" if fmt == "bullet" else "decimal")
    lvl.append(num_fmt)
    lvl_text = OxmlElement("w:lvlText")
    lvl_text.set(qn("w:val"), "\u2022" if fmt == "bullet" else "%1.")
    lvl.append(lvl_text)
    suff = OxmlElement("w:suff")
    suff.set(qn("w:val"), "tab")
    lvl.append(suff)
    p_pr = OxmlElement("w:pPr")
    tabs = OxmlElement("w:tabs")
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "num")
    tab.set(qn("w:pos"), "720")
    tabs.append(tab)
    p_pr.append(tabs)
    ind = OxmlElement("w:ind")
    ind.set(qn("w:left"), "720")
    ind.set(qn("w:hanging"), "360")
    p_pr.append(ind)
    lvl.append(p_pr)
    abstract.append(lvl)
    # OOXML requires every abstractNum before the concrete num instances.
    # Insert here rather than appending after an existing <w:num>.
    first_num = numbering.find(qn("w:num"))
    if first_num is None:
        numbering.append(abstract)
    else:
        numbering.insert(numbering.index(first_num), abstract)

    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(num_id))
    abstract_num_id = OxmlElement("w:abstractNumId")
    abstract_num_id.set(qn("w:val"), str(abstract_id))
    num.append(abstract_num_id)
    if fmt == "decimal":
        lvl_override = OxmlElement("w:lvlOverride")
        lvl_override.set(qn("w:ilvl"), "0")
        start_override = OxmlElement("w:startOverride")
        start_override.set(qn("w:val"), "1")
        lvl_override.append(start_override)
        num.append(lvl_override)
    numbering.append(num)
    return num_id


def apply_numbering(paragraph, num_id):
    p_pr = paragraph._p.get_or_add_pPr()
    num_pr = p_pr.find(qn("w:numPr"))
    if num_pr is None:
        num_pr = OxmlElement("w:numPr")
        p_pr.append(num_pr)
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    num_id_node = OxmlElement("w:numId")
    num_id_node.set(qn("w:val"), str(num_id))
    num_pr.append(ilvl)
    num_pr.append(num_id_node)


def add_list_item(doc, text, bold_prefix=None):
    # A compact evidence-card paragraph avoids fragile cross-application list
    # numbering while preserving genuine visual grouping without fake glyphs.
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.14)
    p.paragraph_format.right_indent = Inches(0.06)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.167
    set_paragraph_border(p, color="D8DEE6", size=4, space=3)
    if bold_prefix and text.startswith(bold_prefix):
        run = p.add_run(bold_prefix)
        set_run_font(run, bold=True)
        run = p.add_run(text[len(bold_prefix):])
        set_run_font(run)
    else:
        run = p.add_run(text)
        set_run_font(run)
    return p


def add_source(doc, text):
    p = doc.add_paragraph()
    p.style = doc.styles["Source Note"]
    r = p.add_run(text)
    set_run_font(r, size=8, color=MUTED, italic=True)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.style = doc.styles["Caption"]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    set_run_font(r, size=9, color=MUTED, italic=True)
    return p


def add_callout(doc, label, text, fill=PALE_BLUE, accent=BLUE):
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [9360])
    prevent_row_split(table.rows[0])
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    set_cell_border(
        cell,
        start={"val": "single", "sz": "18", "color": accent},
        top={"val": "nil"},
        bottom={"val": "nil"},
        end={"val": "nil"},
    )
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.10
    r = p.add_run(label + " ")
    set_run_font(r, bold=True, color=accent)
    r = p.add_run(text)
    set_run_font(r)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)
    return table


def add_table(doc, headers, rows, widths, aligns=None, highlights=None, font_size=9):
    table = doc.add_table(rows=1, cols=len(headers))
    set_table_geometry(table, widths)
    header = table.rows[0]
    set_repeat_table_header(header)
    prevent_row_split(header)
    for i, text in enumerate(headers):
        cell = header.cells[i]
        set_cell_shading(cell, LIGHT)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        p = cell.paragraphs[0]
        p.alignment = aligns[i] if aligns else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(str(text))
        set_run_font(r, size=font_size, color=NAVY, bold=True)
    for row_idx, values in enumerate(rows):
        row = table.add_row()
        prevent_row_split(row)
        if highlights and row_idx in highlights:
            fill, color = highlights[row_idx]
        else:
            fill, color = (WHITE, INK)
        for i, value in enumerate(values):
            cell = row.cells[i]
            set_cell_shading(cell, fill)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.alignment = aligns[i] if aligns else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            r = p.add_run(str(value))
            set_run_font(r, size=font_size, color=color, bold=(highlights and row_idx in highlights))
    return table


def add_picture_with_alt(doc, path, width_inches, alt_text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    inline_shape = run.add_picture(str(path), width=Inches(width_inches))
    doc_pr = inline_shape._inline.docPr
    doc_pr.set("descr", alt_text)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    return p


def chart_font(size, bold=False):
    candidates = [
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def save_bar_chart(path, title, labels, values, colors, max_value, tick=0.1, value_fmt="{:.3f}"):
    width, height = 1600, 760
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font_title = chart_font(42, True)
    font_label = chart_font(30)
    font_value = chart_font(28, True)
    font_tick = chart_font(23)
    # Leave enough room for the longest category label.  The prior 310 px
    # margin clipped "Exact correspondence (test)" after DOCX scaling.
    left, right, top, bottom = 430, 80, 110, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    draw.text((left, 25), title, font=font_title, fill="#" + NAVY)
    n = len(labels)
    gap = 28
    bar_h = (plot_h - gap * (n - 1)) / n
    for i, (label, val, color) in enumerate(zip(labels, values, colors)):
        y0 = top + i * (bar_h + gap)
        y1 = y0 + bar_h
        x1 = left + plot_w * (val / max_value)
        draw.rounded_rectangle((left, y0, x1, y1), radius=12, fill="#" + color)
        bbox = draw.textbbox((0, 0), label, font=font_label)
        draw.text((left - 18 - (bbox[2] - bbox[0]), y0 + (bar_h - (bbox[3] - bbox[1])) / 2 - 2),
                  label, font=font_label, fill="#" + INK)
        value_text = value_fmt.format(val)
        draw.text((x1 + 12, y0 + (bar_h - 28) / 2 - 2), value_text, font=font_value, fill="#" + NAVY)
    steps = int(math.floor(max_value / tick))
    for i in range(steps + 1):
        val = i * tick
        x = left + plot_w * val / max_value
        draw.line((x, top - 8, x, height - bottom + 8), fill="#D8DEE6", width=2)
        draw.text((x - 18, height - bottom + 20), f"{val:.1f}", font=font_tick, fill="#" + MUTED)
    img.save(path, dpi=(180, 180))


def save_effect_chart(path):
    width, height = 1600, 820
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = chart_font(42, True)
    label_font = chart_font(28)
    value_font = chart_font(25, True)
    tick_font = chart_font(22)
    draw.text((360, 26), "Branch C paired effects (inner validation)", font=title_font, fill="#" + NAVY)
    items = [
        ("Full-range pose", -0.0022, -0.0119, 0.0085, MUTED),
        ("Uniform fusion", 0.0019, -0.0025, 0.0057, MUTED),
        ("Learned reliability", 0.0015, -0.0007, 0.0041, MUTED),
        ("Observability losses", 0.0041, 0.0010, 0.0070, GOLD),
        ("Whole fusion stack", 0.0072, 0.0026, 0.0116, GOLD),
        ("Training configuration*", 0.0406, 0.0289, 0.0521, GREEN),
    ]
    left, right, top, bottom = 380, 120, 130, 100
    plot_w = width - left - right
    xmin, xmax = -0.02, 0.06
    def xpos(v):
        return left + (v - xmin) / (xmax - xmin) * plot_w
    for tick in [-0.02, 0.0, 0.02, 0.04, 0.06]:
        x = xpos(tick)
        draw.line((x, top - 8, x, height - bottom), fill="#D8DEE6", width=2)
        draw.text((x - 30, height - bottom + 20), f"{tick:+.02f}", font=tick_font, fill="#" + MUTED)
    draw.line((xpos(0), top - 12, xpos(0), height - bottom), fill="#" + NAVY, width=4)
    draw.line((xpos(0.02), top - 12, xpos(0.02), height - bottom), fill="#" + RED, width=4)
    draw.text((xpos(0.02) + 8, top - 50), "+0.02 go gate", font=tick_font, fill="#" + RED)
    row_gap = (height - top - bottom) / len(items)
    for i, (label, mean, lo, hi, color) in enumerate(items):
        y = top + row_gap * (i + 0.5)
        bbox = draw.textbbox((0, 0), label, font=label_font)
        draw.text((left - 18 - (bbox[2] - bbox[0]), y - 18), label, font=label_font, fill="#" + INK)
        draw.line((xpos(lo), y, xpos(hi), y), fill="#" + color, width=7)
        draw.ellipse((xpos(mean) - 10, y - 10, xpos(mean) + 10, y + 10), fill="#" + color)
        draw.text((xpos(hi) + 12, y - 16), f"{mean:+.4f}", font=value_font, fill="#" + color)
    draw.text((left, height - 40), "*Unregistered follow-up; configuration-level effect, not attributed to one hyperparameter.",
              font=tick_font, fill="#" + MUTED)
    img.save(path, dpi=(180, 180))


def set_keep_with_next(paragraph):
    paragraph.paragraph_format.keep_with_next = True


def configure_document(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(11)
    normal.font.color.rgb = rgb(INK)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10

    for name, size, color, before, after in (
        ("Heading 1", 16, BLUE, 16, 8),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 12, DARK_BLUE, 8, 4),
    ):
        style = doc.styles[name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.font.size = Pt(size)
        style.font.color.rgb = rgb(color)
        style.font.bold = True
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.keep_together = True

    caption = doc.styles["Caption"]
    caption.font.name = "Calibri"
    caption._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    caption._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    caption.font.size = Pt(9)
    caption.font.color.rgb = rgb(MUTED)
    caption.font.italic = True
    caption.paragraph_format.space_before = Pt(4)
    caption.paragraph_format.space_after = Pt(4)
    caption.paragraph_format.keep_with_next = True

    styles = doc.styles
    source_style = styles.add_style("Source Note", 1)
    source_style.font.name = "Calibri"
    source_style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    source_style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    source_style.font.size = Pt(8)
    source_style.font.color.rgb = rgb(MUTED)
    source_style.font.italic = True
    source_style.paragraph_format.space_before = Pt(4)
    source_style.paragraph_format.space_after = Pt(4)
    source_style.paragraph_format.keep_with_next = True

    header = section.header
    p = header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run("LLMSTU  |  MASTER'S THESIS PROGRESS REPORT")
    set_run_font(r, size=8.5, color=MUTED, bold=True)
    set_paragraph_border(p, color=MID, size=6, space=4)

    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.paragraph_format.space_before = Pt(2)
    r = p.add_run("Progress report  |  10 August 2026  |  Page ")
    set_run_font(r, size=8.5, color=MUTED)
    add_page_field(p)
    r = p.add_run(" of ")
    set_run_font(r, size=8.5, color=MUTED)
    add_total_pages_field(p)

    doc.core_properties.title = "LLMSTU Master's Thesis Progress Report"
    doc.core_properties.subject = "Advisor-facing progress and evidence review"
    doc.core_properties.author = "Wael Kokach"
    doc.core_properties.keywords = "master's thesis, LLMSTU, progress report, visible cues, grounding, temporal segmentation"


def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    heading_specs = {
        1: (16, BLUE),
        2: (13, BLUE),
        3: (12, DARK_BLUE),
    }
    size, color = heading_specs.get(level, (11, INK))
    for run in p.runs:
        set_run_font(run, size=size, color=color, bold=True)
    set_keep_with_next(p)
    return p


def add_para(doc, text, bold_lead=None, italic=False, align=None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    if bold_lead and text.startswith(bold_lead):
        r = p.add_run(bold_lead)
        set_run_font(r, bold=True)
        r = p.add_run(text[len(bold_lead):])
        set_run_font(r, italic=italic)
    else:
        r = p.add_run(text)
        set_run_font(r, italic=italic)
    return p


def build_report():
    OUT_DIR.mkdir(exist_ok=True)
    ASSET_DIR.mkdir(exist_ok=True)

    chart_a = ASSET_DIR / "branch_a_r1.png"
    chart_b = ASSET_DIR / "branch_b_f1.png"
    chart_c = ASSET_DIR / "branch_c_effects.png"
    save_bar_chart(
        chart_a,
        "Grounding Recall@1: controlled improvements",
        ["Ordinal binding", "Hungarian binding", "Exact correspondence (test)"],
        [0.3230, 0.4954, 0.6462],
        [MUTED, BLUE, GREEN],
        0.70,
        tick=0.1,
    )
    save_bar_chart(
        chart_b,
        "Six-cue test macro-F1",
        ["Transformer 552", "Transformer 556", "ASRF 556", "ASRF 570", "MS-TCN 556", "MS-TCN 570"],
        [0.377, 0.407, 0.477, 0.492, 0.500, 0.488],
        [MUTED, MUTED, BLUE, BLUE, GREEN, BLUE],
        0.60,
        tick=0.1,
    )
    save_effect_chart(chart_c)

    doc = Document()
    configure_document(doc)
    # First-page masthead
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("MASTER'S THESIS PROGRESS REPORT")
    set_run_font(r, size=24, color=NAVY, bold=True)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(14)
    r = p.add_run("Deep Learning-Based Real-Time Student Behavior Analysis and Attention Loss Detection")
    set_run_font(r, size=14, color=BLUE, bold=True)
    r = p.add_run("\nA near-real-time, leak-free, uncertainty-aware visible-cue study in a computer laboratory")
    set_run_font(r, size=11.5, color=MUTED, italic=True)

    for label, value in (
        ("Prepared for", "Master's thesis advisor"),
        ("Prepared by", "Wael Kokach"),
        ("Reporting date", "10 August 2026"),
        ("Repository state", "main at 5826e1cb8a48dc5adedb5f7fb9b9e941a709631d"),
        ("Project status", "Core experiments complete; thesis drafting and closure work in progress"),
    ):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(label + ": ")
        set_run_font(r, size=10.5, color=NAVY, bold=True)
        r = p.add_run(value)
        set_run_font(r, size=10.5, color=INK)
    rule = doc.add_paragraph()
    rule.paragraph_format.space_before = Pt(6)
    rule.paragraph_format.space_after = Pt(12)
    set_paragraph_border(rule, color=BLUE, size=14, space=4)

    add_callout(
        doc,
        "RECOMMENDATION FOR ADVISOR:",
        "Approve progression to final thesis writing and defence preparation. The project now has a defensible empirical core, a functioning near-real-time prototype, pre-registered test evidence, and a strong audit trail. Approval should be conditional on closing ethics/consent documentation, obtaining a second annotator for the human diagnostic set, and completing the public artifact-release plan.",
        fill=PALE_GREEN,
        accent=GREEN,
    )

    add_heading(doc, "Executive summary", 1)
    add_para(
        doc,
        "The project is ready to be written and defended as a setting-specific system for detecting, attributing, temporally modelling, and presenting observable student behavioural cues. It is not defensible as a system that directly measures internal attention, comprehension, boredom, curiosity, or learning outcomes. This narrower framing is supported by the experiments rather than imposed only as an ethical disclaimer."
    )
    add_para(
        doc,
        "The strongest positive result is the controlled Branch-A attribution experiment: replacing detector-order assignment with content-based Hungarian matching raises grounding Recall@1 from 0.3230 to 0.4954 while Recall@10 changes by only 0.014. A model trained on exact correspondences reaches test R@1 of 0.6462 on a pre-registered 27-video split. Branch B then shows that temporal architecture matters more than adding feature families: MS-TCN reaches 0.500 +/- 0.011 test macro-F1, improving on the transformer by 0.0933 in all three seed pairs."
    )
    add_para(
        doc,
        "Branch C provides disciplined negative evidence. A validated full-range pose estimator, task-relative canonicalisation, uniform multimodal fusion, and learned reliability fusion all fail the registered practical-effect gate. The branch nevertheless makes two valuable contributions: it demonstrates that pose is redundant with the existing appearance representation, and it catches label leakage in an initially impressive reliability-fusion result before that result could enter the thesis. An unregistered follow-up also finds that a different training configuration improves macro-F1 by 0.0406, a practical effect twice the registered threshold, although the responsible hyperparameter remains unidentified."
    )

    add_heading(doc, "Advisor decision requested", 2)
    for text in (
        "Approve the thesis contribution as a controlled systems-and-methods study centred on visible cues, label attribution, temporal segmentation, uncertainty, and deployment trade-offs.",
        "Approve retention of Branch C as a negative-result chapter or substantial discussion section, not as a claim that OVERT-Cue improves performance.",
        "Approve a bounded final experiment set: isolate the training-configuration gain, complete the registered robustness arms if time permits, and avoid further head-pose model shopping on this corpus.",
        "Require documentary closure of ethics/consent and a second human annotator before submission.",
    ):
        add_list_item(doc, text)

    add_heading(doc, "1. Project scope and defensible claim", 1)
    add_para(
        doc,
        "Registered title:",
        bold_lead="Registered title:"
    )
    p = doc.paragraphs[-1]
    p.add_run(" Deep Learning-Based Real-Time Student Behavior Analysis and Attention Loss Detection")
    set_run_font(p.runs[-1], italic=True)
    add_para(
        doc,
        "Operational thesis claim: the system detects and temporally aggregates visible cues such as screen orientation, looking away, head-down posture, turning towards a peer, phone-use evidence, and uncertainty. It supports instructor review; it does not infer a student's internal mental state."
    )
    add_para(
        doc,
        "The corpus audit further narrows the setting. The recordings come from one computer-laboratory room and one fixed corner-mounted camera at approximately 20-30 degrees elevation, not from multiple overhead cameras. Viewpoint invariance can be proved and synthetically tested but cannot be claimed empirically from this dataset."
    )

    add_heading(doc, "Research questions now supported by evidence", 2)
    questions = [
        "RQ1 - Attribution: How much does description-to-student binding quality affect open-vocabulary grounding?",
        "RQ2 - Cue modelling: Which feature families and temporal architectures improve six-cue classification?",
        "RQ3 - Temporal behaviour: How well do frame predictions form sustained human-referenced events?",
        "RQ4 - Deployment: What accuracy, calibration, latency, and scaling trade-offs arise in an instructor dashboard?",
        "RQ5 - Construct validity: How context-dependent is the relationship between visible cues and engagement labels?",
        "RQ6 - Extension study: Do full-range pose, task-relative canonicalisation, and observability-aware fusion add measurable value beyond appearance?",
    ]
    for q in questions:
        add_list_item(doc, q, bold_prefix=q.split(":")[0] + ":")

    add_heading(doc, "2. Progress against major work packages", 1)
    add_caption(doc, "Table 1. Current completion and evidence status")
    rows = [
        ("Data reconstruction and audit", "Complete", "128 recovered recordings; 4,660/4,660 anchors verified; deduplication and video-wise splits frozen."),
        ("Branch A grounding", "Complete", "Controlled assignment ablation and one permitted 27-video test evaluation."),
        ("Branch B cue modelling", "Complete", "Leak-free 73/27/27 rebuild; transformer, MS-TCN and ASRF; three seeds; calibration and events."),
        ("Human-reference study", "Diagnostic complete", "754 accepted frame labels, 10 tracks, 16 distinct episodes; second annotator still missing."),
        ("Dashboard and runtime", "Operational", "Model selector, uploaded/recorded video, calibration, abstention and replay verified; near-real-time GPU mode."),
        ("Branch C extension", "Selection stage complete", "180 training runs across 12 arms; four registered nulls; outer folds preserved unopened."),
        ("Reproducibility", "Substantial, incomplete", "Artifact lock, bootstrap, pyproject, CI design and 234-test HPC suite; public checkpoint/data hosting unresolved."),
        ("Thesis writing", "In progress", "Full first draft and separate thesis-ready Branch-C draft available."),
        ("Ethics and governance", "Open blocker", "No ethics/consent record found; identifiable annotation history requires a repository-policy decision."),
    ]
    add_table(
        doc,
        ["Work package", "Status", "Evidence / remaining condition"],
        rows,
        [2500, 1600, 5260],
        aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
        highlights={0: (PALE_GREEN, GREEN), 1: (PALE_GREEN, GREEN), 2: (PALE_GREEN, GREEN),
                    4: (PALE_GREEN, GREEN), 8: (PALE_RED, RED)},
        font_size=8.6,
    )
    add_source(doc, "Sources: FINDINGS.md Sections 1-12; THESIS_FIRST_DRAFT.md; THESIS_BRANCH_C_DRAFT.md; docs/REPRODUCIBILITY.md.")

    add_heading(doc, "3. Dataset reconstruction and evaluation discipline", 1)
    add_para(
        doc,
        "The largest improvement in scientific validity came from rebuilding the data pipeline rather than changing the neural architecture. Earlier March experiments were invalidated because the frame-wise split was temporally leaked, some checkpoints were missing or duplicated, and several evaluators produced plausible but incorrect numbers. The rebuilt pipeline recovers recording identity, removes near-duplicate crops, preserves every retained frame's student annotations, and groups all splits by video."
    )

    add_caption(doc, "Table 2. Audited corpus and split")
    data_rows = [
        ("Raw per-student records", "283,913", "One crop, caption and structured label record per detected student."),
        ("Recovered recordings", "128", "127 used in the main split; the remaining recording contains four frames."),
        ("Verified identity anchors", "4,660 / 4,660", "100% of available anchors matched recovered appearance chains."),
        ("Deduplicated selections", "84,950", "Approximately 70% of near-duplicate crops removed."),
        ("Main video split", "73 / 27 / 27", "Train / validation / test; video-wise and shared across Branch A/B rebuild."),
        ("Detector frames", "36,339 / 9,352 / 9,405", "Train / validation / test."),
        ("Temporal sequences", "4,390 / 1,085 / 1,056", "6,531 total sequences."),
        ("Temporal frames", "185,050 / 42,702 / 43,733", "Train / validation / test."),
        ("Camera setting", "1 room, 1 fixed corner camera", "Audit sample indicates approximately 20-30 degree elevation."),
        ("Students per video", "Median 7 (range 2-13)", "Branch-C repository/data audit."),
    ]
    add_table(doc, ["Audit item", "Result", "Interpretation"], data_rows, [2500, 2300, 4560],
              aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
              font_size=8.8)
    add_source(doc, "Sources: THESIS_FIRST_DRAFT.md Tables 3.1 and 5.1; FINDINGS.md Sections 2 and 11.1; docs/branch_c/REPOSITORY_AND_DATA_AUDIT.md.")

    add_callout(
        doc,
        "SCIENTIFIC VALUE:",
        "The project does not rely on a single best score. Its evidence base includes controlled ablations, pre-registered test access, grouped splits, multiple seeds, paired video-level comparisons, human diagnostic labels, calibration, runtime profiling, and explicit invalidation of compromised results.",
        fill=PALE_BLUE,
        accent=BLUE,
    )

    add_heading(doc, "4. Branch A - description-to-student grounding", 1)
    add_para(
        doc,
        "Branch A addresses a first-order supervision problem: a multi-student caption can describe several people, but the description units must be bound to the correct boxes before detector training. The original detector-confidence ordering has no semantic relationship to the caption order. Content-based Hungarian matching provides a clean, controlled improvement while holding images, boxes, captions, schedule, learning rate, seed, and evaluator fixed."
    )

    add_heading(doc, "4.1 Content-only assignment benchmark", 2)
    add_caption(doc, "Table 3. Description-unit to box assignment")
    add_table(
        doc,
        ["Binding method", "Assignment accuracy", "Wrong-assignment rate"],
        [
            ("Detector-confidence ordinal", "0.2607", "0.7393"),
            ("Hungarian, CLIP content only", "0.7896", "0.2104"),
            ("Sinkhorn, content only", "0.7896", "0.2104"),
        ],
        [4600, 2380, 2380],
        aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER],
        highlights={1: (PALE_GREEN, GREEN)},
        font_size=9.2,
    )
    add_source(doc, "72,398 frames and 261,397 assignments; spatial_weight=0.0. The earlier 0.9231 result is contaminated by a ground-truth ordering shortcut and is excluded.")

    add_para(
        doc,
        "A logistic assignment calibrator reaches AUROC 0.760 and ECE 0.011. At probability >= 0.9 it retains 70.1% of assignments at 96.2% held-out precision. This confirms that confidence ranking is meaningful, although the later training ablation shows that aggressive filtering sacrifices too much data."
    )

    add_heading(doc, "4.2 Controlled detector ablation and held-out result", 2)
    add_caption(doc, "Table 4. Grounding experiment")
    branch_a_rows = [
        ("Ordinal binding", "165,013", "0.3230", "0.8978", "0.9811", "Controlled baseline"),
        ("Hungarian binding", "165,017", "0.4954", "0.9595", "0.9952", "+0.1724 R@1"),
        ("Hungarian + p>=0.9", "98,526", "0.4787", "0.9441", "0.9913", "Negative: below unfiltered"),
        ("Exact correspondence, validation", "141,522", "0.6343", "0.9808", "0.9963", "Main model"),
        ("Exact correspondence, test", "141,522", "0.6462", "0.9891", "0.9982", "Pre-registered; protocol closed"),
    ]
    add_table(doc, ["Training labels", "Regions", "R@1", "R@5", "R@10", "Interpretation"],
              branch_a_rows, [2600, 1250, 1000, 1000, 1000, 2510],
              aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER,
                      WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
              highlights={1: (PALE_BLUE, BLUE), 4: (PALE_GREEN, GREEN)}, font_size=8.3)
    add_source(doc, "Sources: TEST_SPLIT_PROTOCOL.md; outputs/FINAL_RESULTS_REGISTER.*; FINDINGS.md Sections 3.2-3.5.")

    add_picture_with_alt(
        doc, chart_a, 6.35,
        "Horizontal bar chart showing grounding Recall at 1 increasing from 0.323 for ordinal binding to 0.495 for Hungarian binding and 0.646 for exact correspondence on test."
    )
    add_caption(doc, "Figure 1. Cleaner attribution improves top-1 grounding; exact correspondences provide the strongest supervision.")

    add_callout(
        doc,
        "CORE THESIS RESULT:",
        "Hungarian binding improves R@1 by 17.24 percentage points (+53% relative), while R@10 changes by only 1.41 points. Both models locate students; the gain is primarily correct attribution of the description to the student.",
        fill=PALE_GREEN,
        accent=GREEN,
    )

    add_heading(doc, "5. Branch B - visible-cue temporal modelling", 1)
    add_para(
        doc,
        "Branch B maps each tracked student sequence to one of six observable cues. The feature base combines CLIP appearance, box geometry, colour statistics and posture proxies. Controlled column-slice ablations then add head pose, facial-expression probabilities, and handcrafted motion/gaze dynamics. All final comparisons use the shared leak-free 73/27/27 video partition and three seeds."
    )

    add_heading(doc, "5.1 Feature ablation", 2)
    add_caption(doc, "Table 5. Isolated feature-family effects")
    add_table(
        doc,
        ["Added block", "Delta macro-F1", "Significant seed pairs", "Conclusion"],
        [
            ("Head-pose block (556-552)", "+0.0294 validation; +0.0296 test", "3/3", "Supported aggregate gain"),
            ("Expression only", "+0.0009", "0/3", "No supported gain"),
            ("Motion/gaze dynamics only", "-0.0105", "0/3", "No supported gain"),
            ("Expression + dynamics", "-0.0031", "0/3", "No supported gain"),
            ("Pose angles over face_found control", "+0.0058 test", "0/3", "Angles not supported"),
        ],
        [2600, 2300, 1900, 2560],
        aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
        highlights={0: (PALE_BLUE, BLUE), 4: (PALE_GOLD, GOLD)}, font_size=8.8,
    )
    add_source(doc, "Sources: THESIS_FIRST_DRAFT.md Section 5.3; FINDINGS.md Sections 11.4 and 11.10.")
    add_para(
        doc,
        "Decomposition changes the interpretation of the head-pose result. Approximately four fifths of the useful block-level gain is associated with face_found, while yaw, pitch and roll add only a small non-significant increment. Branch C later confirms that the legacy angles are not reliable orientation measurements and that a valid full-range estimator is redundant with appearance."
    )

    add_heading(doc, "5.2 Temporal architecture comparison", 2)
    add_caption(doc, "Table 6. Untouched Branch-B test split, mean over three seeds")
    branch_b_rows = [
        ("Transformer", "552", "0.657 +/- 0.003", "0.396", "0.377 +/- 0.007", "0.357", "0.065"),
        ("Transformer", "556", "0.699 +/- 0.011", "0.424", "0.407 +/- 0.004", "0.392", "0.068"),
        ("Transformer*", "570", "0.660 +/- 0.011", "0.424", "0.390 +/- 0.005", "0.388", "0.075"),
        ("ASRF", "556", "0.710 +/- 0.022", "0.513", "0.477 +/- 0.016", "0.473", "0.033"),
        ("ASRF", "570", "0.735 +/- 0.023", "0.521", "0.492 +/- 0.019", "0.482", "0.031"),
        ("MS-TCN", "556", "0.759 +/- 0.002", "0.519", "0.500 +/- 0.011", "0.489", "0.032"),
        ("MS-TCN*", "570", "0.765 +/- 0.018", "0.481", "0.488 +/- 0.023", "0.477", "0.025"),
    ]
    add_table(doc, ["Model", "Features", "Accuracy", "Balanced acc.", "Macro-F1", "Macro-AUPRC", "ECE"],
              branch_b_rows, [1500, 900, 1550, 1250, 1550, 1400, 1210],
              aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER,
                      WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER,
                      WD_ALIGN_PARAGRAPH.CENTER],
              highlights={5: (PALE_GREEN, GREEN)}, font_size=7.8)
    add_source(doc, "*The additional 570-dimensional feature block did not provide a repeatable aggregate gain. Source: BRANCH_B_TEST_PROTOCOL.md and outputs/FINAL_RESULTS_REGISTER.*.")

    add_picture_with_alt(
        doc, chart_b, 6.35,
        "Horizontal bar chart comparing six-cue test macro F1. MS-TCN with 556 features is highest at 0.500, followed by ASRF 570 at 0.492 and MS-TCN 570 at 0.488."
    )
    add_caption(doc, "Figure 2. Temporal architecture contributes substantially more than adding expression and handcrafted dynamics.")
    add_para(
        doc,
        "With identical 556-dimensional inputs, MS-TCN improves test macro-F1 over the transformer by 0.0933, with all three seed-pair differences significant. The pre-registered ASRF-556 primary remains reported at 0.477 +/- 0.016; the fact that MS-TCN scores higher is retained rather than used to rewrite the selection rule."
    )

    add_heading(doc, "5.3 Human-referenced temporal and event evaluation", 2)
    add_para(
        doc,
        "The dense human diagnostic contains 754 accepted frame annotations across ten tracks and 16 distinct episodes after removing duplicated event aliases. It is deliberately small and excludes detector/tracker errors, so its event scores are diagnostic rather than classroom-wide estimates."
    )
    add_caption(doc, "Table 7. Human-referenced segmentation and events")
    event_rows = [
        ("Transformer 552", "0.738", "0.244", "30.9", "0.560", "0.292", "0.383", "9.4"),
        ("Transformer 556", "0.782", "0.265", "38.2", "0.622", "0.208", "0.312", "5.1"),
        ("MS-TCN 556", "0.788", "0.291", "48.5", "0.554", "0.271", "0.353", "11.1"),
        ("MS-TCN 570", "0.771", "0.362", "59.9", "0.454", "0.312", "0.368", "15.3"),
        ("ASRF 570", "0.785", "0.291", "53.4", "0.506", "0.229", "0.315", "9.4"),
        ("Pseudo-label teacher", "0.874", "0.700", "76.5", "0.588", "0.625", "0.606", "17.9"),
        ("Majority control", "0.341", "0.067", "34.9", "0.000", "0.000", "0.000", "0.0"),
    ]
    add_table(doc, ["System", "Frame acc.", "F1@25", "Edit", "Event P", "Event R", "Event F1", "FA/hour"],
              event_rows, [1900, 1000, 950, 850, 1050, 1050, 1050, 1510],
              aligns=[WD_ALIGN_PARAGRAPH.LEFT] + [WD_ALIGN_PARAGRAPH.CENTER] * 7,
              highlights={3: (PALE_BLUE, BLUE), 5: (PALE_GREEN, GREEN)}, font_size=7.7)
    add_source(doc, "tIoU=0.30 for events. Detector and tracker excluded. Source: THESIS_FIRST_DRAFT.md Section 5.5 and outputs/FINAL_RESULTS_REGISTER.*.")
    add_para(
        doc,
        "MS-TCN clearly reduces fragmentation: its best edit score is 59.9 versus 28.8-38.2 for transformer configurations. Event recall remains the unresolved weakness, reaching at most 0.312 compared with 0.625 for the pseudo-label teacher. This makes temporal segmentation a demonstrated improvement while preventing an exaggerated event-detection claim."
    )

    add_heading(doc, "5.4 Calibration, abstention and alert operating points", 2)
    add_caption(doc, "Table 8. Temperature calibration")
    add_table(
        doc,
        ["Model", "Temperature", "Test ECE before -> after", "Test NLL before -> after"],
        [
            ("Transformer 556", "1.311", "0.0739 -> 0.0191", "0.9660 -> 0.9121"),
            ("ASRF 556", "0.759", "0.0570 -> 0.0306", "0.7854 -> 0.7791"),
            ("MS-TCN 556", "0.924", "0.0287 -> 0.0155", "0.7033 -> 0.7043"),
        ],
        [2200, 1400, 2840, 2920],
        aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER],
        highlights={2: (PALE_GREEN, GREEN)}, font_size=9,
    )
    add_source(doc, "Thresholds are fit on validation and frozen. A slight NLL increase for MS-TCN is reported rather than hidden.")
    add_para(
        doc,
        "For the deployed path, the display threshold retains approximately 90% coverage; the alert threshold targets at least 85% selective accuracy. Raw predictions are always recorded, while low-confidence predictions are withheld from the interface or alerts. Across 13 dashboard variants, alert coverage at the same 85% precision target spans 30.5% to 78.7%, showing that operational usefulness is not ranked by macro-F1 alone."
    )

    doc.add_page_break()

    add_heading(doc, "6. Construct validity and external evidence", 1)
    add_para(
        doc,
        "External experiments are used to bound the thesis claim, not to manufacture a universal engagement result. Their central lesson is that identical visible movements can mean different things in different learning settings."
    )
    add_caption(doc, "Table 9. External and construct-validity experiments")
    add_table(
        doc,
        ["Experiment", "Result", "Interpretation"],
        [
            ("DIPSER cue-engagement association", "Spearman rho = +0.172; p<0.0001; n=1,825", "Weak and opposite to the naive disengagement hypothesis; cue meaning is context-dependent."),
            ("Basic expressions -> expert boredom", "AUROC 0.544; n=1,176", "Near chance; no boredom-detection claim."),
            ("SCB zero-shot transfer", "Localisation recall 0.392; R@1 confounded", "Task/label mismatch prevents a valid transfer score."),
            ("Teacher vs human frame labels", "0.874 agreement on 754 accepted frames", "Empirical teacher benchmark, not a theoretical ceiling."),
            ("Pseudo-label field accuracy", "0.939 over 311 held-out crops", "Activity is weakest at 78.9%."),
        ],
        [2600, 2550, 4210],
        aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
        font_size=8.6,
    )
    add_source(doc, "Sources: outputs/FINAL_RESULTS_REGISTER.*; FINDINGS.md Sections 5-6; DIPSER and SCB analyses.")

    add_heading(doc, "6.1 CMOSE split-policy experiment", 2)
    add_para(
        doc,
        "CMOSE is treated as a separate four-level ordinal engagement task. The released random-segment split places 101 of 103 subjects in more than one split. Rebuilding a subject-disjoint split produces a large performance drop, quantifying how split policy changes the apparent difficulty."
    )
    add_caption(doc, "Table 10. CMOSE official versus subject-disjoint evaluation")
    cmose_rows = [
        ("Accuracy", "0.7179 +/- 0.0030", "0.6006 +/- 0.0058", "-0.1173"),
        ("Average accuracy", "0.6007 +/- 0.0086", "0.4347 +/- 0.0084", "-0.1660"),
        ("Macro-F1", "0.5733 +/- 0.0027", "0.4111 +/- 0.0051", "-0.1622"),
        ("MAE", "0.3123 +/- 0.0029", "0.4459 +/- 0.0123", "+0.1336"),
        ("QWK", "0.5369 +/- 0.0039", "0.3167 +/- 0.0319", "-0.2202"),
        ("Spearman rho", "0.5354 +/- 0.0086", "0.3221 +/- 0.0305", "-0.2133"),
    ]
    add_table(doc, ["Metric", "Official split", "Subject-disjoint", "Change"], cmose_rows,
              [2200, 2500, 2500, 2160],
              aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER],
              font_size=8.8)
    add_source(doc, "Separate task; these values must not be compared directly with six-cue macro-F1 or grounding R@k.")

    add_heading(doc, "7. Deployment, dashboard and runtime", 1)
    add_para(
        doc,
        "The prototype is operational and demonstrable, but it is correctly described as near-real-time. The deployment work replaced the expensive face landmark mesh with a detector-based face-presence signal, then introduced detector and temporal striding with measured cue-agreement controls."
    )
    add_caption(doc, "Table 11. Runtime trade-off on one A100, approximately six students")
    runtime_rows = [
        ("1:1", "3.69", "270.7 ms", "295.9 ms", "100% reference", "100%"),
        ("3:2 adopted", "5.77", "157.6 ms", "291.7 ms", "94.7%", "100%"),
        ("5:3", "7.03", "117.9 ms", "257.5 ms", "92.2%", "100%"),
        ("10:5", "-", "-", "-", "88.3%", "100%"),
    ]
    add_table(doc, ["Detector:temporal stride", "FPS", "p50", "p95", "Cue agreement", "Track coverage"],
              runtime_rows, [2450, 900, 1300, 1300, 1700, 1710],
              aligns=[WD_ALIGN_PARAGRAPH.LEFT] + [WD_ALIGN_PARAGRAPH.CENTER] * 5,
              highlights={1: (PALE_GREEN, GREEN)}, font_size=8.8)
    add_source(doc, "Source: FINDINGS.md Sections 11.16-11.17; LLMDet/configs/attention_runtime.yaml.")
    add_para(
        doc,
        "The detector backend improves like-for-like throughput from 2.84 to 3.69 FPS (+30%) with no repeatable accuracy loss. The adopted 3:2 stride raises throughput by a further 56% and preserves 94.7% cue agreement. Because the p95 tail remains approximately 292 ms and the system does not reach a 10 FPS budget, the report avoids the stronger real-time claim."
    )

    add_heading(doc, "7.1 Dashboard evidence", 2)
    dashboard_items = [
        "Model selector across 13 deployable variants, each linked to measured validation metrics and its own calibration.",
        "Uploaded/recorded video as a first-class source, plus precomputed session replay and camera/live-path support.",
        "Eight of eight HTTP checks and all model-card consistency checks passed on the frozen dashboard verification.",
        "Four replayed models disagree on 8.5-15.0% of shared frame-track pairs, confirming that switching changes the actual model rather than only the label shown in the interface.",
        "On CPU, the temporal head runs at approximately 25-26 FPS for MS-TCN but 5.7-5.8 FPS for ASRF; ASRF is roughly 4.5 times slower.",
        "Branch-C dashboard integration remains a reviewed design, not a deployed mode. It was not activated after the scientific go gate failed, protecting the validated legacy path from unnecessary complexity.",
    ]
    for item in dashboard_items:
        add_list_item(doc, item)

    add_callout(
        doc,
        "DEPLOYMENT POSITION:",
        "The project demonstrates a calibrated, auditable instructor-facing prototype with measured latency and fallback behaviour. It should be presented as decision support and research infrastructure, not as an autonomous classroom surveillance or intervention system.",
        fill=PALE_GOLD,
        accent=GOLD,
    )

    doc.add_page_break()

    add_heading(doc, "8. Branch C - OVERT-Cue extension study", 1)
    add_para(
        doc,
        "Branch C was designed to test an overhead-specific contribution based on task-relative orientation and reliability-aware multimodal fusion. The initial premise was corrected before experimentation: the dataset contains one fixed corner camera, not multiple overhead views. The branch therefore retained only claims that could be supported in this setting and used nested grouped cross-validation because the original Branch-B test had already been spent."
    )
    add_para(
        doc,
        "The five outer folds remain unopened. All Branch-C performance values below are inner-validation selection statistics over 15 matched fold-seed pairs, not generalisation estimates and not directly comparable to the Branch-B test table."
    )

    add_heading(doc, "8.1 Methodological contribution retained", 2)
    for item in (
        "A task-relative SO(3) representation with an exact coordinate-invariance proof, verified over 256 random rotations to numerical tolerance.",
        "A validated 6DRepNet360 full-range pose instrument selected after license review; Ultralytics and DirectMHP candidates were rejected because of copyleft implications.",
        "Pre-registered gates for canonicalisation, pose, uniform fusion, learned fusion and observability losses.",
        "Leakage-aware quality controls that distinguish deployable pipeline signals from teacher-side annotation fields.",
        "Independent recomputation, error analysis and proposer-reviewer sign-off.",
    ):
        add_list_item(doc, item)

    add_heading(doc, "8.2 Experimental sweep", 2)
    add_caption(doc, "Table 12. Branch-C inner-validation arms under the common project selection rule")
    branch_c_arm_rows = [
        ("0b", "Five pipeline-measured quality signals only", "0.3208", "0.028", "Valid shortcut baseline"),
        ("0", "+ three teacher-side annotation fields", "0.4602", "0.023", "Upper bound; not deployable"),
        ("1", "Appearance 552 + face_found", "0.4837", "0.027", "Deployed-style baseline"),
        ("2", "+ legacy MediaPipe angles", "0.4887", "0.033", "Pose baseline"),
        ("5", "+ validated 6DRepNet360 rotation", "0.4865", "0.027", "No aggregate gain"),
        ("3b", "Appearance 552, Branch-C training configuration", "0.5243", "0.032", "Same architecture; configuration change"),
        ("3", "3b + deeper stack", "0.5288", "0.029", "Small architecture increment"),
        ("8", "+ head/motion experts, uniform fusion", "0.5311", "0.025", "Fusion null"),
        ("9c", "+ leakage-free learned reliability", "0.5326", "0.024", "Learned weighting null"),
        ("10c", "+ observability losses", "0.5360", "0.023", "Small effect below gate"),
    ]
    add_table(doc, ["Arm", "Features / model", "Macro-F1", "SD", "Interpretation"], branch_c_arm_rows,
              [700, 3800, 1300, 900, 2660],
              aligns=[WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER,
                      WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
              highlights={0: (PALE_GOLD, GOLD), 5: (PALE_GREEN, GREEN)}, font_size=8.1)
    add_source(doc, "Source: THESIS_BRANCH_C_DRAFT.md; 12 arms x 5 folds x 3 seeds = 180 training runs. Leaky arms 9/10 are excluded from this table.")

    add_heading(doc, "8.3 Registered contrasts and verdict", 2)
    add_caption(doc, "Table 13. Paired Branch-C effects")
    contrast_rows = [
        ("Full-range pose: arm5-arm2", "-0.0022", "[-0.0119, +0.0085]", "5/15", "Null"),
        ("Full-range pose vs no angles: arm5-arm1", "+0.0028", "[-0.0072, +0.0139]", "8/15", "Null"),
        ("Uniform fusion: arm8-arm3", "+0.0019", "[-0.0025, +0.0057]", "11/15", "Null"),
        ("Learned vs uniform: arm9c-arm8", "+0.0015", "[-0.0007, +0.0041]", "10/15", "Null"),
        ("Observability losses: arm10c-arm9c", "+0.0041", "[+0.0010, +0.0070]", "11/15", "Significant; one fifth of gate"),
        ("Whole fusion stack: arm10c-arm3", "+0.0072", "[+0.0026, +0.0116]", "12/15", "Below +0.02 gate"),
        ("Training configuration*: arm3b-arm1", "+0.0406", "[+0.0289, +0.0521]", "14/15", "Positive; twice the gate"),
    ]
    add_table(doc, ["Contrast", "Delta", "95% CI", "Wins", "Verdict"], contrast_rows,
              [3300, 1100, 2000, 900, 2060],
              aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER,
                      WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
              highlights={6: (PALE_GREEN, GREEN), 4: (PALE_GOLD, GOLD)}, font_size=8.2)
    add_source(doc, "*Unregistered configuration-level finding; six hyperparameters differ and no individual cause is yet established.")

    add_picture_with_alt(
        doc, chart_c, 6.35,
        "Effect-size chart for Branch C. Pose, uniform fusion, and learned reliability intervals cross zero. Observability losses and the whole fusion stack remain below the plus 0.02 go gate. The unregistered training configuration effect is plus 0.0406 and exceeds the gate."
    )
    add_caption(doc, "Figure 3. Every registered architectural hypothesis fails the practical-effect gate; the training configuration is the only substantial positive.")

    add_heading(doc, "8.4 Why the negative pose result is informative", 2)
    add_para(
        doc,
        "The pose instrument was validated before use. It increases coverage from approximately 63% to 100%, recovers applied in-plane rotation to about one degree, and produces a standalone scalar that separates head_down from screen_oriented at AUC 0.830. Nevertheless, replacing the legacy pose block changes predictions slightly less than changing the random seed: arm2 versus arm5 agrees on 81.9% of frames, while two seeds of the same arm agree on 82.7%."
    )
    add_caption(doc, "Table 14. Appearance already encodes orientation-relevant evidence")
    add_table(
        doc,
        ["Cue", "Appearance AUROC, no pose", "Standalone pose scalar"],
        [
            ("head_down", "0.915", "0.830"),
            ("uncertain", "0.930", "0.852"),
            ("turned_to_peer", "0.825", "0.625"),
            ("phone_use", "0.889", "0.519"),
        ],
        [3000, 3180, 3180],
        aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER],
        highlights={0: (PALE_BLUE, BLUE), 1: (PALE_BLUE, BLUE)}, font_size=9,
    )
    add_source(doc, "The standalone-pose scalar is a development pre-test and should be contextual rather than presented as a controlled held-out contrast. The arm-level pose null is the citable controlled result.")
    add_para(
        doc,
        "The defensible conclusion is not merely that one estimator failed. A full-range estimator was validated, given complete coverage, and shown discriminative in isolation, but the appearance representation already contained stronger orientation-related information. Pose is redundant with appearance in this task."
    )

    add_heading(doc, "8.5 Leakage caught before thesis reporting", 2)
    add_para(
        doc,
        "The first learned-fusion result appeared to improve macro-F1 by approximately 0.09. A required shortcut audit showed that three quality fields - occluded, head_kpts and face_kpts - came from the same annotation record as the cue label. Zeroing them at inference collapsed learned fusion from 0.6566 to 0.2567 while leaving uniform fusion unchanged. After leakage-free retraining, the learned-fusion effect reduced to approximately +0.0015 and became null."
    )
    add_caption(doc, "Table 15. Shortcut and leakage correction")
    add_table(
        doc,
        ["Comparison", "Before correction", "After correction", "Conclusion"],
        [
            ("Quality-only shortcut", "0.4602 with eight signals", "0.3208 with five deployable signals", "Teacher-side fields contributed +0.1395"),
            ("Learned vs uniform fusion", "+0.0902", "+0.0015", "The apparent gain was label leakage"),
            ("Appearance over genuine shortcut", "Initially read as +0.023", "+0.1629", "Appearance contribution is substantial"),
        ],
        [2550, 2100, 2100, 2610],
        aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
        highlights={1: (PALE_RED, RED), 2: (PALE_GREEN, GREEN)}, font_size=8.5,
    )
    add_source(doc, "Sources: FINDINGS.md Sections 12.19-12.22; THESIS_BRANCH_C_DRAFT.md Sections 4.4-4.5.")

    add_callout(
        doc,
        "BRANCH-C VERDICT:",
        "Four registered hypotheses are null. This is suitable for the thesis because the study validates its instrument, follows pre-registered gates, exposes a data-contract leak, and prevents an unjustified architecture claim. The practical follow-up is the +0.0406 training-configuration effect, not additional pose or fusion complexity.",
        fill=PALE_GREEN,
        accent=GREEN,
    )

    doc.add_page_break()

    add_heading(doc, "9. Reproducibility, software quality and independent review", 1)
    add_para(
        doc,
        "The repository now contains a machine-readable artifact lock, a cross-platform bootstrap entry point, a pyproject dependency declaration, profile-specific download logic, CI checks, frozen protocols, run records, hashes, and extensive documentation. The final Branch-C commit records 234 passing tests on the reference HPC environment. Dashboard verification adds frozen replay fixtures and model-card checks."
    )
    add_caption(doc, "Table 16. Reproducibility status")
    repro_rows = [
        ("Code and protocols", "Strong", "Branch A/B/C protocols, split hashes, test discipline, evaluator and source are tracked."),
        ("Model provenance", "Strong on HPC", "Checkpoint/config hashes and artifact manifest recorded; Branch-C run records preserved."),
        ("Bootstrap and dependency declaration", "Implemented", "bootstrap.py, artifacts.lock.json and pyproject.toml provide demo/research/full-data/HPC profiles."),
        ("Test evidence", "Reference environment pass", "Latest commit reports 234 tests passing; dashboard checks 8/8 with model-card consistency."),
        ("Public demo checkpoint", "Open", "Fine-tuned detector/temporal weights have no public immutable download URL yet."),
        ("Restricted human/student data", "Not publicly distributable", "Access depends on ethics, consent and repository policy."),
        ("Container / CUDA lock", "Open", "No complete Docker/Apptainer image or bit-reproducible CUDA environment."),
        ("Public repository license", "Open", "A top-level license is still required before public release."),
        ("Cross-platform suite", "Open", "Reference code assumes Linux/HPC components; Windows collection requires fcntl/mmcv compatibility work."),
    ]
    add_table(doc, ["Area", "Status", "Evidence / gap"], repro_rows, [2600, 1750, 5010],
              aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
              highlights={0: (PALE_GREEN, GREEN), 1: (PALE_GREEN, GREEN), 2: (PALE_GREEN, GREEN),
                          4: (PALE_GOLD, GOLD), 5: (PALE_RED, RED), 7: (PALE_GOLD, GOLD)},
              font_size=8.4)
    add_source(doc, "Sources: docs/REPRODUCIBILITY.md; artifacts.lock.json; pyproject.toml; docs/branch_c/INDEPENDENT_REVIEW_LOG.md.")

    add_heading(doc, "Independent verification and defect prevention", 2)
    quality_items = [
        "Multiple specialist agents were separated by file ownership and review context; the lead inspected evidence before accepting conclusions.",
        "Unit tests caught an incorrect 6D-rotation column layout and a missing-modality NaN propagation bug before training.",
        "A smoke test caught temporal-model collapse caused by missing deep supervision before a 60-run fusion sweep.",
        "Independent recomputation reproduced the pose null and identified a duplicate evaluation directory, missing convenience-ledger rows and batch-dependent evaluation differences.",
        "Batch-size-one evaluation was selected to eliminate cross-sequence contamination at temporal convolution boundaries.",
        "License audit rejected components whose copyleft terms conflicted with the intended distribution model.",
    ]
    for item in quality_items:
        add_list_item(doc, item)

    add_heading(doc, "10. Thesis contribution and defensibility", 1)
    add_para(
        doc,
        "The thesis is conditionally defensible now. Its strength is not a claim of state-of-the-art classroom engagement recognition. Its strength is a coherent research progression from compromised early experiments to a controlled, reproducible, human-audited and deployment-measured visible-cue system."
    )
    add_caption(doc, "Table 17. Defensible contribution package")
    contributions = [
        ("1. Attribution-aware grounding", "Controlled evidence that description-to-box binding quality is a first-order training variable; +0.1724 R@1 with detection largely unchanged."),
        ("2. Leak-free temporal cue benchmark", "Shared video-wise split, three seeds, one evaluator and paired comparisons across transformer, MS-TCN and ASRF."),
        ("3. Feature evidence", "Head-pose block decomposition; null results for expression and handcrafted dynamics; pose redundancy established with a validated estimator."),
        ("4. Event and uncertainty layer", "Human diagnostic events, segmentation/edit metrics, calibration, abstention and false-alert analysis."),
        ("5. Near-real-time prototype", "Dashboard, model selector, video upload/replay, runtime profiling, stage timing and measured stride trade-offs."),
        ("6. Methodological audit contribution", "Documented invalidation of leakage, shortcut metrics, random-model deployment and label-bearing reliability inputs."),
        ("7. Construct-validity boundary", "External evidence shows visible cues are context-dependent and cannot be equated with internal engagement."),
    ]
    add_table(doc, ["Contribution", "Evidence"], contributions, [3000, 6360],
              aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT],
              highlights={0: (PALE_GREEN, GREEN), 1: (PALE_GREEN, GREEN), 4: (PALE_GREEN, GREEN)},
              font_size=8.7)
    add_source(doc, "Advisor-facing synthesis of the evidence registers and thesis drafts.")

    add_heading(doc, "11. Limitations and conditions before submission", 1)
    limitations = [
        ("Ethics/consent documentation", "Potential administrative blocker", "No approval/consent record was found. Obtain and cite the institutional record or formally narrow what can be submitted/published."),
        ("Second human annotator", "High defence exposure", "The dense event set is single-annotator and contains only 16 distinct episodes. Add independent annotation and agreement statistics."),
        ("Construct scope", "Must remain narrow", "Use observable cues and alert evidence; do not claim mental-state, boredom, comprehension or learning-outcome inference."),
        ("Pseudo-label dependence", "Central validity limitation", "Most large-scale cue metrics are teacher agreement. Preserve the distinction from human accuracy."),
        ("Detector/tracker human ground truth", "Missing", "Human event evaluation excludes detector/tracker errors. End-to-end population performance is not established."),
        ("External validity", "Limited", "One room, one camera and unrecoverable subject identity; no cross-camera or subject-disjoint internal claim."),
        ("Class imbalance", "Persistent", "screen_oriented dominates; turned_to_peer has only 63 sequences corpus-wide in Branch C."),
        ("Branch-C outer folds", "Preserved", "Unopened because the inner go gate failed. Do not promote inner-validation values as generalisation estimates."),
        ("Reproducibility release", "Incomplete", "Publish or legally document custom checkpoint distribution, add a top-level license and close hard-coded path/container gaps."),
    ]
    add_table(doc, ["Issue", "Severity", "Required treatment"], limitations, [2700, 1900, 4760],
              aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
              highlights={0: (PALE_RED, RED), 1: (PALE_RED, RED), 2: (PALE_GOLD, GOLD),
                          3: (PALE_GOLD, GOLD)}, font_size=8.4)
    add_source(doc, "Sources: THESIS_DEFENSIBILITY_REVIEW.md; THESIS_BRANCH_C_DRAFT.md; docs/REPRODUCIBILITY.md.")

    add_heading(doc, "12. Recommended completion plan", 1)
    steps = [
        ("Isolate the training-configuration effect.", "Run one-factor changes for epochs, learning rate, schedule, dropout, gradient clipping and AMP. Freeze the comparison plan first and do not attribute the +0.0406 until isolated."),
        ("Close the human-evidence gap.", "Obtain a second annotator for the existing gold sample or a small disjoint sample; report agreement and adjudication."),
        ("Resolve ethics and repository privacy.", "Locate approval/consent records; determine whether tracked identifiable annotations require removal and history rewriting."),
        ("Finish the reproducible release path.", "Assign immutable public/manual artifact URLs, add a top-level license, verify the demo bootstrap from an empty clone and document restricted-data boundaries."),
        ("Update the integrated thesis draft.", "Merge the Branch-C methods, null results, corrected leakage analysis and training-recipe finding into THESIS_FIRST_DRAFT.md without changing the visible-cue framing."),
        ("Prepare defence materials.", "Use the controlled Branch-A ablation, Branch-B architecture result, Branch-C falsification and dashboard demo as the narrative spine."),
    ]
    step_rows = [
        (str(index), f"{title} {detail}")
        for index, (title, detail) in enumerate(steps, start=1)
    ]
    add_table(
        doc,
        ["Step", "Required action"],
        step_rows,
        [780, 8580],
        aligns=[WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
        font_size=9.5,
    )

    add_callout(
        doc,
        "PROPOSED APPROVAL WORDING:",
        "The advisor approves progression to final thesis drafting on the basis that the core technical work and controlled evaluation are sufficient for a master's-level contribution. Final submission remains conditional on ethics/consent documentation, strengthened human annotation evidence, and completion of the reproducibility and artifact-release statement.",
        fill=PALE_GREEN,
        accent=GREEN,
    )

    doc.add_page_break()

    add_heading(doc, "Appendix A. Consolidated headline results", 1)
    consolidated = [
        ("Data recovery", "4,660/4,660 anchors; 128 recordings", "Verified reconstruction and video grouping"),
        ("Deduplication", "283,913 -> 84,950 crops", "Approximately 70% removed"),
        ("Assignment", "0.2607 -> 0.7896 accuracy", "Ordinal -> content-only Hungarian"),
        ("Grounding ablation", "R@1 0.3230 -> 0.4954", "+0.1724 controlled effect"),
        ("Grounding held-out", "R@1/5/10 = 0.6462/0.9891/0.9982", "Pre-registered 27-video test"),
        ("Temporal architecture", "MS-TCN-556 macro-F1 0.500 +/- 0.011", "+0.0933 over transformer-556"),
        ("Feature block", "Head-pose block +0.0296 test", "Mostly face_found; angles not significant"),
        ("Segmentation", "MS-TCN-570 edit 59.9", "Transformer range 28.8-38.2"),
        ("Human events", "Best model event recall 0.312; teacher 0.625", "16 distinct episodes; diagnostic"),
        ("External construct", "DIPSER rho +0.172; boredom AUROC 0.544", "Visible cues are context-dependent"),
        ("Runtime", "5.77 FPS; p95 291.7 ms; 94.7% cue agreement", "Adopted 3:2 stride, one A100"),
        ("Branch C pose", "-0.0022 CI [-0.0119,+0.0085]", "Validated full-range pose is redundant"),
        ("Branch C learned fusion", "+0.0015 CI [-0.0007,+0.0041]", "Null after leakage removal"),
        ("Branch C training recipe", "+0.0406 CI [+0.0289,+0.0521]", "Unregistered; cause not isolated"),
    ]
    add_table(doc, ["Area", "Headline result", "Meaning"], consolidated, [2200, 3150, 4010],
              aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
              highlights={3: (PALE_GREEN, GREEN), 4: (PALE_GREEN, GREEN), 5: (PALE_GREEN, GREEN),
                          13: (PALE_GREEN, GREEN)}, font_size=8.4)
    add_source(doc, "Only compare values within their task and protocol; grounding R@k, cue macro-F1, event recall and ordinal engagement metrics answer different questions.")

    add_heading(doc, "Appendix B. Important invalidated or non-citable results", 1)
    invalid_rows = [
        ("March detector R@1 approximately 0.613", "Frame-wise temporal leakage; best checkpoint unavailable."),
        ("March temporal accuracy 0.6442", "Two classes had zero support and aggregation was inflated."),
        ("Matching accuracy 0.9231", "Ground-truth left-to-right order leaked through the spatial matching cost."),
        ("DDP macro-F1 0.4096/0.4364/0.4400", "Computed on rank-local validation shards, not the full set."),
        ("Earlier 570 best-model claim", "Overturned by correct single-process and isolated-feature evaluation."),
        ("First dashboard end-to-end claim", "Checkpoint width mismatch was swallowed and a random temporal model ran."),
        ("Archived 7.1 FPS", "Omitted the useful face-presence block and was not reproduced like-for-like."),
        ("Branch-C learned fusion +0.0902", "Reliability gate read label-bearing teacher-side quality fields."),
        ("95% performance without looking at student", "Corrected: deployable five-signal shortcut is 0.3208, not 0.4602."),
        ("Branch-C inner validation as a test result", "Outer folds remain unopened; selection-stage values are not generalisation estimates."),
    ]
    add_table(doc, ["Result", "Why it must not be promoted"], invalid_rows, [3550, 5810],
              aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT],
              highlights={7: (PALE_RED, RED), 8: (PALE_RED, RED)}, font_size=8.6)
    add_source(doc, "Sources: MARCH_2026_POSTMORTEM.md; outputs/FINAL_RESULTS_REGISTER.*; FINDINGS.md Sections 11-12.")

    add_heading(doc, "Appendix C. Repository evidence map", 1)
    evidence_rows = [
        ("Chronological experimental log", "FINDINGS.md"),
        ("Citable/non-citable register", "outputs/FINAL_RESULTS_REGISTER.md and .json"),
        ("Branch-A test protocol", "TEST_SPLIT_PROTOCOL.md"),
        ("Branch-B test protocol", "BRANCH_B_TEST_PROTOCOL.md"),
        ("Branch-C frozen protocol", "BRANCH_C_PROTOCOL.md"),
        ("Branch-C thesis-ready text", "THESIS_BRANCH_C_DRAFT.md"),
        ("Branch-C audits", "docs/branch_c/REPOSITORY_AND_DATA_AUDIT.md, NOVELTY_AUDIT.md, LICENSE_AUDIT.md"),
        ("Branch-C results and ablations", "outputs/branch_c/RESULTS_REGISTER.*, ABLATIONS.md, ERROR_ANALYSIS.md"),
        ("Runtime configuration", "LLMDet/configs/attention_runtime.yaml"),
        ("Dashboard verification/design", "FINDINGS.md Sections 11.18-11.22; docs/branch_c/DASHBOARD_INTEGRATION.md"),
        ("Reproducibility", "docs/REPRODUCIBILITY.md; artifacts.lock.json; bootstrap.py; pyproject.toml"),
        ("First full thesis draft", "THESIS_FIRST_DRAFT.md"),
        ("Defensibility review", "THESIS_DEFENSIBILITY_REVIEW.md"),
        ("Invalid early work", "MARCH_2026_POSTMORTEM.md"),
    ]
    add_table(doc, ["Evidence category", "Primary repository artifact"], evidence_rows, [3100, 6260],
              aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT], font_size=8.7)

    # Final paragraph and table hygiene.
    for paragraph in doc.paragraphs:
        if paragraph.style.name.startswith("Heading"):
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.keep_together = True

    doc.save(OUT_PATH)
    return OUT_PATH


if __name__ == "__main__":
    path = build_report()
    print(path)
