"""Convert the measured Markdown report to an editable Chinese Word report.

Run with python-docx installed. Final visual QA is separate from file creation.
"""
import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


def font(run, size=None, bold=None):
    run.font.name = 'Times New Roman'
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), '宋体')
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def inline(paragraph, text, size=None):
    # Plain inline code stays editable and long path text wraps naturally.
    parts = re.split(r'(`[^`]+`|\*\*.+?\*\*)', text)
    for value in parts:
        bold = value.startswith('**') and value.endswith('**')
        value = value[2:-2] if bold else value.strip('`')
        run = paragraph.add_run(value)
        font(run, size, bold)


def create(source, output):
    source, output = Path(source), Path(output)
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin = section.bottom_margin = Inches(0.8)
    section.left_margin = section.right_margin = Inches(0.85)
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(11)
    style.element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), '宋体')
    style.paragraph_format.line_spacing = 1.2
    style.paragraph_format.space_after = Pt(6)
    for name, size in [('Title', 19), ('Subtitle', 11), ('Heading 1', 14), ('Heading 2', 12)]:
        s = doc.styles[name]
        s.font.name = 'Times New Roman'
        s.font.size = Pt(size)
        s.font.color.rgb = RGBColor(0, 0, 0)
        s.font.italic = False
        s.element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), '黑体')
        s.paragraph_format.space_before = Pt(12)
        s.paragraph_format.space_after = Pt(7)
    # The bundled Word template may carry a blue rule in an inherited style.
    # Remove paragraph borders from every style so headings remain plain black.
    for s in doc.styles:
        for border in list(s.element.iter(qn('w:pBdr'))):
            border.getparent().remove(border)
        for fonts in s.element.iter(qn('w:rFonts')):
            for key in ('asciiTheme', 'hAnsiTheme', 'eastAsiaTheme', 'cstheme'):
                fonts.attrib.pop(qn('w:' + key), None)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    lines = source.read_text(encoding='utf-8').splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('# '):
            p = doc.add_paragraph(line[2:], 'Title')
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            subtitle = doc.add_paragraph('神经网络与深度学习课程大作业', style='Subtitle')
            subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
        elif line.startswith('## '):
            doc.add_heading(line[3:], level=1)
            i += 1
        elif line.startswith('### '):
            doc.add_heading(line[4:], level=2)
            i += 1
        elif line.startswith('!['):
            match = re.match(r'!\[(.*?)\]\((.*?)\)', line)
            image_path = source.parent / match.group(2)
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            from PIL import Image
            with Image.open(image_path) as img:
                ratio = img.height / img.width
            width = min(6.65, 4.7 / ratio)
            p.add_run().add_picture(str(image_path), width=Inches(width))
            p.paragraph_format.keep_with_next = True
            caption = doc.add_paragraph(match.group(1))
            caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in caption.runs:
                font(run, 10)
            i += 1
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                values = [x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', value) for value in values):
                    rows.append(values)
                i += 1
            table = doc.add_table(rows=1, cols=len(rows[0]))
            table.autofit = False
            widths = {
                2: [1.4, 5.25], 3: [2.05, 2.3, 2.3],
                5: [0.6, 1.7, 1.7, 1.35, 1.3],
                6: [1.4, 0.65, 1.15, 1.15, 1.15, 1.15],
            }.get(len(rows[0]), [6.65 / len(rows[0])] * len(rows[0]))
            # Numeric clean-results table and per-run table have different width needs.
            if len(rows[0]) == 5 and 'Accuracy %' in rows[0]:
                widths = [0.5, 1.35, 1.35, 1.65, 1.8]
            if len(rows[0]) == 6 and '种子' in rows[0]:
                widths = [0.5, 0.5, 1.15, 1.15, 2.15, 1.2]
            for column, width in zip(table.columns, widths):
                column.width = Inches(width)
            borders = OxmlElement('w:tblBorders')
            for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
                item = OxmlElement('w:' + side)
                for key, value in [('val', 'single'), ('sz', '4'), ('color', 'D9D9D9')]:
                    item.set(qn('w:' + key), value)
                borders.append(item)
            table._tbl.tblPr.append(borders)
            for n, values in enumerate(rows):
                row = table.rows[0] if n == 0 else table.add_row()
                prevent = OxmlElement('w:cantSplit')
                row._tr.get_or_add_trPr().append(prevent)
                if n == 0:
                    row._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
                for col, value in enumerate(values):
                    cell = row.cells[col]
                    cell.width = Inches(widths[col])
                    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    pr = cell._tc.get_or_add_tcPr()
                    shade = OxmlElement('w:shd')
                    shade.set(qn('w:fill'), 'DCE6F1' if n == 0 else ('F5F7FA' if n % 2 == 0 else 'FFFFFF'))
                    pr.append(shade)
                    margins = OxmlElement('w:tcMar')
                    for side in ('top', 'left', 'bottom', 'right'):
                        margin = OxmlElement('w:' + side)
                        margin.set(qn('w:w'), '75')
                        margin.set(qn('w:type'), 'dxa')
                        margins.append(margin)
                    pr.append(margins)
                    p = cell.paragraphs[0]
                    p.paragraph_format.space_after = Pt(2)
                    p.paragraph_format.space_before = Pt(2)
                    p.paragraph_format.line_spacing = 1.1
                    if len(rows[0]) > 2 or n == 0:
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    inline(p, value, 10)
                    for run in p.runs:
                        run.bold = n == 0
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
        elif line.startswith('```'):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                p = doc.add_paragraph()
                inline(p, lines[i], 9)
                p.paragraph_format.space_after = Pt(2)
                i += 1
            i += 1
        else:
            group = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '|', '![', '```')):
                group.append(lines[i].strip())
                i += 1
            p = doc.add_paragraph()
            inline(p, ''.join(group))
    doc.core_properties.title = '轻量化 ResNet 与 SE 注意力在交通标志识别中的应用研究'
    doc.core_properties.subject = 'GTSRB 分类与结构消融实验'
    doc.core_properties.author = ''
    doc.core_properties.keywords = 'CNN ResNet GTSRB SE depthwise'
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='report/大作业报告.md')
    parser.add_argument('--output', default='../大作业报告_交通标志识别.docx')
    args = parser.parse_args()
    create(args.source, args.output)
