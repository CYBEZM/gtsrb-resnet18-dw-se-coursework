"""Export the measured Markdown report as a self-contained Chinese PDF."""
import argparse
from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                               Table, TableStyle, KeepTogether)


def inline(value):
    # SimSun lacks the mathematical minus glyph; preserve the operation visibly.
    value = value.translate(str.maketrans({'\u2212': '-', '\u2013': '-', '\u2014': '-'}))
    value = escape(value.replace('`', ''))
    return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', value)


def create(source, output, regular, bold):
    source, output = Path(source), Path(output)
    pdfmetrics.registerFont(TTFont('ReportBody', str(regular), subfontIndex=0))
    pdfmetrics.registerFont(TTFont('ReportBold', str(bold), subfontIndex=0))
    pdfmetrics.registerFontFamily('ReportBody', normal='ReportBody', bold='ReportBold',
                                  italic='ReportBody', boldItalic='ReportBold')
    body = ParagraphStyle('Body', fontName='ReportBody', fontSize=10.5, leading=16,
                          spaceAfter=7, wordWrap='CJK', splitLongWords=True)
    styles = {
        'body': body,
        'title': ParagraphStyle('Title', parent=body, fontName='ReportBold', fontSize=18,
                                leading=27, alignment=TA_CENTER, spaceAfter=10),
        'subtitle': ParagraphStyle('Subtitle', parent=body, fontSize=10,
                                   alignment=TA_CENTER, spaceAfter=16),
        'h1': ParagraphStyle('H1', parent=body, fontName='ReportBold', fontSize=13,
                             leading=20, spaceBefore=12, spaceAfter=8, keepWithNext=True),
        'h2': ParagraphStyle('H2', parent=body, fontName='ReportBold', fontSize=11.5,
                             leading=18, spaceBefore=9, spaceAfter=6, keepWithNext=True),
        'cell': ParagraphStyle('Cell', parent=body, fontSize=9, leading=13,
                               spaceAfter=0, alignment=TA_CENTER),
        'cell_left': ParagraphStyle('CellLeft', parent=body, fontSize=9, leading=13,
                                    spaceAfter=0),
        'caption': ParagraphStyle('Caption', parent=body, fontSize=9, leading=14,
                                  alignment=TA_CENTER, spaceAfter=9),
        'code': ParagraphStyle('Code', parent=body, fontSize=8.5, leading=13,
                               spaceAfter=2),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(output), pagesize=letter, leftMargin=0.8*inch,
        rightMargin=0.8*inch, topMargin=0.72*inch, bottomMargin=0.7*inch,
        title='轻量化 ResNet 与 SE 注意力在交通标志识别中的应用研究', author='')
    width = document.width
    story = []
    lines = source.read_text(encoding='utf-8').splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line:
            continue
        if line.startswith('# '):
            story.extend([Paragraph(inline(line[2:]), styles['title']),
                          Paragraph('神经网络与深度学习课程大作业', styles['subtitle'])])
        elif line.startswith('## '):
            story.append(Paragraph(inline(line[3:]), styles['h1']))
        elif line.startswith('### '):
            story.append(Paragraph(inline(line[4:]), styles['h2']))
        elif line.startswith('!['):
            caption, filename = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', line).groups()
            picture = Image(str(source.parent / filename))
            scale = min(width / picture.imageWidth, 4.45*inch / picture.imageHeight)
            picture.drawWidth = picture.imageWidth * scale
            picture.drawHeight = picture.imageHeight * scale
            story.append(KeepTogether([Spacer(1, 5), picture,
                                       Paragraph(inline(caption), styles['caption'])]))
        elif line.startswith('|'):
            rows = []
            while True:
                row = [x.strip() for x in line.strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', value) for value in row):
                    rows.append(row)
                if index >= len(lines) or not lines[index].strip().startswith('|'):
                    break
                line = lines[index].strip()
                index += 1
            count = len(rows[0])
            fractions = {2: [0.25, 0.75], 3: [0.38, 0.31, 0.31],
                         4: [0.13, 0.35, 0.35, 0.17],
                         5: [0.08, 0.23, 0.23, 0.24, 0.22],
                         6: [0.18, 0.07, 0.1875, 0.1875, 0.1875, 0.1875]}[count]
            if count == 6 and '种子' in rows[0]:
                fractions = [0.07, 0.07, 0.19, 0.19, 0.28, 0.20]
            cells = [[Paragraph(inline(value), styles['cell_left'] if count == 2
                                 else styles['cell']) for value in row] for row in rows]
            table = Table(cells, colWidths=[width*f for f in fractions], repeatRows=1,
                          hAlign='CENTER')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#DCE6F1')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.white, colors.HexColor('#F5F7FA')]),
                ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#D9D9D9')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.extend([table, Spacer(1, 9)])
        elif line.startswith('```'):
            while index < len(lines) and not lines[index].strip().startswith('```'):
                story.append(Paragraph(inline(lines[index]), styles['code']))
                index += 1
            index += 1
            story.append(Spacer(1, 7))
        else:
            group = [line]
            while index < len(lines) and lines[index].strip() and not lines[index].startswith(('#', '|', '![', '```')):
                group.append(lines[index].strip())
                index += 1
            story.append(Paragraph(inline(''.join(group)), body))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('ReportBody', 9)
        canvas.drawCentredString(letter[0]/2, 0.38*inch, str(doc.page))
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='report/大作业报告.md')
    parser.add_argument('--output', default='../大作业报告_交通标志识别_三种子.pdf')
    parser.add_argument('--font', default='C:/Windows/Fonts/simsun.ttc')
    parser.add_argument('--bold-font', default='C:/Windows/Fonts/simhei.ttf')
    args = parser.parse_args()
    create(args.source, args.output, args.font, args.bold_font)
