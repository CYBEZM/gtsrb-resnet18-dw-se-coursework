"""Grad-CAM for four prespecified VALIDATION examples, A and D, seed 11."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch

from infer import load_model, predict
from src.utils import read_json, save_json, sha256


def make(data_root, runs_root, output):
    torch.set_num_threads(2)
    data_root, runs_root, output = Path(data_root), Path(runs_root), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = read_json('report/data/dataset_summary.json')
    examples = data['visual_examples']
    cell, top, label_height = 180, 50, 38
    canvas = Image.new('RGB', (cell * 5, top + (cell + label_height) * len(examples)), 'white')
    draw = ImageDraw.Draw(canvas)
    font = None
    for name in ('C:/Windows/Fonts/arial.ttf', 'DejaVuSans.ttf'):
        try:
            font = ImageFont.truetype(name, 17)
            break
        except OSError:
            pass
    if font is None:
        font = ImageFont.load_default(size=17)
    headings = ['Input', 'A Grad-CAM', 'D Grad-CAM', 'A prediction', 'D prediction']
    for col, text in enumerate(headings):
        draw.text((col * cell + 8, 15), text, fill='black', font=font)
    models = {v: load_model(runs_root / v / '11/best.ckpt')[0] for v in ('A', 'D')}
    records = []
    for row, sample in enumerate(examples):
        y = top + row * (cell + label_height)
        with Image.open(data_root / sample['path']) as img:
            predictions = {v: predict(models[v], img, heatmap=True) for v in ('A', 'D')}
        canvas.paste(predictions['A'][1].resize((cell, cell)), (0, y))
        canvas.paste(predictions['A'][2].resize((cell, cell)), (cell, y))
        canvas.paste(predictions['D'][2].resize((cell, cell)), (cell * 2, y))
        draw.text((8, y + cell + 5), f'True class: {sample["label"]}', fill='black', font=font)
        record = dict(sample_id=sample['path'], true_class=sample['label'], subset='validation')
        for col, v in enumerate(('A', 'D'), start=3):
            pred = predictions[v][0]
            record[v] = pred
            draw.multiline_text((col * cell + 10, y + 30),
                '\n'.join(f'Class {r["class_id"]}: {100*r["probability"]:.1f}%' for r in pred),
                font=font, spacing=14, fill='black')
        records.append(record)
    canvas.save(output)
    save_json(output.with_suffix('.json'), dict(subset='prespecified validation examples',
              source='report/data/dataset_summary.json', target_layer='stage3 (12x12)',
              image_sha256=sha256(output), examples=records))
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data/gtsrb'))
    parser.add_argument('--runs-root', type=Path, default=Path('runs'))
    parser.add_argument('--output', type=Path, default=Path('report/figures/gradcam_examples.png'))
    args = parser.parse_args()
    make(args.data_root, args.runs_root, args.output)
