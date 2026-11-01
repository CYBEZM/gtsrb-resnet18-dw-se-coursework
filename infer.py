import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.data import canonical_image, normalize
from src.labels import CLASS_NAMES_ZH
from src.models import TrafficResNet
from src.utils import save_json


def load_model(checkpoint, device='cpu'):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = TrafficResNet(state['config']['variant']).to(device).eval()
    model.load_state_dict(state['model'])
    return model, state


def predict(model, image, heatmap=False):
    canonical = canonical_image(image)
    x = torch.from_numpy(np.asarray(canonical).transpose(2, 0, 1).copy()).unsqueeze(0)
    x = normalize(x.to(next(model.parameters()).device).float() / 255)
    activation = []
    handle = None
    if heatmap:
        def capture(module, inputs, output):
            output.retain_grad()
            activation.append(output)
        handle = model.stages[2].register_forward_hook(capture)
    try:
        with torch.set_grad_enabled(heatmap):
            logits = model(x)
            probabilities = logits.softmax(1)[0]
            values, indices = probabilities.topk(3)
            top = [dict(class_id=int(i), name=CLASS_NAMES_ZH[int(i)], probability=float(v))
                   for i, v in zip(indices.detach().cpu(), values.detach().cpu())]
            overlay = None
            if heatmap:
                model.zero_grad(set_to_none=True)
                logits[0, indices[0]].backward()
                feature = activation[0]
                weights = feature.grad.mean((2, 3), keepdim=True)
                cam = torch.relu((weights * feature).sum(1, keepdim=True))
                cam = torch.nn.functional.interpolate(cam, size=(48, 48), mode='bilinear', align_corners=False)[0, 0]
                cam = cam.detach().cpu().numpy()
                cam = cam / max(float(cam.max()), 1e-8)
                # A simple blue-to-red RGB ramp avoids importing a second OpenMP
                # runtime through plotting libraries in the inference process.
                colors = np.stack([cam, np.zeros_like(cam), 1 - cam], axis=-1)
                blended = 0.6 * np.asarray(canonical) / 255 + 0.4 * colors
                overlay = Image.fromarray(np.uint8(np.clip(blended * 255, 0, 255)))
        return top, canonical, overlay
    finally:
        if handle:
            handle.remove()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('prediction.json'))
    parser.add_argument('--cam', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, _ = load_model(args.checkpoint)
    with Image.open(args.image) as image:
        top, _, overlay = predict(model, image, bool(args.cam))
    save_json(args.output, dict(top3=top, note='Uncalibrated closed-set probabilities; cropped signs only'))
    if args.cam:
        args.cam.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(args.cam)
    print(top)
