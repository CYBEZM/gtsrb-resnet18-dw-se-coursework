"""FP32 GPU timing, kept separate from training and input preprocessing."""
import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from src.models import TrafficResNet, complexity
from src.utils import save_json, seed_all, sha256


def profile(variant, checkpoint=None):
    torch.set_num_threads(4)
    seed_all(2026)
    model = TrafficResNet(variant).cuda().to(memory_format=torch.channels_last).eval()
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    result = complexity(model)
    result.update(dict(variant=variant, device=torch.cuda.get_device_name(0), torch=torch.__version__,
                       cuda=torch.version.cuda, precision='FP32', tf32=False, channels_last=True,
                       warmup=50, samples=200, repeats=3, timing='CUDA Events, GPU-resident input',
                       checkpoint_sha256=sha256(checkpoint) if checkpoint else None))
    latencies, throughputs = [], []
    with torch.inference_mode():
        for repeat in range(3):
            x = torch.randn(1, 3, 48, 48, device='cuda').to(memory_format=torch.channels_last)
            for _ in range(50):
                model(x)
            torch.cuda.synchronize()
            timings = []
            for _ in range(200):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                model(x)
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end))
            latencies.append(dict(median_ms=float(np.median(timings)), p95_ms=float(np.percentile(timings, 95)), raw_ms=timings))
            x = torch.randn(32, 3, 48, 48, device='cuda').to(memory_format=torch.channels_last)
            for _ in range(50):
                model(x)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(200):
                model(x)
            end.record()
            end.synchronize()
            throughputs.append(6400 / (start.elapsed_time(end) / 1000))
    result.update(dict(latency_runs=latencies, throughput_runs_images_s=throughputs,
                       latency_median_ms=statistics.median(v['median_ms'] for v in latencies),
                       latency_p95_ms=statistics.median(v['p95_ms'] for v in latencies),
                       throughput_images_s=statistics.median(throughputs),
                       weight_file_bytes=Path(checkpoint).stat().st_size if checkpoint else None))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=list('ABCD'), required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    save_json(args.output, profile(args.variant, args.checkpoint))
    print(args.output)
