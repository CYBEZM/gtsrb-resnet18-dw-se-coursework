import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # grid_sample backward on CUDA is not fully deterministic; input tensors here
    # require no gradients and augmentations run before the model. Reproducibility
    # is scoped to the recorded device/software, not arbitrary platforms.


def load_config(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def provenance(folder):
    import torch
    import PIL
    folder = Path(folder)
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    result = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  torch=torch.__version__, cuda_runtime=torch.version.cuda,
                  cudnn=torch.backends.cudnn.version(), numpy=np.__version__, pillow=PIL.__version__,
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  git_commit=git('rev-parse', 'HEAD'), git_status=git('status', '--porcelain'),
                  argv=sys.argv, deterministic_cudnn=True, tf32=False)
    try:
        result['nvidia_smi'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,power.limit,pstate',
                                                       '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        result['nvidia_smi'] = None
    # Snapshot even untracked source/config files; exclude data, weights and secrets.
    sources = []
    with zipfile.ZipFile(folder / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for pattern in ('*.py', 'src/*.py', 'configs/*.yaml', 'requirements.txt', 'README.md'):
            for path in sorted(Path('.').glob(pattern)):
                z.write(path, path.as_posix())
                sources.append({'path': path.as_posix(), 'sha256': sha256(path)})
    result['source_files'] = sources
    result['source_snapshot_sha256'] = sha256(folder / 'source_snapshot.zip')
    save_json(folder / 'environment.json', result)
    return result


def metrics_from_confusion(cm):
    cm = cm.astype(np.int64)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    return dict(accuracy=float(tp.sum() / max(1, cm.sum())), macro_f1=float(f1.mean()),
                samples=int(cm.sum()), correct=int(tp.sum()), per_class_precision=precision.tolist(),
                per_class_recall=recall.tolist(), per_class_f1=f1.tolist(), support=support.tolist(),
                confusion_matrix=cm.tolist())
