"""Train one reproducible model/seed; validation selects the checkpoint."""
import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data import augment, build_cache, normalize
from src.models import TrafficResNet, complexity
from src.utils import load_config, metrics_from_confusion, provenance, read_json, save_json, seed_all, sha256


def atomic_checkpoint(path, state):
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def learning_rate(epoch, config):
    warm = config['warmup_epochs']
    if epoch < warm:
        return config['lr'] * (epoch + 1) / warm
    progress = (epoch - warm) / max(1, config['epochs'] - warm - 1)
    return config['min_lr'] + 0.5 * (config['lr'] - config['min_lr']) * (1 + math.cos(math.pi * progress))


@torch.inference_mode()
def validate(model, images, labels, indices, batch_size, device):
    model.eval()
    cm = torch.zeros(43 * 43, dtype=torch.long, device=device)
    total_loss = torch.zeros((), device=device)
    for begin in range(0, len(indices), batch_size):
        ix = indices[begin:begin + batch_size]
        x = normalize(images[ix].float() / 255).contiguous(memory_format=torch.channels_last)
        y = labels[ix]
        logits = model(x)
        total_loss += nn.functional.cross_entropy(logits, y, reduction='sum')
        cm += torch.bincount(43 * y + logits.argmax(1), minlength=43 * 43)
    result = metrics_from_confusion(cm.reshape(43, 43).cpu().numpy())
    result['loss'] = float(total_loss.item() / len(indices))
    return result


def train(config, data_root, out, resume=False):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use the CUDA-enabled interpreter documented in README.')
    torch.set_num_threads(4)
    config = dict(config)
    seed = config['seed']
    seed_all(seed)
    device = torch.device('cuda')
    out = Path(out)
    if out.exists() and (out / 'config.json').exists():
        if read_json(out / 'config.json') != config:
            raise ValueError('Refusing to reuse a run directory with different config')
        if (out / 'complete.json').exists():
            print('Already complete:', out, flush=True)
            return read_json(out / 'complete.json')
        if not resume:
            raise FileExistsError('Run exists; use --resume or a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_cache(data_root)
    manifest_path = Path(data_root) / 'cache48/manifest.json'
    save_json(out / 'config.json', config)
    if not (out / 'environment.json').exists():
        provenance(out)
    manifest_hash = sha256(manifest_path)
    save_json(out / 'data_reference.json', dict(manifest_sha256=manifest_hash,
              data_root=str(Path(data_root).resolve()), split_seed=manifest['split_seed'],
              train_count=len(manifest['train_indices']), val_count=len(manifest['val_indices']),
              raw_content_sha256=manifest['raw_content_sha256']))
    # The canonical uint8 cache is small enough to keep on the 8 GB GPU.
    # This removes Windows multiprocessing and per-image decode overhead.
    images = torch.from_numpy(np.load(Path(data_root) / 'cache48/train_images.npy')).to(device)
    labels = torch.tensor([r['label'] for r in manifest['train_rows']], device=device)
    train_ix = torch.tensor(manifest['train_indices'], device=device)
    val_ix = torch.tensor(manifest['val_indices'], device=device)
    model = TrafficResNet(config['variant']).to(device, memory_format=torch.channels_last)
    save_json(out / 'complexity.json', complexity(model))
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'],
                                weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=config['amp'])
    start_epoch, best_score, best_epoch = 0, (-1.0, -float('inf')), -1
    history = []
    if resume and (out / 'last.ckpt').exists():
        state = torch.load(out / 'last.ckpt', map_location=device, weights_only=False)
        if state['manifest_sha256'] != manifest_hash:
            raise ValueError('Dataset manifest changed since checkpoint')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start_epoch = state['epoch'] + 1
        best_score, best_epoch = tuple(state['best_score']), state['best_epoch']
        if (out / 'log.csv').exists():
            with (out / 'log.csv').open(newline='', encoding='utf-8') as f:
                history = [r for r in csv.DictReader(f) if int(r['epoch']) < start_epoch]
    fields = ['epoch', 'lr', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy',
              'val_macro_f1', 'seconds', 'peak_cuda_bytes', 'best_epoch']
    for epoch in range(start_epoch, config['epochs']):
        torch.cuda.synchronize()
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        lr = learning_rate(epoch, config)
        for group in optimizer.param_groups:
            group['lr'] = lr
        # Separate data RNG streams from model construction / optimizer state.
        generator = torch.Generator(device=device).manual_seed(seed * 100000 + epoch)
        aug_generator = torch.Generator(device=device).manual_seed(seed * 100000 + epoch + 50000000)
        order = train_ix[torch.randperm(len(train_ix), device=device, generator=generator)]
        loss_sum, correct = torch.zeros((), device=device), torch.zeros((), device=device)
        batch_size = config['batch_size']
        for begin in range(0, len(order), batch_size):
            ix = order[begin:begin + batch_size]
            x = augment(images[ix].float() / 255, aug_generator)
            x = normalize(x).contiguous(memory_format=torch.channels_last)
            y = labels[ix]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=config['amp']):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f'Nonfinite loss in epoch {epoch}')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach().float() * len(ix)
            correct += (logits.detach().argmax(1) == y).sum()
        val = validate(model, images, labels, val_ix, config['eval_batch_size'], device)
        score = (val['macro_f1'], -val['loss'])
        if score > best_score:
            best_score, best_epoch = score, epoch
            atomic_checkpoint(out / 'best.ckpt', dict(model=model.state_dict(), config=config,
                              epoch=epoch, validation=val, manifest_sha256=manifest_hash))
        torch.cuda.synchronize()
        row = dict(epoch=epoch, lr=lr, train_loss=loss_sum.item() / len(order),
                   train_accuracy=correct.item() / len(order), val_loss=val['loss'],
                   val_accuracy=val['accuracy'], val_macro_f1=val['macro_f1'],
                   seconds=time.perf_counter() - started,
                   peak_cuda_bytes=torch.cuda.max_memory_allocated(), best_epoch=best_epoch)
        history.append(row)
        with (out / 'log.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(history)
        atomic_checkpoint(out / 'last.ckpt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          scaler=scaler.state_dict(), epoch=epoch, best_score=best_score,
                          best_epoch=best_epoch, manifest_sha256=manifest_hash, config=config))
        save_json(out / 'progress.json', row)
        print(f'{config["variant"]} seed={seed} epoch={epoch+1}/{config["epochs"]} '
              f'loss={row["train_loss"]:.4f} val_acc={val["accuracy"]:.5f} '
              f'val_f1={val["macro_f1"]:.5f} {row["seconds"]:.1f}s', flush=True)
    result = dict(status='complete', variant=config['variant'], seed=seed, epochs=config['epochs'],
                  best_epoch=best_epoch, best_validation_macro_f1=best_score[0],
                  checkpoint_sha256=sha256(out / 'best.ckpt'), manifest_sha256=manifest_hash,
                  note='Validation only; test set has not been used by train.py')
    save_json(out / 'complete.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/A.yaml')
    parser.add_argument('--data-root', type=Path, default=Path('data/gtsrb'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg['seed'] = args.seed
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    train(cfg, args.data_root, args.output or Path(f'runs/{cfg["variant"]}/{args.seed}'), args.resume)
