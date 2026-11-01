"""Read progress without initializing CUDA or touching running checkpoints."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
queue = root / 'runs/queue_status.json'
if queue.exists():
    print(queue.read_text(encoding='utf-8'))
for variant in 'ABCD':
    for seed in (11, 22, 33):
        run = root / 'runs' / variant / str(seed)
        progress = run / 'progress.json'
        if progress.exists():
            row = json.loads(progress.read_text(encoding='utf-8'))
            state = 'complete' if (run / 'complete.json').exists() else 'in progress'
            print(f'{variant}/{seed}: epoch {row["epoch"]+1}/60, '
                  f'validation F1={row["val_macro_f1"]:.5f}, {state}')
