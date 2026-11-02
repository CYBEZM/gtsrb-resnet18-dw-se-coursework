"""Read progress without initializing CUDA or touching running checkpoints."""
import json
from pathlib import Path
from src.utils import sha256

root = Path(__file__).resolve().parent
queue = root / 'runs/queue_status.json'
if queue.exists():
    print(queue.read_text(encoding='utf-8'))
protocol = json.loads((root / 'protocol_frozen.json').read_text(encoding='utf-8'))
seeds = protocol['seeds']
print(f'Planned runs: {4 * len(seeds)}, seeds: {seeds}')
completed = 0
finished_epochs = 0
evaluated = 0
for variant in 'ABCD':
    for seed in seeds:
        run = root / 'runs' / variant / str(seed)
        progress = run / 'progress.json'
        if (run / 'complete.json').exists():
            completed += 1
        evaluation = run / 'robustness_metrics.json'
        conditions = len(json.loads(evaluation.read_text(encoding='utf-8'))) if evaluation.exists() else 0
        evaluated += conditions
        if progress.exists():
            row = json.loads(progress.read_text(encoding='utf-8'))
            state = 'complete' if (run / 'complete.json').exists() else 'in progress'
            epochs = protocol['configs'][variant]['epochs']
            finished_epochs += row['epoch'] + 1
            print(f'{variant}/{seed}: epoch {row["epoch"]+1}/{epochs}, '
                  f'validation F1={row["val_macro_f1"]:.5f}, {state}')
            profile = root / 'report/profiles' / f'{variant}.json'
            print(f'  test conditions: {conditions}/16; GPU profiling: {"complete" if profile.exists() else "pending"}')
        else:
            print(f'{variant}/{seed}: pending, epoch 0/{protocol["configs"][variant]["epochs"]}')
print(f'Total training: {completed}/{4 * len(seeds)} runs; '
      f'{finished_epochs}/{sum(protocol["configs"][v]["epochs"] for v in "ABCD") * len(seeds)} epochs; '
      f'test conditions: {evaluated}/{64 * len(seeds)}')
report = root / 'report/大作业报告.md'
audit = root / 'report/final_audit.json'
active_hash = sha256(root / 'protocol_frozen.json')
summary = root / 'report/tables/summary.json'
summary_current = summary.exists() and json.loads(summary.read_text(encoding='utf-8')).get('protocol_sha256') == active_hash
audit_current = audit.exists() and json.loads(audit.read_text(encoding='utf-8')).get('protocol_sha256') == active_hash
report_current = report.exists() and active_hash in report.read_text(encoding='utf-8')
print(f'Active-plan summary: {"ready" if summary_current else "pending"}; '
      f'report: {"ready" if report_current else "pending"}; '
      f'result audit: {"ready" if audit_current else "pending"}')
if not summary_current:
    print('Existing seed-11 report is historical; three-seed statistics are not yet complete.')
