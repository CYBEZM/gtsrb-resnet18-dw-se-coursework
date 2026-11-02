"""Recompute every final metric from sample predictions and verify provenance."""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.utils import read_json, save_json, sha256
from experiment_protocol import evaluation_protocol_path


def audit(root, protocol_path, variants='ABCD'):
    root = Path(root)
    protocol = read_json(protocol_path)
    comparisons = []
    hashes = {}
    sample_ids_reference = None
    for variant in variants:
        for seed in protocol['seeds']:
            evaluation_protocol = evaluation_protocol_path(protocol_path, seed)
            run = root / variant / str(seed)
            complete = read_json(run / 'complete.json')
            assert complete['epochs'] == protocol['configs'][variant]['epochs']
            assert complete['manifest_sha256'] == protocol['manifest_sha256']
            assert complete['checkpoint_sha256'] == sha256(run / 'best.ckpt')
            with (run / 'log.csv').open(newline='', encoding='utf-8') as file:
                history = list(csv.DictReader(file))
            assert [int(row['epoch']) for row in history] == list(range(complete['epochs']))
            best = max(history, key=lambda row: (float(row['val_macro_f1']), -float(row['val_loss']), -int(row['epoch'])))
            assert int(best['epoch']) == complete['best_epoch']
            assert float(best['val_macro_f1']) == complete['best_validation_macro_f1']
            results = read_json(run / 'robustness_metrics.json')
            expected = {'clean_0'} | {f'{kind}_{level}' for kind in
                ('brightness', 'motion', 'noise', 'jpeg', 'resolution') for level in (1, 2, 3)}
            assert set(results) == expected
            assert read_json(run / 'test_metrics.json') == results['clean_0']
            for condition, recorded in results.items():
                prediction = run / recorded['prediction_file']
                assert sha256(prediction) == recorded['prediction_sha256']
                assert recorded['checkpoint_sha256'] == complete['checkpoint_sha256']
                assert recorded['protocol_sha256'] == sha256(evaluation_protocol)
                assert recorded['manifest_sha256'] == protocol['manifest_sha256']
                with np.load(prediction) as values:
                    truth, predicted = values['labels'], values['predicted']
                    probs, ids = values['probabilities'], values['sample_ids']
                    assert probs.shape == (12630, 43)
                    assert np.all(np.isfinite(probs)) and np.all(probs >= 0)
                    assert np.allclose(probs.sum(1), 1, atol=1e-5)
                    assert np.array_equal(probs.argmax(1), predicted)
                    if sample_ids_reference is None:
                        sample_ids_reference = (ids.copy(), truth.copy())
                    assert np.array_equal(ids, sample_ids_reference[0])
                    assert np.array_equal(truth, sample_ids_reference[1])
                    accuracy = accuracy_score(truth, predicted)
                    macro_f1 = f1_score(truth, predicted, labels=list(range(43)), average='macro', zero_division=0)
                    cm = confusion_matrix(truth, predicted, labels=list(range(43)))
                    assert np.isclose(accuracy, recorded['accuracy'], atol=1e-14, rtol=0)
                    assert np.isclose(macro_f1, recorded['macro_f1'], atol=1e-14, rtol=0)
                    assert np.array_equal(cm, recorded['confusion_matrix'])
                    assert recorded['correct'] == int((truth == predicted).sum())
                if condition in hashes:
                    assert hashes[condition] == recorded['input_cache_sha256']
                hashes[condition] = recorded['input_cache_sha256']
                comparisons.append(dict(variant=variant, seed=seed, condition=condition,
                                        accuracy=float(accuracy), macro_f1=float(macro_f1),
                                        protocol_sha256=sha256(evaluation_protocol), verified=True))
    return dict(status='PASS', verified_conditions=len(comparisons),
                samples_per_condition=12630, protocol_sha256=sha256(protocol_path),
                checks=['sklearn accuracy/F1', 'full confusion matrices', 'prediction file hashes',
                        'all model/label/sample alignment', 'shared input hashes across models',
                        'checkpoint and protocol provenance', 'probability sums and argmax',
                        '60 unique epochs and validation-based checkpoint selection'],
                conditions=comparisons)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-root', default='runs')
    parser.add_argument('--protocol', default='protocol_frozen.json')
    parser.add_argument('--output', default='report/final_audit.json')
    args = parser.parse_args()
    result = audit(args.runs_root, args.protocol)
    save_json(args.output, result)
    print('PASS:', result['verified_conditions'], 'conditions independently recomputed')
