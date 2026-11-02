"""Provenance rejection and three-run aggregation integration checks.

Synthetic numbers exist only in a temporary test directory, never report/.
"""
import copy
from pathlib import Path
import shutil
import statistics
import tempfile
import unittest

from experiment_protocol import evaluation_protocol_path
from src.utils import read_json, save_json, sha256


class RepeatExtensionTests(unittest.TestCase):
    def test_inherited_protocol_rejects_changed_rules(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            parent_path, active_path = folder / 'parent.json', folder / 'active.json'
            parent = read_json('protocols/protocol_v2_single_seed.json')
            save_json(parent_path, parent)
            active = copy.deepcopy(parent)
            active['seeds'] = [11, 22, 33]
            active['inherited_evaluations'] = {'11': dict(path=str(parent_path), sha256=sha256(parent_path))}
            save_json(active_path, active)
            self.assertEqual(evaluation_protocol_path(active_path, 11), parent_path)
            self.assertEqual(evaluation_protocol_path(active_path, 22), active_path)
            with self.assertRaises(ValueError):
                evaluation_protocol_path(active_path, 44)
            active['configs']['A']['epochs'] = 10
            save_json(active_path, active)
            with self.assertRaisesRegex(ValueError, 'differs in configs'):
                evaluation_protocol_path(active_path, 11)
            parent['seeds'] = [44]
            save_json(parent_path, parent)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                evaluation_protocol_path(active_path, 11)

    def test_three_run_statistics_and_report(self):
        from make_tables import make
        from audit_summary import audit
        from build_report import build
        with tempfile.TemporaryDirectory(prefix='synthetic-repeat-check-') as folder:
            folder = Path(folder)
            run_root, report_root = folder / 'runs', folder / 'report'
            protocol_path = folder / 'protocol.json'
            protocol = read_json('protocol_frozen.json')
            protocol.pop('inherited_evaluations', None)
            save_json(protocol_path, protocol)
            shutil.copytree('report/profiles', report_root / 'profiles')
            shutil.copytree('report/data', report_root / 'data')
            for vi, variant in enumerate('ABCD'):
                for si, seed in enumerate((11, 22, 33)):
                    dest = run_root / variant / str(seed)
                    dest.mkdir(parents=True)
                    source = Path('runs') / variant / '11'
                    for name in ('complete.json', 'complexity.json', 'log.csv', 'environment.json'):
                        shutil.copy2(source / name, dest / name)
                    complete = read_json(dest / 'complete.json')
                    complete['seed'] = seed
                    save_json(dest / 'complete.json', complete)
                    robust = read_json(source / 'robustness_metrics.json')
                    for condition in robust.values():
                        condition.update(seed=seed, protocol_sha256=sha256(protocol_path),
                            accuracy=0.7 + 0.08 * si - 0.01 * vi + 0.005 * vi * si,
                            macro_f1=0.6 + 0.11 * si - 0.01 * vi + 0.003 * vi * si)
                    save_json(dest / 'robustness_metrics.json', robust)
                    save_json(dest / 'test_metrics.json', robust['clean_0'])
            make(run_root, report_root, protocol_path)
            summary_path = report_root / 'tables/summary.json'
            result = audit(summary_path, protocol_path)
            self.assertEqual(result['status'], 'PASS')
            summary = read_json(summary_path)
            self.assertEqual(len(summary['per_seed']), 12)
            a = summary['models'][0]
            self.assertAlmostEqual(a['accuracy_mean'], 0.78)
            self.assertAlmostEqual(a['accuracy_std'], 0.08)
            self.assertAlmostEqual(a['macro_f1_std'], statistics.stdev([0.6, 0.71, 0.82]))
            dw = next(r for r in summary['contrasts'] if r['contrast'] == 'DW effect')
            self.assertAlmostEqual(dw['mean_accuracy_difference_pp'], -0.5)
            self.assertAlmostEqual(dw['std_accuracy_difference_pp'], 0.5)
            build(run_root, report_root, protocol_path)
            text = (report_root / '大作业报告.md').read_text(encoding='utf-8')
            self.assertIn('78.00 ± 8.00', text)
            self.assertIn('查看结果后的重复实验扩展', text)
            self.assertNotIn('C取得最高的干净准确率', text)
            summary['models'][0]['accuracy_std'] = 0.0
            save_json(summary_path, summary)
            with self.assertRaises(AssertionError):
                audit(summary_path, protocol_path)


if __name__ == '__main__':
    unittest.main()
