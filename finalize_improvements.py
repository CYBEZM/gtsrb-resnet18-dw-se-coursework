"""Finish local supplemental deliverables; publishing remains an explicit step."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def finalize(data_root,external_root,report_python=None):
    def run(*args,python=None):
        subprocess.run([str(python or sys.executable),'-u',*map(str,args)],check=True)
    run('summarize_supplement.py')
    run('audit_deep_analysis.py')
    if (Path(external_root)/'download_manifest.json').exists():
        run('evaluate_external.py','--external-root',external_root,'--gtsrb-root',data_root)
        run('audit_external.py','--external-root',external_root)
        run('make_external_scenes.py','--external-root',external_root)
    run('build_improved_report.py')
    exporter=report_python or os.environ.get('GTSRB_REPORT_PYTHON')
    if exporter or importlib.util.find_spec('reportlab'):
        run('export_pdf.py','--source','report/大作业报告_改进版.md',
            '--output','report/大作业报告_改进版.pdf',python=exporter)
    else:
        print('PDF not exported: install reportlab or set GTSRB_REPORT_PYTHON. Markdown is complete.')
    run('package_results.py','--output','../traffic_sign_cnn_改进版_代码与结果.zip')
    from src.utils import save_json
    save_json('runs/improvement_completion.json',dict(computation='complete',
        original_A_D_preserved=True,E_conditions_audited=48,
        report='report/大作业报告_改进版.md',
        final_visual_review='Pending review of the newly generated E section and PDF',
        git_publication='Newly completed E evidence still requires final review and publication'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True)
    p.add_argument('--external-root',default='../../work/gtsdb_external')
    p.add_argument('--report-python')
    a=p.parse_args();finalize(a.data_root,a.external_root,a.report_python)
