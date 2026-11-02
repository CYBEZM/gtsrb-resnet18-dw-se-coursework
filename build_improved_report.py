"""Compose reviewed supplemental sections without rewriting the published A-D report."""
from pathlib import Path
from src.utils import read_json, save_json, sha256


def build():
    root=Path('report')
    original=root/'大作业报告.md'
    text=original.read_text(encoding='utf-8')
    sections=[]
    sources=[original]
    for directory,audit_name in [('deep_analysis','analysis.json'),('external','audit.json'),
                                  ('supplement','audit.json')]:
        section=root/directory/'section.md'
        audit=root/directory/audit_name
        if section.exists():
            assert read_json(audit)['status']=='PASS', f'{directory} has not passed audit'
            if directory=='external':
                assert read_json(audit)['summary_sha256']==sha256(root/directory/'summary.json')
            if directory=='deep_analysis':
                independent=root/directory/'independent_audit.json'
                assert read_json(independent)['status']=='PASS'
                assert read_json(independent)['analysis_sha256']==sha256(audit)
                sources.append(independent)
            sections.append(section.read_text(encoding='utf-8'))
            sources.extend([section,audit])
    if not (root/'supplement/section.md').exists():
        sections.append('## 补充实验进度说明\n\nE组仅在stage3/4采用深度可分离卷积，'
            '按独立的事后探索协议执行种子11、22、33。当前训练与评估尚未全部完成，'
            '本版本不列入其精度、鲁棒性或硬件性能结论；完成并通过审计后自动更新。\n')
    text=text.replace('## 附录 结果来源映射','\n\n'.join(sections)+'\n\n## 附录 结果来源映射',1)
    text=text.replace('后续可在新增独立场景数据和更严格统计设计下验证结论，再考虑stage局部替换或低数据量实验。',
        '新增的真实道路裁剪分析与stage局部替换实验见补充章节。独立地点和时间的场景采集、'
        '更充分的随机重复仍需后续验证。')
    dest=root/'大作业报告_改进版.md'
    dest.write_text(text,encoding='utf-8')
    save_json(root/'improved_report_sources.json',dict(status='PASS',
        output=dest.as_posix(),output_sha256=sha256(dest),
        sources=[dict(path=p.as_posix(),sha256=sha256(p)) for p in sources],
        supplemental_E_complete=(root/'supplement/section.md').exists()))
    print(dest)


if __name__=='__main__':
    build()
