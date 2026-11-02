"""Evaluate the frozen GTSDB crop selection on CPU without training or tuning."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from src.data import canonical_image, normalize
from src.models import TrafficResNet
from src.utils import read_json, save_json, sha256, metrics_from_confusion

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'report/external'


def canonical_hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def prepare_inputs(external_root, gtsrb_root):
    selection = read_json(external_root / 'selection_frozen.json')
    manifest = read_json(external_root / 'download_manifest.json')
    for row in manifest:
        if sha256(external_root / row['path']) != row['sha256']:
            raise ValueError('External downloaded image changed')
    known_hashes = set()
    for name in ('train_images.npy', 'test_images.npy'):
        for array in np.load(gtsrb_root / 'cache48' / name, mmap_mode='r'):
            known_hashes.add(canonical_hash(array))
    pixels, rows, excluded = [], [], []
    for index, annotation in enumerate(selection['annotations']):
        name, left, top, right, bottom, label = annotation
        box = list(map(int, (left, top, right, bottom)))
        with Image.open(external_root / 'scenes' / name) as scene:
            if not (0 <= box[0] < box[2] < scene.width and 0 <= box[1] < box[3] < scene.height):
                raise ValueError('Invalid external annotation box')
            # Explicit inclusive ROI convention, frozen before model inference.
            crop = scene.crop((box[0],box[1],box[2]+1,box[3]+1))
            array = np.asarray(canonical_image(crop)).transpose(2,0,1).copy()
        row = dict(sample_id=f'{name}:{index}',scene=name,class_id=int(label),box_inclusive=box,
                   canonical_sha256=canonical_hash(array),crop_width=box[2]-box[0]+1,
                   crop_height=box[3]-box[1]+1)
        if row['canonical_sha256'] in known_hashes:
            excluded.append(row)
        else:
            rows.append(row); pixels.append(array)
    inputs = np.stack(pixels)
    plan = dict(selection_sha256=sha256(external_root/'selection_frozen.json'),
        download_manifest_sha256=sha256(external_root/'download_manifest.json'),
        crop_convention='Inclusive gt.txt ROI -> edge square padding -> 48x48 bilinear; no context margin',
        duplicate_rule='Exclude byte-identical canonical RGB arrays matching GTSRB train/validation/test before inference',
        excluded_exact_duplicates=excluded, samples=rows,
        input_sha256=canonical_hash(inputs),
        limitation='No near-duplicate/physical-sign independence guarantee; GTSDB train split is evaluation-only here; '
                   '100 selected scenes do not represent the full benchmark; classification on supplied boxes, no detection')
    dest = OUT/'input_protocol.json'
    if dest.exists() and read_json(dest) != plan:
        raise ValueError('External evaluation inputs changed')
    save_json(dest,plan)
    return inputs, rows


def evaluate(external_root, gtsrb_root):
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.mkldnn.enabled = True
    inputs, rows = prepare_inputs(external_root,gtsrb_root)
    labels = np.array([r['class_id'] for r in rows],dtype=np.int64)
    variants = list('ABCD')
    if all((ROOT/f'runs/E/{seed}/best.ckpt').exists() and
           (ROOT/f'runs/E/{seed}/complete.json').exists() for seed in (11,22,33)):
        variants.append('E')
    all_results=[]
    for variant in variants:
        for seed in (11,22,33):
            checkpoint = ROOT/f'runs/{variant}/{seed}/best.ckpt'
            target = OUT/f'{variant}_{seed}.json'
            predictions = OUT/f'{variant}_{seed}.npz'
            if target.exists():
                result=read_json(target)
                assert result['checkpoint_sha256']==sha256(checkpoint)
                assert result['input_protocol_sha256']==sha256(OUT/'input_protocol.json')
                assert result['prediction_sha256']==sha256(predictions)
                all_results.append(result); continue
            model=TrafficResNet(variant).eval().to(memory_format=torch.channels_last)
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(state['model'])
            probabilities=[]
            with torch.inference_mode():
                for begin in range(0,len(inputs),32):
                    batch=torch.from_numpy(inputs[begin:begin+32]).float()/255
                    logits=model(normalize(batch).contiguous(memory_format=torch.channels_last))
                    probabilities.append(logits.softmax(1).numpy())
            probs=np.concatenate(probabilities); predicted=probs.argmax(1)
            cm=np.bincount(43*labels+predicted,minlength=43*43).reshape(43,43)
            result=metrics_from_confusion(cm)
            present=sorted(set(labels.tolist()))
            result.update(variant=variant,seed=seed,device='CPU FP32',
                present_classes=present,macro_f1_present_classes=float(np.mean(np.array(result['per_class_f1'])[present])),
                checkpoint_sha256=sha256(checkpoint),input_protocol_sha256=sha256(OUT/'input_protocol.json'),
                top3_accuracy=float(np.mean(np.any(np.argsort(probs,axis=1)[:,-3:]==labels[:,None],axis=1))))
            np.savez_compressed(predictions,sample_ids=np.array([r['sample_id'] for r in rows]),
                                labels=labels,predicted=predicted,probabilities=probs)
            result['prediction_sha256']=sha256(predictions)
            save_json(target,result); all_results.append(result)
            print(variant,seed,'external_accuracy',result['accuracy'],flush=True)
            del model,state
    from statistics import mean,stdev
    summary=[]
    for variant in variants:
        selected=[r for r in all_results if r['variant']==variant]
        row=dict(variant=variant)
        for key in ('accuracy','macro_f1_present_classes','top3_accuracy'):
            values=[r[key] for r in selected]
            row[key+'_mean']=mean(values); row[key+'_sample_sd']=stdev(values)
        summary.append(row)
    save_json(OUT/'summary.json',dict(samples=len(rows),scenes=len(set(r['scene'] for r in rows)),
        class_support=dict(Counter(labels.tolist())),models=summary,results=all_results,
        input_protocol_sha256=sha256(OUT/'input_protocol.json'),
        metric_note='Accuracy over crops; macro-F1 averaged only across classes present in this subset, not directly comparable to GTSRB macro-F1'))
    make_examples(inputs,rows,all_results)
    parts=['## 补充实验 真实道路图像中的标志裁剪分类','',
        '使用GTSDB的600张有标注道路图像中的固定子集。按文件名排序选择前100张有标注图像，'
        '保留其中全部标志框，选择规则在模型推理前保存。类别ID与GTSRB的43类定义逐项对应。'
        '原数据ReadMe明确允许任意用途；图片来自固定版本的公开镜像，仅下载数据，未执行镜像代码。','',
        f'按标注框裁剪并执行统一预处理，排除与GTSRB训练、验证或测试输入完全相同的标准化图像后，'
        f'保留{len(rows)}张标志裁剪，来自{len(set(r["scene"] for r in rows))}张道路图像、'
        f'{len(set(labels.tolist()))}个有支持样本的类别。', '',
        '| 组别 | Accuracy % | 已出现类别Macro-F1 % | Top-3 Accuracy % |',
        '| --- | --- | --- | --- |']
    for row in summary:
        parts.append('| '+row['variant']+' | '+' | '.join(
            f'{100*row[key+"_mean"]:.2f} ± {100*row[key+"_sample_sd"]:.2f}'
            for key in ('accuracy','macro_f1_present_classes','top3_accuracy'))+' |')
    parts += ['', '表中为全部三个训练种子的均值±样本标准差。外部子集的Macro-F1仅对有真实支持样本的类别取平均，'
        '与原43类测试集的Macro-F1口径不同。模型只对已知标注框裁剪进行分类，未评价道路图像中的目标定位。', '',
        '![道路原图与人工标注框：仅对框内裁剪分类](external/road_scenes.png)', '',
        '道路原图展示固定C组种子11的首个错误样本所在场景。框来自数据集人工标注，'
        '框旁仅展示裁剪分类预测，不表示系统具备自动检测能力。','',
        '![固定规则选取的真实场景裁剪与预测](external/examples.png)', '',
        '图中按样本ID选取最先出现的四个正确与四个错误案例（若不足则全部），固定展示C组种子11。'
        '这是用于解释成功与失败的事后案例展示，不用于代表整体成功比例；全部样本均纳入上表。', '',
        'GTSDB是不同数据集，但公开资料与精确像素去重不足以证明它与GTSRB在实体标志或相邻视频帧层面完全独立。'
        '因此，本节作为额外真实场景案例和分布变化分析，不称为严格独立的泛化基准；'
        '标注框裁剪方式也与原GTSRB图片中的背景范围不同。后续若要验证真正跨域性能，仍需采集独立地点和时间的数据。', '',
        '来源与逐图校验值见`report/external/source_manifest.json`；裁剪坐标、排除规则与输入哈希见'
        '`input_protocol.json`；逐模型指标和逐样本概率全部保留。','']
    (OUT/'section.md').write_text('\n'.join(parts),encoding='utf-8')


def make_examples(inputs,rows,results):
    import json
    with np.load(OUT/'C_11.npz') as raw:
        predicted=raw['predicted']; labels=raw['labels']
    selected=list(np.flatnonzero(predicted==labels)[:4])+list(np.flatnonzero(predicted!=labels)[:4])
    sheet=Image.new('RGB',(4*250,2*205),'white'); draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    cases=[]
    for cell,index in enumerate(selected):
        row,col=divmod(cell,4); x,y=col*250,row*205
        crop=Image.fromarray(inputs[index].transpose(1,2,0)).resize((144,144))
        sheet.paste(crop,(x+50,y+5))
        draw.text((x+10,y+152),f'{rows[index]["scene"]}  true={labels[index]}',font=font,fill='black')
        draw.text((x+10,y+176),f'C seed11 prediction={predicted[index]}',font=font,
                  fill='#007a44' if predicted[index]==labels[index] else '#ba2525')
        cases.append(dict(**rows[index],predicted=int(predicted[index])))
    sheet.save(OUT/'examples.png'); save_json(OUT/'examples.json',cases)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--external-root',type=Path,default=Path('../../work/gtsdb_external'))
    parser.add_argument('--gtsrb-root',type=Path,default=Path('../../work/gtsrb'))
    args=parser.parse_args(); evaluate(args.external_root,args.gtsrb_root)
