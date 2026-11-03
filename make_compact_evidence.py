"""Measured component analysis and two compact, white-background case figures.

No training or checkpoint selection. Existing formal results remain untouched.
"""
import argparse
import csv
import math
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image, ImageDraw
import torch

from infer import load_model
from make_model_comparisons import gradcam, font, center, pixels_hash
from review_data import ROOT, VARIANTS, SEEDS
from src.models import TrafficResNet
from src.data import canonical_image, corrupt
from src.utils import read_json, save_json, sha256

OUT=ROOT/'report/integrated'


def receptive_fields(standard=False):
    """Theoretical longest convolutional path; no SE/global dependencies."""
    model=TrafficResNet('A'); side=48; receptive=1; jump=1; trace=[]
    def apply(kernel,stride,padding):
        nonlocal side,receptive,jump
        side=(side+2*padding-kernel)//stride+1
        receptive+=(kernel-1)*jump; jump*=stride
    if standard:
        apply(7,2,3); apply(3,2,1)
    else:
        apply(3,1,1)
    trace.append(dict(component='stem',size=side,receptive_field=receptive,jump=jump))
    for i,stage in enumerate(model.stages):
        for block in stage:
            for layer in block.branch.modules():
                if isinstance(layer,torch.nn.Conv2d):
                    apply(layer.kernel_size[0],layer.stride[0],layer.padding[0])
        trace.append(dict(component=f'stage{i+1}',size=side,receptive_field=receptive,jump=jump))
    return trace


def component_analysis():
    models={}; sources=set()
    for v in VARIANTS:
        model=TrafficResNet(v)
        c=read_json(ROOT/f'runs/{v}/11/complexity.json'); sources.add(f'runs/{v}/11/complexity.json')
        groups={name:sum(p.numel() for p in module.parameters()) for name,module in
                [('stem',model.stem)]+[(f'stage{i+1}',s) for i,s in enumerate(model.stages)]+[('fc',model.fc)]}
        assert sum(groups.values())==c['parameters']
        models[v]=dict(parameters=c['parameters'],parameters_by_component=groups,
            conv_linear_macs=c['conv_linear_macs'],stage_macs=c['stage_macs'],
            stage12_parameter_percent=100*(groups['stage1']+groups['stage2'])/c['parameters'],
            stage12_macs_percent=100*sum(c['stage_macs'][:2])/c['conv_linear_macs'])
    validation=[]
    for v in VARIANTS:
        for s in SEEDS:
            path=f'runs/{v}/{s}/log.csv'; sources.add(path)
            with (ROOT/path).open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
            changes=[100*(float(rows[i]['val_macro_f1'])-float(rows[i-1]['val_macro_f1'])) for i in range(5,len(rows))]
            validation.append(dict(variant=v,seed=s,max_f1_drop_pp=min(changes),
                count_drop_over_10pp=sum(x < -10 for x in changes),last_train_loss=float(rows[-1]['train_loss'])))
    save_json(OUT/'component_analysis.json',dict(status='PASS',models=models,
        receptive_field_adapted=receptive_fields(),receptive_field_standard=receptive_fields(True),
        receptive_field_scope='Analytic longest Conv/Pool path on 48x48; original stem is a shape calculation, not a trained model; excludes global SE dependencies',
        validation_events=validation,
        idealized_arithmetic_intensity=dict(kernel=3,channels=64,bytes_per_fp32=4,
            dw_flops_per_byte=3**2/4,standard_flops_per_byte=3**2*64/4,
            scope='2 FLOPs/MAC; each input/output element transferred once; weights/cache/fusion ignored; not a hardware measurement'),
        sources=[dict(path=p,sha256=sha256(ROOT/p)) for p in sorted(sources)],
        model_source_sha256=sha256(ROOT/'src/models.py'),generator_sha256=sha256(__file__)))


def compact_cams(data_root):
    # Preserve the prior prespecified validation examples, take two by fixed index.
    examples=read_json(ROOT/'report/data/dataset_summary.json')['visual_examples'][2:4]
    cell=240; canvas=Image.new('RGB',(7*cell,510),'white'); d=ImageDraw.Draw(canvas)
    for col,title in enumerate(['干净输入','降至12×12']+list(VARIANTS)):
        center(d,col*cell+cell//2,14,title,29)
    cases=[]; sources=set()
    for i,row in enumerate(examples):
        path=data_root/row['path']
        with Image.open(path) as source: clean=canonical_image(source)
        array=np.asarray(clean).transpose(2,0,1).copy()
        degraded=corrupt(array,'resolution',3,row['path'])
        changed=Image.fromarray(degraded.transpose(1,2,0))
        y=65+i*220
        for col,image in enumerate([clean,changed]):
            canvas.paste(image.resize((126,126),Image.Resampling.NEAREST),(col*cell+57,y))
        center(d,cell//2,y+144,f"真值 {row['label']}",25)
        center(d,cell+cell//2,y+144,'统一退化输入',25)
        cases.append(dict(sample_id=row['path'],label=row['label'],original_sha256=sha256(path),
            degraded_chw_sha256=pixels_hash(degraded),models={}))
    for col,v in enumerate(VARIANTS,2):
        path=ROOT/f'runs/{v}/11/best.ckpt'; sources.add(path)
        model,_=load_model(path)
        for i,row in enumerate(examples):
            with Image.open(data_root/row['path']) as source: clean=canonical_image(source)
            arr=np.asarray(clean).transpose(2,0,1).copy(); arr=corrupt(arr,'resolution',3,row['path'])
            image=Image.fromarray(arr.transpose(1,2,0)); overlay,record=gradcam(model,image)
            record['correct']=record['class_id']==row['label']; cases[i]['models'][v]=record
            y=65+i*220; canvas.paste(overlay.resize((126,126),Image.Resampling.NEAREST),(col*cell+57,y))
            cc='#25704b' if record['correct'] else '#a64225'
            center(d,col*cell+cell//2,y+144,f"预测 {record['class_id']}",25,cc)
            center(d,col*cell+cell//2,y+177,f"{100*record['probability']:.1f}%",24,cc)
        del model
    canvas.save(OUT/'compact_cams.png')
    return cases,sources


def road_composite(external_root):
    plan=read_json(ROOT/'report/external/input_protocol.json')
    source_path=ROOT/'report/external/C_11.npz'
    with np.load(source_path) as saved:
        ids=saved['sample_ids'].copy(); labels=saved['labels'].copy(); predicted=saved['predicted'].copy()
    ordered=sorted(range(len(ids)),key=lambda i:str(ids[i]))
    chosen=[i for i in ordered if predicted[i]==labels[i]][:2]+[i for i in ordered if predicted[i]!=labels[i]][:2]
    first_error=next(i for i in ordered if predicted[i]!=labels[i]); scene_name=plan['samples'][first_error]['scene']
    path=external_root/'scenes'/scene_name
    with Image.open(path) as source: scene=source.convert('RGB')
    # Left annotated road scene, right two successes and two failures.
    canvas=Image.new('RGB',(1800,650),'white'); d=ImageDraw.Draw(canvas)
    center(d,510,12,'道路原图与人工标注框',30)
    display=scene.copy(); display.thumbnail((1000,555)); origin=(10,60); canvas.paste(display,origin)
    sx,sy=display.width/scene.width,display.height/scene.height
    for row in plan['samples']:
        if row['scene']!=scene_name:continue
        l,t,r,b=row['box_inclusive']; d.rectangle((10+l*sx,60+t*sy,10+(r+1)*sx,60+(b+1)*sy),outline='#a64225',width=3)
    cases=[]
    for j,i in enumerate(chosen):
        row=plan['samples'][i]; raw_path=external_root/'scenes'/row['scene']
        with Image.open(raw_path) as source:
            l,t,r,b=row['box_inclusive']; image=canonical_image(source.crop((l,t,r+1,b+1)))
        assert pixels_hash(np.asarray(image).transpose(2,0,1).copy())==row['canonical_sha256']
        x=1060+(j%2)*360; y=55+(j//2)*295
        canvas.paste(image.resize((165,165),Image.Resampling.NEAREST),(x+80,y))
        color='#25704b' if predicted[i]==labels[i] else '#a64225'
        center(d,x+162,y+180,f'真值 {labels[i]} / 预测 {predicted[i]}',26,color)
        center(d,x+162,y+218,str(ids[i]),22)
        cases.append(dict(sample_id=str(ids[i]),label=int(labels[i]),predicted=int(predicted[i]),
            canonical_sha256=row['canonical_sha256'],scene_sha256=sha256(raw_path)))
    canvas.save(OUT/'road_composite.png')
    return dict(model='C/11',selection='First two correct and first two erroneous sample IDs; left scene contains first error',
        scene=scene_name,scene_sha256=sha256(path),cases=cases),{source_path,ROOT/'report/external/input_protocol.json'}


def main(data_root,external_root):
    OUT.mkdir(exist_ok=True,parents=True); torch.set_num_threads(2)
    component_analysis(); cams,sources=compact_cams(data_root); roads,other=road_composite(external_root)
    sources|=other
    save_json(OUT/'compact_sources.json',dict(status='PASS',validation_cases=cams,road_cases=roads,
        selection='Prespecified validation example indices 2,3 (construction / keep right), fixed severity3 resolution, seed11; not selected by predicted outcomes',
        inference='New CPU FP32 Grad-CAM inference on validation images only; original formal metrics untouched',
        target='Each model explains its own top1 logit at stage3; uncalibrated probabilities',
        sources=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha256(p)) for p in sorted(sources)],
        generator_sha256=sha256(__file__),
        outputs=[dict(path=f'report/integrated/{name}.png',sha256=sha256(OUT/f'{name}.png')) for name in ['compact_cams','road_composite']]))
    print('PASS: component counts, receptive fields, two compact evidence figures.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--external-root',type=Path,required=True); args=parser.parse_args()
    main(args.data_root.resolve(),args.external_root.resolve())
