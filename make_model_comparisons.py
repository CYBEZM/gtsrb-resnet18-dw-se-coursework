"""Actual same-input model comparisons; deterministic cases and source hashes.

Prediction labels in test panels come directly from saved FP32 evaluation NPZs.
Only the four prespecified validation Grad-CAM examples require new CPU inference.
No training, checkpoint selection or metric files are changed.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from infer import load_model
from src.data import canonical_image, normalize, corrupt
from src.labels import CLASS_NAMES_ZH
from src.utils import read_json, save_json, sha256

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'report/integrated'
VARIANTS='ABCDE'
FONT='C:/Windows/Fonts/simhei.ttf'
CELL=280


def font(size=25):
    return ImageFont.truetype(FONT,size)


def center(draw, x, y, value, size=25, color='#222222'):
    draw.text((x,y),str(value),font=font(size),fill=color,anchor='mt')


def label(draw,x,y,value,size=24):
    parts=[value[i:i+11] for i in range(0,len(value),11)]
    for j,part in enumerate(parts[:2]): center(draw,x,y+26*j,part,size)


def pixels_hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_predictions(condition,sources):
    result={}
    for v in VARIANTS:
        path=ROOT/f'runs/{v}/11/predictions/{condition}.npz'
        with np.load(path,allow_pickle=False) as data:
            result[v]={k:data[k].copy() for k in data.files}
        sources.add(path)
        if v!='A':
            assert np.array_equal(result[v]['sample_ids'],result['A']['sample_ids'])
            assert np.array_equal(result[v]['labels'],result['A']['labels'])
        assert np.array_equal(result[v]['probabilities'].argmax(1),result[v]['predicted'])
    return result


def model_record(data,index):
    pred=int(data['predicted'][index]); truth=int(data['labels'][index])
    return dict(class_id=pred,class_name=CLASS_NAMES_ZH[pred],
                probability=float(data['probabilities'][index,pred]),correct=pred==truth)


def prediction_cell(canvas,draw,col,top,image,record):
    x=col*CELL+CELL//2
    canvas.paste(image.resize((138,138),Image.Resampling.NEAREST),(x-69,top+8))
    color='#25704b' if record['correct'] else '#a64225'
    label(draw,x,top+155,f"{record['class_id']} {record['class_name']}",23)
    center(draw,x,top+211,f"{record['probability']*100:.1f}%  {'正确' if record['correct'] else '错误'}",23,color)


def test_cases(data_root,sources):
    data=load_predictions('clean_0',sources); y=data['A']['labels']
    specs=[('C修正A的错误','A','C',False),('C引入的新错误','A','C',True),
           ('E修正B的错误','B','E',False),('E引入的新错误','B','E',True)]
    rows=[]; canvas=Image.new('RGB',(CELL*6,70+4*300),'white'); draw=ImageDraw.Draw(canvas)
    for col,title in enumerate(['同一输入 / 真值']+list(VARIANTS)):
        center(draw,col*CELL+CELL//2,15,title,30)
    for j,(title,ref,other,reference_correct) in enumerate(specs):
        mask=(data[ref]['predicted']==y)==reference_correct
        mask&=((data[other]['predicted']==y)!=reference_correct)
        indices=np.flatnonzero(mask)
        assert len(indices)>0
        index=min(indices,key=lambda i:str(data['A']['sample_ids'][i]))
        sample_id=str(data['A']['sample_ids'][index]); path=data_root/sample_id
        with Image.open(path) as original: image=canonical_image(original)
        top=70+j*300
        center(draw,CELL//2,top,title,24)
        canvas.paste(image.resize((138,138),Image.Resampling.NEAREST),(CELL//2-69,top+38))
        label(draw,CELL//2,top+187,f'真值 {int(y[index])} {CLASS_NAMES_ZH[int(y[index])]}',23)
        center(draw,CELL//2,top+248,Path(sample_id).name,22)
        records={v:model_record(data[v],index) for v in VARIANTS}
        for col,v in enumerate(VARIANTS,1): prediction_cell(canvas,draw,col,top+20,image,records[v])
        draw.line((15,top+290,CELL*6-15,top+290),fill='#dedede')
        rows.append(dict(case=title,selection='lexicographically first seed11 sample satisfying the paired outcome',
            eligible_samples=int(len(indices)),sample_id=sample_id,label=int(y[index]),
            original_sha256=sha256(path),canonical_hwc_sha256=pixels_hash(np.asarray(image)),models=records))
    canvas.save(OUT/'paired_cases.png')
    return rows


def corruption_comparison(data_root,sources):
    conditions=['clean_0','brightness_3','motion_3','noise_3','jpeg_3','resolution_3']
    all_data={c:load_predictions(c,sources) for c in conditions}
    clean=all_data['clean_0']; y=clean['A']['labels']
    assert all(np.array_equal(d['A']['sample_ids'],clean['A']['sample_ids']) for d in all_data.values())
    eligible=np.ones(len(y),dtype=bool)
    for v in VARIANTS: eligible&=clean[v]['predicted']==y
    disagreement=np.zeros(len(y),dtype=bool)
    for c in conditions[1:]:
        preds=np.stack([all_data[c][v]['predicted'] for v in VARIANTS])
        disagreement|=(preds!=preds[0]).any(0)
    candidates=np.flatnonzero(eligible & disagreement)
    index=min(candidates,key=lambda i:str(clean['A']['sample_ids'][i]))
    sample_id=str(clean['A']['sample_ids'][index]); path=data_root/sample_id
    with Image.open(path) as image: canonical=canonical_image(image)
    array=np.asarray(canonical).transpose(2,0,1).copy()
    canvas=Image.new('RGB',(CELL*6,885),'white'); draw=ImageDraw.Draw(canvas)
    names=['干净输入','亮度×0.2','模糊核长9','噪声σ=0.12','JPEG Q=10','降至12×12']
    rows=[]
    for col,(condition,title) in enumerate(zip(conditions,names)):
        kind,severity=condition.rsplit('_',1)
        changed=array if kind=='clean' else corrupt(array,kind,int(severity),sample_id)
        center(draw,col*CELL+CELL//2,18,title,28)
        image=Image.fromarray(changed.transpose(1,2,0))
        canvas.paste(image.resize((168,168),Image.Resampling.NEAREST),(col*CELL+56,66))
        models={v:model_record(all_data[condition][v],index) for v in VARIANTS}
        for j,v in enumerate(VARIANTS):
            rec=models[v]; yy=264+j*116; cc='#25704b' if rec['correct'] else '#a64225'
            center(draw,col*CELL+CELL//2,yy,f"{v}: {rec['class_id']} ({rec['probability']*100:.1f}%)",26,cc)
            label(draw,col*CELL+CELL//2,yy+34,rec['class_name'],23)
        rows.append(dict(condition=condition,canonical_chw_sha256=pixels_hash(changed),models=models))
    canvas.save(OUT/'same_image_corruptions.png')
    return dict(sample_id=sample_id,label=int(y[index]),class_name=CLASS_NAMES_ZH[int(y[index])],
        selection='First sample ID correct for every seed11 model on clean input and showing model disagreement in at least one level3 corruption',
        eligible_samples=int(len(candidates)),original_sha256=sha256(path),conditions=rows)


def gradcam(model,image):
    canonical=canonical_image(image); x=torch.from_numpy(np.asarray(canonical).transpose(2,0,1).copy())
    x=normalize(x.unsqueeze(0).float()/255); features=[]
    def hook(module,inputs,output):
        output.retain_grad(); features.append(output)
    handle=model.stages[2].register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        logits=model(x); prob=logits.softmax(1)[0]; target=int(prob.argmax())
        logits[0,target].backward(); feature=features[0]
        cam=torch.relu((feature.grad.mean((2,3),keepdim=True)*feature).sum(1,keepdim=True))
        cam=torch.nn.functional.interpolate(cam,(48,48),mode='bilinear',align_corners=False)[0,0].detach().numpy()
        cam/=max(float(cam.max()),1e-8)
        # Zero activation preserves the photograph; orange denotes positive attribution.
        alpha=(cam*.6)[...,None]; rgb=np.asarray(canonical).astype(float)
        overlay=Image.fromarray(np.uint8(np.clip(rgb*(1-alpha)+np.array([255,99,0])*alpha,0,255)))
        return overlay,dict(class_id=target,class_name=CLASS_NAMES_ZH[target],probability=float(prob[target].detach()),
                            target='predicted top1 logit',activation_shape=list(feature.shape),heatmap_max=float(cam.max()))
    finally:
        handle.remove()


def validation_cams(data_root,sources):
    examples=read_json(ROOT/'report/data/dataset_summary.json')['visual_examples']
    images=[]; rows=[]
    for row in examples:
        with Image.open(data_root/row['path']) as image: images.append(image.convert('RGB').copy())
        rows.append(dict(sample_id=row['path'],label=row['label'],original_sha256=sha256(data_root/row['path']),models={}))
    canvas=Image.new('RGB',(CELL*6,70+len(rows)*265),'white'); draw=ImageDraw.Draw(canvas)
    for col,title in enumerate(['固定验证输入']+list(VARIANTS)): center(draw,col*CELL+CELL//2,14,title,30)
    for j,(row,image) in enumerate(zip(rows,images)):
        yy=70+j*265; canonical=canonical_image(image)
        canvas.paste(canonical.resize((138,138),Image.Resampling.NEAREST),(71,yy+8))
        label(draw,CELL//2,yy+155,f"真值 {row['label']} {CLASS_NAMES_ZH[row['label']]}",23)
    for col,v in enumerate(VARIANTS,1):
        checkpoint=ROOT/f'runs/{v}/11/best.ckpt'; sources.add(checkpoint)
        model,_=load_model(checkpoint)
        for j,(row,image) in enumerate(zip(rows,images)):
            overlay,record=gradcam(model,image); record['correct']=record['class_id']==row['label']
            row['models'][v]=record
            prediction_cell(canvas,draw,col,70+j*265,overlay,record)
        del model
        print('Grad-CAM',v,flush=True)
    canvas.save(OUT/'validation_cams.png')
    return rows


def preprocessing(data_root):
    row=read_json(ROOT/'report/data/dataset_summary.json')['visual_examples'][-1]
    with Image.open(data_root/row['path']) as original: image=original.convert('RGB')
    a=np.asarray(image); h,w=a.shape[:2]; size=max(h,w)
    padded=Image.fromarray(np.pad(a,(((size-h)//2,size-h-(size-h)//2),((size-w)//2,size-w-(size-w)//2),(0,0)),mode='edge'))
    canvas=Image.new('RGB',(1260,315),'white'); draw=ImageDraw.Draw(canvas)
    for i,(picture,title) in enumerate([(image,f'原图 {w}×{h}'),(padded,'边缘像素补成正方形'),(canonical_image(image),'双线性缩放 48×48')]):
        x=210+i*420
        scale=min(180/picture.width,180/picture.height)
        resized=picture.resize((round(picture.width*scale),round(picture.height*scale)),Image.Resampling.NEAREST)
        canvas.paste(resized,(x-resized.width//2,35+(180-resized.height)//2)); center(draw,x,246,title,27)
        if i<2:
            draw.line((x+125,125,x+290,125),fill='#333333',width=3)
            draw.polygon([(x+290,125),(x+278,118),(x+278,132)],fill='#333333')
    canvas.save(OUT/'preprocessing.png')


def main(root):
    OUT.mkdir(parents=True,exist_ok=True); sources=set(); torch.set_num_threads(2)
    rows=test_cases(root,sources); corruptions=corruption_comparison(root,sources)
    cams=validation_cams(root,sources); preprocessing(root)
    manifest_path=root/'cache48/manifest.json'; manifest=read_json(manifest_path)
    assert sha256(manifest_path)==read_json(ROOT/'report/data/dataset_summary.json')['manifest_sha256']
    indices={row['path']:i for i,row in enumerate(manifest['test_rows'])}
    cache=np.load(root/'cache48/test_images.npy',mmap_mode='r')
    for row in rows:
        index=indices[row['sample_id']]
        assert pixels_hash(cache[index].transpose(1,2,0).copy())==row['canonical_hwc_sha256']
    index=indices[corruptions['sample_id']]
    for row in corruptions['conditions']:
        kind,severity=row['condition'].rsplit('_',1)
        original=cache if kind=='clean' else np.load(root/f'cache48/test_{kind}_{severity}.npy',mmap_mode='r')
        assert pixels_hash(original[index])==row['canonical_chw_sha256'],row['condition']
    save_json(OUT/'comparison_sources.json',dict(status='PASS',seed=11,
        illustrated_test_pixels_match_original_evaluation_cache=True,
        probability_note='uncalibrated top1 probabilities; test panels use original saved evaluation arrays',
        paired_cases=rows,corruption_case=corruptions,validation_cams=cams,
        cam_note='stage3 (12x12), each model explains its own predicted class; orange positive attribution, no causal interpretation',
        generator_sha256=sha256(__file__),
        sources=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha256(p)) for p in sorted(sources)],
        outputs=[dict(path=(OUT/name).relative_to(ROOT).as_posix(),sha256=sha256(OUT/name))
                 for name in ['paired_cases.png','same_image_corruptions.png','validation_cams.png','preprocessing.png']]))
    print('Generated four real-input comparison figures.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--data-root',type=Path,required=True)
    main(parser.parse_args().data_root.resolve())
