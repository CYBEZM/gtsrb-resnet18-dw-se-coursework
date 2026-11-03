"""Three explanatory figures grounded in the frozen data and implemented modules.

Real images illustrate the task/split; drawn feature planes are conceptual.
No inference, training, or modification of original measurements is performed.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from review_data import ROOT
from src.data import canonical_image, corrupt
from src.utils import read_json, save_json, sha256

OUT=ROOT/'report/integrated'
INK='#292929'
GRAY='#777777'
ACCENT='#9c572e'
FONT='C:/Windows/Fonts/simhei.ttf'


def font(size=34):
    return ImageFont.truetype(FONT,size)


def text(draw,x,y,value,size=34,anchor='mt',color=INK):
    lines=value.split('\n'); height=len(lines)*(size+10)-10
    top=y-height/2 if anchor=='mm' else y
    for j,line in enumerate(lines):
        draw.text((x,top+j*(size+10)),line,font=font(size),fill=color,anchor='mt')


def box(draw,bounds,label,size=34):
    draw.rectangle(bounds,outline=INK,width=2)
    x1,y1,x2,y2=bounds
    text(draw,(x1+x2)/2,(y1+y2)/2,label,size,anchor='mm')


def arrow(draw,start,end,color=GRAY,width=3):
    draw.line([start,end],fill=color,width=width)
    v=np.asarray(end,dtype=float)-start; v/=max(float(np.linalg.norm(v)),1)
    normal=np.array([-v[1],v[0]])
    a=np.asarray(end)-v*15+normal*7; b=np.asarray(end)-v*15-normal*7
    draw.polygon([tuple(end),tuple(a),tuple(b)],fill=color)


def fit(canvas,image,bounds):
    x,y,w,h=bounds; ratio=min(w/image.width,h/image.height)
    im=image.resize((round(image.width*ratio),round(image.height*ratio)),Image.Resampling.NEAREST)
    offset=(round(x+(w-im.width)/2),round(y+(h-im.height)/2))
    canvas.paste(im,offset)
    return offset,im.size


def open_rgb(path):
    with Image.open(path) as image:
        return image.convert('RGB')


def task_scene(external_root):
    plan_path=ROOT/'report/external/input_protocol.json'
    plan=read_json(plan_path)
    compact=read_json(OUT/'compact_sources.json')
    # Reuse the scene already documented in the road section; select its first box.
    name=compact['road_cases']['scene']
    row=next(row for row in plan['samples'] if row['scene']==name)
    path=external_root/'scenes'/name; scene=open_rgb(path)
    l,t,r,b=row['box_inclusive']
    crop=canonical_image(scene.crop((l,t,r+1,b+1)))
    pixels=np.asarray(crop).transpose(2,0,1).copy()
    assert sha256(path)==compact['road_cases']['scene_sha256']
    import hashlib
    assert hashlib.sha256(pixels.tobytes()).hexdigest()==row['canonical_sha256']
    changed=[pixels,corrupt(pixels,'brightness',3,row['sample_id']),
             corrupt(pixels,'motion',3,row['sample_id'])]
    canvas=Image.new('RGB',(1800,660),'white'); d=ImageDraw.Draw(canvas)
    text(d,300,22,'真实道路与标志裁剪',39)
    text(d,910,22,'相同标志的输入条件',39)
    text(d,1500,22,'结构研究与评价',39)
    offset,size=fit(canvas,scene,(25,100,550,345))
    sx,sy=size[0]/scene.width,size[1]/scene.height
    d.rectangle((offset[0]+l*sx,offset[1]+t*sy,offset[0]+(r+1)*sx,offset[1]+(b+1)*sy),
                outline=ACCENT,width=4)
    text(d,300,470,'框来自数据集人工标注\n任务：已知框内的43类识别',32)
    arrow(d,(585,272),(635,272))
    for j,(array,label) in enumerate(zip(changed,['清晰','低亮度','运动模糊'])):
        x=650+j*175; image=Image.fromarray(array.transpose(1,2,0))
        canvas.paste(image.resize((145,145),Image.Resampling.NEAREST),(x,176))
        text(d,x+72,338,label,31)
    text(d,905,418,'保留轮廓、颜色与内部图案\n退化会改变细节可辨性',31)
    arrow(d,(1188,272),(1240,272))
    box(d,(1260,117,1740,297),'A 标准     B 全DW     C SE\nD DW+SE     E 深层DW',33)
    arrow(d,(1500,309),(1500,368))
    box(d,(1260,383,1740,553),'分类精度 / 参数与MACs\n退化表现 / GPU延迟',33)
    text(d,900,592,'统一输入与训练规则 → 比较精度、资源和场景适配的取舍',34)
    canvas.save(OUT/'task_scene.png')
    return dict(scene=name,scene_sha256=sha256(path),sample_id=row['sample_id'],
                box_inclusive=row['box_inclusive'],canonical_sha256=row['canonical_sha256'],
                label=row['class_id'],conditions=['clean_0','brightness_3','motion_3'],
                note='Illustration only; supplied annotation, no detector output or newly measured predictions')


def grouping(data_root):
    path=data_root/'cache48/manifest.json'; m=read_json(path)
    assert sha256(path)==read_json(ROOT/'report/data/dataset_summary.json')['manifest_sha256']
    canvas=Image.new('RGB',(1800,800),'white'); d=ImageDraw.Draw(canvas)
    text(d,900,15,'按完整(class_id, track_id)分组，同一序列的帧进入同一子集',36)
    groups=[]
    for j,(label,indices) in enumerate([('训练子集',m['train_indices']),('验证子集',m['val_indices'])]):
        rows=[m['train_rows'][i] for i in indices if m['train_rows'][i]['label']==1]
        track=min(row['track'] for row in rows)
        chosen=sorted((r for r in rows if r['track']==track),key=lambda r:r['path'])[:3]
        x=25+j*900
        d.rectangle((x,75,x+850,355),outline=GRAY,width=2)
        text(d,x+425,90,f'类别1 / track {track.split("/")[1]} → {label}',32)
        source=[]
        for k,row in enumerate(chosen):
            image=open_rgb(data_root/row['path']); canonical=canonical_image(image)
            px=x+100+k*260
            canvas.paste(canonical.resize((125,125),Image.Resampling.NEAREST),(px,153))
            text(d,px+62,292,Path(row['path']).name,25)
            source.append(dict(path=row['path'],original_sha256=sha256(data_root/row['path'])))
        groups.append(dict(subset=label,track=track,samples=source))
    text(d,900,380,'真实分组示例；中间帧不跨训练 / 验证',32)
    row=read_json(ROOT/'report/data/dataset_summary.json')['visual_examples'][-1]
    original=open_rgb(data_root/row['path']); array=np.asarray(original); h,w=array.shape[:2]; side=max(h,w)
    padded=Image.fromarray(np.pad(array,(((side-h)//2,side-h-(side-h)//2),
                                     ((side-w)//2,side-w-(side-w)//2),(0,0)),mode='edge'))
    for j,(image,label) in enumerate([(original,f'原图 {w}×{h}'),(padded,'边缘补方'),(canonical_image(original),'双线性缩放 48×48')]):
        cx=300+j*600
        fit(canvas,image,(cx-100,458,200,200))
        text(d,cx,682,label,34)
        if j<2:arrow(d,(cx+155,560),(cx+445,560))
    text(d,900,742,'固定归一化；同一预处理用于所有模型与种子',30)
    canvas.save(OUT/'grouping_preprocessing.png')
    return dict(manifest_sha256=sha256(path),selection='Class1, lexicographically first track and first three frames within each actual subset',
                groups=groups,preprocessing_sample=dict(path=row['path'],original_sha256=sha256(data_root/row['path'])))


def planes(draw,x,y,w=115,h=115,count=3):
    for i in reversed(range(count)):
        dx=i*17; dy=-i*12
        draw.rectangle((x+dx,y+dy,x+w+dx,y+h+dy),fill='white',outline=INK,width=2)
        draw.line((x+dx+20,y+dy+30,x+w+dx-18,y+dy+70),fill=GRAY,width=2)


def mechanism():
    canvas=Image.new('RGB',(1800,900),'white'); d=ImageDraw.Draw(canvas)
    text(d,900,10,'卷积如何处理空间与通道',39)
    text(d,205,70,'输入通道',32); text(d,1575,70,'输出通道',32)
    planes(d,125,155); planes(d,1510,155)
    arrow(d,(310,205),(580,205)); arrow(d,(1210,205),(1465,205))
    box(d,(600,120,1190,290),'标准3×3卷积\n联合学习空间与跨通道组合',35)
    text(d,900,308,'每个输入—输出通道对可学习独立空间核',32)
    planes(d,125,410); planes(d,1510,410)
    for k in range(3):
        yy=390+k*62
        arrow(d,(310,450),(480,yy+22))
        box(d,(500,yy,840,yy+46),f'DW 3×3：通道{k+1}',29)
        arrow(d,(860,yy+22),(980,450))
    box(d,(1000,385,1300,515),'PW 1×1\n混合通道',34)
    arrow(d,(1320,450),(1465,450))
    text(d,900,590,'DW：逐通道空间滤波 → PW：跨通道混合；两者之间不增加BN或激活',33)
    d.line((30,650,1770,650),fill='#bbbbbb',width=2)
    text(d,900,668,'SE：由全局通道摘要生成权重，再缩放主分支特征',35)
    labels=['特征U\nH×W×C','GAP\nC','FC + ReLU\nC/16','FC + Sigmoid\nC','通道权重\nC','通道缩放\nU × 权重']
    for j,label in enumerate(labels):
        x=30+j*295; box(d,(x,740,x+245,865),label,30)
        if j<5:arrow(d,(x+250,802),(x+287,802))
    canvas.save(OUT/'component_mechanisms.png')
    return dict(note='Conceptual feature planes, not measured activations; depth multiplier1; no BN/activation at DW/PW boundary',
                se_reduction=16,se_position='after final branch BN, before residual addition',
                source='src/models.py',source_sha256=sha256(ROOT/'src/models.py'))


def main(data_root,external_root):
    OUT.mkdir(exist_ok=True,parents=True)
    scene=task_scene(external_root)
    groups=grouping(data_root); modules=mechanism()
    sources=['report/data/dataset_summary.json','report/external/input_protocol.json',
             'report/integrated/compact_sources.json','src/models.py','src/data.py']
    save_json(OUT/'explanatory_sources.json',dict(status='PASS',task_scene=scene,grouping=groups,modules=modules,
        generator_sha256=sha256(__file__),sources=[dict(path=p,sha256=sha256(ROOT/p)) for p in sources],
        outputs=[dict(path=f'report/integrated/{name}.png',sha256=sha256(OUT/f'{name}.png'))
                 for name in ['task_scene','grouping_preprocessing','component_mechanisms']]))
    print('Generated three white-background explanatory figures from actual inputs and model code.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--external-root',type=Path,required=True); args=parser.parse_args()
    main(args.data_root.resolve(),args.external_root.resolve())
