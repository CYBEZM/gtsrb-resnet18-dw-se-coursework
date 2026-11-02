"""Show supplied GTSDB boxes on road scenes; these are not detector outputs."""
from pathlib import Path
import argparse
from PIL import Image,ImageDraw,ImageFont
import numpy as np
from src.utils import read_json,save_json,sha256


def make(external_root):
    root=Path(external_root);out=Path('report/external')
    plan=read_json(out/'input_protocol.json')
    with np.load(out/'C_11.npz') as raw:predicted=raw['predicted'];truth=raw['labels']
    chosen=[int(np.flatnonzero(predicted!=truth)[0])]
    scenes=list(dict.fromkeys(plan['samples'][i]['scene'] for i in chosen))
    sheet=Image.new('RGB',(1200,770*len(scenes)),'white')
    draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',26)
    records=[]
    for row,name in enumerate(scenes):
        scene=Image.open(root/'scenes'/name).convert('RGB')
        scale=min(1180/scene.width,700/scene.height)
        image=scene.resize((round(scene.width*scale),round(scene.height*scale)))
        x,y=10,row*770+45;sheet.paste(image,(x,y))
        draw.text((10,row*770+8),f'{name} - supplied annotation boxes; C seed11 crop classification',font=font,fill='black')
        cases=[]
        for i,sample in enumerate(plan['samples']):
            if sample['scene']!=name:continue
            left,top,right,bottom=sample['box_inclusive']
            box=(x+left*scale,y+top*scale,x+(right+1)*scale,y+(bottom+1)*scale)
            color='#008044' if predicted[i]==truth[i] else '#c62020'
            draw.rectangle(box,outline=color,width=3)
            draw.text((box[0],max(y,box[1]-24)),f'GT {truth[i]} / pred {predicted[i]}',font=font,fill=color,stroke_width=1,stroke_fill='white')
            cases.append(dict(sample_id=sample['sample_id'],box_inclusive=sample['box_inclusive'],
                              class_id=int(truth[i]),predicted=int(predicted[i])))
        records.append(dict(scene=name,scene_sha256=sha256(root/'scenes'/name),cases=cases))
    sheet.save(out/'road_scenes.png')
    save_json(out/'road_scenes.json',dict(model='C/11',selection='First erroneous crop scene in sample order',
        note='Boxes are ground truth, not predictions from an object detector',scenes=records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--external-root',default='../../work/gtsdb_external')
    make(p.parse_args().external_root)
