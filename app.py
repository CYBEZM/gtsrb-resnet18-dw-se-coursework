"""Local, CPU-default demonstration; does not interfere with GPU training."""
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import torch

from infer import load_model, predict
from src.data import canonical_image, corrupt

st.set_page_config(page_title='交通标志识别', layout='wide')
st.title('交通标志识别与退化对比')
st.write('适用于已裁剪的 GTSRB 43 类交通标志图片。预测概率未做置信度校准。')
torch.set_num_threads(4)
paths = sorted(Path('runs').glob('*/*/best.ckpt'))
if not paths:
    st.info('训练产生 best.ckpt 后可使用本页面。')
    st.stop()
choice = st.sidebar.selectbox('模型检查点', [str(p) for p in paths])
checkpoint_path = Path(choice)

@st.cache_resource
def cached_model(path, modified):
    return load_model(path)[0]

model = cached_model(choice, checkpoint_path.stat().st_mtime_ns)
upload = st.file_uploader('上传交通标志图片', type=['png', 'jpg', 'jpeg', 'ppm', 'bmp'])
kind_labels = {'原图': 'clean', '亮度降低': 'brightness', '运动模糊': 'motion', '高斯噪声': 'noise',
               'JPEG压缩': 'jpeg', '降低分辨率': 'resolution'}
kind_name = st.sidebar.selectbox('退化类型', list(kind_labels))
severity = st.sidebar.slider('强度', 1, 3, 1)
show_cam = st.sidebar.checkbox('显示 stage3 Grad-CAM', True)
if upload:
    with Image.open(upload) as opened:
        original = canonical_image(opened)
    image = original
    kind = kind_labels[kind_name]
    if kind != 'clean':
        pixels = np.asarray(original).transpose(2, 0, 1)
        image = Image.fromarray(corrupt(pixels, kind, severity, upload.name).transpose(1, 2, 0))
    top, canonical, heatmap = predict(model, image, show_cam)
    columns = st.columns(3 if show_cam else 2)
    columns[0].image(original.resize((240, 240)), caption='原始输入')
    columns[1].image(canonical.resize((240, 240)), caption='当前识别输入')
    if show_cam:
        columns[2].image(heatmap.resize((240, 240)), caption='stage3 Grad-CAM')
    st.dataframe([{'类别ID': r['class_id'], '类别': r['name'], '预测概率': r['probability']} for r in top], hide_index=True)
    st.caption('热力图用于辅助观察关注区域；合成退化结果不等同于真实道路部署验证。')
