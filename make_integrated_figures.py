"""White-background A-E diagrams/plots from immutable saved measurements.

Run separately from PyTorch inference to avoid loading two OpenMP runtimes.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, NullFormatter

from review_data import ROOT, VARIANTS, SEEDS, KINDS, collect, csv_rows, write_summary
from src.utils import save_json, sha256

OUT = ROOT/'report/integrated'
COLORS = ['#222222', '#bd5b25', '#28734e', '#965780', '#85731d']
MARKERS = ['o', 's', '^', 'D', 'v']


def save(fig, name):
    fig.savefig(OUT/f'{name}.png', dpi=210, facecolor='white', bbox_inches='tight')
    fig.savefig(OUT/f'{name}.svg', facecolor='white', bbox_inches='tight')
    plt.close(fig)


def box(ax, x, y, w, h, text, fontsize=11):
    ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor='#333333',linewidth=1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,linespacing=1.5)


def arrow(ax, a, b):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#444444',lw=1.2))


def diagrams(data):
    fig, ax=plt.subplots(figsize=(12,4.3)); ax.set(xlim=(0,12),ylim=(0,4.3)); ax.axis('off')
    for x,w,text in [(0.1,2.2,'交通标志图像\nGTSRB / 道路标注裁剪'),
                     (2.8,2.2,'统一输入\n边缘补方 + 48×48'),
                     (5.5,2.9,'结构对照\nA 标准  B DW  C SE\nD DW+SE  E 深层DW'),
                     (8.9,2.9,'固定训练规则\n种子11 / 22 / 33\n每次60轮，验证集选模')]:
        box(ax,x,2.5,w,1.45,text,11)
    for a,b in [(2.3,2.8),(5,5.5),(8.4,8.9)]: arrow(ax,(a,3.2),(b,3.2))
    arrow(ax,(10.35,2.5),(10.35,1.8)); arrow(ax,(10.35,1.8),(1.6,1.8))
    for x,w,text in [(0.1,3.0,'干净测试与复杂度\nAccuracy / F1 / 参数 / MACs'),
                     (3.8,3.9,'退化与逐图对照\n低亮度 / 模糊 / 噪声 / 压缩 / 降采样'),
                     (8.4,3.4,'设备与应用验证\nGPU计时 + 道路裁剪分类')]:
        arrow(ax,(x+w/2,1.8),(x+w/2,1.4)); box(ax,x,.2,w,1.2,text,11)
    save(fig,'workflow')

    fig, axes=plt.subplots(2,1,figsize=(12,4.7),gridspec_kw={'height_ratios':[1,1.4]})
    ax=axes[0]; ax.axis('off'); ax.set(xlim=(0,12),ylim=(0,2.3))
    labels=['输入\n3×48×48','入口3×3\n64×48×48','stage1 ×2\n64×48×48',
            'stage2 ×2\n128×24×24','stage3 ×2\n256×12×12','stage4 ×2\n512×6×6','GAP + FC\n43类']
    for i,label in enumerate(labels):
        x=.05+i*1.72; box(ax,x,.65,1.5,1.0,label,10)
        if i<6: arrow(ax,(x+1.5,1.15),(x+1.72,1.15))
    ax.text(6,2.05,'共同骨架：3×3入口、stride=1、无maxpool；每阶段两个BasicBlock',ha='center',fontsize=12)
    ax=axes[1]; ax.axis('off'); ax.set(xlim=(0,12),ylim=(.85,3.5))
    seq=[(.1,1.45,'输入 x'),(2.0,1.65,'卷积1\nBN → ReLU'),(4.1,1.65,'卷积2\nBN'),
         (6.2,1.55,'SE / 恒等\n相加前'),(8.2,1.2,'相加'),(10,1.7,'ReLU\n输出')]
    for x,w,text in seq: box(ax,x,1.6,w,1.05,text,11)
    for i in range(len(seq)-1): arrow(ax,(seq[i][0]+seq[i][1],2.125),(seq[i+1][0],2.125))
    ax.plot([.83,.83,8.8],[1.6,1.1,1.1],color='#444444',lw=1)
    arrow(ax,(8.8,1.1),(8.8,1.6)); ax.text(4.7,1.22,'shortcut：恒等，或1×1投影 + BN',ha='center',fontsize=10)
    ax.text(6,3.2,'B/D：替换全部残差主分支卷积；E：仅stage3/4；C/D：每块加入SE',ha='center',fontsize=12)
    fig.tight_layout(); save(fig,'architecture_v3')


def plots(data):
    fig,axes=plt.subplots(1,2,figsize=(11.8,3.7))
    for v,color in zip(VARIANTS,COLORS):
        logs=[csv_rows(f'runs/{v}/{s}/log.csv') for s in SEEDS]
        for ax,key,mult in [(axes[0],'val_macro_f1',100),(axes[1],'val_loss',1)]:
            a=np.array([[float(row[key])*mult for row in log] for log in logs]); m=a.mean(0); sd=a.std(0,ddof=1)
            ax.plot(np.arange(1,61),m,label=v,color=color,lw=1.4)
            ax.fill_between(np.arange(1,61),m-sd,np.minimum(m+sd,100) if mult==100 else m+sd,color=color,alpha=.08)
    axes[0].set(xlabel='训练轮次',ylabel='验证Macro-F1 (%)',ylim=(0,102))
    axes[1].set(xlabel='训练轮次',ylabel='验证交叉熵损失',yscale='log')
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
    axes[1].yaxis.set_minor_formatter(NullFormatter())
    axes[0].legend(ncol=5,loc='lower right'); fig.tight_layout(); save(fig,'learning_curves')

    fig,axes=plt.subplots(1,3,figsize=(12.5,4.0),gridspec_kw={'width_ratios':[1,1.12,1]})
    for v,color,marker in zip(VARIANTS,COLORS,MARKERS):
        m=data['models'][v]
        axes[0].errorbar(m['parameters']/1e6,m['accuracy_mean']*100,
            yerr=m['accuracy_sd']*100,fmt=marker,capsize=4,color=color,ms=8)
        axes[0].annotate(v,(m['parameters']/1e6,m['accuracy_mean']*100),xytext=(7,5),textcoords='offset points')
    axes[0].set(xlabel='参数量 (M)',ylabel='测试Accuracy (%)',xscale='log',ylim=(97.4,99.0),title='(a) 精度与存储取舍')
    axes[0].set_xticks([1.5,3,6,12]); axes[0].set_xticklabels(['1.5','3','6','12'])
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    contrasts=data['contrasts']; y=np.arange(len(contrasts))
    axes[1].axvline(0,color='#888888',lw=1,linestyle='--')
    for j,(seed,marker) in enumerate(zip(SEEDS,['o','s','^'])):
        axes[1].scatter([r['seed_differences_pp'][j] for r in contrasts],y+(j-1)*.12,
                        marker=marker,s=30,label=f'seed{seed}',color=COLORS[j+1])
    axes[1].scatter([r['mean_pp'] for r in contrasts],y,marker='|',s=130,c='black',label='配对均值')
    axes[1].set(yticks=y,yticklabels=[r['contrast'] for r in contrasts],
                xlabel='Accuracy差值 (pp)',title='(b) 相同种子配对差值',xlim=(-1.05,.55),ylim=(5.6,-.6))
    axes[1].legend(ncol=2,loc='lower left',fontsize=8)
    xs=np.arange(5); bottom=np.zeros(5)
    for i,hatch in enumerate(['///','...','xxx','\\\\']):
        heights=np.array([data['complexity'][v]['stage_macs'][i]/1e6 for v in VARIANTS])
        axes[2].bar(xs,heights,bottom=bottom,label=f'stage{i+1}',color='white',edgecolor=f'{.15+i*.15:.2f}',hatch=hatch,width=.65)
        bottom+=heights
    axes[2].set(xticks=xs,xticklabels=list(VARIANTS),ylabel='阶段Conv/Linear MACs (M)',title='(c) 分阶段计算量')
    axes[2].legend(ncol=2,fontsize=8); fig.tight_layout(w_pad=1.2); save(fig,'tradeoffs_v3')

    fig,axes=plt.subplots(2,3,figsize=(12,6.3)); lookup={(r['variant'],r['kind'],r['severity']):r for r in data['robustness']}
    for ax,kind in zip(axes.flat,KINDS):
        for v,color,marker in zip(VARIANTS,COLORS,MARKERS):
            rows=[dict(accuracy_mean=data['models'][v]['accuracy_mean'],accuracy_sd=data['models'][v]['accuracy_sd'])]+[lookup[v,kind,s] for s in (1,2,3)]
            avg=np.array([r['accuracy_mean']*100 for r in rows]); sd=np.array([r['accuracy_sd']*100 for r in rows])
            ax.plot(range(4),avg,label=v,color=color,marker=marker,ms=4,lw=1.4)
            ax.fill_between(range(4),avg-sd,avg+sd,color=color,alpha=.07)
        ax.set(title=KINDS[kind],xticks=range(4),xticklabels=['干净','1','2','3'],ylim=(60,101),ylabel='Accuracy (%)',xlabel='退化强度')
    axes.flat[-1].axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles,labels,loc='center',ncol=2,frameon=False,title='模型 / 三种子均值±样本标准差')
    fig.tight_layout(); save(fig,'robustness')

    fig,axes=plt.subplots(1,2,figsize=(11.8,3.5)); profiles=data['profiles']
    axes[0].bar(xs,[profiles[v]['latency_median_ms'] for v in VARIANTS],color='white',edgecolor=COLORS,width=.6,label='中位数')
    axes[0].plot(xs,[profiles[v]['latency_p95_ms'] for v in VARIANTS],'kx',label='P95',ms=7)
    axes[0].set(xticks=xs,xticklabels=list(VARIANTS),ylabel='batch=1 延迟 (ms)'); axes[0].legend()
    axes[1].bar(xs,[profiles[v]['throughput_images_s'] for v in VARIANTS],color='white',edgecolor=COLORS,width=.6)
    axes[1].set(xticks=xs,xticklabels=list(VARIANTS),ylabel='batch=32 吞吐 (images/s)')
    fig.tight_layout(); save(fig,'hardware')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['SimHei','DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','legend.frameon':False})
    data=collect(); write_summary(data); diagrams(data); plots(data)
    save_json(OUT/'figure_sources.json',dict(status='PASS',summary_sha256=sha256(OUT/'summary.json'),
        generator_sha256=sha256(__file__),
        outputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha256(p)) for p in sorted(OUT.glob('*.png'))
                 if p.stem in ('workflow','architecture_v3','learning_curves','tradeoffs_v3','robustness','hardware')]))
    print('Generated six unified A-E figures with white backgrounds.',flush=True)


if __name__=='__main__':
    main()
