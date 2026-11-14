import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch
INK='#1d1d1f'; BG='#e8e8ea'
benches=['MMLU-Pro\n(n=80)','GPQA-Diamond\n(n=50)','BFCL non-live\n(n=50)']
acc_q=[71.3,42.0,96.0]; acc_b=[77.5,58.0,98.0]
tr_q=[17.5,54.0,0.0]; tr_b=[13.8,34.0,0.0]
fig,axes=plt.subplots(1,2,figsize=(10,4.6),dpi=200,facecolor=BG,gridspec_kw={'width_ratios':[3,3]})
def draw(ax,a,b,title,ymax):
    ax.set_facecolor(BG); x=np.arange(3); w=0.36
    for xs,vals,col,lab in [(x-w/2,a,(0.114,0.114,0.122,0.5),'Qwen3.5-4B'),(x+w/2,b,INK,'Baphomet-1.0-4B')]:
        bars=ax.bar(xs,vals,w,color=col,label=lab)
        for r,v in zip(bars,vals):
            ax.text(r.get_x()+r.get_width()/2,v+ymax*0.015,f'{v:.1f}%',ha='center',va='bottom',fontsize=9,family='monospace',color=INK)
    ax.set_xticks(x); ax.set_xticklabels(benches,fontsize=10,color=(0.114,0.114,0.122,0.6))
    ax.set_ylim(0,ymax); ax.yaxis.set_major_formatter(lambda v,p:f'{v:.0f}%')
    ax.tick_params(length=0,labelsize=9,colors=(0.114,0.114,0.122,0.5))
    for s in ['top','right','left']: ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color((0.114,0.114,0.122,0.14))
    ax.grid(axis='y',color=(0.114,0.114,0.122,0.07)); ax.set_axisbelow(True)
    ax.set_title(title,loc='left',fontsize=11,color=INK,pad=10)
draw(axes[0],acc_q,acc_b,'Accuracy',110)
draw(axes[1],tr_q,tr_b,'Truncated at 8,192 tokens',65)
axes[0].legend(frameon=False,fontsize=9,loc='upper left',labelcolor=INK)
fig.suptitle('Baphomet beats base Qwen3.5-4B on all three, mostly by truncating less',x=0.02,ha='left',fontsize=13,color=INK,y=0.99)
fig.text(0.02,0.905,'Paired 180 items, T4, thinking on, temp 0.6. GPQA gap p=0.021; MMLU-Pro p=0.18 (exact McNemar)',fontsize=9,color=(0.114,0.114,0.122,0.55))
plt.tight_layout(rect=[0,0,1,0.88]); plt.savefig('bench_chart.png',facecolor=BG)
