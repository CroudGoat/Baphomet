import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
INK='#1d1d1f'; BG='#e8e8ea'; GREY='#8e8e93'; BLUE='#3b6fb6'
benches=['MMLU-Pro\n(n=80)','GPQA-Diamond\n(n=50)','BFCL non-live\n(n=50)']
q=[71.3,42.0,96.0]; b=[77.5,58.0,98.0]
fig,ax=plt.subplots(figsize=(7,4.4),dpi=200,facecolor=BG); ax.set_facecolor(BG)
x=np.arange(3); w=0.36
for xs,vals,col,lab in [(x-w/2,q,GREY,'Qwen3.5-4B'),(x+w/2,b,BLUE,'Baphomet-1.0-4B')]:
    bars=ax.bar(xs,vals,w,color=col,label=lab)
    for r,v in zip(bars,vals):
        ax.text(r.get_x()+r.get_width()/2,v+1.5,f'{v:.1f}%',ha='center',va='bottom',fontsize=9,family='monospace',color=INK)
ax.set_xticks(x); ax.set_xticklabels(benches,fontsize=10,color='#555558')
ax.set_ylim(0,110); ax.set_yticks([0,20,40,60,80,100]); ax.yaxis.set_major_formatter(lambda v,p:f'{v:.0f}%')
ax.tick_params(length=0,labelsize=9,colors='#77777b')
for s in ['top','right','left']: ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color('#c8c8cc'); ax.grid(axis='y',color='#d8d8db'); ax.set_axisbelow(True)
ax.legend(frameon=False,fontsize=9,loc='upper left',labelcolor=INK)
plt.tight_layout(); plt.savefig('bench_chart_scores.png',facecolor=BG)
