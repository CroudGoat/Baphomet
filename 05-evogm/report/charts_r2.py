# Round-2+ charts. Two styles from the same data (data_r2.json, built by analysis_r2.py):
#   report: serif, greyscale, PDF -> report/fig_*_{en,ja}.pdf  (matches figs.py)
#   chat:   Hark light theme, PNG at 2x -> report/charts/*_{en,ja}.png
import json, os, math, numpy as np, matplotlib
def R1(v): return f"{math.floor(v*10+0.5+1e-9)/10:.1f}"
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import to_rgba
D=json.load(open("data_r2.json")); os.makedirs("charts",exist_ok=True)
TASKS=["mmlupro","gpqa","bfcl"]
INK="#1d1d1f"; BG="#e8e8ea"
def ink(a): return to_rgba(INK,a)
SKY,SAGE,CLAY,DUSK,GOLD,TEAL,ROSE="#b1bcd2","#607868","#dbb691","#a79fb8","#d9c489","#7e9c9b","#c9a3a8"
NAMES=dict(
 en=dict(bap11="Baphomet-1.1-4B",r1_winner="EvoGM round-1 winner",r1="EvoGM round-1 winner",bap10="Baphomet-1.0-4B",baphomet="Baphomet-1.0-4B",qwen="Qwen3.5-4B",average="Uniform average (1/8)",
         onehot_agents_a1="Agents-A1",onehot_dsv4pro="DeepSeek-V4-Pro-distill",onehot_q38distill="Qwen3.8-Distill",onehot_qwopus_coder="Qwopus-Coder",onehot_qwopus_v3="Qwopus-v3",
         onehot_arex_turbo="AREX-Turbo",onehot_neohorse="NeoHorse",onehot_tmax="T-Max",
         all="All",mmlupro="MMLU-Pro",gpqa="GPQA",bfcl="BFCL",fit="Fitness (mean of 3)"),
 ja=dict(bap11="Baphomet-1.1-4B",r1_winner="EvoGM ラウンド1勝者",r1="EvoGM ラウンド1勝者",bap10="Baphomet-1.0-4B",baphomet="Baphomet-1.0-4B",qwen="Qwen3.5-4B",average="均等平均（1/8）",
         onehot_agents_a1="Agents-A1",onehot_dsv4pro="DeepSeek-V4-Pro-distill",onehot_q38distill="Qwen3.8-Distill",onehot_qwopus_coder="Qwopus-Coder",onehot_qwopus_v3="Qwopus-v3",
         onehot_arex_turbo="AREX-Turbo",onehot_neohorse="NeoHorse",onehot_tmax="T-Max",
         all="全体",mmlupro="MMLU-Pro",gpqa="GPQA",bfcl="BFCL",fit="fitness（3 タスク平均）"))
T=dict(
 en=dict(acc="Accuracy",trunc="Truncated answers",tok="Mean completion tokens",
   t1="Baphomet-1.1-4B leads the 180-item test numerically, within overlapping CIs",s1="180-item test, accuracy with Wilson 95% CI, 8,192-token cap (expert donors not run on this test)",
   t2="Baphomet-1.1-4B truncates less than 1.0 but more than the round-1 winner",s2="180-item test, 8,192-token cap",
   t3="Baphomet-1.1-4B tops the 110-item fitness set it was selected on",s3="110-item fitness set, 4,096-token cap; 1.1 was chosen on these items, so its score is optimistic",
   t4="Truncation and length on the 110-item fitness set",s4="110-item fitness set, 4,096-token cap; truncated = wrong",
   t5="Best full-110 fitness rose to 0.764 in round 3",s5="Left: best exact 110-item fitness so far; right: IRT estimate vs exact fitness of the 12 completed candidates",
   step="Round / iteration",bestfull="Best full-110 fitness",bestirt="Best IRT estimate among new candidates",irtx="IRT estimate (40-item subset)",fully="Exact 110-item fitness",
   yx="y = x",mae="MAE",win="Baphomet-1.1-4B",
   t6="Baphomet-1.1-4B shifts weight toward DeepSeek-V4-Pro-distill and Qwopus-v3",s6="Merge coefficient λ per donor",lam="λ"),
 ja=dict(acc="正答率",trunc="打ち切られた回答の割合",tok="平均生成トークン数",
   t1="180 問テストで Baphomet-1.1-4B が数値上は最高（信頼区間は重なる）",s1="180 問テスト，正答率と Wilson 95% 信頼区間，上限 8,192 トークン（エキスパート単体は未評価）",
   t2="Baphomet-1.1-4B の打ち切りは 1.0 より少なく，ラウンド1勝者より多い",s2="180 問テスト，上限 8,192 トークン",
   t3="選抜に使った 110 問 fitness セットで Baphomet-1.1-4B が最高",s3="110 問 fitness セット，上限 4,096 トークン．1.1 はこの問題で選ばれたため楽観的な値",
   t4="110 問 fitness セットでの打ち切りと生成長",s4="110 問 fitness セット，上限 4,096 トークン．打ち切り＝不正解",
   t5="ラウンド3で最良の full-110 fitness が 0.764 に上昇",s5="左：その時点までの最良の 110 問 fitness．右：完全評価した 12 候補の IRT 推定値と実測値",
   step="ラウンド / 反復",bestfull="最良の full-110 fitness",bestirt="新規候補の最良 IRT 推定値",irtx="IRT 推定値（40 問サブセット）",fully="110 問 fitness（実測）",
   yx="y = x",mae="MAE",win="Baphomet-1.1-4B",
   t6="Baphomet-1.1-4B は DeepSeek-V4-Pro-distill と Qwopus-v3 の重みを増やした",s6="ドナーごとのマージ係数 λ",lam="λ"))
ORDER_FIT=["bap11","r1_winner","onehot_q38distill","onehot_dsv4pro","onehot_qwopus_v3","baphomet","average","onehot_qwopus_coder","onehot_neohorse","onehot_tmax","onehot_arex_turbo","onehot_agents_a1"]
TEST=["bap11","r1","bap10","qwen"]

def setup(style,lang):
    plt.rcdefaults()
    if style=="report":
        plt.rcParams.update({"font.family":"serif","font.serif":["IPAexMincho"] if lang=="ja" else ["Nimbus Roman","DejaVu Serif"],
                             "font.size":8,"pdf.fonttype":42,"axes.unicode_minus":False})
    else:
        plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":(["IPAexGothic"] if lang=="ja" else ["DejaVu Sans"]),
            "font.size":11,"axes.facecolor":BG,"figure.facecolor":BG,"savefig.facecolor":BG,"text.color":INK,
            "axes.labelcolor":ink(.5),"xtick.color":ink(.5),"ytick.color":ink(.5),"axes.edgecolor":ink(.3),"axes.unicode_minus":False})
def col(style,m):
    """one colour per model, same on every chart"""
    m={"r1_winner":"r1","baphomet":"bap10"}.get(m,m)
    if style=="report":
        return {"bap11":"0.0","r1":"0.38","bap10":"0.62","qwen":"0.86","average":"0.80"}.get(m,"0.80")
    return {"bap11":INK,"r1":CLAY,"bap10":SKY,"qwen":SAGE,"average":DUSK}.get(m,ink(.3))
def clean(a,style,axis="y"):
    for s in ["top","right"]: a.spines[s].set_visible(False)
    if style=="chat":
        a.tick_params(length=0); a.grid(axis=axis,color=ink(.07),lw=.8); a.set_axisbelow(True)
    else:
        a.grid(axis=axis,alpha=.3,lw=.4); a.set_axisbelow(True)
def num(style):  # monospace numbers on chat charts
    return dict(family="DejaVu Sans Mono") if style=="chat" else {}
def title(f,style,lang,t,s):
    if style=="chat":
        f.text(0.012,0.985,T[lang][t],fontsize=15,fontweight="bold",color=INK,va="top",ha="left")
        f.text(0.012,0.935,T[lang][s],fontsize=11,color=ink(.5),va="top",ha="left")
def save(f,style,name,lang,top=0.86):
    if style=="report":
        f.tight_layout(pad=0.4); f.savefig(f"fig_{name}_{lang}.pdf"); 
    else:
        f.tight_layout(rect=(0,0,1,top)); f.savefig(f"charts/{name}_{lang}.png",dpi=200)
    plt.close(f)

def ch_test_acc(style,lang):
    setup(style,lang); N=NAMES[lang]; Td=D["test"]
    f,a=plt.subplots(figsize=(6.3,2.6) if style=="report" else (10,5.6))
    groups=["all"]+TASKS; w=0.2; x=np.arange(len(groups))
    for j,m in enumerate(TEST):
        acc=[Td[m][g]["acc"]*100 for g in groups]; lo=[Td[m][g]["lo"]*100 for g in groups]; hi=[Td[m][g]["hi"]*100 for g in groups]
        xs=x+(j-1.5)*w
        a.bar(xs,acc,w*0.9,color=col(style,m),edgecolor=("k" if style=="report" else "none"),lw=.4,label=N[m])
        a.errorbar(xs,acc,yerr=[np.array(acc)-lo,np.array(hi)-acc],fmt="none",ecolor=("k" if style=="report" else ink(.55)),elinewidth=.7,capsize=1.8)
        for xx,v,h in zip(xs,acc,hi):
            a.text(xx,h+1.2,R1(v),ha="center",va="bottom",fontsize=(5.3 if style=="report" else 8.5),color=(INK if style=="chat" else "k"),**num(style))
    a.set_xticks(x); a.set_xticklabels([N[g]+f" (n={Td['bap11'][g]['n']})" for g in groups],fontsize=(8 if style=="report" else 12))
    a.set_ylim(0,110); a.set_yticks(range(0,101,20)); a.set_yticklabels([f"{v}%" for v in range(0,101,20)])
    a.set_ylabel(T[lang]["acc"]); clean(a,style)
    a.legend(ncol=4,frameon=False,fontsize=(7 if style=="report" else 10.5),loc="upper center",bbox_to_anchor=(0.5,1.13 if style=="report" else 1.1))
    title(f,style,lang,"t1","s1"); save(f,style,"test180_acc",lang,top=0.84)

def ch_test_eff(style,lang):
    setup(style,lang); N=NAMES[lang]; Td=D["test"]
    f,ax=plt.subplots(1,2,figsize=(6.3,2.5) if style=="report" else (11,5.4))
    groups=["all"]+TASKS; w=0.2; x=np.arange(len(groups))
    for k,(key,lab,fmt,sc) in enumerate([("trunc",T[lang]["trunc"],"{:.0f}",100),("tok",T[lang]["tok"],"{:,.0f}",1)]):
        a=ax[k]
        for j,m in enumerate(TEST):
            v=[Td[m][g][key]*sc for g in groups]; xs=x+(j-1.5)*w
            a.bar(xs,v,w*0.9,color=col(style,m),edgecolor=("k" if style=="report" else "none"),lw=.4,label=N[m])
            for xx,vv in zip(xs,v):
                a.text(xx,vv+(1 if key=="trunc" else 90),fmt.format(vv) if key=="trunc" else (f"{vv/1000:.1f}k" if vv>=1000 else f"{vv:.0f}"),
                       ha="center",va="bottom",fontsize=(4.3 if style=="report" else 6.8),rotation=0,**num(style))
        a.set_xticks(x); a.set_xticklabels([N[g] for g in groups],fontsize=(7.5 if style=="report" else 11))
        if key=="trunc":
            a.set_ylim(0,62); a.set_yticks(range(0,61,15)); a.set_yticklabels([f"{t}%" for t in range(0,61,15)])
        else:
            a.set_ylim(0,8000); a.set_yticks(range(0,8001,2000)); a.set_yticklabels([f"{t:,}" for t in range(0,8001,2000)])
        a.set_title(lab,fontsize=(8 if style=="report" else 12),color=(INK if style=="chat" else "k"),loc="left"); clean(a,style)
    h,l=ax[0].get_legend_handles_labels()
    f.legend(h,l,ncol=4,frameon=False,fontsize=(7 if style=="report" else 10.5),loc="lower center",bbox_to_anchor=(0.5,0.0))
    title(f,style,lang,"t2","s2")
    if style=="report":
        f.tight_layout(pad=0.4,rect=(0,0.08,1,1)); f.savefig(f"fig_test180_eff_{lang}.pdf"); plt.close(f)
    else:
        f.tight_layout(rect=(0,0.07,1,0.86)); f.savefig(f"charts/test180_eff_{lang}.png",dpi=200); plt.close(f)

def rows110():
    R={r["name"]:r for r in D["fit110"]}; return [R[m] for m in ORDER_FIT]
def ch_fit110(style,lang):
    setup(style,lang); N=NAMES[lang]; rs=rows110(); y=np.arange(len(rs))[::-1]
    f,ax=plt.subplots(1,4,figsize=(6.3,3.3) if style=="report" else (13,6.6),sharey=True)
    for k,key in enumerate(["fit"]+TASKS):
        a=ax[k]; v=[(r["fit"] if key=="fit" else r["acc"][key])*100 for r in rs]
        a.barh(y,v,0.72,color=[col(style,r["name"]) for r in rs],edgecolor=("k" if style=="report" else "none"),lw=.3)
        for yy,vv,r in zip(y,v,rs):
            a.text(vv+1.5,yy,R1(vv),va="center",fontsize=(5.2 if style=="report" else 8.5),
                   fontweight=("bold" if r["name"]=="bap11" else "normal"),**num(style))
        a.set_xlim(0,118); a.set_xticks([0,50,100]); a.set_xticklabels(["0","50","100%"])
        a.set_title(N[key],fontsize=(8 if style=="report" else 12),loc="left",color=(INK if style=="chat" else "k")); clean(a,style,"x")
    ax[0].set_yticks(y); ax[0].set_yticklabels([N[r["name"]] for r in rs],fontsize=(7 if style=="report" else 11))
    for t in ax[0].get_yticklabels():
        if t.get_text()==N["bap11"]: t.set_fontweight("bold"); t.set_color("k" if style=="report" else INK)
    title(f,style,lang,"t3","s3"); save(f,style,"fit110_acc",lang,top=0.87)

def ch_fit110_eff(style,lang):
    setup(style,lang); N=NAMES[lang]; rs=rows110(); y=np.arange(len(rs))[::-1]
    f,ax=plt.subplots(1,2,figsize=(6.3,3.1) if style=="report" else (11,6.4),sharey=True)
    for k,(key,lab) in enumerate([("trunc_all",T[lang]["trunc"]),("tok_all",T[lang]["tok"])]):
        a=ax[k]; v=[r[key]*(100 if key=="trunc_all" else 1) for r in rs]
        a.barh(y,v,0.72,color=[col(style,r["name"]) for r in rs],edgecolor=("k" if style=="report" else "none"),lw=.3)
        mx=max(v)
        for yy,vv,r in zip(y,v,rs):
            a.text(vv+mx*0.015,yy,(R1(vv)+"%" if key=="trunc_all" else f"{vv:,.0f}"),va="center",fontsize=(5.5 if style=="report" else 9),
                   fontweight=("bold" if r["name"]=="bap11" else "normal"),**num(style))
        if key=="trunc_all": a.set_xlim(0,60); a.set_xticks([0,15,30,45]); a.set_xticklabels(["0%","15%","30%","45%"])
        else: a.set_xlim(0,3000); a.set_xticks([0,1000,2000]); a.set_xticklabels(["0","1,000","2,000"])
        a.set_title(lab,fontsize=(8 if style=="report" else 12),loc="left",color=(INK if style=="chat" else "k")); clean(a,style,"x")
    ax[0].set_yticks(y); ax[0].set_yticklabels([N[r["name"]] for r in rs],fontsize=(7 if style=="report" else 11))
    for t in ax[0].get_yticklabels():
        if t.get_text()==N["bap11"]: t.set_fontweight("bold")
    title(f,style,lang,"t4","s4"); save(f,style,"fit110_eff",lang,top=0.87)

def ch_search(style,lang):
    setup(style,lang); L=T[lang]
    f,ax=plt.subplots(1,2,figsize=(6.3,2.6) if style=="report" else (12,5.6),gridspec_kw=dict(width_ratios=[1.5,1]))
    tr=D["traj"]; x=np.arange(len(tr)); bf=[t["best_full"] for t in tr]
    nb=[(t["new_best"] if t["round"]>=2 else None) for t in tr]
    a=ax[0]
    a.step(x,bf,where="post",color=(INK if style=="chat" else "k"),lw=1.5,label=L["bestfull"])
    a.plot(x,bf,"o",ms=3.5,color=(INK if style=="chat" else "k"))
    xi=[i for i,v in enumerate(nb) if v is not None]; a.plot(xi,[nb[i] for i in xi],"^",ms=4,ls="none",color=(CLAY if style=="chat" else "0.55"),label=L["bestirt"])
    for i in [0,4,8]:
        if i: a.axvline(i-0.5,color=(ink(.14) if style=="chat" else "0.7"),ls="--",lw=.8)
        a.text(i+1.5,0.79,f"R{tr[i]['round']}",ha="center",fontsize=(7 if style=="report" else 11),color=(ink(.5) if style=="chat" else "0.3"))
    for i in [2,11]:
        a.annotate(f"{bf[i]:.3f}",(x[i],bf[i]),xytext=(-16 if i==2 else 0,5),textcoords="offset points",ha="center",fontsize=(6.5 if style=="report" else 10),**num(style))
    a.set_xticks(x); a.set_xticklabels([str(t["iter"]) for t in tr],fontsize=(7 if style=="report" else 10))
    a.set_xlabel(L["step"]); a.set_ylim(0.67,0.80); a.set_yticks([0.68,0.70,0.72,0.74,0.76,0.78])
    a.legend(frameon=False,fontsize=(6.5 if style=="report" else 10),loc="lower right"); clean(a,style)
    b=ax[1]; c=D["completed"]; ix=[q["irt"] for q in c]; fy=[q["full"] for q in c]
    lo,hi=0.68,0.79
    b.plot([lo,hi],[lo,hi],ls="--",color=(ink(.3) if style=="chat" else "0.5"),lw=.9)
    b.text(0.785,0.787,L["yx"],ha="right",va="bottom",fontsize=(6.5 if style=="report" else 10),color=(ink(.5) if style=="chat" else "0.3"))
    isw=[abs(q["full"]-D["bap11_full"])<1e-9 for q in c]
    b.scatter([v for v,w in zip(ix,isw) if not w],[v for v,w in zip(fy,isw) if not w],s=(16 if style=="report" else 55),
              c=(SKY if style=="chat" else "0.6"),edgecolors=("k" if style=="report" else "none"),lw=.4,alpha=.9)
    wi=isw.index(True); b.scatter([ix[wi]],[fy[wi]],marker="*",s=(70 if style=="report" else 220),c=(INK if style=="chat" else "k"),zorder=3)
    b.annotate(L["win"],(ix[wi],fy[wi]),xytext=(0.789,0.744),textcoords="data",ha="right",va="top",fontsize=(6.5 if style=="report" else 10))
    b.text(0.685,0.775,f"{L['mae']} {D['irt_full_mae']:.3f}\nSpearman ρ {D['irt_full_spearman']:.2f}\nn = {len(c)}",fontsize=(6.5 if style=="report" else 10),va="top",**num(style))
    b.set_xlim(lo,hi); b.set_ylim(lo,hi); b.set_xticks([0.70,0.73,0.76,0.79]); b.set_yticks([0.70,0.73,0.76,0.79])
    b.set_xlabel(L["irtx"]); b.set_ylabel(L["fully"]); clean(b,style,"both")
    ax[0].set_ylabel(L["bestfull"])
    title(f,style,lang,"t5","s5"); save(f,style,"search_r2",lang,top=0.85)

def ch_lam(style,lang):
    setup(style,lang); N=NAMES[lang]
    donors=["onehot_agents_a1","onehot_dsv4pro","onehot_q38distill","onehot_qwopus_coder","onehot_qwopus_v3","onehot_arex_turbo","onehot_neohorse","onehot_tmax"]
    f,a=plt.subplots(figsize=(6.3,2.4) if style=="report" else (11,5.4)); x=np.arange(8); w=0.27
    for j,(m,lam) in enumerate([("bap11",D["lam_bap11"]),("r1",D["lam_r1"]),("bap10",D["lam_bap10"])]):
        xs=x+(j-1)*w
        a.bar(xs,lam,w*0.9,color=col(style,m),edgecolor=("k" if style=="report" else "none"),lw=.4,label=N[m]+f" (Σλ={sum(lam):.2f})")
        for xx,v in zip(xs,lam):
            if abs(v)>=0.09 and m!="bap10": a.text(xx,v+0.012,f"{v:.2f}",ha="center",va="bottom",fontsize=(5.5 if style=="report" else 9),**num(style))
    a.axhline(0,color=(ink(.3) if style=="chat" else "k"),lw=.6)
    a.set_xticks(x); a.set_xticklabels([N[d].replace("DeepSeek-V4-Pro-distill","DeepSeek-V4-Pro\n-distill").replace("Qwen3.8-Distill","Qwen3.8\n-Distill") for d in donors],fontsize=(6.5 if style=="report" else 10))
    a.set_ylim(-0.05,0.7); a.set_yticks([0,0.2,0.4,0.6]); a.set_ylabel(T[lang]["lam"]); clean(a,style)
    a.legend(frameon=False,fontsize=(7 if style=="report" else 10.5),loc="upper right")
    title(f,style,lang,"t6","s6"); save(f,style,"lambda",lang,top=0.85)

for style in ["report","chat"]:
    for lang in ["en","ja"]:
        for fn in [ch_test_acc,ch_test_eff,ch_fit110,ch_fit110_eff,ch_search,ch_lam]: fn(style,lang)
print(sorted(os.listdir("charts")))
