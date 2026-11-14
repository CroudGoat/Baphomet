import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["Nimbus Roman","DejaVu Serif"],"font.size":9,"pdf.fonttype":42})
R=[json.loads(l) for l in open("../kout/full/evo_log.jsonl")]
n={"mmlupro":40,"gpqa":30,"bfcl":40}
def tok(r): return sum(r["mean_tokens"][k]*n[k] for k in n)/110
H=json.load(open("../kout/full/history.json"))["hist"]
it=[h["iter"] for h in H]; best=[h["best"] for h in H]; pm=[h.get("pop_mean",h.get("mean")) for h in H]; nm=[h.get("new_mean",h.get("mean")) for h in H]
for lang in ["en","ja"]:
    if lang=="ja": plt.rcParams["font.serif"]=["IPAexMincho"]
    L=dict(en=("Iteration","Fitness","Best so far","Population mean","New candidates mean","Mean completion tokens (fitness items)","Fitness","Initial population","Generated","Baphomet λ","Winner"),
           ja=("反復","fitness","最良値","集団の平均","新規候補の平均","平均生成トークン数（fitness項目）","fitness","初期集団","生成された候補","Baphomet の λ","勝者"))[lang]
    f,a=plt.subplots(figsize=(3.3,2.3)); a.plot(it,best,"k-o",ms=4,label=L[2]); a.plot(it,pm,"k--s",ms=3.5,label=L[3]); a.plot(it,nm,":^",ms=3.5,color="0.4",label=L[4])
    a.set_xlabel(L[0]); a.set_ylabel(L[1]); a.set_xticks(it); a.legend(frameon=False,fontsize=7); a.grid(alpha=.3,lw=.4); f.tight_layout(); f.savefig(f"fig_traj_{lang}.pdf"); plt.close(f)
    f,a=plt.subplots(figsize=(3.3,2.4))
    ok=[r for r in R if not r.get("failed")]
    i0=[r for r in ok if r.get("iter",0)==0 and r.get("kind")!="baphomet"]; g=[r for r in ok if r.get("iter",0)>0]
    a.scatter([tok(r) for r in i0],[r["fitness"] for r in i0],s=14,facecolors="none",edgecolors="k",label=L[7])
    a.scatter([tok(r) for r in g],[r["fitness"] for r in g],s=10,c="0.55",label=L[8])
    b=[r for r in ok if r.get("kind")=="baphomet"][0]; a.scatter([tok(b)],[b["fitness"]],marker="s",s=28,c="k",label=L[9])
    w=max(ok,key=lambda r:r["fitness"]); a.scatter([tok(w)],[w["fitness"]],marker="*",s=70,c="k",label=L[10])
    a.set_xlabel(L[5]); a.set_ylabel(L[6]); a.legend(frameon=False,fontsize=6.5,loc="lower left"); a.grid(alpha=.3,lw=.4); f.tight_layout(); f.savefig(f"fig_scatter_{lang}.pdf"); plt.close(f)
x=np.array([tok(r) for r in ok]); y=np.array([r["fitness"] for r in ok])
from scipy.stats import spearmanr; print(len(ok), spearmanr(x,y), round(tok(w)), round(tok(b)))
print(set(r.get("kind") for r in R))
