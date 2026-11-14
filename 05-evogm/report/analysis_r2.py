# Recompute every number used in the round-2+ report update from the raw/scored files.
import json, math
from math import comb
from scipy.stats import spearmanr, pearsonr
E="/workspace/work/evogm/"
FR=E+"final_r2/"
def rd(p): return [json.loads(l) for l in open(p)]
ids=set(json.load(open(FR+"subset_ids.json")))
TASKS=["mmlupro","gpqa","bfcl"]
def wilson(k,n,z=1.959964):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h
def mcn(b,d):
    n=b+d
    return 1.0 if n==0 else min(1,2*sum(comb(n,i) for i in range(min(b,d)+1))/2**n)
# ---- 180-item test ----
scored={}; raw={}
scored["bap11"]={r["id"]:r for r in rd(FR+"scored_evogm.jsonl") if r["id"] in ids}
raw["bap11"]={r["id"]:r for r in rd(FR+"raw_evogm.jsonl")}
scored["r1"]={r["id"]:r for r in rd(E+"../merge3/final_bench/scored_evogm.jsonl") if r["id"] in ids}
raw["r1"]={r["id"]:r for r in rd(FR+"r1_raw_evogm.jsonl.bak")}
for t,f in [("bap10","baphomet"),("qwen","qwen35")]:
    scored[t]={r["id"]:r for r in rd(FR+f"scored_{f}.jsonl") if r["id"] in ids}
    raw[t]={r["id"]:r for r in rd(FR+f"raw_{f}.jsonl")}
common=set.intersection(*[set(v) for v in scored.values()]); assert len(common)==180
test={}
for m in scored:
    test[m]={}
    for b in TASKS+["all"]:
        ii=[i for i in common if b=="all" or scored[m][i]["bench"]==b]
        k=sum(bool(scored[m][i]["correct"]) for i in ii); n=len(ii)
        lo,hi=wilson(k,n)
        test[m][b]=dict(n=n,k=k,acc=k/n,lo=lo,hi=hi,trunc=sum(bool(scored[m][i]["truncated"]) for i in ii)/n,
                        tok=sum(raw[m][i]["completion_tokens"] for i in ii)/n)
mc={}
for a,o in [("bap11","bap10"),("bap11","qwen"),("bap11","r1"),("r1","bap10"),("r1","qwen"),("bap10","qwen")]:
    for b in TASKS+["all"]:
        ii=[i for i in common if b=="all" or scored[a][i]["bench"]==b]
        x=sum(bool(scored[a][i]["correct"]) and not scored[o][i]["correct"] for i in ii)
        y=sum(bool(scored[o][i]["correct"]) and not scored[a][i]["correct"] for i in ii)
        mc[f"{a}_vs_{o}_{b}"]=dict(win=x,loss=y,p=mcn(x,y))
# ---- 110-item fitness set ----
N={"mmlupro":40,"gpqa":30,"bfcl":40}
R1=rd(E+"kout/full/evo_log.jsonl")
def row(name,acc,trunc,tok,fit):
    nt=sum(trunc[t]*N[t] for t in TASKS)
    return dict(name=name,acc={t:acc[t] for t in TASKS},fit=fit,trunc_all=nt/110,trunc=trunc,
                tok_all=sum(tok[t]*N[t] for t in TASKS)/110,tok=tok)
fit110=[]
order=["onehot_agents_a1","onehot_dsv4pro","onehot_q38distill","onehot_qwopus_coder","onehot_qwopus_v3","onehot_arex_turbo","onehot_neohorse","onehot_tmax","average","baphomet"]
for k in order:
    r=[x for x in R1 if x.get("kind")==k][0]
    fit110.append(row(k,r["acc"],r["trunc"],r["mean_tokens"],r["fitness"]))
w1=max([x for x in R1 if not x.get("failed")],key=lambda x:x["fitness"])
fit110.append(row("r1_winner",w1["acc"],w1["trunc"],w1["mean_tokens"],w1["fitness"]))
res=json.load(open(E+"kout/r2/evo_result.json")); B=res["best"]
fit110.append(row("bap11",B["full_acc"],B["full_trunc"],B["full_mean_tokens"],B["full_fitness"]))
# ---- search trajectory & IRT validity ----
comp=res["completed"]
irt=[c["irt"] for c in comp]; full=[c["full"] for c in comp]
mae=sum(abs(a-b) for a,b in zip(irt,full))/len(comp)
bias=sum(a-b for a,b in zip(irt,full))/len(comp)
traj=[]
for h in res["hist"]:
    bf=h.get("best_full", h.get("best"))
    traj.append(dict(round=h["round"],iter=h["iter"],best_full=bf,new_best=h.get("new_best"),elapsed_h=h["elapsed_h"]))
out=dict(test=test,mcnemar=mc,fit110=fit110,completed=comp,irt_full_mae=mae,irt_full_bias=bias,
         irt_full_spearman=spearmanr(irt,full)[0],irt_full_pearson=pearsonr(irt,full)[0],traj=traj,
         lam_bap11=B["lams"],lam_r1=w1["lams"],lam_bap10=[0.1375]*8,sum_bap11=sum(B["lams"]),sum_r1=sum(w1["lams"]),
         r1_full_fit=w1["fitness"],r1_acc=w1["acc"],n_new=res["n_new"],n_raced=res["n_raced"],n_failed=res["n_failed"],n_cached_total=res["n_cached_total"],
         bap11_irt=B["fitness"],bap11_full=B["full_fitness"])
json.dump(out,open(E+"report/data_r2.json","w"),indent=1)
if __name__=="__main__":
    for m in test:
        print(m," ".join(f"{b}:{test[m][b]['acc']*100:.1f}[{test[m][b]['lo']*100:.1f},{test[m][b]['hi']*100:.1f}] tr{test[m][b]['trunc']*100:.1f} tok{test[m][b]['tok']:.0f}" for b in TASKS+["all"]))
    for k,v in mc.items(): print(k,v["win"],v["loss"],f"{v['p']:.4g}")
    for r in fit110: print(f"{r['name']:22s} fit {r['fit']:.4f} "+" ".join(f"{t} {r['acc'][t]:.3f}" for t in TASKS)+f" trunc {r['trunc_all']*100:.1f} tok {r['tok_all']:.0f}")
    print("IRT vs full: MAE",round(mae,4),"bias",round(bias,4),"spearman",round(out['irt_full_spearman'],3),"pearson",round(out['irt_full_pearson'],3))
    for t in traj: print(t)
    print("sum lam bap11",sum(B["lams"]),"r1",sum(w1["lams"]),w1["lams"])
