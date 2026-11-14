import json, numpy as np
from irt import *
params={}; thetas={}
for t in TASKS:
    th,a,b=fit2pl(C[:,TI[t]]); params[t]=(a,b); thetas[t]=th
cost={t:T[:,TI[t]].mean(0) for t in TASKS}
top=np.argsort(-FULLFIT)[:15]
K=dict(mmlupro=16,gpqa=10,bfcl=14)
sel=select(params,K,{t:thetas[t][top] for t in TASKS},cost,0.5)
out=dict(tasks=TASKS, n_task={t:len(TI[t]) for t in TASKS}, created="2026-11-13 from evogm-full v1 round-1 (40 candidates x 110 items), per-task 2PL joint MAP (theta~N(0,1), log a~N(0,.5^2), b~N(0,2^2)); subset = top Fisher info at the top-15 candidates' theta / sqrt(mean tokens), per task 16 MMLU-Pro / 10 GPQA / 14 BFCL",
         items={t:[ids[TI[t][j]] for j in range(len(TI[t]))] for t in TASKS}, a={t:params[t][0].tolist() for t in TASKS}, b={t:params[t][1].tolist() for t in TASKS},
         subset={t:sorted(int(j) for j in sel[t]) for t in TASKS})
out['subset_ids']=[ids[TI[t][j]] for t in TASKS for j in out['subset'][t]]
json.dump(out,open('kds/irt_params.json','w'),indent=0)
print(len(out['subset_ids']), {t:np.round(cost[t][sel[t]].mean()) for t in TASKS})
# in-sample estimator vs full + racing simulation on round-1 generated candidates
def est(c, obs_mask=None, up=0.0):
    e=[]
    for t in TASKS:
        a,b=params[t]; sub=np.array(sel[t]); y=C[c,TI[t]][sub]
        if obs_mask is not None:
            m=obs_mask[np.array(TI[t])[sub]]; 
            th,sd=theta_map(y[m],a[sub][m],b[sub][m]) if m.any() else (0.0,1.0)
            th=th+up*sd; rest=np.setdiff1d(np.arange(len(TI[t])),sub[m]); e.append((y[m].sum()+sig(a[rest]*(th-b[rest])).sum())/len(TI[t]))
        else:
            e.append(est_task(y,sub,a,b,len(TI[t]))[0])
    return np.mean(e)
E=np.array([est(c) for c in range(40)]); print('in-sample MAE',np.abs(E-FULLFIT).mean().round(4),'max',np.abs(E-FULLFIT).max().round(3))
order=np.concatenate([np.array(TI[t])[sel[t]] for t in ['gpqa','mmlupro','bfcl']])
R=[json.loads(l) for l in open('kout/full/evo_log.jsonl')]
for up in [1.5,2.0]:
  pr=0; saved=0; tot=0; wrong=0
  for c in range(20,40):
    thr=np.sort(FULLFIT[:c])[::-1][9]  # P-th best so far
    L=T[c]; full_steps=L[order].max(); fin=None
    for s in range(200,int(full_steps),100):
        obs=np.zeros(110,bool); obs[order]=L[order]<=s
        if obs[order].mean()<0.5: continue
        if est(c,obs,up)<thr: fin=s; break
    tot+=np.array(feats(L[order]))@coef
    if fin:
        pr+=1; Lc=np.minimum(L[order],fin); saved+=np.array(feats(L[order]))@coef-np.array(feats(Lc))@coef
        if FULLFIT[c]>=thr: wrong+=1
  print('up',up,'pruned',pr,'/20 time saved frac',round(saved/tot,3),'wrongly pruned',wrong)
