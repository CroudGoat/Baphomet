# per-task 2PL IRT (joint MAP) on the 40x110 round-1 correctness matrix; subset selection + LOO validation of the p-IRT fitness estimator
import json, numpy as np, torch
from simt import feats
D=np.load('irt_mat.npz'); C=D['C'].astype(float); T=D['T']; ids=list(D['ids']); FULLFIT=D['fit']
FIT=json.load(open('kds/fitness_items.json')); bench={i['id']:i['bench'] for i in FIT}
TASKS=['mmlupro','gpqa','bfcl']; TI={t:[k for k,i in enumerate(ids) if bench[i]==t] for t in TASKS}
coef=np.load('timecoef.npy')
def fit2pl(Y, iters=600):
    Y=torch.tensor(Y,dtype=torch.float64); n,m=Y.shape
    th=torch.zeros(n,requires_grad=True,dtype=torch.float64); la=torch.zeros(m,requires_grad=True,dtype=torch.float64); b=torch.zeros(m,requires_grad=True,dtype=torch.float64)
    opt=torch.optim.LBFGS([th,la,b],max_iter=iters,line_search_fn='strong_wolfe')
    def cl():
        opt.zero_grad(); z=torch.exp(la)[None]*(th[:,None]-b[None])
        nll=torch.nn.functional.binary_cross_entropy_with_logits(z,Y,reduction='sum')
        l=nll+0.5*(th**2).sum()+0.5*(la**2).sum()/0.5**2+0.5*(b**2).sum()/2.0**2
        l.backward(); return l
    opt.step(cl)
    return th.detach().numpy(), np.exp(la.detach().numpy()), b.detach().numpy()
def theta_map(y, a, b, prior_sd=1.0):
    g=np.linspace(-5,5,2001); z=a[None]*(g[:,None]-b[None]); 
    ll=(y[None]*-np.logaddexp(0,-z)+(1-y[None])*-np.logaddexp(0,z)).sum(1)-0.5*(g/prior_sd)**2
    k=np.argmax(ll); w=np.exp(ll-ll.max()); w/=w.sum(); mu=(w*g).sum(); sd=np.sqrt((w*(g-mu)**2).sum())
    return g[k], sd
def sig(x): return 1/(1+np.exp(-x))
def est_task(y_sub, sub, a, b, n_t):
    """p-IRT: observed subset correct + predicted p for the unobserved items, / n_t. sub = local item idx within task."""
    th,sd=theta_map(y_sub, a[sub], b[sub]); rest=np.setdiff1d(np.arange(n_t), sub)
    return (y_sub.sum()+sig(a[rest]*(th-b[rest])).sum())/n_t, th, sd
def select(params, k, top_th, cost, alpha):
    out={}
    for t in TASKS:
        a,b=params[t]; th=top_th[t]
        P=sig(a[None]*(th[:,None]-b[None])); I=(a**2*P*(1-P)).mean(0)
        score=I/np.power(cost[t],alpha)
        out[t]=np.argsort(-score)[:k[t]]
    return out
if __name__=='__main__':
    params={}; thetas={}
    for t in TASKS:
        th,a,b=fit2pl(C[:,TI[t]]); params[t]=(a,b); thetas[t]=th
    cost={t:T[:,TI[t]].mean(0) for t in TASKS}
    top=np.argsort(-FULLFIT)[:15]
    res=[]
    for k in [dict(mmlupro=16,gpqa=10,bfcl=14),dict(mmlupro=18,gpqa=8,bfcl=14),dict(mmlupro=16,gpqa=12,bfcl=12),dict(mmlupro=20,gpqa=10,bfcl=10),dict(mmlupro=14,gpqa=14,bfcl=12)]:
      for alpha in [0,0.5,1.0]:
        sel=select(params,k,{t:thetas[t][top] for t in TASKS},cost,alpha)
        # LOO: refit IRT without candidate c, estimate c from its subset answers
        errs=[]; ests=[]
        for c in range(40):
            m=np.arange(40)!=c; e=[]
            for t in TASKS:
                th,a,b=fit2pl(C[m][:,TI[t]],iters=200)
                sub=sel[t]; y=C[c,TI[t]][sub]; e.append(est_task(y,sub,a,b,len(TI[t]))[0])
            ests.append(np.mean(e))
        ests=np.array(ests); err=ests-FULLFIT
        # naive subset accuracy for comparison
        naive=np.mean([C[:,np.array(TI[t])[sel[t]]].mean(1) for t in TASKS],0)
        # time: subset in one batch, gpqa first then mmlu then bfcl
        order=np.concatenate([np.array(TI[t])[sel[t]] for t in ['gpqa','mmlupro','bfcl']])
        secs=np.mean([np.array(feats(T[c,order]))@coef for c in range(40)])
        tok=T[:,order].sum(1).mean()
        rho=np.corrcoef(np.argsort(np.argsort(ests)),np.argsort(np.argsort(FULLFIT)))[0,1]
        toperr=np.abs(err[top]).mean()
        res.append((k,alpha,round(np.abs(err).mean(),4),round(toperr,4),round(rho,3),round(np.abs(naive-FULLFIT).mean(),4),round(secs),round(tok)))
        print(res[-1],flush=True)
    full_secs=np.mean([np.array(feats(T[c,np.concatenate([TI['gpqa'],TI['mmlupro']])]))@coef+np.array(feats(T[c,TI['bfcl']]))@coef for c in range(40)])
    print('full110 secs',full_secs)
