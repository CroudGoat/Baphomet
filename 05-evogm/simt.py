import json,numpy as np
D=np.load('irt_mat.npz'); T=D['T']; ids=list(D['ids'])
FIT=json.load(open('kds/fitness_items.json')); bench={i['id']:i['bench'] for i in FIT}
R=[json.loads(l) for l in open('kout/full/evo_log.jsonl')]
def feats(lens, maxseq=48):
    # continuous batching FCFS: returns (steps, sum_active, sum_ctx)
    lens=list(lens); q=0; act=[]; steps=0; sa=0; sc=0
    while q<len(lens) or act:
        while len(act)<maxseq and q<len(lens): act.append([lens[q],0]); q+=1
        steps+=1; sa+=len(act); sc+=sum(a[1] for a in act)
        for a in act: a[1]+=1
        act=[a for a in act if a[1]<a[0]]
    return steps,sa,sc
if __name__=='__main__':
    bf=[k for k,i in enumerate(ids) if bench[i]=='bfcl']
    mc=[k for k,i in enumerate(ids) if bench[i]=='gpqa']+[k for k,i in enumerate(ids) if bench[i]=='mmlupro']
    X=[];y=[]
    for c,r in enumerate(R):
        for idx,sec in ((bf,r['t_gen']-r['mc_secs']),(mc,r['mc_secs'])):
            X.append(feats(T[c,idx])); y.append(sec)
    X=np.array(X,float);y=np.array(y)
    A=np.c_[X]; coef,*_=np.linalg.lstsq(A,y,rcond=None); print('coef step,act,ctx',coef)
    pred=A@coef; print('rel err', np.round(np.abs(pred-y)/y,3)[:20], 'mean',np.mean(np.abs(pred-y)/y))
    np.save('timecoef.npy',coef)
