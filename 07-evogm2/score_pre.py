# Score BFCL prescreen outputs and pick 10 per category for the search set (0<p<1 closest to 0.5; fill with p=1 lowest-token if short).
import json, sys, glob, collections
sys.path.insert(0, "/workspace/work/evogm/kds"); sys.path.insert(1, "/workspace/work/bench/.pylib")
import fitscore
C={i["id"]:i for i in json.load(open("bfcl_cands.json"))}
hits=collections.defaultdict(list); acc=collections.defaultdict(list)
for f in glob.glob("pre_out/pre_*.jsonl"):
    for l in open(f):
        r=json.loads(l)
        try: ok=fitscore.score(C[r["id"]],r["output"],r["finish_reason"])["correct"]
        except Exception: ok=False
        hits[r["id"]].append(bool(ok)); acc[r["model"]].append(bool(ok))
p={k:sum(v)/len(v) for k,v in hits.items()}
models=sorted({json.loads(open(f).readline())["model"] for f in glob.glob("pre_out/pre_*.jsonl")})
print("models",models,"items scored",len(p)); print({m:round(sum(v)/len(v),3) for m,v in acc.items()})
bycat=collections.defaultdict(list)
for k,i in C.items(): bycat[i["category"]].append(k)
pick=[]
for cat,ks in bycat.items():
    ks=[k for k in ks if k in p]
    mid=sorted([k for k in ks if 0<p[k]<1],key=lambda k:abs(p[k]-.5))
    rest=sorted([k for k in ks if p[k]==1 or p[k]==0],key=lambda k:-p[k])
    sel=(mid+rest)[:10]; pick+=sel
    print(cat,"n",len(ks),"all-right",sum(p[k]==1 for k in ks),"all-wrong",sum(p[k]==0 for k in ks),"mid",len(mid),"picked mean p",round(sum(p[k] for k in sel)/len(sel),2))
S=json.load(open("search_items.json")); S=[i for i in S if i["bench"]!="bfcl"]+[C[k] for k in pick]
assert len(S)==200; json.dump(S,open("search_items.json","w")); json.dump(p,open("bfcl_pre_p.json","w"))
print("search_items.json updated", collections.Counter(i["bench"] for i in S))
