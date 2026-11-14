import json
from math import comb
B="/workspace/work/merge3/final_bench/"
ids=set(json.load(open(B+"subset_ids.json")))
L=lambda t:{r["id"]:r for r in map(json.loads,open(B+f"scored_{t}.jsonl")) if r["id"] in ids}
S={t:L(t) for t in ["evogm","baphomet","qwen35"]}
c=set.intersection(*[set(v) for v in S.values()]);print("n",len(c))
def p(b,d):
    n=b+d
    return 1.0 if n==0 else min(1,2*sum(comb(n,i) for i in range(min(b,d)+1))/2**n)
raw={t:{r["id"]:r for r in map(json.loads,open(B+f"raw_{t}.jsonl"))} for t in S}
for t in S:
    for b in ["mmlupro","gpqa","bfcl","all"]:
        ii=[i for i in c if b=="all" or S[t][i]["bench"]==b]
        print(t,b,len(ii),round(sum(S[t][i]["correct"] for i in ii)/len(ii),3),"trunc",round(sum(S[t][i]["truncated"] for i in ii)/len(ii),3),"tok",round(sum(raw[t][i]["completion_tokens"] for i in ii)/len(ii)))
for o in ["baphomet","qwen35"]:
    for b in ["mmlupro","gpqa","bfcl","all"]:
        ii=[i for i in c if b=="all" or S["evogm"][i]["bench"]==b]
        x=sum(S["evogm"][i]["correct"] and not S[o][i]["correct"] for i in ii);y=sum(S[o][i]["correct"] and not S["evogm"][i]["correct"] for i in ii)
        print("evogm vs",o,b,x,y,round(p(x,y),3))
