# EvoGM v2 item sets: search 200 (MMLU-Pro 100 / GPQA 60 / BFCL 40) + validation 150 (75 / 45 / 30).
# All random stratified (no loglik-IRT selection). Disjoint from: bench pool (incl. 180 test), v1 fitness 110, each other.
import json, re, random, sys, os, pyarrow.parquet as pq
from collections import defaultdict, Counter
sys.path.insert(0, "/workspace/work/bench/.pylib")
BENCH="/workspace/work/bench"; M3="/workspace/work/merge3"
SPLIT={"search":dict(mmlupro=100,gpqa=60,bfcl=40),"val":dict(mmlupro=75,gpqa=45,bfcl=30)}
rng=random.Random(20261115)
bench=[json.loads(l) for l in open(f"{BENCH}/items.jsonl")]; bench_ids={b["id"] for b in bench}
v1=json.load(open("/workspace/work/evogm/kds/fitness_items.json")); v1_ids={i["id"] for i in v1}|{i.get("bfcl_id") for i in v1}
out={"search":[],"val":[]}
def alloc(groups,n):
    tot=sum(len(v) for v in groups.values()); q={c:n*len(v)/tot for c,v in groups.items()}; a={c:int(x) for c,x in q.items()}
    for c in sorted(q,key=lambda c:-(q[c]-a[c]))[:n-sum(a.values())]: a[c]+=1
    return a
# MMLU-Pro (full test set, stratified by category)
L="ABCDEFGHIJ"; rows=pq.read_table(f"{BENCH}/data/mmlupro_test.parquet").to_pylist()
byc=defaultdict(list)
for r in rows:
    if f"mmlupro_{r['question_id']}" in bench_ids or f"fit_mmlupro_{r['question_id']}" in v1_ids: continue
    byc[r["category"]].append(r)
for c in byc: rng.shuffle(byc[c])
for s in ["search","val"]:
    for c,k in alloc(byc,SPLIT[s]["mmlupro"]).items():
        for r in [byc[c].pop() for _ in range(k)]:
            opts="\n".join(f"{L[i]}. {o}" for i,o in enumerate(r["options"]))
            q=("The following is a multiple choice question about "+c+". Think step by step and then finish your answer with \"The answer is (X)\" where X is the correct letter choice.\n\n"
               f"Question: {r['question']}\nOptions:\n{opts}")
            out[s].append(dict(id=f"v2_mmlupro_{r['question_id']}",bench="mmlupro",category=c,messages=[{"role":"user","content":q}],answer=r["answer"],nopts=len(r["options"])))
# GPQA main/extended, no diamond, no v1
norm=lambda x:re.sub(r"[^a-z0-9]","",x.lower())
dia={norm(r["Question"]) for r in pq.read_table(f"{BENCH}/data/gpqa_diamond_full.parquet").to_pylist()}
v1q={norm(re.search(r"Question: (.*?)\n\nA\)",i["messages"][0]["content"],re.S).group(1)) for i in v1 if i["bench"]=="gpqa"}
uniq={}
for p in json.load(open(f"{M3}/kdata/fitness_pool.json")):
    if p["task"]!="gpqa": continue
    m=re.match(r"What is the correct answer to this question:(.*)\nChoices:\n\(A\) (.*)\n\(B\) (.*)\n\(C\) (.*)\n\(D\) (.*)\nAnswer: ?$",p["ctx"],re.S)
    qn=norm(m.group(1))
    if qn in dia or qn in v1q or qn in uniq: continue
    uniq[qn]=(p,m)
gp=sorted(uniq.values(),key=lambda v:v[0]["key"]); rng.shuffle(gp); print("gpqa eligible",len(gp))
for s in ["search","val"]:
    for p,m in [gp.pop() for _ in range(SPLIT[s]["gpqa"])]:
        ch=[m.group(i).strip() for i in range(2,6)]; ans="ABCD"[int(p["gold"])]
        q=("Answer the following multiple choice question. Think step by step and then finish your answer with \"The answer is (X)\" where X is one of A, B, C, D.\n\n"
           f"Question: {m.group(1).strip()}\n\nA) {ch[0]}\nB) {ch[1]}\nC) {ch[2]}\nD) {ch[3]}")
        out[s].append(dict(id=f"v2_{p['key'].replace(':','_')}",bench="gpqa",category=p["subtask"],messages=[{"role":"user","content":q}],answer=ans))
# BFCL non-live, stratified over 4 categories
from bfcl_eval.utils import load_dataset_entry
import bfcl_eval
pa=os.path.join(os.path.dirname(bfcl_eval.__file__),"data/possible_answer")
cats=["simple_python","multiple","parallel","parallel_multiple"]; pool={}; gt={}
for cat in cats:
    for line in open(f"{pa}/BFCL_v4_{cat}.json"):
        d=json.loads(line); gt[d["id"]]=d["ground_truth"]
    e=[x for x in load_dataset_entry(cat,include_prereq=False,include_language_specific_hint=True) if x["id"] not in bench_ids and x["id"] not in v1_ids]
    rng.shuffle(e); pool[cat]=e; print("bfcl",cat,"eligible",len(e))
for s in ["search","val"]:
    for cat,k in alloc({c:[0]*1 for c in cats},SPLIT[s]["bfcl"]).items():
        for e in [pool[cat].pop() for _ in range(k)]:
            out[s].append(dict(id=f"v2_{e['id']}",bfcl_id=e["id"],bench="bfcl",category=cat,messages=e["question"][0],tools=e["function"],answer=gt[e["id"]]))
S={i["id"] for i in out["search"]}; V={i["id"] for i in out["val"]}
assert not S&V and len(S)==200 and len(V)==150
allb={i["bfcl_id"] for i in out["search"]+out["val"] if "bfcl_id" in i}
assert not (allb&bench_ids) and not (allb&v1_ids)
for s in out:
    json.dump(out[s],open(f"{s}_items.json","w"))
    print(s,Counter(i["bench"] for i in out[s]),Counter(i["category"] for i in out[s] if i["bench"]=="bfcl"))
