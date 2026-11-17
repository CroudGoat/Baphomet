# Keep only the 20 mid-difficulty BFCL items (0<p<1 in prescreen); refill 20 slots with MMLU-Pro +10 / GPQA +10.
# New items are disjoint from bench pool (incl. 180 test), v1 fitness set, current search and val sets.
import json, re, random, sys
sys.path.insert(0, "/workspace/work/bench/.pylib")
import pyarrow.parquet as pq
from collections import defaultdict, Counter
sys.path.insert(0, "/workspace/work/bench/.pylib")
BENCH="/workspace/work/bench"; M3="/workspace/work/merge3"
rng=random.Random(20261116)
S=json.load(open("search_items.json")); V=json.load(open("val_items.json")); P=json.load(open("bfcl_pre_p.json"))
keep=[i for i in S if i["bench"]!="bfcl" or 0<P[i["id"]]<1]
assert sum(i["bench"]=="bfcl" for i in keep)==20
used={i["id"] for i in S+V}
bench_ids={json.loads(l)["id"] for l in open(f"{BENCH}/items.jsonl")}
v1=json.load(open("/workspace/work/evogm/kds/fitness_items.json")); v1_ids={i["id"] for i in v1}
L="ABCDEFGHIJ"; byc=defaultdict(list)
for r in pq.read_table(f"{BENCH}/data/mmlupro_test.parquet").to_pylist():
    q=r["question_id"]
    if f"mmlupro_{q}" in bench_ids or f"fit_mmlupro_{q}" in v1_ids or f"v2_mmlupro_{q}" in used: continue
    byc[r["category"]].append(r)
cats=sorted(byc); rng.shuffle(cats)
for c in byc: rng.shuffle(byc[c])
new=[]
for c in cats[:10]:  # 10 items, one each from 10 random categories
    r=byc[c].pop(); opts="\n".join(f"{L[i]}. {o}" for i,o in enumerate(r["options"]))
    q=("The following is a multiple choice question about "+c+". Think step by step and then finish your answer with \"The answer is (X)\" where X is the correct letter choice.\n\n"
       f"Question: {r['question']}\nOptions:\n{opts}")
    new.append(dict(id=f"v2_mmlupro_{r['question_id']}",bench="mmlupro",category=c,messages=[{"role":"user","content":q}],answer=r["answer"],nopts=len(r["options"])))
norm=lambda x:re.sub(r"[^a-z0-9]","",x.lower())
dia={norm(r["Question"]) for r in pq.read_table(f"{BENCH}/data/gpqa_diamond_full.parquet").to_pylist()}
v1q={norm(re.search(r"Question: (.*?)\n\nA\)",i["messages"][0]["content"],re.S).group(1)) for i in v1 if i["bench"]=="gpqa"}
usedq={norm(re.search(r"Question: (.*?)\n\nA\)",i["messages"][0]["content"],re.S).group(1)) for i in S+V if i["bench"]=="gpqa"}
uniq={}
for p in json.load(open(f"{M3}/kdata/fitness_pool.json")):
    if p["task"]!="gpqa": continue
    m=re.match(r"What is the correct answer to this question:(.*)\nChoices:\n\(A\) (.*)\n\(B\) (.*)\n\(C\) (.*)\n\(D\) (.*)\nAnswer: ?$",p["ctx"],re.S)
    qn=norm(m.group(1))
    if qn in dia or qn in v1q or qn in usedq or qn in uniq: continue
    uniq[qn]=(p,m)
gp=sorted(uniq.values(),key=lambda v:v[0]["key"]); rng.shuffle(gp); print("gpqa eligible",len(gp))
for p,m in gp[:10]:
    ch=[m.group(i).strip() for i in range(2,6)]; ans="ABCD"[int(p["gold"])]
    q=("Answer the following multiple choice question. Think step by step and then finish your answer with \"The answer is (X)\" where X is one of A, B, C, D.\n\n"
       f"Question: {m.group(1).strip()}\n\nA) {ch[0]}\nB) {ch[1]}\nC) {ch[2]}\nD) {ch[3]}")
    new.append(dict(id=f"v2_{p['key'].replace(':','_')}",bench="gpqa",category=p["subtask"],messages=[{"role":"user","content":q}],answer=ans))
out=keep+new; ids=[i["id"] for i in out]
assert len(out)==200 and len(set(ids))==200 and not set(ids)&{i["id"] for i in V}
json.dump(out,open("search_items.json","w"))
print(Counter(i["bench"] for i in out), Counter(i["category"] for i in out if i["bench"]=="bfcl"))
