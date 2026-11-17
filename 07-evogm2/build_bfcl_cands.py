# 40 BFCL candidates per category (incl. the 10 currently in search), excluding val, bench pool, v1.
import json, random, sys, os
sys.path.insert(0, "/workspace/work/bench/.pylib")
from bfcl_eval.utils import load_dataset_entry
import bfcl_eval
bench_ids={json.loads(l)["id"] for l in open("/workspace/work/bench/items.jsonl")}
v1={i.get("bfcl_id") for i in json.load(open("/workspace/work/evogm/kds/fitness_items.json"))}
S=json.load(open("search_items.json")); V=json.load(open("val_items.json"))
val_b={i["bfcl_id"] for i in V if i["bench"]=="bfcl"}; cur={i["bfcl_id"]:i for i in S if i["bench"]=="bfcl"}
pa=os.path.join(os.path.dirname(bfcl_eval.__file__),"data/possible_answer"); rng=random.Random(777); out=[]
for cat in ["simple_python","multiple","parallel","parallel_multiple"]:
    gt={}
    for line in open(f"{pa}/BFCL_v4_{cat}.json"):
        d=json.loads(line); gt[d["id"]]=d["ground_truth"]
    ents=load_dataset_entry(cat,include_prereq=False,include_language_specific_hint=True)
    el=[e for e in ents if e["id"] not in bench_ids|v1|val_b|set(cur)]
    pick=[cur[k] for k in cur if cur[k]["category"]==cat]
    for e in rng.sample(el,40-len(pick)):
        pick.append(dict(id=f"v2_{e['id']}",bfcl_id=e["id"],bench="bfcl",category=cat,messages=e["question"][0],tools=e["function"],answer=gt[e["id"]]))
    out+=pick
json.dump(out,open("bfcl_cands.json","w")); print(len(out), os.path.getsize("bfcl_cands.json"))
