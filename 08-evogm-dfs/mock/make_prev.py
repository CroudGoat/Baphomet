# mock v2_prev.json consistent with gpuworker_mock (same correctness functions): 16 "v2" full rows (B11 = the plain-B11 plan),
# base val (BFCL from the mock unless MOCK_BASEBFCL), v2 r1i1 / B11 val records.  No "irt" -> the kernel fits the 2PL in-kernel.
# usage: make_prev.py OUT_JSON
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/workspace/work/evogm-dfs/src")
import gpuworker_mock as M, dfslib
K = "/workspace/work/evogm2/kds"
SEARCH = json.load(open(f"{K}/search_items.json")); VALI = json.load(open(f"{K}/val_items.json"))
TASKS = ["mmlupro", "gpqa", "bfcl"]
real = json.load(open("/workspace/work/evogm-dfs/kds_dfs/v2_prev.json"))
def acc(items, recs):
    by = {r["id"]: r for r in recs}
    return {t: sum(1 for i in items if i["bench"] == t and by[i["id"]]["output"] == "OK") / sum(1 for i in items if i["bench"] == t) for t in TASKS}
def mean(d): return sum(d.values()) / len(d)
b11plan = dfslib.plan_of(dfslib.decode(dfslib.geno_plain(1)))
rows = []
thetas = [("b11", M.theta_of(b11plan), M.pkey(b11plan))] + [(f"v2m{k}", 0.25 + 0.06 * k, f"v2m{k}") for k in range(15)]
for name, th, key in thetas:
    recs = M.gen_recs(SEARCH, th, key, 4096)
    lk = real["donors"]["b11"]["lkey"] if name == "b11" else f"mock-{name}"
    a = acc(SEARCH, recs)
    rows.append(dict(lkey=lk, tag=name, kind=name, full_fitness=mean(a), full_acc=a, t_gen=M.simsecs([r["completion_tokens"] for r in recs], cap=48, slow=1.0),
                     t_swap=120.0, correct={r["id"]: int(r["output"] == "OK") for r in recs}, tokens={r["id"]: r["completion_tokens"] for r in recs},
                     finish={r["id"]: r["finish_reason"] for r in recs}))
def val(th, key):
    recs = M.gen_recs(VALI, th, key, 4096); a = acc(VALI, recs)
    return dict(val_fitness=mean(a), val_acc=a, t_total=M.simsecs([r["completion_tokens"] for r in recs], cap=48, slow=1.0))
bv = val(0.3, "base")
if os.environ.get("MOCK_BASEBFCL"):
    bv["val_acc"]["bfcl"] = float(os.environ["MOCK_BASEBFCL"]); bv["val_fitness"] = mean(bv["val_acc"])
b11row = rows[0]
out = dict(note="MOCK", donors=real["donors"], base_val=dict(lkey="base", kind="qwen35_base", **bv),
           v2_records=dict(r1i1=dict(evo=dict(lkey=real["donors"]["r1i1"]["lkey"], full_fitness=mean(acc(SEARCH, M.gen_recs(SEARCH, 1.15, "v2r1i1", 4096)))),
                                     val=dict(lkey=real["donors"]["r1i1"]["lkey"], **val(1.15, "v2r1i1"))),
                           b11=dict(evo=dict(lkey=b11row["lkey"], full_fitness=b11row["full_fitness"], full_acc=b11row["full_acc"]),
                                    val=dict(lkey=b11row["lkey"], **val(M.theta_of(b11plan), M.pkey(b11plan))))),
           rows=rows)
json.dump(out, open(sys.argv[1], "w"))
print("mock prev: b11 full", round(b11row["full_fitness"], 4), "v2r1i1 full", round(out["v2_records"]["r1i1"]["evo"]["full_fitness"], 4),
      "val b11", round(out["v2_records"]["b11"]["val"]["val_fitness"], 4), "v2r1i1", round(out["v2_records"]["r1i1"]["val"]["val_fitness"], 4), "base bfcl", bv["val_acc"]["bfcl"])
