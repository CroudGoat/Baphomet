# Build kds_dfs/v2_prev.json (dataset cloudgoat/evogm-dfs-data) from the finished EvoGM v2 run (kout/main):
#   donors' 16-dim lambdas (r1i1, B11, B10), Qwen3.5 base validation record (BFCL floor), v2 records of B11 / r1i1 (search full + val),
#   all v2 full-200 rows (per-item correct + tokens; IRT calibration rows) and the IRT fit in use at the end of v2 (irt_v2_refit_r1.json).
import json, gzip, sys
K = "/workspace/work/evogm2/kout/main"
sys.path.insert(0, "/workspace/work/evogm2/kds")
ev = {}
for l in open(f"{K}/evo_log.jsonl"):
    r = json.loads(l); ev.setdefault(r["lkey"], {}).update(r)
vl = {}
for l in open(f"{K}/val_log.jsonl"):
    r = json.loads(l); vl[r["lkey"]] = r
B11 = [0.0116, 0.2039, 0.5901, 0.0013, 0.2972, 0.0323, 0.0030, -0.0007]
lams_b11 = B11 + B11; lams_b10 = [0.1375] * 16
r1 = [r for r in ev.values() if r["lkey"].startswith("0.027353,0.149812")]
assert len(r1) == 1; r1 = r1[0]
assert r1.get("round") == 1 and r1.get("iter") == 1 and r1["kind"] == "generated"
lkey = lambda x: ",".join(f"{float(v):.6f}" for v in x)
kb11 = lkey(lams_b11); assert kb11 in ev and ev[kb11]["kind"] == "baphomet11"
kb10 = lkey(lams_b10); assert kb10 in ev
base = [v for v in vl.values() if v.get("kind") == "qwen35_base"][0]
SEARCH = json.load(open("/workspace/work/evogm2/kds/search_items.json")); TASKS = ["mmlupro", "gpqa", "bfcl"]
rows = []
for r in sorted(ev.values(), key=lambda r: r["lkey"]):
    if r.get("full_fitness") is None or r.get("failed"): continue
    d = json.load(gzip.open(f"{K}/fit_outputs/{r['tag']}_full.json.gz", "rt"))
    sc = d["scores"]; tk = {o["id"]: o["completion_tokens"] for o in d["outputs"]}
    fr = {o["id"]: o["finish_reason"] for o in d["outputs"]}
    assert sorted(sc) == sorted(i["id"] for i in SEARCH)
    rows.append(dict(lkey=r["lkey"], tag=r["tag"], kind=r.get("kind"), round=r.get("round"), iter=r.get("iter"), full_fitness=r["full_fitness"],
                     full_acc=r.get("full_acc"), t_gen=r.get("t_gen"), t_swap=r.get("t_swap"), t_total=r.get("t_total"), t_complete=r.get("t_complete"),
                     correct={i: int(bool(s["correct"]) and not s["truncated"]) for i, s in sc.items()}, tokens=tk, finish={i: fr[i] for i in sc}))
irt = json.load(open(f"{K}/irt_v2_refit_r1.json"))
assert irt["n_cal"] == len(rows), (irt["n_cal"], len(rows))
assert set(irt["cal_lkeys"]) == {r["lkey"] for r in rows}
def strip(r, keys): return {k: r.get(k) for k in keys}
EK = ("lkey", "lams", "tag", "kind", "round", "iter", "full_fitness", "full_acc", "full_trunc", "full_mean_tokens", "full_gen_tokens", "t_total", "t_swap", "t_gen", "t_complete")
VK = ("lkey", "lams", "tag", "kind", "val_fitness", "val_acc", "val_trunc", "val_mean_tokens", "val_gen_tokens", "t_total", "t_swap")
out = dict(
    note="EvoGM v2 (cloudgoat/evogm2-main + evogm2-final) results reused by EvoGM-DFS. Built by /workspace/work/evogm-dfs/build_prev.py on 2026-11-17.",
    donors=dict(r1i1=dict(lams=r1["lams"], lkey=r1["lkey"]), b11=dict(lams=lams_b11, lkey=kb11), b10=dict(lams=lams_b10, lkey=kb10)),
    base_val=strip(base, VK),
    v2_records=dict(r1i1=dict(evo=strip(r1, EK), val=strip(vl.get(r1["lkey"]), VK) if r1["lkey"] in vl else None,
                              final180=dict(mmlupro=[59, 80], gpqa=[28, 50], bfcl=[47, 50], total=[134, 180])),
                    b11=dict(evo=strip(ev[kb11], EK), val=strip(vl.get(kb11), VK) if kb11 in vl else None,
                             final180=dict(mmlupro=[62, 80], gpqa=[29, 50], bfcl=[49, 50], total=[140, 180])),
                    b10=dict(evo=strip(ev[kb10], EK), val=None)),
    rows=rows, irt=irt)
json.dump(out, open("/workspace/work/evogm-dfs/kds_dfs/v2_prev.json", "w"))
json.dump({"title": "evogm-dfs-data", "id": "cloudgoat/evogm-dfs-data", "licenses": [{"name": "CC0-1.0"}]},
          open("/workspace/work/evogm-dfs/kds_dfs/dataset-metadata.json", "w"))
print("rows", len(rows), "r1i1", r1["lkey"][:40], r1["full_fitness"], "val", (vl.get(r1["lkey"]) or {}).get("val_fitness"),
      "b11", ev[kb11]["full_fitness"], (vl.get(kb11) or {}).get("val_fitness"), "b10", ev[kb10]["full_fitness"], "base bfcl", base["val_acc"]["bfcl"])
