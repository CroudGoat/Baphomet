# Build the MERGE^3 fitness pool D (leaderboard v2 MMLU-Pro + GPQA diamond/main/extended, minus every test item),
# the random reduced set D-bar (20 items per task, seed 420; MERGE^3 Extract = uniform random sampling),
# and the leaderboard response matrix used to fit the IRT item parameters.
import json, glob, os, numpy as np, collections
os.chdir("/workspace/work/merge3")
SEED, N_ANCHOR = 420, 20
allit = json.load(open("irt/fitness_items_all.json"))
short = {"leaderboard_mmlu_pro": "mmlu_pro", "leaderboard_gpqa_diamond": "gpqa_diamond", "leaderboard_gpqa_main": "gpqa_main",
         "leaderboard_gpqa_extended": "gpqa_extended"}
order = ["leaderboard_mmlu_pro", "leaderboard_gpqa_diamond", "leaderboard_gpqa_main", "leaderboard_gpqa_extended"]
pool = []
for t in order:
    rows = sorted([i for i in allit if i["task"] == t and not i["excluded"]], key=lambda r: r["doc_id"])
    for r in rows:
        pool.append(dict(key=f"{short[t]}:{r['doc_id']}", task="mmlu_pro" if t.endswith("mmlu_pro") else "gpqa", subtask=short[t],
                         doc_id=r["doc_id"], ctx=r["ctx"], conts=r["conts"], gold=r["gold"], qid=r["qid"]))
for i, p in enumerate(pool): p["pos"] = i
scen = {t: [p["pos"] for p in pool if p["task"] == t] for t in ["mmlu_pro", "gpqa"]}
print({t: len(v) for t, v in scen.items()}, collections.Counter(p["subtask"] for p in pool))
os.makedirs("kdata", exist_ok=True)
json.dump(pool, open("kdata/fitness_pool.json", "w"))
# reduced dataset: uniform random sample per task (pandas df.sample(n, random_state=seed) equivalent)
rs = np.random.RandomState(SEED)
anchors = {}
for t in ["mmlu_pro", "gpqa"]:
    idx = rs.choice(len(scen[t]), N_ANCHOR, replace=False)
    anchors[t] = [pool[scen[t][i]] for i in idx]
json.dump(anchors, open("kdata/anchors.json", "w"))
print({t: [a["key"] for a in v] for t, v in anchors.items()})
# response matrix from leaderboard details
files = sorted(glob.glob("irt/resp/*.json"))
keypos = {p["key"]: p["pos"] for p in pool}
Y, names, hashes = [], [], collections.defaultdict(collections.Counter)
for f in files:
    d = json.load(open(f))
    y = np.full(len(pool), np.nan, dtype=np.float32)
    for t, v in d.items():
        st = short[t]
        for j, (di, val) in enumerate(zip(v["doc_id"], v["y"])):
            k = f"{st}:{di}"
            if k in keypos: y[keypos[k]] = float(val)
            if v.get("doc_hash"): hashes[k][v["doc_hash"][j]] += 1
    if np.isnan(y).mean() > 0.01: print("skip (missing)", f, np.isnan(y).mean()); continue
    Y.append(y); names.append(os.path.basename(f)[:-5])
Y = np.array(Y)
print("response matrix", Y.shape, "nan", np.isnan(Y).sum(), "mean acc mmlu", np.nanmean(Y[:, scen["mmlu_pro"]]), "gpqa", np.nanmean(Y[:, scen["gpqa"]]))
cons = [max(c.values()) / sum(c.values()) for c in hashes.values()]
print("GPQA doc_hash agreement across models: mean", np.mean(cons), "min", np.min(cons))
np.savez_compressed("kdata/lb_responses.npz", Y=Y, names=np.array(names), scen_mmlu_pro=np.array(scen["mmlu_pro"]), scen_gpqa=np.array(scen["gpqa"]))
