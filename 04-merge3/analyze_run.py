# Post-run analysis of kernel merge3-run outputs (kout/run) -> kout/run/analysis.json
import json, os, sys, numpy as np
K = "/workspace/work/merge3/kout/run"
DON = ["InternScience/Agents-A1-4B", "Jackrong/DeepSeek-V4-Pro-Qwen3.5-4B", "empero-ai/Qwen3.8-4B-Distill",
       "Qwopus3.5-4B-v3", "AREX-Turbo"]
sys.path.insert(0, "/workspace/work/merge3/kaggle_src")
import m3lib
DON = m3lib.DONORS
out = {}
evo = json.load(open(f"{K}/evo_result.json"))
recs = [json.loads(l) for l in open(f"{K}/evo_log.jsonl")]
out["status"] = evo["status"]; out["n_evaluated"] = evo["n_evaluated"]; out["evo_hours"] = evo["hours"]
out["n_log_lines"] = len(recs); out["n_unique"] = len({r["gkey"] for r in recs})
b = evo["best"]; x = b["x"]
out["winner"] = dict(tag=b["tag"], gpu=b["gpu"], dare_seed=b["dare_seed"], mean_fit=b["mean_fit"],
                     fit={t: {k: v for k, v in b["fit"][t].items() if k != "lam"} for t in b["fit"]},
                     coeffs=[dict(donor=DON[i], w=x[i], d=x[5 + i]) for i in range(5)])
def mf(r): return float(np.mean([r["fit"][t]["gmpirt"] for t in ["mmlu_pro", "gpqa"]]))
seeds = {"seed_equal": [0.2] * 5 + [1.0] * 5, "seed_base": [0.0] * 5 + [1.0] * 5}
gk = lambda v: ",".join(f"{a:.12f}" for a in v)
byk = {r["gkey"]: r for r in recs}
out["seeds"] = {n: (dict(fit={t: byk[gk(v)]["fit"][t]["gmpirt"] for t in ["mmlu_pro", "gpqa"]}, mean=mf(byk[gk(v)])) if gk(v) in byk else None)
                for n, v in seeds.items()}
gens = {}
for r in recs: gens.setdefault(r["tag"], []).append(mf(r))
out["per_gen"] = {g: dict(n=len(v), best=max(v), mean=float(np.mean(v))) for g, v in gens.items()}
out["swap_s_median"] = float(np.median([r["t_swap"] for r in recs])); out["cand_s_median"] = float(np.median([r["t_total"] for r in recs]))
summ = json.load(open(f"{K}/endpoint_summary.json"))["summary"]
out["endpoints"] = summ
if os.path.exists(f"{K}/endpoint_merge3_winner.json"):
    R = json.load(open(f"{K}/endpoint_merge3_winner.json"))["results"]
    ia = json.load(open("/workspace/work/merge3/kdata/irt_anchors.json"))
    ra = json.load(open("/workspace/work/merge3/kdata/anchors.json"))
    w = {}
    for t in ["mmlu_pro", "gpqa"]:
        w[t] = dict(anchor_weighted_acc=float(sum(R[k]["correct"] * wt for k, wt in zip(ia[t]["keys"], ia[t]["weights"]))),
                    reduced_acc=float(np.mean([R[a["key"]]["correct"] for a in ra[t]])))
    out["winner_on_endpoint_items"] = w
json.dump(out, open(f"{K}/analysis.json", "w"), indent=1)
print(json.dumps(out, indent=1))
