# MERGE^3 Estimate step (run locally after kernels I and E):
#   gamma_m (endpoint abilities) = fit_theta on each endpoint's responses over the FULL fitness pool D, with the IRT item
#   parameters fitted on the Open LLM Leaderboard. Writes kdata/irt_estimate.npz + kdata/evo_config.json.
import json, sys, os, numpy as np
sys.path.insert(0, "/workspace/work/merge3/kaggle_src")
import m3lib
os.chdir("/workspace/work/merge3")
P = np.load("irt_out/irt_params.npz"); A, B = P["A"], P["B"]
pool = json.load(open("kdata/fitness_pool.json"))
N = len(pool); assert A.shape[2] == N, (A.shape, N)
scen = {t: [p["pos"] for p in pool if p["task"] == t] for t in ["mmlu_pro", "gpqa"]}
order = m3lib.SHORT + ["base"]
out = dict(A=A, B=B, scen_mmlu_pro=np.array(scen["mmlu_pro"]), scen_gpqa=np.array(scen["gpqa"]))
summ = {}
for name in order + ["baphomet"]:
    R = json.load(open(f"endpoints_out/endpoint_{name}.json"))["results"]
    y = np.array([R[p["key"]]["correct"] for p in pool], dtype=float)
    th = m3lib.fit_theta(y, list(range(N)), A, B)
    out["theta_" + name] = th
    pred = m3lib.item_curve(th.reshape(1, A.shape[1], 1), A, B)[0]
    summ[name] = {t: dict(acc=float(y[scen[t]].mean()), irt_fit_acc=float(pred[scen[t]].mean())) for t in scen}
    print(name, summ[name], flush=True)
np.savez_compressed("kdata/irt_estimate.npz", **out)
cfg = dict(tasks=["mmlu_pro", "gpqa"], theta_order=order, mode="gmpirt", seed=420, pop_size=25, n_gen=7,
           seed_equal=[0.2] * 5 + [1.0] * 5, seed_base=[0.0] * 5 + [1.0] * 5, n_anchor=20)
json.dump(cfg, open("kdata/evo_config.json", "w"), indent=1)
json.dump(summ, open("endpoint_accuracy_summary.json", "w"), indent=1)
