# tinyBenchmarks anchor points (tutorials/anchor_points.ipynb, clustering='irt'): per scenario, KMeans on item IRT
# parameters X = [A_1..A_D, B] (rows = items), sample_weight = balance weights normalised to sum 1 within the scenario
# (uniform here: leaderboard MMLU-Pro acc and our pooled GPQA acc are plain item means), random_state=42, n_init="auto";
# anchor = item closest to each cluster centre; anchor weight = total balance weight of its cluster.
# 200 anchors MMLU-Pro, 100 GPQA, from the fitness pool (test items already excluded).
import json, numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
P = np.load("irt_out/irt_params.npz"); A, B = P["A"], P["B"]
pool = json.load(open("kdata/fitness_pool.json"))
scen = {t: np.array([p["pos"] for p in pool if p["task"] == t]) for t in ["mmlu_pro", "gpqa"]}
NUM = {"mmlu_pro": 200, "gpqa": 100}
X_all = np.vstack((A.squeeze(), B.squeeze().reshape((1, -1)))).T
out = {}
for t, pos in scen.items():
    X = X_all[pos]
    w = np.ones(len(pos)); w /= w.sum()
    km = KMeans(n_clusters=NUM[t], n_init="auto", random_state=42).fit(X, sample_weight=w)
    ap = pairwise_distances(km.cluster_centers_, X, metric="euclidean").argmin(axis=1)
    aw = np.array([w[km.labels_ == c].sum() for c in range(NUM[t])])
    assert len(set(ap.tolist())) == len(ap), "duplicate anchors"
    out[t] = dict(pos=[int(pos[i]) for i in ap], weights=aw.tolist(), keys=[pool[int(pos[i])]["key"] for i in ap])
    print(t, len(ap), "weights sum", aw.sum(), "min/max", aw.min(), aw.max())
json.dump(out, open("kdata/irt_anchors.json", "w"), indent=1)
# validation on leaderboard models (in-sample for IRT): weighted-anchor accuracy vs true scenario accuracy
Y = np.load("kdata/lb_responses.npz")["Y"]
for t, pos in scen.items():
    yh = (Y[:, out[t]["pos"]] * np.array(out[t]["weights"])).sum(1); yt = Y[:, pos].mean(1)
    print(t, "anchor-weighted acc MAE on leaderboard models", np.abs(yh - yt).mean())
