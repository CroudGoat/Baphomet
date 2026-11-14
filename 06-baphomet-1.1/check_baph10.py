# Uniform Baphomet-1.0 merge (lambda 0.1375 x 8) of a few tensors from the same fp8 TVs, compared with the published 1.0 weights.
import sys, json, torch
sys.path.insert(0, "/workspace/work/evogm/kds"); import linmerge
from safetensors import safe_open
BASE = "/workspace/work/baphomet11/m/base"; TVROOT = "/workspace/work/baphomet11/tv"
idx = [json.load(open(f"{TVROOT}/{s}/index.json")) for s in linmerge.SHORT]
bw = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]
def merged(k, lams):
    with safe_open(f"{BASE}/{bw[k]}", "pt") as s: bt = s.get_tensor(k)
    acc = bt.float().clone()
    for i in range(8):
        if k + "::q" not in idx[i] or lams[i] == 0: continue
        with safe_open(f"{TVROOT}/{linmerge.SHORT[i]}/{idx[i][k+'::q']}", "pt") as s: q = s.get_tensor(k + "::q"); sc = s.get_tensor(k + "::s")
        acc.add_(q.float() * sc, alpha=float(lams[i]))
    return bt, acc.to(bt.dtype)
P = "/workspace/work/baphomet11/m/baph10/model-00002-of-00010.safetensors"
with safe_open(P, "pt") as s:
    keys = sorted(s.keys())
    for k in keys:
        pub = s.get_tensor(k); bt, m = merged(k, [0.1375] * 8)
        assert pub.shape == m.shape and pub.dtype == m.dtype, (k, pub.dtype, m.dtype)
        d = (m.float() - pub.float()); dp = (pub.float() - bt.float())
        print(k, str(pub.dtype), "exact%", round(100 * float((m == pub).float().mean()), 2), "max|diff|", float(d.abs().max()),
              "rel_err_vs_delta", round(float(d.norm() / dp.norm().clamp_min(1e-30)), 3))
