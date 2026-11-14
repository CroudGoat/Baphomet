# Validate the fp8-TV linear merge against the published Baphomet-1.0-4B on the tensors of the downloaded Baphomet shards.
#   merge      : base + sum 0.1375*T_i (what the evolution uses)
#   no_neohorse/no_tmax : the same without one donor (shows whether that donor's TV/key mapping matters)
#   base       : the unmerged base (reference: err = 1 by definition of the metric)
# metric: rel_err = ||X - Baph|| / ||Baph - base||, plus max |X - Baph| and #bit-identical tensors.
# usage: CUDA_VISIBLE_DEVICES=1 python valmerge.py BASE TVDIR BAPH OUT
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, linmerge
from safetensors import safe_open
base, tvdir, baph, out = sys.argv[1:5]
t0 = time.time()
M = linmerge.Merger(base, [f"{tvdir}/{s}.safetensors" for s in linmerge.SHORT])
ref_idx = linmerge.tensor_index(baph); keys = set(ref_idx)
H = {}
def h(f):
    if f not in H: H[f] = safe_open(f, "pt", device="cpu")
    return H[f]
V = {"merge": list(linmerge.BAPHOMET_LAMBDA)}
for drop in ["neohorse", "tmax"]:
    l = list(linmerge.BAPHOMET_LAMBDA); l[linmerge.SHORT.index(drop)] = 0.0; V["no_" + drop] = l
res = {}; den = 0.0
for name, lam in V.items():
    num = 0.0; maxabs = 0.0; exact = 0; n = 0; dd = 0.0
    for k, t in M.iter_merged(lam, device="cuda", only=keys):
        ref = h(ref_idx[k]).get_tensor(k).float()
        b = h(M.bidx[k]).get_tensor(k).float()
        d = t.float().cpu() - ref
        num += float((d * d).sum()); dd += float(((ref - b) ** 2).sum()); n += 1
        ma = float(d.abs().max()) if d.numel() else 0.0; maxabs = max(maxabs, ma); exact += int(ma == 0.0)
    den = dd
    res[name] = dict(rel_err=math.sqrt(num / max(dd, 1e-30)), max_abs=maxabs, n_exact=exact, n_tensors=n)
    print(name, res[name], flush=True)
json.dump(dict(variants=res, n_keys=len(keys), secs=time.time() - t0), open(out, "w"), indent=1)
