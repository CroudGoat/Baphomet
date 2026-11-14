import sys, os, json, math, torch
sys.path.insert(0, "/workspace/work/evogm/kds"); import linmerge
exec(open("extract.py").read().split("base_dir, donor_dir, remap, out_dir =")[0].split("torch.set_num_threads(2)")[1].replace("linmerge.tensor_index = _tensor_index",""), globals())
from safetensors import safe_open
k = sys.argv[1]
bidx = _tensor_index("m/base"); b = RawST(bidx[k]).get_slice(k)[:].float()
acc = b.clone(); acc8 = b.clone()
for (s, repo, remap) in linmerge.DONORS:
    di = _tensor_index(f"m/{s}"); dk = k
    if remap: dk = "model." + k[len("model.language_model."):] if k.startswith("model.language_model.") else None
    if dk not in di: continue
    d = RawST(di[dk]).get_slice(dk)[:].float(); t = d - b
    acc.add_(t, alpha=0.1375)
pub = safe_open("m/baph10/model-00002-of-00010.safetensors", "pt").get_tensor(k)
ex = acc.to(pub.dtype)
print(k, "exact-fp32 merge vs published: exact%", round(100*float((ex == pub).float().mean()), 2), "max", float((ex.float()-pub.float()).abs().max()),
      "rel_vs_delta", round(float((ex.float()-pub.float()).norm()/(pub.float()-b).norm()), 3))
