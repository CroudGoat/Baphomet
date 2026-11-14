# CPU port of linmerge.extract_tv (identical math: fp32 diff, per-row absmax fp8 e4m3, all-zero tensors skipped),
# processed in row chunks (row-wise ops -> chunking does not change any value). Output: grouped safetensors + index.
import sys, os, json, math, time, torch
sys.path.insert(0, "/workspace/work/evogm/kds")
import linmerge
from safetensors import safe_open
from safetensors.torch import save_file
torch.set_num_threads(2)

import struct
_DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "F64": torch.float64, "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8}
class _RawSlice:
    def __init__(self, f, data0, meta): self.f, self.data0, self.meta = f, data0, meta
    def get_shape(self): return list(self.meta["shape"])
    def __getitem__(self, sl):
        shp = self.meta["shape"]; dt = _DT[self.meta["dtype"]]; es = torch.empty(0, dtype=dt).element_size()
        s0, s1 = self.meta["data_offsets"]
        if isinstance(sl, slice) and sl.start is not None:
            per = math.prod(shp[1:]) * es; r0, r1 = sl.start, min(sl.stop, shp[0]); a, b = s0 + r0 * per, s0 + r1 * per; out = [r1 - r0] + list(shp[1:])
        else:
            a, b = s0, s1; out = list(shp)
        buf = bytearray(b - a); self.f.seek(self.data0 + a); self.f.readinto(buf)
        return torch.frombuffer(buf, dtype=dt).reshape(out).clone() if b > a else torch.empty(out, dtype=dt)
class RawST:   # header-parsing reader with seek/read (no mmap: 9.3 GB single-file donors fail to mmap in this sandbox)
    def __init__(self, path):
        self.f = open(path, "rb"); n = struct.unpack("<Q", self.f.read(8))[0]; self.h = json.loads(self.f.read(n)); self.data0 = 8 + n
    def get_slice(self, k): return _RawSlice(self.f, self.data0, self.h[k])
import glob
def _tensor_index(model_dir):   # linmerge.tensor_index, but listing single-file checkpoints via the header (no mmap)
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(model_dir, v) for k, v in wm.items() if os.path.exists(os.path.join(model_dir, v))}
    out = {}
    for f in glob.glob(os.path.join(model_dir, "*.safetensors")):
        for k in RawST(f).h:
            if k != "__metadata__": out[k] = f
    return out
linmerge.tensor_index = _tensor_index
base_dir, donor_dir, remap, out_dir = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4]
os.makedirs(out_dir, exist_ok=True)
bidx = linmerge.tensor_index(base_dir); dmap = linmerge.donor_map(donor_dir, remap)
H = {}
def h(f):
    if f not in H: H[f] = RawST(f)
    return H[f]
st = dict(n=0, zero=0, missing=0, sqnorm=0.0, qerr=0.0); group = {}; gbytes = 0; gi = 0; index = {}
GROUP_MAX = 1 << 30; CH = linmerge.CHUNK_ELEMS
def flush():
    global group, gbytes, gi
    if not group: return
    fn = f"tv-{gi:03d}.safetensors"; save_file(group, os.path.join(out_dir, fn))
    for k in group: index[k] = fn
    group = {}; gbytes = 0; gi += 1
t0 = time.time()
for k in sorted(bidx):
    if k not in dmap: st["missing"] += 1; continue
    bs = h(bidx[k]).get_slice(k); ds_ = h(dmap[k][0]).get_slice(dmap[k][1])
    shp = bs.get_shape(); assert shp == ds_.get_shape(), (k, shp, ds_.get_shape())
    st["n"] += 1
    if len(shp) >= 2:
        rows = shp[0]; per_row = math.prod(shp[1:]); rstep = max(1, CH // max(per_row, 1))
        ranges = [(r0, min(rows, r0 + rstep)) for r0 in range(0, rows, rstep)]
    else:
        ranges = [None]
    qs, ss = [], []; nonzero = False; sq = 0.0; qe = 0.0
    for rg in ranges:
        if rg is None: b = bs[:].float(); d = ds_[:].float()
        else: b = bs[rg[0]:rg[1]].float(); d = ds_[rg[0]:rg[1]].float()
        t = d - b
        if not nonzero and torch.any(t != 0): nonzero = True
        q, s = linmerge.quant_fp8(t)
        sq += float((t * t).sum())   # fp32 per chunk (linmerge: fp32 per tensor); qerr not recomputed (speed)
        qs.append(q); ss.append(s); del b, d, t
    if not nonzero: st["zero"] += 1; continue
    q = torch.cat(qs, 0) if len(qs) > 1 else qs[0]; s = torch.cat(ss, 0) if len(ss) > 1 else ss[0]
    st["sqnorm"] += sq; st["qerr"] += qe
    group[k + "::q"] = q.contiguous(); group[k + "::s"] = s.contiguous(); gbytes += q.numel() + 4 * s.numel()
    if gbytes > GROUP_MAX: flush()
flush()
st["rel_qerr"] = math.sqrt(st["qerr"] / max(st["sqnorm"], 1e-30)); st["norm"] = math.sqrt(st["sqnorm"]); st["secs"] = round(time.time() - t0)
json.dump(index, open(os.path.join(out_dir, "index.json"), "w"))
json.dump(st, open(os.path.join(out_dir, "stats.json"), "w"), indent=1)
print("DONE", st, flush=True)
