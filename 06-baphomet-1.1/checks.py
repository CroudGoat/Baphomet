import sys, os, json, struct, math, torch
exec(open("/workspace/work/baphomet11/extract.py").read().split("base_dir, donor_dir, remap, out_dir =")[0].split("torch.set_num_threads(2)")[1].replace("linmerge.tensor_index = _tensor_index",""), globals())
sys.path.insert(0, "/workspace/work/evogm/kds"); import linmerge
from safetensors import safe_open
BASE = "/workspace/work/baphomet11/m/base"; OUT = "/workspace/work/baphomet11/Baphomet-1.1-4B"; TVROOT = "/workspace/work/baphomet11/tv"
def hdr(p):
    with open(p, "rb") as f: n = struct.unpack("<Q", f.read(8))[0]; return f.read(n)
wb = json.load(open(f"{BASE}/model.safetensors.index.json")); wo = json.load(open(f"{OUT}/model.safetensors.index.json"))
assert wb == wo
rep = {}
for sh in sorted(set(wb["weight_map"].values())):
    hb, ho = hdr(f"{BASE}/{sh}"), hdr(f"{OUT}/{sh}")
    rep[sh] = dict(header_identical=hb == ho, size_base=os.path.getsize(f"{BASE}/{sh}"), size_out=os.path.getsize(f"{OUT}/{sh}"))
    with safe_open(f"{OUT}/{sh}", "pt") as so, safe_open(f"{BASE}/{sh}", "pt") as sb:
        assert set(so.keys()) == set(sb.keys())
        for k in so.keys():
            a, b = so.get_slice(k), sb.get_slice(k)
            assert a.get_dtype() == b.get_dtype() and a.get_shape() == b.get_shape(), k
print(json.dumps(rep, indent=1))
# finiteness + norms on all tensors (streamed)
RO = {}; DT_ = _DT; nonfinite = 0; changed = 0; total = 0; rows_ = []
for k, sh in sorted(wb["weight_map"].items()):
    if sh not in RO: RO[sh] = (RawST(f"{OUT}/{sh}"), RawST(f"{BASE}/{sh}"))
    so, sb = RO[sh][0].get_slice(k), RO[sh][1].get_slice(k); shp = so.get_shape()
    rows = shp[0] if len(shp) >= 2 else None; step = max(1, (1 << 23) // max(1, math.prod(shp[1:]))) if rows else None
    ch = False; sel = any(s_ in k for s_ in ["embed_tokens", "layers.0.mlp.down_proj", "layers.31.self_attn.o_proj", "layers.3.self_attn.q_proj", "visual.blocks.0.attn.qkv.weight", "mtp.layers.0.mlp.up_proj", "layers.0.linear_attn.A_log", "model.language_model.norm.weight", "layers.12.linear_attn.norm.weight"])
    b2 = o2 = d2 = 0.0; dmax = 0.0
    for r in (range(0, rows, step) if rows else [None]):
        o = so[r:r + step] if rows else so[:]; b = sb[r:r + step] if rows else sb[:]
        of, bf = o.float(), b.float()
        nonfinite += int((~torch.isfinite(of)).sum()); ch = ch or bool((o != b).any())
        if sel:
            dd = (of - bf).double(); b2 += float((bf.double() ** 2).sum()); o2 += float((of.double() ** 2).sum()); d2 += float((dd ** 2).sum()); dmax = max(dmax, float(dd.abs().max()))
        del o, b, of, bf
    total += 1; changed += ch
    if sel: rows_.append((k, str(DT_[so.meta["dtype"]]), round(math.sqrt(b2), 4), round(math.sqrt(o2), 4), math.sqrt(d2), dmax))
print("tensors", total, "changed", changed, "nonfinite elems", nonfinite)
for r in rows_: print(r)
