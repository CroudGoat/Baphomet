# CPU port of linmerge.Merger.iter_merged with the r2 winner lambda; writes shards with the base's exact header layout.
import sys, os, json, struct, time, math, torch
sys.path.insert(0, "/workspace/work/evogm/kds")
import linmerge
from safetensors import safe_open
torch.set_num_threads(2)
BASE = "/workspace/work/baphomet11/m/base"; TVROOT = "/workspace/work/baphomet11/tv"
out_dir = sys.argv[1]; lams = json.loads(sys.argv[2]); only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[3] else None
os.makedirs(out_dir, exist_ok=True)
idx = [json.load(open(f"{TVROOT}/{s}/index.json")) for s in linmerge.SHORT]
tvkeys = [set(k[:-3] for k in ix if k.endswith("::q")) for ix in idx]
TH = {}
def tv(i, name):   # lazy slice: row chunks are read on demand (identical values, bounded RAM)
    f = f"{TVROOT}/{linmerge.SHORT[i]}/{idx[i][name]}"
    if f not in TH: TH[f] = safe_open(f, "pt", device="cpu")
    return TH[f].get_slice(name)
DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}
wm = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]
stats = {}
for shard in sorted(set(wm.values())):
    bp = f"{BASE}/{shard}"
    with open(bp, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr_bytes = f.read(n)
    hdr = json.loads(hdr_bytes); data0 = 8 + n
    keys = [k for k in hdr if k != "__metadata__"]
    if only is not None and not (set(keys) & only): continue
    op = f"{out_dir}/{shard}"; fo = open(op + ".part", "wb"); fo.write(struct.pack("<Q", n)); fo.write(hdr_bytes)
    bh = safe_open(bp, "pt", device="cpu"); t0 = time.time()
    for k in sorted(keys, key=lambda k: hdr[k]["data_offsets"][0]):
        meta = hdr[k]; bt = bh.get_tensor(k)
        assert list(bt.shape) == meta["shape"] and bt.dtype == DT[meta["dtype"]]
        use = [i for i in range(len(lams)) if k in tvkeys[i] and float(lams[i]) != 0.0]
        do_merge = (only is None or k in only) and bool(use) and bt.is_floating_point()
        qs = [tv(i, k + "::q") for i in use] if do_merge else []; ss = [tv(i, k + "::s") for i in use] if do_merge else []
        if bt.dim() >= 2:
            rows = bt.shape[0]; per_row = bt.numel() // rows; rstep = max(1, linmerge.CHUNK_ELEMS // max(per_row, 1))
        else:
            rows, rstep = 1, 1
        s0, s1 = meta["data_offsets"]; assert fo.tell() == data0 + s0, k
        st = dict(n_donors=len(use) if do_merge else 0, b2=0.0, m2=0.0, d2=0.0, dmax=0.0, n_changed=0)
        for r0 in range(0, rows, rstep):
            r1 = min(rows, r0 + rstep)
            b = bt[r0:r1] if bt.dim() >= 2 else bt
            if do_merge:
                if bt.dim() >= 2: ds = [(q[r0:r1].float() * s[r0:r1]) for q, s in zip(qs, ss)]
                else: ds = [(q[:].float() * s[:]) for q, s in zip(qs, ss)]
                acc = b.float()            # .float() on an fp32 tensor returns the same tensor -> clone so the base is not mutated
                if acc.data_ptr() == b.data_ptr(): acc = acc.clone()
                for j, i in enumerate(use):
                    acc.add_(ds[j], alpha=float(lams[i]))
                o = acc.to(bt.dtype); del ds, acc
            else:
                o = b
            o = o.contiguous()
            if o.numel(): fo.write(o.view(torch.uint8).numpy().tobytes() if o.dtype != torch.uint8 else o.numpy().tobytes())
            if bt.is_floating_point():
                bf, of = b.float(), o.float(); dd = of - bf
                st["b2"] += float((bf * bf).sum(dtype=torch.float64)); st["m2"] += float((of * of).sum(dtype=torch.float64))
                st["d2"] += float((dd * dd).sum(dtype=torch.float64)); st["dmax"] = max(st["dmax"], float(dd.abs().max()) if dd.numel() else 0.0)
                st["n_changed"] += int((o != b).sum()); del bf, of, dd
            del o, b
        assert fo.tell() == data0 + s1, k
        stats[k] = dict(n_donors=st["n_donors"], base_norm=math.sqrt(st["b2"]), merged_norm=math.sqrt(st["m2"]), delta_norm=math.sqrt(st["d2"]),
                        max_abs_delta=st["dmax"], n_changed=st["n_changed"], numel=bt.numel())
        del bt, qs, ss
    fo.close(); os.replace(op + ".part", op); print("shard", shard, round(time.time() - t0), flush=True)
json.dump(stats, open(f"{out_dir}/../merge_stats_{os.path.basename(out_dir)}.json", "w"), indent=0)
print("MERGEDONE", flush=True)
