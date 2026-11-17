# dfslib: Data Flow Space (DFS) layer-path merge for Qwen3.5-4B donors (EvoGM-DFS, 2026-11-17), Sakana "Evolutionary Model
# Merge" style: binary layer-slot indicator + scaling matrix W.  Shared by the controller, the GPU worker, the shard builder and the
# CPU tests.  Pure python/numpy at import; torch / safetensors imported inside the functions that need them.
#
# Donors (each = Qwen3.5 base + sum lambda*TV, the exact linmerge fp8-TV merges of EvoGM v2):
#   0 r1i1 (EvoGM v2 val-best generated candidate), 1 b11 (Baphomet-1.1), 2 b10 (Baphomet-1.0)
# Blocks: 8 blocks of 4 consecutive layers (layers 4b..4b+3) = one period of the hybrid layout (3 linear-attn + 1 full-attn),
#   so any block can sit in any 4-aligned slot and config.layer_types stays the same periodic pattern.
# Genotype x in [-1,1]^57:
#   x[0:48]  indicator scores of the 48 slots, bit = score > 0.  Slot order "rbd": t = rep*24 + block*3 + donor
#            (repetition, then block index, then donor).  The path is the selected slots in t order.
#            (The paper's literal layout is rep -> model -> layer, i.e. all of model A's layers, then model B's ...; with r=2
#            that can only switch donors "backwards" once, so per-block donor choices like A0 B1 A2 B3 are unreachable.  The rbd
#            order makes every same-depth per-block donor choice (3^8) reachable in repetition 0, and repetition 1 adds a second
#            pass (block duplication / loops).  Within one block index the donor order is r1i1, b11, b10.)
#   x[48:57] W (3x3, row = donor of the previous path block, col = donor of the next one): W_ij = 2 ** (W_LOG2 * x), [0.5, 2],
#            log2 W rounded to multiples of 1/8 (W in {2^(k/8)}, k = -8..8; |x| < 1/16 -> W = 1).  At every boundary between consecutive path blocks (including same-donor boundaries) the
#            residual stream is multiplied by W[d_prev, d_next] (the first block takes the embedding unscaled).
# Repair (phenotype only; the genotype is kept): more than MAX_BLOCKS selected -> keep the MAX_BLOCKS highest scores; fewer than
#   MIN_BLOCKS -> add the highest-scoring unselected slots.  Ties by slot index.
# W folding (exact up to the RMSNorm eps): every norm reading the residual stream is scale invariant (input_layernorm,
#   post_attention_layernorm, final norm; GatedDeltaNet / attention only see normed inputs), so scaling the stream by s_i before
#   block i equals dividing the outputs of block i and of all later blocks by C_i = prod_{k<=i} s_k.  We fold 1/C_i into the
#   output projections of block i (self_attn.o_proj, linear_attn.out_proj, mlp.down_proj).  C_i is clamped to [C_MIN, C_MAX]
#   (fp16 range/precision of folded weights and activations); the clamp is part of the phenotype definition.
# Padding: the engine has L_MAX layers; slots after the path are identity layers (B11 layer of the same hybrid position with all
#   three output projections zeroed -> the layer adds exactly 0 to the residual stream).
import re, math, json
import numpy as np

DONORS = ["r1i1", "b11", "b10"]
ND = 3; NB = 8; BL = 4; R = 2
NL = NB * BL                      # 32 donor layers
NBITS = R * NB * ND               # 48
NW = ND * ND                      # 9
DIM = NBITS + NW                  # 57
L_MAX = 40                        # engine layers (10 blocks)
MAX_BLOCKS = L_MAX // BL          # 10
MIN_BLOCKS = 6
W_LOG2 = 1.0                      # W in [2^-1, 2^1]
W_STEPS = 8                       # log2 W on a 1/8 grid (17 levels; |log2 W| < 1/16 -> exactly 1, so near-1 noise is not a new model)
C_MIN, C_MAX = 0.25, 4.0
PAD_DONOR = 1                     # padding layers are B11 layers with zeroed outputs
SLOTS = [(r, b, d) for r in range(R) for b in range(NB) for d in range(ND)]
OUT_SUFFIX = ("self_attn.o_proj.weight", "linear_attn.out_proj.weight", "mlp.down_proj.weight")
LAYER_RE = re.compile(r"^((?:model\.language_model\.|model\.)layers\.)(\d+)\.(.+)$")   # language-model layers only (not mtp/visual)


def slot_index(rep, blk, donor):
    return rep * NB * ND + blk * ND + donor


def geno_from_slots(slots, w=None, on=0.5, off=-0.5):
    """genotype with the given (rep, blk, donor) slots selected; w = 3x3 log2-scale coords in [-1,1] (default 0 -> W = 1)."""
    x = np.full(DIM, off, dtype=np.float64)
    for r, b, d in slots: x[slot_index(r, b, d)] = on
    x[NBITS:] = 0.0 if w is None else np.asarray(w, dtype=np.float64).reshape(-1)
    return x


def geno_plain(d):
    return geno_from_slots([(0, b, d) for b in range(NB)])


def decode(x):
    """genotype -> phenotype dict(path=[(donor, block)], C=[cumulative scale per path block], key, repaired, n_blocks)."""
    x = np.clip(np.asarray(x, dtype=np.float64).reshape(-1), -1.0, 1.0)
    assert x.shape[0] == DIM, x.shape
    s = x[:NBITS]
    on = [t for t in range(NBITS) if s[t] > 0]
    repaired = None
    if len(on) > MAX_BLOCKS:
        keep = sorted(on, key=lambda t: (-s[t], t))[:MAX_BLOCKS]; on = sorted(keep); repaired = "trim"
    elif len(on) < MIN_BLOCKS:
        off = sorted([t for t in range(NBITS) if t not in on], key=lambda t: (-s[t], t))
        on = sorted(on + off[:MIN_BLOCKS - len(on)]); repaired = "fill"
    path = [(SLOTS[t][2], SLOTS[t][1]) for t in on]
    Wm = np.round(2.0 ** (np.round(W_LOG2 * x[NBITS:].reshape(ND, ND) * W_STEPS) / W_STEPS), 4)   # log2 W quantized to 1/8
    C = [1.0]
    for i in range(1, len(path)):
        c = min(C_MAX, max(C_MIN, C[-1] * float(Wm[path[i - 1][0], path[i][0]])))
        C.append(round(c, 4))
    return dict(path=path, C=C, slots=on, repaired=repaired, n_blocks=len(path), key=pheno_key(path, C))


def pheno_key(path, C):
    return " ".join(f"{DONORS[d]}{b}" for d, b in path) + " | " + ",".join(f"{c:.4f}" for c in C)


def is_plain(ph, d=None):
    ok = len(ph["path"]) == NB and all(b == i for i, (_, b) in enumerate(ph["path"])) and len({dd for dd, _ in ph["path"]}) == 1 \
        and all(abs(c - 1.0) < 1e-12 for c in ph["C"])
    return ok and (d is None or ph["path"][0][0] == d)


def plan_of(ph, l_max=L_MAX):
    """engine layer plan, one entry per engine layer j: [donor, src_layer, fold] (fold = 1/C of its block; 0.0 = identity pad)."""
    plan = []
    for (d, b), c in zip(ph["path"], ph["C"]):
        for k in range(BL): plan.append([int(d), int(b * BL + k), 1.0 / float(c)])
    assert len(plan) <= l_max, (len(plan), l_max)
    while len(plan) < l_max:
        j = len(plan); plan.append([PAD_DONOR, int(j % NL), 0.0])
    for j, (d, src, f) in enumerate(plan): assert src % BL == j % BL   # hybrid layer type preserved
    return plan


def summary(ph):
    """short human-readable description of a phenotype."""
    segs = []; cur = None
    for d, b in ph["path"]:
        if cur and cur[0] == d and cur[2] == b - 1: cur[2] = b
        else:
            cur = [d, b, b]; segs.append(cur)
    s = "+".join(f"{DONORS[d]}[{a}-{e}]" if a != e else f"{DONORS[d]}[{a}]" for d, a, e in segs)
    if any(abs(c - 1.0) > 1e-12 for c in ph["C"]): s += " C=" + ",".join(f"{c:g}" for c in ph["C"])
    return s


# ---------------------------------------------------------------- weights (worker / builder / tests)
def layer_of(key):
    m = LAYER_RE.match(key)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None


def shard_name(donor, blk):
    return f"{DONORS[donor]}_b{blk}.safetensors"


NONLAYER = "b11_nonlayer.safetensors"


class ShardSet:
    """donor block shards (layer tensors with their ORIGINAL layer names) + B11 non-layer tensors, read lazily (mmap)."""
    def __init__(self, shard_dir):
        from safetensors import safe_open
        self.dir = shard_dir; self.h = {}; self.keys = {}
        self._open = safe_open

    def handle(self, name):
        if name not in self.h:
            self.h[name] = self._open(f"{self.dir}/{name}", "pt", device="cpu"); self.keys[name] = sorted(self.h[name].keys())
        return self.h[name]

    def layer_keys(self, donor, src):
        name = shard_name(donor, src // BL); h = self.handle(name)
        out = []
        for k in self.keys[name]:
            p = layer_of(k)
            if p and p[1] == src: out.append((k, p))
        assert out, f"no tensors for layer {src} in {name}"
        return name, out


def iter_plan(plan, shards, device=None, with_nonlayer=True, out_dtype=None):
    """(name, tensor) in checkpoint format for an L_MAX-layer engine: B11 non-layer tensors, then engine layers 0..L-1 in order.
    Output projections are multiplied by the fold (fp32 math -> fp16); fold 1.0 tensors are passed through unchanged
    (bit-identical to the v2 merge); fold 0.0 -> zeros (identity pad)."""
    import torch
    od = out_dtype or torch.float16   # engine dtype (vLLM fp16 on T4); the CPU tests pass float32
    if with_nonlayer:
        h = shards.handle(NONLAYER)
        for k in shards.keys[NONLAYER]: yield k, h.get_tensor(k)
    for j, (d, src, fold) in enumerate(plan):
        name, ks = shards.layer_keys(int(d), int(src)); h = shards.handle(name)
        for k, (pre, _, rest) in ks:
            t = h.get_tensor(k)
            if rest.endswith(OUT_SUFFIX) and float(fold) != 1.0:
                if float(fold) == 0.0: t = torch.zeros(t.shape, dtype=od)
                else:
                    t = t.to(device) if device is not None else t
                    ct = torch.float64 if od == torch.float64 else torch.float32
                    t = (t.to(ct) * float(fold)).to(od)
            yield f"{pre}{j}.{rest}", t


def build_shards(merger, lams_by_donor, blocks, out_dir, nonlayer_lams=None, device="cuda", log=print):
    """merge each donor's layers of the given blocks with linmerge.Merger.iter_merged (identical arithmetic to the v2 merges) and
    save one shard per (donor, block); optionally also the non-layer tensors (B11) -> NONLAYER."""
    from safetensors.torch import save_file
    import time, os
    keys = sorted(merger.bidx)
    for b in blocks:
        only = {k for k in keys if layer_of(k) and layer_of(k)[1] // BL == b}
        assert only, f"block {b}: no layer tensors"
        for d, lams in enumerate(lams_by_donor):
            t0 = time.time(); out = {}
            for k, t in merger.iter_merged(lams, device=device, keep_on_device=False, only=only):
                out[k] = t.contiguous()
            save_file(out, f"{out_dir}/{shard_name(d, b)}.tmp"); os.replace(f"{out_dir}/{shard_name(d, b)}.tmp", f"{out_dir}/{shard_name(d, b)}")
            log("shard", shard_name(d, b), len(out), "tensors", round(time.time() - t0, 1), "s"); del out
    if nonlayer_lams is not None:
        only = {k for k in keys if not layer_of(k)}
        t0 = time.time(); out = {k: t.contiguous() for k, t in merger.iter_merged(nonlayer_lams, device=device, keep_on_device=False, only=only)}
        save_file(out, f"{out_dir}/{NONLAYER}.tmp"); os.replace(f"{out_dir}/{NONLAYER}.tmp", f"{out_dir}/{NONLAYER}")
        log("shard", NONLAYER, len(out), "tensors", round(time.time() - t0, 1), "s")


def padded_config(cfg, l_max=L_MAX):
    """config dict of the base checkpoint -> the same with l_max text layers (layer_types extended periodically)."""
    cfg = json.loads(json.dumps(cfg))
    tc = cfg.get("text_config", cfg)
    n0 = tc["num_hidden_layers"]; assert n0 == NL, n0
    lt = tc.get("layer_types")
    if lt is not None:
        assert len(lt) == n0 and all(lt[i] == lt[i % BL] for i in range(n0)), "layer_types not periodic with period 4"
        tc["layer_types"] = [lt[j % BL] for j in range(l_max)]
    else:
        assert tc.get("full_attention_interval", 4) == BL
    tc["num_hidden_layers"] = l_max
    return cfg


def write_boot_dir(base_dir, shard_dir, boot_dir, l_max=L_MAX):
    """engine boot checkpoint = plain B11 padded to l_max layers: config (l_max layers), tokenizer files, symlinked B11 shards +
    non-layer shard, and one file with the pad layers (B11 layers of the same position, outputs zeroed)."""
    import os, shutil, glob
    from safetensors.torch import save_file
    os.makedirs(boot_dir, exist_ok=True)
    for f in glob.glob(f"{base_dir}/*"):
        bn = os.path.basename(f)
        if os.path.isfile(f) and not bn.endswith(".safetensors") and bn != "model.safetensors.index.json" and bn != "config.json":
            shutil.copy(f, f"{boot_dir}/{bn}")
    cfg = json.load(open(f"{base_dir}/config.json"))
    json.dump(padded_config(cfg, l_max), open(f"{boot_dir}/config.json", "w"), indent=2)
    wm = {}
    S = ShardSet(shard_dir)
    names = [NONLAYER] + [shard_name(PAD_DONOR, b) for b in range(NB)]
    for n in names:
        if not os.path.exists(f"{boot_dir}/{n}"): os.symlink(f"{shard_dir}/{n}", f"{boot_dir}/{n}")
        S.handle(n)
        for k in S.keys[n]: wm[k] = n
    pad = {}
    for j in range(NL, l_max):
        _, ks = S.layer_keys(PAD_DONOR, j % NL); h = S.handle(shard_name(PAD_DONOR, (j % NL) // BL))
        for k, (pre, _, rest) in ks:
            t = h.get_tensor(k)
            if rest.endswith(OUT_SUFFIX):
                import torch; t = torch.zeros_like(t)
            pad[f"{pre}{j}.{rest}"] = t.contiguous()
    if pad:
        save_file(pad, f"{boot_dir}/pad_layers.safetensors")
        for k in pad: wm[k] = "pad_layers.safetensors"
    json.dump({"metadata": {}, "weight_map": wm}, open(f"{boot_dir}/model.safetensors.index.json", "w"))
    return len(wm)
