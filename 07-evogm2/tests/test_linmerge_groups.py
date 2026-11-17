# Unit test for the grouped (16-dim) linmerge: key->group assignment on the real Qwen3.5-4B key list, and
# 16-dim merge with equal halves == 8-dim merge bit for bit; unequal halves use the right half per tensor.
import sys, os, json, re, collections, tempfile, torch
sys.path.insert(0, "/workspace/work/evogm2/kds"); import linmerge as LM
from safetensors.torch import save_file
DK = json.load(open("/workspace/work/evogm/donor_keys.json"))
base = DK[LM.BASE_ID]
# 1) assignment counts
cnt = collections.Counter(); part = collections.Counter()
for k in base:
    g = LM.GROUPS[LM.group_of(k)]; cnt[g] += 1
    area = "lm" if k.startswith("model.language_model.") else ("mtp" if k.startswith("mtp.") else "visual")
    part[(area, g)] += 1
print("keys", len(base), dict(cnt)); print("by area", dict(sorted(part.items())))
mlp_pat = collections.Counter(re.sub(r"\.\d+\.", ".N.", k) for k in base if LM.group_of(k) == 1)
print("MLP patterns:", sorted(mlp_pat.items()))
attn_lm = collections.Counter(re.sub(r"layers\.\d+\.", "layers.N.", k).split(".N.")[-1] if ".N." in re.sub(r"layers\.\d+\.", "layers.N.", k) else k
                              for k in base if LM.group_of(k) == 0 and not k.startswith("model.visual"))
print("attention-group (text) patterns:", sorted(attn_lm.items()))
# element counts (parameters) per group, text model only
def numel(shape): n = 1; [n := n * int(x) for x in re.findall(r"\d+", shape)]; return n
pe = collections.Counter()
for k, (dt, sh) in base.items():
    if not k.startswith("model.visual"): pe[LM.GROUPS[LM.group_of(k)]] += numel(sh)
print("text params per group (M):", {g: round(v / 1e6, 1) for g, v in pe.items()})
assert cnt["mlp"] == 96 + 3 + 96 and cnt["attn"] == len(base) - cnt["mlp"]
assert all(".mlp." in k for k in base if LM.group_of(k) == 1)
assert not any(LM.group_of(k) for k in base if "self_attn" in k or "linear_attn" in k or "norm" in k.split(".")[-2] or "embed" in k)

# 2) synthetic checkpoint with the real keys/dtypes/donor coverage (tiny shapes)
torch.manual_seed(0)
tmp = tempfile.mkdtemp(); bdir = f"{tmp}/base"; os.makedirs(bdir)
DT = {"BF16": torch.bfloat16, "F32": torch.float32}
bt = {}
for k, (dt, sh) in base.items():
    nd = len(re.findall(r"\d+", sh)); shape = (5, 7) if nd >= 2 else (7,)
    bt[k] = (torch.randn(shape) * 0.05).to(DT[dt])
save_file(bt, f"{bdir}/model.safetensors")
tvs = []
for s, repo, remap in LM.DONORS:
    keys = set()
    for k in DK[repo]:
        bk = "model.language_model." + k[6:] if remap and k.startswith("model.") and not k.startswith("model.language_model.") else k
        if bk in base: keys.add(bk)
    out = {}
    for k in keys:
        d = (bt[k].float() + torch.randn(bt[k].shape) * 1e-3).to(bt[k].dtype)
        t = d.float() - bt[k].float()
        if not torch.any(t != 0): continue
        q, sc = LM.quant_fp8(t); out[k + "::q"] = q; out[k + "::s"] = sc
    p = f"{tmp}/{s}.safetensors"; save_file(out, p); tvs.append(p)
    print("tv", s, "keys", len(keys))
M = LM.Merger(bdir, tvs)
B11 = [0.0116, 0.2039, 0.5901, 0.0013, 0.2972, 0.0323, 0.0030, -0.0007]
g = torch.Generator().manual_seed(1)
R8 = [float(x) for x in (torch.rand(8, generator=g) * 2 - 1)]
def merged(l, **kw): return dict(M.iter_merged(l, device="cpu", chunk_elems=8, **kw))   # chunk_elems small -> exercises row chunking
nbad = 0
for l8 in (B11, R8, [0.1375] * 8, [0.0] * 8):
    a = merged(l8); b = merged(LM.expand(l8))
    bad = [k for k in a if a[k].dtype != b[k].dtype or not torch.equal(a[k], b[k])]
    nbad += len(bad); print("equal halves vs 8-dim:", l8[:3], "... tensors", len(a), "non-identical", len(bad))
assert nbad == 0
# 3) unequal halves: each tensor must equal the 8-dim merge with its own group's lambda
A, Mm = B11, [0.1375] * 8
m16 = merged(A + Mm); mA = merged(A); mM = merged(Mm)
bad = [k for k in m16 if not torch.equal(m16[k], (mM if LM.group_of(k) else mA)[k])]
ndiff = sum(1 for k in m16 if not torch.equal(mA[k], mM[k]))
print("split halves: tensors", len(m16), "matching own-group 8-dim merge", len(m16) - len(bad), "| tensors where the two halves differ", ndiff)
assert not bad and ndiff > 0
# 4) dtype preserved (F32 A_log/norms stay F32)
assert all(m16[k].dtype == bt[k].dtype for k in m16)
# 5) custom group_fn honoured
cf = merged(A + Mm, group_fn=lambda k: 0); assert all(torch.equal(cf[k], mA[k]) for k in cf)
try: LM.lam_for_key([0.1] * 5, "x"); raise SystemExit("length check failed")
except ValueError: pass
print("ALL LINMERGE GROUP TESTS PASSED")
