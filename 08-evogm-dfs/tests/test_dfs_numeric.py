# Numeric checks of the DFS construction on tiny random Qwen3.5 models (transformers Qwen3_5ForCausalLM, same layer classes as
# Qwen3.5-4B: GatedDeltaNet linear attention x3 + gated full attention x1 per 4 layers, (1+w) RMSNorm, tied embeddings), CPU fp32.
#  A  identity padding: plain B11 path in a 40-layer model (8 zeroed-output pad layers at the end) == the 32-layer donor, and pads
#     placed in the middle of the path == pads at the end.
#  B  W folding: weights assembled by dfslib (plan_of + iter_plan: out-projections x 1/C) vs an independent reference built from
#     the donors' layer MODULES with the residual stream explicitly multiplied by s_i at each block boundary (forward pre-hooks),
#     for random genotypes (trim/fill repairs, clamp hits).  Run with eps = 1e-6 (as Qwen3.5) and eps = 0, small (0.02) and
#     large (1.0) embedding scale.
#  C  plain B11 tensors from build_shards + iter_plan (32 layers) are bit-identical to the linmerge full merge (real linmerge.Merger
#     on tiny fake base / fp8 TVs).
import sys, os, copy, json, math, tempfile
import numpy as np, torch
sys.path.insert(0, "/workspace/work/evogm-dfs/src"); sys.path.insert(1, "/workspace/work/evogm2/kds")
import dfslib as D
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
torch.set_num_threads(2)
RES = {}
OUT_DTYPE = torch.float32   # exact-math check; fp16 rounding of the folded weights is measured separately (run16)


def cfg(n, eps):
    return Qwen3_5TextConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=n, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                             linear_value_head_dim=16, tie_word_embeddings=True, rms_norm_eps=eps, max_position_embeddings=256)


def donor(seed, eps, emb_std):
    torch.manual_seed(seed)
    m = Qwen3_5ForCausalLM(cfg(32, eps)).float().eval()
    if OUT_DTYPE == torch.float64: m = m.double()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n and p.dim() == 1: p.normal_(0, 0.2)            # (1+w) norms: non-trivial gains
            elif "A_log" in n or "dt_bias" in n: p.normal_(0, 0.3)
            elif "embed_tokens" in n: p.normal_(0, emb_std)
            else: p.normal_(0, 0.08)
    m.config.use_cache = False
    return m


def shard_dir(donors, tmp):
    """write the three donors as dfslib shards (+ B11 non-layer)"""
    from safetensors.torch import save_file
    sds = [{k: v.detach().clone().contiguous() for k, v in m.state_dict().items()} for m in donors]
    for d, sd in enumerate(sds):
        for b in range(D.NB):
            save_file({k: v for k, v in sd.items() if D.layer_of(k) and D.layer_of(k)[1] // D.BL == b}, f"{tmp}/{D.shard_name(d, b)}")
    nl = {k: v for k, v in sds[D.PAD_DONOR].items() if not D.layer_of(k) and k != "lm_head.weight"}
    save_file(nl, f"{tmp}/{D.NONLAYER}")
    return D.ShardSet(tmp)


def folded_model(plan, shards, eps):
    m = Qwen3_5ForCausalLM(cfg(len(plan), eps)).float().eval(); m.config.use_cache = False
    if OUT_DTYPE == torch.float64: m = m.double()
    sd = {k: v.to(torch.float64 if OUT_DTYPE == torch.float64 else torch.float32) for k, v in D.iter_plan(plan, shards, out_dtype=OUT_DTYPE)}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert set(missing) <= {"lm_head.weight"}, missing           # tied
    m.tie_weights()
    assert torch.equal(m.lm_head.weight, m.model.embed_tokens.weight)
    return m


def reference_model(ph, donors, eps, l_max, pad_at=None):
    """independent assembly: deep copies of donor layer modules; explicit residual scaling by forward pre-hooks."""
    m = Qwen3_5ForCausalLM(cfg(l_max, eps)).float().eval(); m.config.use_cache = False
    if OUT_DTYPE == torch.float64: m = m.double()
    b11 = donors[D.PAD_DONOR]
    m.model.embed_tokens.weight.data.copy_(b11.model.embed_tokens.weight.data); m.model.norm.weight.data.copy_(b11.model.norm.weight.data)
    m.tie_weights()
    blocks = [("L", d, b, c) for (d, b), c in zip(ph["path"], ph["C"])]
    npad = l_max // D.BL - len(blocks)
    pads = [("P",)] * npad
    seq = blocks + pads if pad_at is None else blocks[:pad_at] + pads + blocks[pad_at:]
    prevC = 1.0; layers = []; hooks = []
    for bi, e in enumerate(seq):
        for k in range(D.BL):
            j = len(layers)
            if e[0] == "L":
                lay = copy.deepcopy(donors[e[1]].model.layers[e[2] * D.BL + k])
            else:
                lay = copy.deepcopy(b11.model.layers[j % D.NL])
                for n, p in lay.named_parameters():
                    if n.endswith(D.OUT_SUFFIX): p.data.zero_()
            for mod in lay.modules():
                if hasattr(mod, "layer_idx"): mod.layer_idx = j
            layers.append(lay)
        if e[0] == "L":
            s = e[3] / prevC; prevC = e[3]
            if abs(s - 1.0) > 0:
                first = layers[-D.BL]
                hooks.append((first, s))
    m.model.layers = torch.nn.ModuleList(layers)
    for lay, s in hooks:
        lay.register_forward_pre_hook(lambda mod, args, kwargs, s=s: ((args[0] * s,) + tuple(args[1:]), kwargs), with_kwargs=True)
    return m


def logits(m, ids):
    with torch.no_grad(): return m(input_ids=ids).logits.double()


def rel(a, b): return float((a - b).abs().max() / b.abs().max())


def run(eps, emb_std, ngeno=12):
    tag = f"eps={eps:g},emb_std={emb_std:g},folded_dtype={str(OUT_DTYPE)[6:]}"
    donors = [donor(100 + d, eps, emb_std) for d in range(3)]
    tmp = tempfile.mkdtemp(); shards = shard_dir(donors, tmp)
    torch.manual_seed(7); ids = torch.randint(0, 300, (2, 24))
    out = {}
    # A: identity padding
    ph = D.decode(D.geno_plain(D.PAD_DONOR)); plan = D.plan_of(ph)
    m40 = folded_model(plan, shards, eps); ref32 = donors[D.PAD_DONOR]
    out["A_plain_b11_padded40_vs_donor32_maxabs"] = float((logits(m40, ids) - logits(ref32, ids)).abs().max())
    mmid = reference_model(ph, donors, eps, D.L_MAX, pad_at=3)
    out["A_pads_middle_vs_end_maxabs"] = float((logits(mmid, ids) - logits(m40, ids)).abs().max())
    # B: folding vs explicit residual scaling
    rs = np.random.RandomState(11); errs = []; info = []
    genos = [D.geno_plain(0), D.geno_plain(1)]
    g = D.geno_plain(0); g[D.NBITS:] = rs.uniform(-1, 1, D.NW); genos.append(g)   # plain path, W != 1 (same-donor boundaries)
    while len(genos) < ngeno: genos.append(rs.uniform(-1, 1, D.DIM))
    for x in genos:
        ph = D.decode(x); plan = D.plan_of(ph)
        mf = folded_model(plan, shards, eps); mr = reference_model(ph, donors, eps, D.L_MAX)
        lf, lr = logits(mf, ids), logits(mr, ids)
        e = rel(lf, lr); errs.append(e)
        # same with the folding switched off (shows the scaling matters)
        plan1 = [[d, s, (1.0 if f != 0 else 0.0)] for d, s, f in plan]
        e_nofold = rel(logits(folded_model(plan1, shards, eps), ids), lr)
        info.append(dict(path=D.summary(ph), repaired=ph["repaired"], Cmin=min(ph["C"]), Cmax=max(ph["C"]), rel_err=e, rel_err_without_fold=e_nofold))
    out["B_fold_rel_err_max"] = max(errs); out["B_cases"] = info
    RES[tag] = out
    print(tag, json.dumps({k: v for k, v in out.items() if k != "B_cases"}))
    for i in info: print("   ", i)


def test_merge_identity():
    """C: build_shards (via the real linmerge.Merger on tiny fake checkpoints) + iter_plan(plain b11, 32 layers) == full merge."""
    import linmerge
    from safetensors.torch import save_file
    tmp = tempfile.mkdtemp(); os.makedirs(f"{tmp}/base"); os.makedirs(f"{tmp}/sh")
    torch.manual_seed(3)
    m = Qwen3_5ForCausalLM(cfg(32, 1e-6)).to(torch.bfloat16)
    base = {("model.language_model." + k[6:] if k.startswith("model.") else k): v.contiguous() for k, v in m.state_dict().items() if k != "lm_head.weight"}
    base["mtp.layers.0.mlp.down_proj.weight"] = torch.randn(64, 128).to(torch.bfloat16)
    save_file(base, f"{tmp}/base/model.safetensors")
    tvs = []
    for i in range(linmerge.N):
        out = {}
        for k, v in base.items():
            q, s = linmerge.quant_fp8(torch.randn(v.shape) * 0.01); out[k + "::q"] = q; out[k + "::s"] = s
        save_file(out, f"{tmp}/tv{i}.safetensors"); tvs.append(f"{tmp}/tv{i}.safetensors")
    M = linmerge.Merger(f"{tmp}/base", tvs)
    rs = np.random.RandomState(5); lams = [list(rs.uniform(-0.3, 0.6, 16)) for _ in range(3)]
    D.build_shards(M, lams, range(D.NB), f"{tmp}/sh", nonlayer_lams=lams[D.PAD_DONOR], device="cpu", log=lambda *a: None)
    S = D.ShardSet(f"{tmp}/sh")
    ok = True; n = 0
    for d in range(3):
        full = dict(M.iter_merged(lams[d], device="cpu"))
        ph = D.decode(D.geno_plain(d)); plan = D.plan_of(ph, l_max=32)
        got = dict(D.iter_plan(plan, S, with_nonlayer=(d == D.PAD_DONOR)))
        want = {k: v for k, v in full.items() if D.layer_of(k) or d == D.PAD_DONOR}
        assert set(got) == set(want), (set(got) ^ set(want))
        for k in want:
            n += 1
            if not (got[k].dtype == want[k].dtype and torch.equal(got[k], want[k])): ok = False; print("MISMATCH", d, k)
    RES["C_merge_identity"] = dict(ok=ok, tensors_compared=n)
    print("C merge identity", ok, n)
    # boot dir (config + index + pad file)
    json.dump({"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": {"num_hidden_layers": 32, "full_attention_interval": 4,
               "layer_types": [("full_attention" if (i + 1) % 4 == 0 else "linear_attention") for i in range(32)]}}, open(f"{tmp}/base/config.json", "w"))
    open(f"{tmp}/base/tokenizer.json", "w").write("{}")
    nk = D.write_boot_dir(f"{tmp}/base", f"{tmp}/sh", f"{tmp}/boot")
    c = json.load(open(f"{tmp}/boot/config.json"))["text_config"]
    idx = json.load(open(f"{tmp}/boot/model.safetensors.index.json"))["weight_map"]
    from safetensors import safe_open
    zero = all(torch.count_nonzero(safe_open(f"{tmp}/boot/pad_layers.safetensors", "pt").get_tensor(k)) == 0
               for k in idx if idx[k] == "pad_layers.safetensors" and k.endswith(D.OUT_SUFFIX))
    layers = sorted({D.layer_of(k)[1] for k in idx if D.layer_of(k)})
    RES["C_boot_dir"] = dict(n_keys=nk, num_hidden_layers=c["num_hidden_layers"], layer_types_ok=c["layer_types"] == [c["layer_types"][j % 4] for j in range(40)] and c["layer_types"][3] == "full_attention",
                             layers=[layers[0], layers[-1], len(layers)], pad_outputs_zero=bool(zero), tokenizer_copied=os.path.exists(f"{tmp}/boot/tokenizer.json"))
    print("C boot dir", RES["C_boot_dir"])


if __name__ == "__main__":
    import sys as _s
    if "f64" in _s.argv:   # exact-math check in float64: the HF module's internal fp32 casts (norms, GDN, softmax, rope) are
        OUT_DTYPE = torch.float64   # redirected to float64 so the only remaining difference is the math (noise floor ~1e-13)
        import transformers.models.qwen3_5.modeling_qwen3_5 as MQ
        class _TP:
            def __getattr__(self, n): return torch.float64 if n in ("float32", "float") else getattr(torch, n)
        MQ.torch = _TP()
        torch.Tensor.float = lambda self, *a, **k: self.double()
        for eps, es in ((0.0, 0.02), (1e-6, 0.02), (1e-6, 1.0)): run(eps, es, ngeno=8)
        json.dump(RES, open("/workspace/work/evogm-dfs/tests/test_dfs_numeric_f64.out.json", "w"), indent=1); _s.exit(0)
    test_merge_identity()
    for eps, es in ((0.0, 0.02), (1e-6, 0.02), (1e-6, 1.0)):
        run(eps, es)
    OUT_DTYPE = torch.float16   # as in the engine: folded out-projections rounded to fp16 (all other weights stay fp32 here)
    run(1e-6, 1.0, ngeno=6)
    json.dump(RES, open("/workspace/work/evogm-dfs/tests/test_dfs_numeric.out.json", "w"), indent=1)
