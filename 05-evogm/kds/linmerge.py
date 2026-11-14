# Linear task-arithmetic merge for Qwen3.5-4B donors, streamed tensor by tensor:
#   W = W_base + sum_i lam_i * T_i,   T_i = W_donor_i - W_base   (fp32 accumulation, output in the base tensor's dtype)
# Task vectors are stored fp8 e4m3 with a per-row absmax scale (as in the MERGE^3 run; relative TV error ~2.6%; since TVs
# are only a few bf16 ulps of the weights, this changes merged bf16 weights by at most ~1 ulp in some elements). Disk: 4.4 GB/TV vs 9 GB
# per donor (v1/v2 runs hit a read-only root filesystem at ~94 GB used).
# A donor lacking a tensor contributes 0. NeoHorse: text-only keys model.* -> model.language_model.* (remap=True).
import os, json, glob, time, math

BASE_ID = "Qwen/Qwen3.5-4B"
DONORS = [  # (short, repo, remap)   order = lambda order everywhere
    ("agents_a1", "InternScience/Agents-A1-4B", False),
    ("dsv4pro", "Jackrong/DeepSeek-V4-Pro-Qwen3.5-4B", False),
    ("q38distill", "empero-ai/Qwen3.8-4B-Distill", False),
    ("qwopus_coder", "Jackrong/Qwopus3.5-4B-Coder", False),
    ("qwopus_v3", "Jackrong/Qwopus3.5-4B-v3", False),
    ("arex_turbo", "BAAI/AREX-Turbo", False),
    ("neohorse", "TokenRhythm/NeoHorse-1-4B", True),
    ("tmax", "allenai/tmax-4b", False),
]
SHORT = [d[0] for d in DONORS]
N = len(DONORS)
BAPHOMET_LAMBDA = [0.1375] * N   # goatmerge linear kernel: base + sum w_i*T_i (normalize/rescale unused for linear, density 1)
CHUNK_ELEMS = 1 << 23
FP8_MAX = 448.0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def download(repo, local_dir, patterns=None):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, local_dir=local_dir, allow_patterns=patterns or ["*.safetensors", "*.json", "*.jinja", "*.txt"])


def tensor_index(model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(model_dir, v) for k, v in wm.items() if os.path.exists(os.path.join(model_dir, v))}
    from safetensors import safe_open
    out = {}
    for f in glob.glob(os.path.join(model_dir, "*.safetensors")):
        with safe_open(f, "pt") as s:
            for k in s.keys(): out[k] = f
    return out


def donor_map(donor_dir, remap):
    out = {}
    for k, f in tensor_index(donor_dir).items():
        bk = k
        if remap and k.startswith("model.") and not k.startswith("model.language_model."):
            bk = "model.language_model." + k[len("model."):]
        out[bk] = (f, k)
    return out


def quant_fp8(d):
    import torch
    if d.dim() >= 2:
        s = d.abs().amax(dim=tuple(range(1, d.dim())), keepdim=True).clamp_min(1e-12) / FP8_MAX
    else:
        s = d.abs().amax().clamp_min(1e-12).reshape(1) / FP8_MAX
    q = (d / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, s.to(torch.float32)


def extract_tv(base_dir, donor_dir, remap, out_path, device="cuda"):
    """T = W_donor - W_base (fp32 diff) -> fp8 e4m3 per-row absmax, base key names. All-zero tensors skipped."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    bidx = tensor_index(base_dir); dmap = donor_map(donor_dir, remap)
    H = {}
    def h(f):
        if f not in H: H[f] = safe_open(f, "pt", device="cpu")
        return H[f]
    out = {}; st = dict(n=0, zero=0, missing=0, sqnorm=0.0, qerr=0.0)
    for k in sorted(bidx):
        if k not in dmap: st["missing"] += 1; continue
        b = h(bidx[k]).get_tensor(k).to(device).float(); d = h(dmap[k][0]).get_tensor(dmap[k][1]).to(device).float()
        assert b.shape == d.shape, (k, b.shape, d.shape)
        t = d - b; st["n"] += 1
        if not torch.any(t != 0): st["zero"] += 1; continue
        q, s = quant_fp8(t)
        st["sqnorm"] += float((t * t).sum()); st["qerr"] += float(((q.float() * s - t) ** 2).sum())
        out[k + "::q"] = q.cpu(); out[k + "::s"] = s.cpu()
        del b, d, t
    save_file(out, out_path)
    st["rel_qerr"] = math.sqrt(st["qerr"] / max(st["sqnorm"], 1e-30)); st["norm"] = math.sqrt(st["sqnorm"])
    return st


class Merger:
    def __init__(self, base_dir, tv_paths):
        from safetensors import safe_open
        self.base_dir = base_dir
        self.bidx = tensor_index(base_dir)
        self.tv_paths = tv_paths
        self.tvkeys = []
        for p in tv_paths:
            with safe_open(p, "pt") as s:
                self.tvkeys.append(set(k[:-3] for k in s.keys() if k.endswith("::q")))
        self.cover = {k: [i for i in range(len(tv_paths)) if k in self.tvkeys[i]] for k in self.bidx}

    def iter_merged(self, lams, device="cuda", keep_on_device=False, max_device_numel=1 << 26, chunk_elems=CHUNK_ELEMS, only=None):
        import torch
        from safetensors import safe_open
        tvh = [safe_open(p, "pt", device="cpu") for p in self.tv_paths]
        bh = {}
        for k in sorted(self.bidx):
            if only is not None and k not in only: continue
            f = self.bidx[k]
            if f not in bh: bh[f] = safe_open(f, "pt", device="cpu")
            use = [i for i in self.cover[k] if float(lams[i]) != 0.0]
            bt = bh[f].get_tensor(k)
            if not use or not bt.is_floating_point():
                res = bt
                if keep_on_device and res.numel() <= max_device_numel: res = res.to(device)
                yield k, res
                continue
            qs = [tvh[i].get_tensor(k + "::q") for i in use]
            ss = [tvh[i].get_tensor(k + "::s") for i in use]
            if bt.dim() >= 2:
                rows = bt.shape[0]; per_row = bt.numel() // rows; rstep = max(1, chunk_elems // max(per_row, 1))
            else:
                rows, rstep = 1, 1
            outs = []
            for r0 in range(0, rows, rstep):
                r1 = min(rows, r0 + rstep)
                if bt.dim() >= 2:
                    b = bt[r0:r1].to(device)
                    ds = [(q[r0:r1].to(device).float() * s[r0:r1].to(device)) for q, s in zip(qs, ss)]
                else:
                    b = bt.to(device)
                    ds = [(q.to(device).float() * s.to(device)) for q, s in zip(qs, ss)]
                acc = b.float()
                for j, i in enumerate(use):
                    acc.add_(ds[j], alpha=float(lams[i]))
                o = acc.to(bt.dtype)
                outs.append(o if (keep_on_device and bt.numel() <= max_device_numel) else o.cpu())
                del b, ds, acc
            res = torch.cat(outs, 0) if len(outs) > 1 else outs[0]
            yield k, res
