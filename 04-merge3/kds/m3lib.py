# MERGE^3 reimplementation helpers for Qwen3.5-4B on Kaggle T4x2.
# - fp8 task-vector extraction (one donor at a time)
# - mergekit-equivalent dare_ties merge (GTA: DARE random sparsification + rescale, sign consensus "sum", normalize)
#   done lazily tensor-by-tensor on GPU
# - loglik multiple-choice fitness evaluation in lm-eval-harness style with vLLM (worker subprocess)
# - IRT estimators (fit_theta / fit_lambda / mp-IRT / gmp-IRT) as in mergenetic/tinyBenchmarks
import os, json, glob, shutil, math, time
import numpy as np

BASE_ID = "Qwen/Qwen3.5-4B"
DONORS = ["InternScience/Agents-A1-4B", "Jackrong/DeepSeek-V4-Pro-Qwen3.5-4B", "empero-ai/Qwen3.8-4B-Distill",
          "Jackrong/Qwopus3.5-4B-v3", "BAAI/AREX-Turbo"]
SHORT = ["agents_a1", "dsv4pro", "qwen38distill", "qwopus_v3", "arex_turbo"]
NONWEIGHT_FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json", "processor_config.json",
                   "special_tokens_map.json"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def download(repo, local_dir):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, local_dir=local_dir, allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt"])


def read_all_tensors_index(model_dir):
    """name -> shard file"""
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(model_dir, v) for k, v in wm.items()}
    from safetensors import safe_open
    out = {}
    for f in glob.glob(os.path.join(model_dir, "*.safetensors")):
        with safe_open(f, "pt") as s:
            for k in s.keys(): out[k] = f
    return out


# ---------------------------------------------------------------- fp8 task vectors
FP8_MAX = 448.0

def quant_fp8(d):
    import torch
    d = d.float()
    if d.dim() >= 2:
        s = d.abs().amax(dim=tuple(range(1, d.dim())), keepdim=True).clamp_min(1e-12) / FP8_MAX
    else:
        s = d.abs().amax().clamp_min(1e-12).reshape(1) / FP8_MAX
    q = (d / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, s.to(torch.float32)


def dequant_fp8(q, s):
    return q.float() * s


def extract_tv(base_dir, donor_dir, out_path, device="cuda"):
    """T = W_donor - W_base, fp8 e4m3 with per-row absmax scale. All-zero tensors skipped."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    bidx = read_all_tensors_index(base_dir); didx = read_all_tensors_index(donor_dir)
    out = {}; stats = dict(n=0, zero=0, missing=0, sqnorm=0.0, qerr=0.0)
    handles = {}
    def get(idx, k):
        f = idx[k]
        if f not in handles: handles[f] = safe_open(f, "pt", device="cpu")
        return handles[f].get_tensor(k)
    for k in bidx:
        if k not in didx: stats["missing"] += 1; continue
        b = get(bidx, k).to(device).float(); d = get(didx, k).to(device).float()
        assert b.shape == d.shape, k
        t = d - b
        stats["n"] += 1
        if not torch.any(t != 0): stats["zero"] += 1; continue
        q, s = quant_fp8(t)
        stats["sqnorm"] += float((t * t).sum()); stats["qerr"] += float(((dequant_fp8(q, s) - t) ** 2).sum())
        out[k + "::q"] = q.cpu(); out[k + "::s"] = s.cpu()
        del b, d, t
    save_file(out, out_path)
    stats["rel_qerr"] = math.sqrt(stats["qerr"] / max(stats["sqnorm"], 1e-30)); stats["norm"] = math.sqrt(stats["sqnorm"])
    return stats


# ---------------------------------------------------------------- dare_ties merge (mergekit GTA semantics)
CHUNK_ELEMS = 1 << 23   # rows per DARE chunk ~8M elements (also part of the DARE RNG seed -> keep fixed for a whole run)

def iter_merged(base_dir, tv_paths, weights, densities, seed=0, device="cuda", chunk_elems=CHUNK_ELEMS, out_dtype="bfloat16",
                keep_on_device=False, max_device_numel=1 << 26):
    """Yield (name, merged tensor) for every base tensor, in sorted-name order.
    mergekit dare_ties: per donor DARE(bernoulli keep=density, rescale 1/density); weighted = w*delta;
    majority sign = sign(sum weighted) (>=0 -> +); mixed = sum(weighted*mask)/sum(w*mask) (divisor 0 -> 1); out = base + mixed.
    Donors lacking a tensor are excluded for that tensor (like mergekit with an absent optional tensor).
    Shared by merge_dare_ties (disk) and the in-place vLLM reload path, so both produce bit-identical tensors.
    keep_on_device: yield GPU tensors, except tensors larger than max_device_numel (embeddings) which are yielded on CPU
    to bound peak GPU memory during vLLM's layerwise reload."""
    import torch
    from safetensors import safe_open
    odt = getattr(torch, out_dtype)
    bidx = read_all_tensors_index(base_dir)
    tvh = [safe_open(p, "pt", device="cpu") for p in tv_paths]
    tvkeys = [set(k[:-3] for k in h.keys() if k.endswith("::q")) for h in tvh]
    bh = {}
    act = [i for i in range(len(tv_paths)) if weights[i] != 0 and densities[i] > 0]
    g = torch.Generator(device=device)
    for ti, k in enumerate(sorted(bidx)):
        f = bidx[k]
        if f not in bh: bh[f] = safe_open(f, "pt", device="cpu")
        bt = bh[f].get_tensor(k)
        use = [i for i in act if k in tvkeys[i]]
        if not use or not bt.is_floating_point():
            res = bt.to(odt) if bt.is_floating_point() else bt
            if keep_on_device and res.numel() <= max_device_numel: res = res.to(device)
        else:
            shape = bt.shape
            flat_b = bt.reshape(-1)
            qs = [tvh[i].get_tensor(k + "::q") for i in use]
            ss = [tvh[i].get_tensor(k + "::s") for i in use]
            rows = shape[0] if bt.dim() >= 2 else 1
            per_row = flat_b.numel() // rows
            rstep = max(1, chunk_elems // max(per_row, 1)) if bt.dim() >= 2 else 1
            outs = []
            for r0 in range(0, rows, rstep):
                r1 = min(rows, r0 + rstep)
                if bt.dim() >= 2:
                    b = bt[r0:r1].to(device).float()
                    deltas = [dequant_fp8(q[r0:r1].to(device), s[r0:r1].to(device)) for q, s in zip(qs, ss)]
                else:
                    b = bt.to(device).float()
                    deltas = [dequant_fp8(q.to(device), s.to(device)) for q, s in zip(qs, ss)]
                wd = []
                for j, i in enumerate(use):
                    d = deltas[j]; dens = float(densities[i])
                    if dens < 1.0:
                        g.manual_seed(int(seed) * 1000003 + ti * 101 + i * 7 + r0)
                        m = torch.bernoulli(torch.full_like(d, dens), generator=g)
                        d = d * m / dens
                    wd.append(d * float(weights[i]))
                W = torch.stack(wd, 0)
                wv = torch.tensor([float(weights[i]) for i in use], device=device).view(-1, *([1] * b.dim()))
                maj = (W.sum(0) >= 0).to(torch.int8) * 2 - 1
                mask = (W.sign().to(torch.int8) == maj)
                mixed = (W * mask).sum(0)
                div = (wv * mask).sum(0); div[div == 0] = 1
                mixed = mixed / div
                o = (b + mixed).to(odt)
                outs.append(o if (keep_on_device and bt.numel() <= max_device_numel) else o.cpu())
                del W, mask, mixed, div, deltas, wd, b
            res = torch.cat(outs, 0) if bt.dim() >= 2 else outs[0]
        yield k, res


def merge_dare_ties(base_dir, tv_paths, weights, densities, out_dir, seed=0, device="cuda", chunk_elems=CHUNK_ELEMS,
                    shard_bytes=4 << 30, out_dtype="bfloat16"):
    """Write the dare_ties merge (see iter_merged) as a HF checkpoint dir."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    shard, shard_sz, shards, wmap = {}, 0, [], {}
    def flush():
        nonlocal shard, shard_sz
        if not shard: return
        name = f"model-{len(shards)+1:05d}.safetensors"
        save_file(shard, os.path.join(out_dir, name), metadata={"format": "pt"})
        for k in shard: wmap[k] = name
        shards.append(name); shard = {}; shard_sz = 0
    for k, res in iter_merged(base_dir, tv_paths, weights, densities, seed=seed, device=device, chunk_elems=chunk_elems,
                              out_dtype=out_dtype):
        shard[k] = res.contiguous(); shard_sz += res.numel() * res.element_size()
        if shard_sz >= shard_bytes: flush()
    flush()
    json.dump({"metadata": {}, "weight_map": wmap}, open(os.path.join(out_dir, "model.safetensors.index.json"), "w"))
    for fn in NONWEIGHT_FILES:
        p = os.path.join(base_dir, fn)
        if os.path.exists(p): shutil.copy(p, out_dir)
    return out_dir


# ---------------------------------------------------------------- IRT (tinyBenchmarks / mergenetic math)
def sigmoid(z): return 1 / (1 + np.exp(-z))

def item_curve(theta, a, b):
    z = np.clip(a * theta - b, -30, 30).sum(axis=1)
    return sigmoid(z)

def fit_theta(y, seen, A, B, eps=1e-10):
    from scipy.optimize import minimize
    D = A.shape[1]
    def nll(x):
        P = item_curve(x.reshape(1, D, 1), A[:, :, seen], B[:, :, seen]).squeeze()
        return -np.sum(y[seen] * np.log(P + eps) + (1 - y[seen]) * np.log(1 - P + eps))
    return minimize(nll, np.zeros(D), method="BFGS").x[None, :, None]

def fit_lambda(y, seen, A, B, thetas, eps=1e-10):
    from scipy.optimize import minimize
    D = A.shape[1]
    def nll(l):
        th = np.sum([l[i] * thetas[i] for i in range(len(thetas))], axis=0)
        P = item_curve(th.reshape(1, D, 1), A[:, :, seen], B[:, :, seen]).squeeze()
        return -np.sum(y[seen] * np.log(P + eps) + (1 - y[seen]) * np.log(1 - P + eps))
    return minimize(nll, np.ones(len(thetas)) / len(thetas), method="BFGS").x

def gmpirt(y_anchor, anchor_pos, scen_pos, A, B, thetas, delta=0.5):
    """y_anchor: 0/1 on anchors (global item positions anchor_pos, all inside scen_pos).
    mergenetic.estimator.utils.estimate_fitness semantics (uniform anchor weights)."""
    N = A.shape[2]
    y = np.zeros(N); y[anchor_pos] = y_anchor
    lam = fit_lambda(y, list(anchor_pos), A, B, thetas)
    th = np.sum([lam[i] * thetas[i] for i in range(len(thetas))], axis=0)
    seen = set(anchor_pos); unseen = [i for i in scen_pos if i not in seen]
    n = len(anchor_pos); w = np.ones(n) / n
    wavg = float((w * y_anchor).sum())
    irt_part = float(item_curve(th.reshape(1, A.shape[1], 1), A, B)[0, unseen].mean())
    tau = n / len(scen_pos)
    mp = tau * wavg + (1 - tau) * irt_part
    return dict(mpirt=mp, gmpirt=delta * wavg + (1 - delta) * mp, weighted_avg=wavg, lam=lam.tolist())
