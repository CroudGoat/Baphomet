"""Tensor-by-tensor SLERP merge for Qwen3.5 (qwen3_5) checkpoints, low RAM.
A = Qwopus3.5-4B-v3, B = Agents-A1-4B. t=0 -> A, t=1 -> B.
- language_model tensors: SLERP (t varies per layer with a V-shaped curve like mergekit's [0,0.5,1,0.5,0] for attn/mlp)
- visual tower: taken from B (Agents-A1 is the VLM-tuned one)
- mtp.* (only in A): copied from A
"""
import json, os, re, sys, shutil, torch
from safetensors import safe_open
from safetensors.torch import save_file

A = "models/Qwopus3.5-4B-v3"; B = "models/Agents-A1-4B"; OUT = sys.argv[1] if len(sys.argv) > 1 else "models/Qwopus-Agents-SLERP-4B"
T_DEFAULT = float(os.environ.get("T", "0.5"))
NLAYERS = 32
os.makedirs(OUT, exist_ok=True)

def wmap(d):
    return json.load(open(f"{d}/{[f for f in os.listdir(d) if f.endswith('index.json')][0]}"))["weight_map"]
wa, wb = wmap(A), wmap(B)

# mmap on the FUSE-backed /workspace raises SIGBUS, so read byte ranges directly
import struct
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8, "BOOL": torch.bool}
headers = {}
def get(d, wm, k):
    f = f"{d}/{wm[k]}"
    if f not in headers:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; headers[f] = (8 + n, json.loads(fh.read(n)))
    off, h = headers[f]; e = h[k]; s, t = e["data_offsets"]
    with open(f, "rb") as fh:
        fh.seek(off + s); buf = bytearray(fh.read(t - s))
    return torch.frombuffer(buf, dtype=DT[e["dtype"]]).reshape(e["shape"]) if t > s else torch.empty(e["shape"], dtype=DT[e["dtype"]])

def interp(curve, x):  # piecewise-linear like mergekit gradients
    pos = x * (len(curve) - 1); i = min(int(pos), len(curve) - 2); f = pos - i
    return curve[i] * (1 - f) + curve[i + 1] * f

def t_for(k):
    m = re.search(r"layers\.(\d+)\.", k)
    if not m: return T_DEFAULT
    x = int(m.group(1)) / (NLAYERS - 1)
    if "mlp" in k: return interp([1, 0.5, 0, 0.5, 1], x) * 2 * T_DEFAULT
    if "attn" in k: return interp([0, 0.5, 1, 0.5, 0], x) * 2 * T_DEFAULT
    return T_DEFAULT

def slerp(t, a, b, eps=1e-8, chunk=1 << 24):
    # chunked: never materialize full fp32 copies (embed_tokens is 635M params)
    dt = a.dtype; af, bf = a.reshape(-1), b.reshape(-1); N = af.numel()
    dot = na = nb = 0.0
    for i in range(0, N, chunk):
        x, y = af[i:i+chunk].double(), bf[i:i+chunk].double()
        dot += (x * y).sum().item(); na += (x * x).sum().item(); nb += (y * y).sum().item()
    c = max(-1.0, min(1.0, dot / ((na ** 0.5) * (nb ** 0.5) + eps)))
    if abs(c) > 0.9995:
        wa, wb = 1 - t, t
    else:
        import math
        th = math.acos(c); s = math.sin(th)
        wa, wb = math.sin((1 - t) * th) / s, math.sin(t * th) / s
    out = torch.empty_like(af)
    for i in range(0, N, chunk):
        out[i:i+chunk] = (wa * af[i:i+chunk].float() + wb * bf[i:i+chunk].float()).to(dt)
    return out.reshape(a.shape)

keys = sorted(set(wa) | set(wb))
shard, size, n, index = {}, 0, 0, {}
LIM = 1 * 1024**3
def flush():
    global shard, size, n
    if not shard: return
    n += 1; name = f"model-{n:05d}.safetensors"
    save_file(shard, f"{OUT}/{name}", metadata={"format": "pt"})
    for k in shard: index[k] = name
    shard, size = {}, 0

stats = {"slerp": 0, "fromA": 0, "fromB": 0}
for i, k in enumerate(keys):
    if k in wa and k in wb and not k.startswith("model.visual"):
        ta, tb = get(A, wa, k), get(B, wb, k)
        assert ta.shape == tb.shape, k
        r = slerp(t_for(k), ta, tb) if ta.is_floating_point() else ta; stats["slerp"] += 1
    elif k in wb:
        r = get(B, wb, k); stats["fromB"] += 1
    else:
        r = get(A, wa, k); stats["fromA"] += 1
    shard[k] = r.contiguous(); size += r.numel() * r.element_size()
    if size >= LIM: flush()
    if i % 100 == 0: print(i, len(keys), k, flush=True)
flush()
# rename to HF convention
total = n; final = {}
for j in range(1, total + 1):
    old = f"model-{j:05d}.safetensors"; new = f"model-{j:05d}-of-{total:05d}.safetensors"
    os.rename(f"{OUT}/{old}", f"{OUT}/{new}")
index = {k: v.replace(".safetensors", f"-of-{total:05d}.safetensors") for k, v in index.items()}
json.dump({"metadata": {}, "weight_map": index}, open(f"{OUT}/model.safetensors.index.json", "w"), indent=1)
for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "processor_config.json"]:
    if os.path.exists(f"{B}/{f}"): shutil.copy(f"{B}/{f}", OUT)
print("done", stats, "shards", total)
