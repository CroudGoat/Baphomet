# Independent recomputation (whole tensor, no row chunks, separate code path) of selected tensors vs the written shards.
import sys, json, torch
exec(open("check_baph10.py").read().split('P = "/workspace')[0])
L = json.load(open("/workspace/work/evogm/kout/r2/evo_result.json"))["best"]["lams"]
wm = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]
for k in sys.argv[1:]:
    _, m = merged(k, L)
    with safe_open(f"/workspace/work/baphomet11/Baphomet-1.1-4B/{wm[k]}", "pt") as s: o = s.get_tensor(k)
    print(k, str(o.dtype), "bit-identical:", torch.equal(m.view(-1).view(torch.uint8) if m.dtype!=torch.float32 else m.view(-1), o.view(-1).view(torch.uint8) if o.dtype!=torch.float32 else o.view(-1)), flush=True)
