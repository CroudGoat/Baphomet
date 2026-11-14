# Step 3: plain-text tokenisation of every fitness-pool prompt exactly as lleval.encode_pair does (ctx, and ctx+cont),
# each donor tokenizer vs base Qwen3.5-4B. Loaders: transformers AutoTokenizer (what vLLM uses) and raw tokenizers.Tokenizer.
import json, time, collections
from transformers import AutoTokenizer
from tokenizers import Tokenizer
import transformers
pool = json.load(open("kdata/fitness_pool.json"))
NAMES = ["base", "agents_a1", "dsv4pro", "qwen38distill", "qwopus_v3", "arex_turbo"]
toks = {}
for s in NAMES:
    a = AutoTokenizer.from_pretrained(f"toks/{s}")
    toks[f"{s}:auto"] = (type(a).__name__, lambda L, a=a: a(L, add_special_tokens=False)["input_ids"])
    r = Tokenizer.from_file(f"toks/{s}/tokenizer.json")
    toks[f"{s}:raw"] = ("raw", lambda L, r=r: [e.ids for e in r.encode_batch(L, add_special_tokens=False)])
REF = "base:auto"
stats = {k: dict(cls=v[0], items_differ=collections.Counter(), examples=[]) for k, v in toks.items()}
t0 = time.time(); CH = 300
for c0i in range(0, len(pool), CH):
    chunk = pool[c0i:c0i + CH]; strs = []; plan = []
    for it in chunk:
        ctx = it["ctx"]; n = len(ctx) - len(ctx.rstrip()); pre = ctx[-n:] if n else ""; c0 = ctx[:-n] if n else ctx
        i0 = len(strs); strs.append(c0); strs += [c0 + pre + c for c in it["conts"]]
        plan.append((it, i0, len(it["conts"])))
    enc = {k: v[1](strs) for k, v in toks.items()}
    R = enc[REF]
    def pairs(E, i0, n):
        ctx = E[i0]; return [tuple(ctx)] + [tuple(E[i0 + 1 + j][len(ctx):]) for j in range(n)]
    for it, i0, n in plan:
        ref = pairs(R, i0, n)
        for k, E in enc.items():
            if k == REF: continue
            if pairs(E, i0, n) != ref:
                st = stats[k]; st["items_differ"][it["task"]] += 1
                if len(st["examples"]) < 3: st["examples"].append(it["key"])
    if c0i % 3000 == 0: print(c0i, f"{time.time()-t0:.0f}s", {k: dict(v["items_differ"]) for k, v in stats.items() if v["items_differ"]}, flush=True)
out = {k: dict(cls=v["cls"], items_differ=dict(v["items_differ"]), examples=v["examples"]) for k, v in stats.items()}
out["_meta"] = dict(n_items=len(pool), transformers=transformers.__version__, secs=time.time() - t0)
json.dump(out, open("tokcheck_result.json", "w"), indent=1); print("DONE", json.dumps(out, indent=1))
