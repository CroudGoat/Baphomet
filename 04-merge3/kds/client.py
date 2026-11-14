# Runs on Colab VM. Sends prompts (formatted with the model's own chat template) to a local vLLM server.
import json, sys, os, asyncio, random, time
import aiohttp
from transformers import AutoTokenizer

model_dir, tag, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
conc = int(os.environ.get("CONC", "48"))
max_tokens = int(os.environ.get("MAXTOK", "8192"))
port = int(os.environ.get("PORT", "8000"))
only = os.environ.get("ONLY")  # optional comma list of benches

tok = AutoTokenizer.from_pretrained(model_dir)
items = [json.loads(l) for l in open("/content/items.jsonl")]
order = list(range(len(items))); random.Random(777).shuffle(order)
items = [items[i] for i in order]
maxphase = int(os.environ.get("MAXPHASE", "3"))
items = [it for it in items if it["phase"] <= maxphase]
items.sort(key=lambda it: it["phase"])  # stable: phase 1 first, shuffled order within
if os.environ.get("SUBSET"):
    keep = set(json.load(open(os.environ["SUBSET"])))
    items = [it for it in items if it["id"] in keep]
if only:
    items = [it for it in items if it["bench"] in only.split(",")]
done = set()
if os.path.exists(out_path):
    for l in open(out_path):
        try: done.add(json.loads(l)["id"])
        except Exception: pass
todo = [it for it in items if it["id"] not in done]
print(f"{tag}: {len(todo)} todo, {len(done)} done", flush=True)

def fmt(it):
    kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=True)
    if it.get("tools"):
        kw["tools"] = it["tools"]
    return tok.apply_chat_template(it["messages"], **kw)

lock = asyncio.Lock()
async def worker(q, sess, fout, stats):
    while True:
        try: it = q.get_nowait()
        except asyncio.QueueEmpty: return
        prompt = fmt(it)
        body = dict(model="m", prompt=prompt, max_tokens=max_tokens, temperature=0.6, top_p=0.95, top_k=20, seed=0,
                    skip_special_tokens=False)
        t0 = time.time()
        for attempt in range(3):
            try:
                async with sess.post(f"http://127.0.0.1:{port}/v1/completions", json=body, timeout=aiohttp.ClientTimeout(total=7200)) as r:
                    res = await r.json()
                break
            except Exception as e:
                res = {"error": repr(e)}; await asyncio.sleep(5)
        if "choices" not in res:
            print("ERR", it["id"], str(res)[:300], flush=True); continue
        ch = res["choices"][0]
        rec = dict(id=it["id"], bench=it["bench"], category=it["category"], model=tag, prompt_tokens=res["usage"]["prompt_tokens"],
                   completion_tokens=res["usage"]["completion_tokens"], finish_reason=ch["finish_reason"], output=ch["text"],
                   secs=round(time.time()-t0, 1))
        async with lock:
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            stats["n"] += 1; stats["tok"] += rec["completion_tokens"]
            if stats["n"] % 10 == 0:
                el = time.time() - stats["t0"]
                print(f"{tag} {stats['n']}/{len(todo)} {stats['tok']/el:.0f} tok/s elapsed {el/60:.1f}m", flush=True)

async def main():
    q = asyncio.Queue()
    for it in todo: q.put_nowait(it)
    stats = dict(n=0, tok=0, t0=time.time())
    async with aiohttp.ClientSession() as sess:
        with open(out_path, "a") as fout:
            await asyncio.gather(*[worker(q, sess, fout, stats) for _ in range(conc)])
    print("ALLDONE", tag, flush=True)
asyncio.run(main())
