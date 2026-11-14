# Kernel F: rebuild the selected MERGE^3 merge (bf16, saved normally), run the fixed 180-item test with yesterday's
# exact generation settings (vLLM 0.33.0 serve fp16, max-model-len 12288, gpu-mem 0.95, max-num-seqs 28, mm off,
# client: own chat template, enable_thinking=True, temp 0.6 top_p 0.95 top_k 20 seed 0, max_tokens 8192, CONC 28),
# then evaluate the winner on the full fitness pool (true loglik accuracy, for estimator diagnostics).
import os, sys, json, glob, time, subprocess, shutil, threading, hashlib
T00 = time.time()
def find(pat):
    g = glob.glob(pat, recursive=True); return g[0] if g else None
DATA = os.path.dirname(find("/kaggle/input/**/m3lib.py"))
sys.path.insert(0, DATA)
W = "/kaggle/working"
def sh(c, t=None):
    print("$", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
    print(r.stdout[-3000:], r.stderr[-3000:], "rc", r.returncode, flush=True); return r
sh("nvidia-smi -L; df -h /tmp /kaggle/working")
sh("pip install -q vllm==0.33.0 aiohttp 2>&1 | tail -2", t=2400)
import m3lib
from m3lib import log
best = json.load(open(f"{DATA}/final_choice.json"))
x = best["x"]; w, d = x[:5], x[5:]
seed = int(hashlib.md5(best["gkey"].encode()).hexdigest()[:8], 16)
m3lib.download(m3lib.BASE_ID, "/tmp/base")
os.makedirs("/tmp/tv", exist_ok=True)
for i, (short, repo) in enumerate(zip(m3lib.SHORT, m3lib.DONORS)):
    if w[i] == 0 or d[i] <= 0: continue
    dd = f"/tmp/donor_{short}"; m3lib.download(repo, dd)
    st = m3lib.extract_tv("/tmp/base", dd, f"/tmp/tv/{short}.safetensors"); shutil.rmtree(dd, ignore_errors=True); log("TV", short, st)
TV = [f"/tmp/tv/{s}.safetensors" for s in m3lib.SHORT]
MD = f"{W}/merge3_final"
m3lib.merge_dare_ties("/tmp/base", TV, w, d, MD, seed=seed, device="cuda:0")
json.dump(dict(method="dare_ties (mergekit GTA semantics, DARE random+rescale, sign consensus sum, normalize)", base=m3lib.BASE_ID,
               donors=m3lib.DONORS, weights=w, densities=d, dare_seed=seed, task_vectors="fp8 e4m3 per-row absmax"), open(f"{MD}/merge3_recipe.json", "w"), indent=1)
log("merged", time.time() - T00)
# ---- 180-item test, one vLLM server per GPU, items split by alternating order
os.makedirs("/content", exist_ok=True)
shutil.copy(f"{DATA}/items.jsonl", "/content/items.jsonl"); shutil.copy(f"{DATA}/subset_ids.json", "/content/subset_ids.json")
ids = json.load(open("/content/subset_ids.json"))
halves = [ids[0::2], ids[1::2]]
for g in (0, 1): json.dump(halves[g], open(f"/content/subset_{g}.json", "w"))
for g in (0, 1):
    subprocess.Popen(f"CUDA_VISIBLE_DEVICES={g} nohup vllm serve {MD} --served-model-name m --dtype float16 --max-model-len 12288 "
                     f"--gpu-memory-utilization 0.95 --max-num-seqs 28 --limit-mm-per-prompt '{{\"image\":0,\"video\":0}}' --port {8000+g} "
                     f"> {W}/vllm_{g}.log 2>&1", shell=True)
for g in (0, 1):
    for i in range(240):
        if subprocess.run(f"curl -s localhost:{8000+g}/health", shell=True, capture_output=True).returncode == 0: break
        time.sleep(10)
log("servers up")
procs = [subprocess.Popen(f"SUBSET=/content/subset_{g}.json CONC=28 MAXPHASE=1 PORT={8000+g} python {DATA}/client.py {MD} merge3 {W}/raw_merge3_{g}.jsonl > {W}/client_{g}.log 2>&1", shell=True) for g in (0, 1)]
[p.wait() for p in procs]
with open(f"{W}/raw_merge3.jsonl", "w") as f:
    for g in (0, 1):
        for l in open(f"{W}/raw_merge3_{g}.jsonl"): f.write(l)
log("test done", sum(1 for _ in open(f"{W}/raw_merge3.jsonl")))
sh("pkill -f 'vllm serve'; sleep 15; pkill -9 -f 'vllm serve'; sleep 5; nvidia-smi --query-gpu=memory.used --format=csv")
# ---- winner on the full fitness pool (diagnostic: true loglik accuracy vs gmp-IRT estimate)
pool = json.load(open(f"{DATA}/fitness_pool.json"))
items = [dict(key=it["key"], ctx=it["ctx"], conts=it["conts"], gold=it["gold"]) for it in pool]
half = [items[0::2], items[1::2]]
for g in (0, 1): json.dump(half[g], open(f"/tmp/pool_{g}.json", "w"))
ps = [subprocess.Popen(f"CUDA_VISIBLE_DEVICES={g} python {DATA}/evalworker.py {MD} /tmp/pool_{g}.json /tmp/pool_res_{g}.json {os.environ.get('PREFIX_CACHE','1')} > {W}/pool_eval_{g}.log 2>&1", shell=True) for g in (0, 1)]
[p.wait() for p in ps]
R = {}
for g in (0, 1):
    if os.path.exists(f"/tmp/pool_res_{g}.json"): R.update(json.load(open(f"/tmp/pool_res_{g}.json"))["results"])
json.dump(dict(results=R), open(f"{W}/endpoint_merge3_winner.json", "w"))
log("ALL DONE hours", (time.time() - T00) / 3600)
