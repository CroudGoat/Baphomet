# Kernel R: MERGE^3 Estimate + Evolve (+ optional Final) on Kaggle T4x2 with one persistent vLLM engine per GPU.
#  1. download base + 5 donors (+ Baphomet), fp8 task vectors
#  2. Estimate: endpoints (base, 5 donors, Baphomet) on tinyBenchmarks IRT anchors (200 MMLU-Pro + 100 GPQA) + the
#     reduced set; gamma_m = fit_theta on the 300 anchor responses  [DEVIATION: MERGE^3 uses the full pool; user decision]
#  3. Evolve: NSGA-II (pymoo defaults), pop 25, n_gen 7, seed 420, dare_ties genome [w1..w5,d1..d5], objectives
#     -gmp-IRT per task from the reduced set (20/task, seed 420). Resumable via evo_log.jsonl / endpoint_*.json in inputs.
#  4. If time allows: Final = merge winner (bf16) + 180-item generation test (same settings as final.py).
import os, sys, json, glob, time, threading, queue, subprocess, shutil, hashlib
import numpy as np
T00 = time.time()
TIME_LIMIT = float(os.environ.get("TIME_LIMIT_H", "11.2")) * 3600
def find(pat):
    g = sorted(glob.glob(pat, recursive=True)); return g[0] if g else None
DATA = os.path.dirname(find("/kaggle/input/**/m3lib.py"))
sys.path.insert(0, DATA)
W = "/kaggle/working"
def sh(c, t=None):
    print("$", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
    print(r.stdout[-3000:], r.stderr[-3000:], "rc", r.returncode, flush=True); return r
sh("nvidia-smi -L; df -h /tmp /kaggle/working; free -g")
import m3lib
from m3lib import log
pool = json.load(open(f"{DATA}/fitness_pool.json")); bykey = {p["key"]: p for p in pool}
IRT = np.load(f"{DATA}/irt_params.npz"); A, B = IRT["A"], IRT["B"]
assert A.shape[2] == len(pool)
SCEN = {t: [p["pos"] for p in pool if p["task"] == t] for t in ["mmlu_pro", "gpqa"]}
ianch = json.load(open(f"{DATA}/irt_anchors.json"))       # tinyBenchmarks anchors (estimate)
ranch = json.load(open(f"{DATA}/anchors.json"))           # MERGE^3 reduced set (evolution), 20/task seed 420
def strip(p): return dict(key=p["key"], ctx=p["ctx"], conts=p["conts"], gold=p["gold"])
ekeys = list(dict.fromkeys([k for t in ianch for k in ianch[t]["keys"]] + [a["key"] for t in ranch for a in ranch[t]]))
json.dump([strip(bykey[k]) for k in ekeys], open("/tmp/endpoint_items.json", "w"))
json.dump([strip(a) for t in ["mmlu_pro", "gpqa"] for a in ranch[t]], open("/tmp/reduced_items.json", "w"))
log("endpoint items", len(ekeys))

# ---------------------------------------------------------------- setup
def dl_all():
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(m3lib.BASE_ID, "/tmp/base")] + [(r, f"/tmp/donor_{s}") for s, r in zip(m3lib.SHORT, m3lib.DONORS)] + \
           [("CloudGoat/Baphomet-1.0-4B", "/tmp/donor_baphomet")]
    with ThreadPoolExecutor(3) as ex: list(ex.map(lambda j: m3lib.download(*j), jobs))
    log("downloads done")
th = threading.Thread(target=dl_all); th.start()
sh("pip install -q vllm==0.33.0 pymoo==0.6.1.5 2>&1 | tail -2", t=2400)
th.join()
ext = r'''
import sys, json, time, threading, os
sys.path.insert(0, sys.argv[1]); import m3lib, torch
os.makedirs("/tmp/tv", exist_ok=True); st = {}
def lane(g, idx):
    torch.cuda.set_device(g)
    for i in idx:
        s = m3lib.SHORT[i]; t0 = time.time()
        st[s] = m3lib.extract_tv("/tmp/base", f"/tmp/donor_{s}", f"/tmp/tv/{s}.safetensors", device=f"cuda:{g}"); st[s]["secs"] = time.time() - t0
        print(s, st[s], flush=True)
ts = [threading.Thread(target=lane, args=(g, list(range(g, 5, 2)))) for g in (0, 1)]
[t.start() for t in ts]; [t.join() for t in ts]
json.dump(st, open("/kaggle/working/tv_stats.json", "w"), indent=1)
'''
open("/tmp/ext.py", "w").write(ext)
sh(f"python /tmp/ext.py {DATA}", t=3600)
TV = [f"/tmp/tv/{s}.safetensors" for s in m3lib.SHORT]

# ---------------------------------------------------------------- persistent workers
class Worker:
    def __init__(self, g):
        self.g, self.n = g, 0
        self.dir = f"/tmp/w{g}"; shutil.rmtree(self.dir, ignore_errors=True); os.makedirs(self.dir)
        self.p = subprocess.Popen(f"CUDA_VISIBLE_DEVICES={g} python {DATA}/gpuworker.py /tmp/base {self.dir} > {W}/worker_{g}.log 2>&1", shell=True)
        self.loaded = None
    def wait_ready(self):
        while not os.path.exists(f"{self.dir}/ready.json"):
            if self.p.poll() is not None: raise RuntimeError(f"worker {self.g} died; see worker_{self.g}.log")
            time.sleep(2)
        return json.load(open(f"{self.dir}/ready.json"))
    def call(self, **cmd):
        n = self.n; self.n += 1
        json.dump(cmd, open(f"{self.dir}/cmd_{n}.tmp", "w")); os.rename(f"{self.dir}/cmd_{n}.tmp", f"{self.dir}/cmd_{n}.json")
        while not os.path.exists(f"{self.dir}/res_{n}.json"):
            if self.p.poll() is not None: raise RuntimeError(f"worker {self.g} died; see worker_{self.g}.log")
            time.sleep(0.3)
        r = json.load(open(f"{self.dir}/res_{n}.json"))
        if not r["ok"]: raise RuntimeError(f"worker {self.g} {cmd.get('op')}: {r['error']}\n{r.get('tb')}")
        return r
workers = [Worker(0), Worker(1)]
log("worker init", [w.wait_ready() for w in workers])

# ---------------------------------------------------------------- Estimate: endpoints on anchors
ENDP = [("base", "/tmp/base")] + [(s, f"/tmp/donor_{s}") for s in m3lib.SHORT] + [("baphomet", "/tmp/donor_baphomet")]
for f in glob.glob("/kaggle/input/**/endpoint_*.json", recursive=True):
    if not os.path.exists(f"{W}/{os.path.basename(f)}") and "summary" not in f: shutil.copy(f, W)
eq = queue.Queue()
for e in ENDP: eq.put(e)
def elane(w):
    while True:
        try: name, d = eq.get_nowait()
        except queue.Empty: return
        out = f"{W}/endpoint_{name}.json"
        if os.path.exists(out): continue
        ts = w.call(op="load_dir", path=d)["t"]; w.loaded = name
        r = w.call(op="eval", items="/tmp/endpoint_items.json")
        json.dump(dict(results=r["results"], stats=r["stats"], t_swap=ts), open(out, "w"))
        log(f"endpoint {name} gpu{w.g} swap {ts:.1f}s eval {r['secs']:.0f}s {r['stats']}")
ths = [threading.Thread(target=elane, args=(w,)) for w in workers]; [t.start() for t in ths]; [t.join() for t in ths]
for s in m3lib.SHORT + ["baphomet"]: shutil.rmtree(f"/tmp/donor_{s}", ignore_errors=True)
# gamma_m = fit_theta on the anchor responses (both tasks jointly, as the full-pool fit was joint)
apos = [p for t in ["mmlu_pro", "gpqa"] for p in ianch[t]["pos"]]
order = m3lib.SHORT + ["base"]
est = dict(A=A, B=B, scen_mmlu_pro=np.array(SCEN["mmlu_pro"]), scen_gpqa=np.array(SCEN["gpqa"]))
summ = {}
for name, _ in ENDP:
    R = json.load(open(f"{W}/endpoint_{name}.json"))["results"]
    y = np.zeros(len(pool))
    for k in ekeys: y[bykey[k]["pos"]] = R[k]["correct"]
    th = m3lib.fit_theta(y, apos, A, B); est["theta_" + name] = th
    pred = m3lib.item_curve(th.reshape(1, A.shape[1], 1), A, B)[0]
    summ[name] = {}
    for t in ["mmlu_pro", "gpqa"]:
        ap, aw = ianch[t]["pos"], np.array(ianch[t]["weights"])
        red = [a["pos"] for a in ranch[t]]
        summ[name][t] = dict(anchor_weighted_acc=float((y[ap] * aw).sum()), irt_pred_acc_pool=float(pred[SCEN[t]].mean()),
                             reduced_acc=float(y[red].mean()))
    log("estimate", name, summ[name])
np.savez_compressed(f"{W}/irt_estimate.npz", **est)
THETAS = [est["theta_" + o] for o in order]
for name, _ in ENDP:   # gmp-IRT of each endpoint from the reduced set (what evolution would have estimated for it)
    R = json.load(open(f"{W}/endpoint_{name}.json"))["results"]
    for t in ["mmlu_pro", "gpqa"]:
        y = np.array([R[a["key"]]["correct"] for a in ranch[t]], dtype=float)
        summ[name][t]["gmpirt_reduced"] = m3lib.gmpirt(y, [a["pos"] for a in ranch[t]], SCEN[t], A, B, THETAS)["gmpirt"]
    log("estimate", name, summ[name])
json.dump(dict(summary=summ, theta_order=order), open(f"{W}/endpoint_summary.json", "w"), indent=1)

# ---------------------------------------------------------------- Evolve
cfg = dict(tasks=["mmlu_pro", "gpqa"], mode="gmpirt", seed=420, pop_size=25, n_gen=7,
           seed_equal=[0.2] * 5 + [1.0] * 5, seed_base=[0.0] * 5 + [1.0] * 5)
LOG = f"{W}/evo_log.jsonl"; cache = {}
prev = find("/kaggle/input/**/evo_log.jsonl")
for p in [prev, LOG]:
    if p and os.path.exists(p):
        for l in open(p):
            r = json.loads(l); cache[r["gkey"]] = r
if prev and not os.path.exists(LOG): shutil.copy(prev, LOG)
log("cached evaluations:", len(cache))
def gkey(x): return ",".join(f"{v:.12f}" for v in x)
class TimeUp(Exception): pass
def evaluate_genome(x, w, tag):
    wts, dens = [float(v) for v in x[:5]], [float(v) for v in x[5:]]
    seed = int(hashlib.md5(gkey(x).encode()).hexdigest()[:8], 16)
    t0 = time.time()
    tm = w.call(op="load_merge", tv=TV, w=wts, d=dens, seed=seed)["t"]; w.loaded = gkey(x)
    r = w.call(op="eval", items="/tmp/reduced_items.json"); R = r["results"]
    fit = {}
    for t in cfg["tasks"]:
        y = np.array([R[a["key"]]["correct"] for a in ranch[t]], dtype=float)
        fit[t] = m3lib.gmpirt(y, [a["pos"] for a in ranch[t]], SCEN[t], A, B, THETAS)
    return dict(gkey=gkey(x), x=list(map(float, x)), tag=tag, gpu=w.g, fit=fit, dare_seed=seed,
                F=[-fit[t][cfg["mode"]] for t in cfg["tasks"]], t_swap=round(tm, 1), t_total=round(time.time() - t0, 1),
                elapsed_h=round((time.time() - T00) / 3600, 3))
log_lock = threading.Lock()
def evaluate_batch(X, tag):
    q = queue.Queue()
    for x in X:
        if gkey(x) not in cache: q.put(x)
    err = []
    def lane(w):
        while not err:
            try: x = q.get_nowait()
            except queue.Empty: return
            if gkey(x) in cache: continue
            if time.time() - T00 > TIME_LIMIT: err.append("timeup"); return
            try: rec = evaluate_genome(x, w, tag)
            except Exception as e: err.append(repr(e)); return
            with log_lock:
                cache[rec["gkey"]] = rec
                with open(LOG, "a") as f: f.write(json.dumps(rec) + "\n")
                log(f"[{tag}] n={len(cache)} gpu{w.g} F={rec['F']} swap {rec['t_swap']}s total {rec['t_total']}s")
    ths = [threading.Thread(target=lane, args=(w,)) for w in workers]; [t.start() for t in ths]; [t.join() for t in ths]
    if err:
        if "timeup" in err: raise TimeUp()
        raise RuntimeError(err)
    return np.array([cache[gkey(x)]["F"] for x in X])
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
class MergeProblem(Problem):
    def __init__(self):
        super().__init__(n_var=10, n_obj=2, xl=0.0, xu=1.0); self.gen = 0
    def _evaluate(self, X, out, *args, **kwargs):
        self.gen += 1; out["F"] = evaluate_batch(X, f"gen{self.gen}")
rng = np.random.default_rng(cfg["seed"])
X0 = np.vstack([np.array(cfg["seed_equal"]), np.array(cfg["seed_base"]), rng.random((cfg["pop_size"] - 2, 10))])
algo = NSGA2(pop_size=cfg["pop_size"], sampling=X0, eliminate_duplicates=True)
status = "done"
try:
    minimize(MergeProblem(), algo, ("n_gen", cfg["n_gen"]), seed=cfg["seed"], verbose=True, save_history=False)
except TimeUp:
    status = "timeup"
log("status", status, "evaluated", len(cache))
recs = list(cache.values())
for r in recs: r["mean_fit"] = float(np.mean([r["fit"][t][cfg["mode"]] for t in cfg["tasks"]]))
best = max(recs, key=lambda r: (r["mean_fit"], -recs.index(r)))
json.dump(dict(status=status, n_evaluated=len(recs), best=best, cfg=cfg,
               selection_rule="max mean gmp-IRT over MMLU-Pro and GPQA among all evaluated",
               hours=round((time.time() - T00) / 3600, 3)), open(f"{W}/evo_result.json", "w"), indent=1)
log("BEST", best)
# winner on the endpoint items (anchors + reduced): diagnostic of the estimator
if status == "done":
    w0 = workers[0]
    x = best["x"]; w0.call(op="load_merge", tv=TV, w=x[:5], d=x[5:], seed=best["dare_seed"])
    r = w0.call(op="eval", items="/tmp/endpoint_items.json")
    json.dump(dict(results=r["results"], stats=r["stats"]), open(f"{W}/endpoint_merge3_winner.json", "w"))
for w in workers:
    try: w.call(op="exit")
    except Exception: pass
    w.p.wait(timeout=120)
sh("nvidia-smi --query-gpu=memory.used --format=csv")

# ---------------------------------------------------------------- Final (only if enough time remains)
if status != "done" or time.time() - T00 > float(os.environ.get("FINAL_DEADLINE_H", "9.0")) * 3600:
    log("skipping final in this kernel; run final.py separately"); sys.exit(0)
x = best["x"]; MD = f"{W}/merge3_final"
m3lib.merge_dare_ties("/tmp/base", TV, x[:5], x[5:], MD, seed=best["dare_seed"], device="cuda:0")
json.dump(dict(method="dare_ties (mergekit GTA semantics, DARE random+rescale, sign consensus sum, normalize)", base=m3lib.BASE_ID,
               donors=m3lib.DONORS, weights=x[:5], densities=x[5:], dare_seed=best["dare_seed"], task_vectors="fp8 e4m3 per-row absmax"),
          open(f"{MD}/merge3_recipe.json", "w"), indent=1)
log("merged winner")
os.makedirs("/content", exist_ok=True)
shutil.copy(f"{DATA}/items.jsonl", "/content/items.jsonl")
ids = json.load(open(f"{DATA}/subset_ids.json")); halves = [ids[0::2], ids[1::2]]
for g in (0, 1): json.dump(halves[g], open(f"/content/subset_{g}.json", "w"))
for g in (0, 1):
    subprocess.Popen(f"CUDA_VISIBLE_DEVICES={g} nohup vllm serve {MD} --served-model-name m --dtype float16 --max-model-len 12288 "
                     f"--gpu-memory-utilization 0.95 --max-num-seqs 28 --limit-mm-per-prompt '{{\"image\":0,\"video\":0}}' --port {8000+g} "
                     f"> {W}/vllm_{g}.log 2>&1", shell=True)
for g in (0, 1):
    for i in range(240):
        if subprocess.run(f"curl -sf localhost:{8000+g}/health", shell=True, capture_output=True).returncode == 0: break
        time.sleep(10)
log("servers up")
procs = [subprocess.Popen(f"SUBSET=/content/subset_{g}.json CONC=28 MAXPHASE=1 PORT={8000+g} python {DATA}/client.py {MD} merge3 {W}/raw_merge3_{g}.jsonl > {W}/client_{g}.log 2>&1", shell=True) for g in (0, 1)]
[p.wait() for p in procs]
with open(f"{W}/raw_merge3.jsonl", "w") as f:
    for g in (0, 1):
        for l in open(f"{W}/raw_merge3_{g}.jsonl"): f.write(l)
log("test done", sum(1 for _ in open(f"{W}/raw_merge3.jsonl")))
sh("pkill -f 'vllm serve'; sleep 15; pkill -9 -f 'vllm serve'")
log("ALL DONE hours", (time.time() - T00) / 3600)
