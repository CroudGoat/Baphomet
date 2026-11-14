# Kernel V: MERGE^3 Evolve on Kaggle T4x2.
# NSGA-II (pymoo defaults, as mergenetic end2end_multilingual), pop 25, n_gen 7 (=175 evaluations), seed 420,
# merge = dare_ties (mergenetic TiesDareMerger genome: [w_1..w_5, d_1..d_5] in [0,1]), objectives = -gmp-IRT per task
# (MMLU-Pro, GPQA) estimated from the reduced (random) anchor set.  Resumable via evo_log.jsonl cache.
import os, sys, json, glob, time, threading, queue, subprocess, shutil, hashlib
import numpy as np
T00 = time.time()
TIME_LIMIT = float(os.environ.get("TIME_LIMIT_H", "11.0")) * 3600
def find(pat):
    g = glob.glob(pat, recursive=True); return g[0] if g else None
DATA = os.path.dirname(find("/kaggle/input/**/m3lib.py"))
sys.path.insert(0, DATA)
W = "/kaggle/working"
def sh(c, t=None):
    print("$", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
    print(r.stdout[-3000:], r.stderr[-3000:], "rc", r.returncode, flush=True); return r
sh("nvidia-smi -L; df -h /tmp /kaggle/working; free -g; ls -R /kaggle/input | head -50")
sh("pip install -q vllm==0.33.0 pymoo==0.6.1.5 2>&1 | tail -2", t=2400)
import m3lib
from m3lib import log
cfg = json.load(open(f"{DATA}/evo_config.json"))
irt = np.load(f"{DATA}/irt_estimate.npz")
A, B = irt["A"], irt["B"]; THETAS = [irt["theta_" + n] for n in cfg["theta_order"]]
anchors = json.load(open(f"{DATA}/anchors.json"))      # {task: [ {key, ctx, conts, gold, pos} ]}
SCEN = {t: list(map(int, irt["scen_" + t])) for t in cfg["tasks"]}
anchor_items = [dict(key=a["key"], ctx=a["ctx"], conts=a["conts"], gold=a["gold"]) for t in cfg["tasks"] for a in anchors[t]]
json.dump(anchor_items, open("/tmp/anchor_items.json", "w"))

# ---- resume cache
LOG = f"{W}/evo_log.jsonl"
cache = {}
prev = find("/kaggle/input/**/evo_log.jsonl")
for p in [prev, LOG]:
    if p and os.path.exists(p):
        for l in open(p):
            r = json.loads(l); cache[r["gkey"]] = r
if prev and not os.path.exists(LOG):
    shutil.copy(prev, LOG)
log("cached evaluations:", len(cache))
def gkey(x): return ",".join(f"{v:.12f}" for v in x)

# ---- base + fp8 task vectors (one donor at a time)
m3lib.download(m3lib.BASE_ID, "/tmp/base")
os.makedirs("/tmp/tv", exist_ok=True)
tvstats = {}
def extract_lane(gpu, idxs):
    import torch
    torch.cuda.set_device(gpu)
    for i in idxs:
        short, repo = m3lib.SHORT[i], m3lib.DONORS[i]
        t0 = time.time(); d = f"/tmp/donor_{short}"
        m3lib.download(repo, d)
        st = m3lib.extract_tv("/tmp/base", d, f"/tmp/tv/{short}.safetensors", device=f"cuda:{gpu}")
        shutil.rmtree(d, ignore_errors=True)
        st["secs"] = round(time.time() - t0); tvstats[short] = st; log("TV", short, st)
ths = [threading.Thread(target=extract_lane, args=(g, list(range(g, 5, 2)))) for g in (0, 1)]
[t.start() for t in ths]; [t.join() for t in ths]
json.dump(tvstats, open(f"{W}/tv_stats.json", "w"), indent=1)
sh("du -sh /tmp/tv/*; df -h /tmp")
TV = [f"/tmp/tv/{s}.safetensors" for s in m3lib.SHORT]

# ---- fitness of one genome on one GPU
merge_lock = threading.Lock()
class TimeUp(Exception): pass
def evaluate_genome(x, gpu, tag):
    import torch
    w, d = [float(v) for v in x[:5]], [float(v) for v in x[5:]]
    t0 = time.time()
    if all(v == 0 for v in w):
        mdir = "/tmp/base"
    else:
        mdir = f"/tmp/cand_{gpu}"
        shutil.rmtree(mdir, ignore_errors=True)
        seed = int(hashlib.md5(gkey(x).encode()).hexdigest()[:8], 16)
        with merge_lock:
            torch.cuda.set_device(gpu)
            m3lib.merge_dare_ties("/tmp/base", TV, w, d, mdir, seed=seed, device=f"cuda:{gpu}")
            torch.cuda.empty_cache()
    tm = time.time() - t0
    out = f"/tmp/res_{gpu}.json"
    r = subprocess.run(f"CUDA_VISIBLE_DEVICES={gpu} python {DATA}/evalworker.py {mdir} /tmp/anchor_items.json {out} 1 > /tmp/ev_{gpu}.log 2>&1", shell=True)
    if r.returncode != 0:
        print(open(f"/tmp/ev_{gpu}.log").read()[-4000:], flush=True); raise RuntimeError("evalworker failed")
    R = json.load(open(out))["results"]
    fit = {}
    for t in cfg["tasks"]:
        y = np.array([R[a["key"]]["correct"] for a in anchors[t]], dtype=float)
        pos = [a["pos"] for a in anchors[t]]
        fit[t] = m3lib.gmpirt(y, pos, SCEN[t], A, B, THETAS)
    if mdir != "/tmp/base": shutil.rmtree(mdir, ignore_errors=True)
    rec = dict(gkey=gkey(x), x=list(map(float, x)), tag=tag, gpu=gpu, fit=fit,
               F=[-fit[t][cfg["mode"]] for t in cfg["tasks"]], t_merge=round(tm), t_total=round(time.time() - t0),
               elapsed_h=round((time.time() - T00) / 3600, 3))
    return rec

log_lock = threading.Lock()
def evaluate_batch(X, tag):
    todo = [x for x in X if gkey(x) not in cache]
    q = queue.Queue()
    for x in todo: q.put(x)
    err = []
    def worker(gpu):
        while not err:
            try: x = q.get_nowait()
            except queue.Empty: return
            if time.time() - T00 > TIME_LIMIT: err.append("timeup"); return
            try:
                rec = evaluate_genome(x, gpu, tag)
            except Exception as e:
                err.append(repr(e)); return
            with log_lock:
                cache[rec["gkey"]] = rec
                with open(LOG, "a") as f: f.write(json.dumps(rec) + "\n")
                log(f"[{tag}] n={len(cache)} gpu{gpu} F={rec['F']} merge {rec['t_merge']}s total {rec['t_total']}s")
    ths = [threading.Thread(target=worker, args=(g,)) for g in (0, 1)]
    [t.start() for t in ths]; [t.join() for t in ths]
    if err:
        if "timeup" in err: raise TimeUp()
        raise RuntimeError(err)
    return np.array([cache[gkey(x)]["F"] for x in X])

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
class MergeProblem(Problem):
    def __init__(self):
        super().__init__(n_var=10, n_obj=2, xl=0.0, xu=1.0)
        self.gen = 0
    def _evaluate(self, X, out, *args, **kwargs):
        self.gen += 1
        out["F"] = evaluate_batch(X, f"gen{self.gen}")
rng = np.random.default_rng(cfg["seed"])
X0 = np.vstack([np.array(cfg["seed_equal"]), np.array(cfg["seed_base"]), rng.random((cfg["pop_size"] - 2, 10))])
algo = NSGA2(pop_size=cfg["pop_size"], sampling=X0, eliminate_duplicates=True)
status = "done"
try:
    res = minimize(MergeProblem(), algo, ("n_gen", cfg["n_gen"]), seed=cfg["seed"], verbose=True, save_history=False)
except TimeUp:
    status = "timeup"
log("status", status, "evaluated", len(cache))
recs = list(cache.values())
for r in recs: r["mean_fit"] = float(np.mean([r["fit"][t][cfg["mode"]] for t in cfg["tasks"]]))
best = max(recs, key=lambda r: (r["mean_fit"], -recs.index(r)))
json.dump(dict(status=status, n_evaluated=len(recs), best=best, selection_rule="max mean gmp-IRT over MMLU-Pro and GPQA among all evaluated",
               hours=round((time.time() - T00) / 3600, 3)), open(f"{W}/evo_result.json", "w"), indent=1)
log("BEST", best)
