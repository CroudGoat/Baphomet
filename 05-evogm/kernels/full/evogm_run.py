# EvoGM-style evolutionary linear merge over the 8 Baphomet donors (Qwen3.5-4B base) on Kaggle T4x2.  HARDENED v3 (2026-11-11).
# Search algorithm ported from github.com/JiangTao97/evogm (fusion_bench/method/evogm.py, MIT):
#   coefficients in [-1,1]^N (generator output tanh + clip), init = one-hot experts + average + random U(-1,1)
#   (+ Baphomet's effective lambda, user request), winners = top 30% of the round history, dual MLP generators
#   lose2win / win2lose (hidden 128-256-256-256-128, ReLU, Tanh), loss = 100*(cycle MSE both ways) + 1*MSE(fake_win, winner
#   centroid), Adam lr 1e-4, 200 full-batch epochs per iteration (generator + optimizer persist within a round),
#   new = clip(lose2win(population)), population = top-P of (population U new); rounds with basis shift
#   (top-N of the round history become the next round's experts; coefficients map through the basis).
# Fitness = mean accuracy over {MMLU-Pro, GPQA, BFCL non-live} by GENERATION (own chat template, thinking on, temp .6,
#   top_p .95, top_k 20, seed 0, max_tokens 4096); truncated = wrong. Final: winner on the 180-item test set at 8192.
# Hardening (v3): race-free extraction completion, download retries + disk guard, per-candidate try/except with engine
#   restart + one retry (then fitness 0, failed=True), call timeouts, checkpoint/resume (evo_log.jsonl = every evaluated
#   candidate; evo_state.pt = search state incl. generator/optimizer/RNG; prior copies are read from /kaggle/input),
#   MODE=smoke dry run (extraction, 2 evals, restart test, mini final).
import os, sys, json, time, glob, gzip, shutil, subprocess, threading, queue, hashlib, math, traceback
from concurrent.futures import ThreadPoolExecutor
T00 = time.time()
MODE = "full"                                           # "smoke" | "full"  (set per kernel at push time)
HARD_H = 8.6                                             # whole-kernel ceiling for this session
FIT_MAXTOK = 4096
REJECT_BFCL = 0.25                                       # early stop: BFCL acc < 25% -> MC items not generated (counted wrong)
GPUMEM = [0.84, 0.78]; MAXSEQS = 48                      # eager engines (torch.compile boot died silently in v1)
DISK_CAP_GB = 70.0                                       # our /tmp/m + /tmp/tv usage ceiling (FS went read-only at ~94 GB)
T_LOAD, T_GEN_FIT, T_GEN_FINAL, T_BOOT = 1500, 3600, 4 * 3600, 1800   # worker call timeouts (s)
MAX_BOOTS = 6                                            # per worker per session
W = "/kaggle/working"
def find_all(pat): return sorted(glob.glob(pat, recursive=True))
def find(pat):
    g = find_all(pat); return g[0] if g else None
DATA = os.path.dirname(find("/kaggle/input/**/linmerge.py"))
if os.path.exists(f"{DATA}/bfcl_eval.tar.gz"):
    os.makedirs("/tmp/pylib", exist_ok=True); subprocess.run(f"tar -xzf {DATA}/bfcl_eval.tar.gz -C /tmp/pylib", shell=True, check=True); PYLIB = "/tmp/pylib"
else:
    PYLIB = f"{DATA}/bfcl_eval"   # Kaggle auto-extracts archives
assert os.path.exists(f"{PYLIB}/bfcl_eval/constants/enums.py"), PYLIB
sys.path.insert(0, DATA); sys.path.insert(1, PYLIB)
os.environ["PYTHONPATH"] = f"{DATA}:{PYLIB}:" + os.environ.get("PYTHONPATH", "")
_PL = open(f"{W}/progress.log", "a")
def log(*a):
    line = time.strftime("%H:%M:%S") + f" [{(time.time()-T00)/3600:.3f}h] " + " ".join(str(x) for x in a)
    print(line, flush=True); _PL.write(line + "\n"); _PL.flush(); os.fsync(_PL.fileno())
def sh(c, t=None):
    try:
        r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
        log("$", c, "->", r.stdout[-1500:], r.stderr[-1500:], "rc", r.returncode); return r
    except Exception as e:
        log("$", c, "EXC", repr(e)[:300]); return None
log("MODE", MODE, "HARD_H", HARD_H)
sh("nvidia-smi -L; df -h /tmp /kaggle/working; free -g; nproc", t=60)
import linmerge
linmerge.log = log
def disk_used_gb():
    tot = 0
    for d in ("/tmp/m", "/tmp/tv"):
        for r, _, fs in os.walk(d):
            for f in fs:
                try: tot += os.path.getsize(os.path.join(r, f))
                except OSError: pass
    return round(tot / 1e9, 1)
def working_gb():
    tot = 0
    for r, _, fs in os.walk(W):
        for f in fs:
            try: tot += os.path.getsize(os.path.join(r, f))
            except OSError: pass
    return round(tot / 1e9, 2)
def _monitor():
    while True:
        try:
            m = subprocess.run("free -m | awk 'NR==2{print $3\"/\"$2\" MB used, avail \"$7}'; timeout 20 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\\n' ' '",
                               shell=True, capture_output=True, text=True, timeout=40).stdout.replace("\n", " | ")
            log("MON", m, "| ours", disk_used_gb(), "GB | working", working_gb(), "GB")
        except Exception as e: log("MON err", repr(e)[:200])
        time.sleep(180)
threading.Thread(target=_monitor, daemon=True).start()
STATUS = dict(stage="start", mode=MODE)
def status(**kw):
    STATUS.update(kw); STATUS["elapsed_h"] = round((time.time() - T00) / 3600, 3)
    try:
        json.dump(STATUS, open(f"{W}/status.tmp", "w"), indent=1, default=str); os.replace(f"{W}/status.tmp", f"{W}/status.json")
    except Exception as e: log("status write err", repr(e))
status()

# ---------------------------------------------------------------- setup: pip || downloads -> fp8 TV extraction (bounded disk)
DIRS = {"base": "/tmp/m/base", **{s: f"/tmp/m/{s}" for s in linmerge.SHORT}, "baphomet": "/tmp/m/baphomet"}
REPOS = {"base": linmerge.BASE_ID, **{d[0]: d[1] for d in linmerge.DONORS}, "baphomet": "CloudGoat/Baphomet-1.0-4B"}
REMAP = {d[0]: d[2] for d in linmerge.DONORS}
TVDIR = "/tmp/tv"; os.makedirs(TVDIR, exist_ok=True)
TV = [f"{TVDIR}/{s}.safetensors" for s in linmerge.SHORT]
log("models+tv disk at start", disk_used_gb())
pip_ok = threading.Event(); pip_res = {}
def pip_thread():
    t = time.time()
    for a in range(2):
        r = subprocess.run(f"pip install --no-cache-dir vllm==0.33.0 >> {W}/pip.log 2>&1", shell=True)
        c = subprocess.run("python -c 'import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)'",
                           shell=True, capture_output=True, text=True)
        pip_res.update(rc=r.returncode, versions=c.stdout.strip(), err=c.stderr[-800:], secs=round(time.time() - t), attempt=a)
        if pip_res["versions"].startswith("0.33.0"): break
        log("pip attempt failed", pip_res)
    log("pip finished", pip_res); pip_ok.set()
threading.Thread(target=pip_thread, daemon=True).start()

def dl_retry(repo, d, patterns=None, tries=3):
    for a in range(tries):
        try: return linmerge.download(repo, d, patterns)
        except Exception as e:
            log("download error", repo, "attempt", a, repr(e)[:300])
            if a == tries - 1: raise
            time.sleep(30)
def disk_guard(need_gb, tag):
    t = time.time()
    while disk_used_gb() + need_gb > DISK_CAP_GB:
        if time.time() - t > 2400: raise RuntimeError(f"disk guard timeout before {tag}: {disk_used_gb()} GB used")
        log("disk guard: waiting before", tag, disk_used_gb(), "GB used"); time.sleep(20)

dl_done = {}; dl_err = {}; slots = threading.Semaphore(3); ready_q = queue.Queue()   # <=3 donors on disk (peak ~57 GB < DISK_CAP_GB; guard enforces)
def dl_all():
    t = time.time(); dl_retry(REPOS["base"], DIRS["base"]); dl_done["base"] = round(time.time() - t); log("downloaded base", dl_done["base"], "disk", disk_used_gb())
    def one(k):
        slots.acquire()
        try:
            disk_guard(12, k)
            t = time.time(); dl_retry(REPOS[k], DIRS[k]); dl_done[k] = round(time.time() - t)
            log("downloaded", k, dl_done[k], "disk", disk_used_gb()); ready_q.put((k, True))
        except Exception as e:
            dl_err[k] = repr(e)[:500]; log("DOWNLOAD FAILED", k, dl_err[k]); shutil.rmtree(DIRS[k], ignore_errors=True)
            slots.release(); ready_q.put((k, False))
    with ThreadPoolExecutor(2) as ex: list(ex.map(one, linmerge.SHORT))
th = threading.Thread(target=dl_all); th.start()
pip_ok.wait()
if not pip_res.get("versions", "").startswith("0.33.0"):
    sh(f"tail -30 {W}/pip.log"); status(stage="pip_failed", pip=pip_res); raise SystemExit("pip install vllm failed")
open("/tmp/ext1.py", "w").write(
    "import sys, json; sys.path.insert(0, sys.argv[1]); import linmerge\n"
    "st = linmerge.extract_tv(sys.argv[2], sys.argv[3], sys.argv[4] == '1', sys.argv[5], device='cuda')\n"
    "json.dump(st, open(sys.argv[5] + '.json', 'w'))\n")
tv_stats = {}; ext_fail = {}; ext_lock = threading.Lock(); inflight = set()
def ext_lane(g):
    # race-free completion: an item is taken from the queue and marked in-flight atomically; a lane only finishes when the
    # download thread is dead, the queue is empty and no other lane is extracting. Donors that never arrived = dl failures.
    while True:
        with ext_lock:
            try:
                item = ready_q.get_nowait(); inflight.add(item[0]); finished = False
            except queue.Empty:
                item = None; finished = (not th.is_alive()) and ready_q.empty() and not inflight
        if item is None:
            if finished: return
            time.sleep(3); continue
        k, ok = item
        if ok:
            for a in range(2):
                t = time.time()
                try:
                    r = subprocess.run(f"CUDA_VISIBLE_DEVICES={g} python /tmp/ext1.py {DATA} {DIRS['base']} {DIRS[k]} {int(REMAP[k])} {TVDIR}/{k}.safetensors",
                                       shell=True, capture_output=True, text=True, timeout=1800)
                    rc, err = r.returncode, r.stderr[-1500:]
                except Exception as e: rc, err = -1, repr(e)
                if rc == 0:
                    st = json.load(open(f"{TVDIR}/{k}.safetensors.json")); st["secs"] = round(time.time() - t)
                    with ext_lock: tv_stats[k] = st; ext_fail.pop(k, None)
                    log("tv", k, {x: st[x] for x in ("n", "zero", "missing", "rel_qerr", "secs")}); break
                with ext_lock: ext_fail[k] = err
                log("EXTRACT FAILED", k, "attempt", a, err)
            shutil.rmtree(DIRS[k], ignore_errors=True); slots.release()
        log("disk", disk_used_gb())
        with ext_lock: inflight.discard(k)
els = [threading.Thread(target=ext_lane, args=(g,)) for g in (0, 1)]; [t.start() for t in els]; [t.join() for t in els]; th.join()
json.dump(tv_stats, open(f"{W}/tv_stats.json", "w"), indent=1)
missing = [k for k in linmerge.SHORT if k not in tv_stats]
if missing:
    status(stage="extract_failed", missing=missing, dl_err=dl_err, ext_fail=ext_fail); raise SystemExit(f"TV extraction failed: {missing}")
status(stage="tvs_done", dl=dl_done, pip=pip_res)
log("all 8 TVs extracted; disk", disk_used_gb())

# ---------------------------------------------------------------- workers
class WorkerError(Exception): pass
def gpu_mem_used(g):
    try:
        r = subprocess.run(f"timeout 20 nvidia-smi -i {g} --query-gpu=memory.used --format=csv,noheader,nounits", shell=True, capture_output=True, text=True, timeout=30)
        return int(r.stdout.strip().split()[0])
    except Exception: return -1
class Worker:
    def __init__(self, g):
        self.g = g; self.boots = 0; self.fails = 0; self.p = None; self.dead = False; self.gpumem = None; self.eager = True
    def alive(self): return self.p is not None and self.p.poll() is None
    def start(self, gpumem):
        self.n = 0; self.gpumem = gpumem
        self.dir = f"/tmp/w{self.g}_{self.boots}"; shutil.rmtree(self.dir, ignore_errors=True); os.makedirs(self.dir)
        self.logf = f"{W}/worker_{self.g}_{self.boots}.log"
        self.p = subprocess.Popen(f"CUDA_VISIBLE_DEVICES={self.g} exec python {DATA}/gpuworker2.py {DIRS['base']} {self.dir} {gpumem} 1 {MAXSEQS} "
                                  f"> {self.logf} 2>&1", shell=True)
        self.boots += 1
    def wait_ready(self):
        t = time.time()
        while not os.path.exists(f"{self.dir}/ready.json"):
            if self.p.poll() is not None: raise WorkerError(f"worker {self.g} died during init (boot {self.boots})")
            if time.time() - t > T_BOOT: raise WorkerError(f"worker {self.g} init timeout")
            time.sleep(2)
        r = json.load(open(f"{self.dir}/ready.json")); log("worker", self.g, "ready", r); return r
    def call(self, timeout, **cmd):
        if not self.alive(): raise WorkerError(f"worker {self.g} not running")
        n = self.n; self.n += 1; t = time.time()
        json.dump(cmd, open(f"{self.dir}/cmd_{n}.tmp", "w")); os.rename(f"{self.dir}/cmd_{n}.tmp", f"{self.dir}/cmd_{n}.json")
        while not os.path.exists(f"{self.dir}/res_{n}.json"):
            if self.p.poll() is not None: raise WorkerError(f"worker {self.g} died during {cmd.get('op')}; see {self.logf}")
            if time.time() - t > timeout: raise WorkerError(f"worker {self.g} {cmd.get('op')} timeout {timeout}s")
            time.sleep(0.5)
        r = json.load(open(f"{self.dir}/res_{n}.json"))
        if not r["ok"]: raise WorkerError(f"worker {self.g} {cmd.get('op')}: {r['error']}\n{r.get('tb')}")
        return r
    def kill(self):
        """exit command, then SIGKILL anything left, then wait until the GPU's memory is actually released."""
        if self.alive():
            try:
                n = self.n; self.n += 1
                json.dump(dict(op="exit"), open(f"{self.dir}/cmd_{n}.tmp", "w")); os.rename(f"{self.dir}/cmd_{n}.tmp", f"{self.dir}/cmd_{n}.json")
                self.p.wait(timeout=60)
            except Exception: pass
        if self.p is not None:
            try: self.p.kill(); self.p.wait(timeout=30)
            except Exception: pass
        if self.p is not None: subprocess.run(f"pkill -9 -f '[g]puworker2.py \\S+ {self.dir} '", shell=True)
        t = time.time(); m = gpu_mem_used(self.g)
        while m > 700 and time.time() - t < 180:
            time.sleep(5); m = gpu_mem_used(self.g)
        if m > 700:   # some other process still holds the GPU: kill whatever nvidia-smi lists on it
            sh(f"nvidia-smi -i {self.g} --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9; sleep 10")
            m = gpu_mem_used(self.g)
        log("worker", self.g, "killed; GPU mem used now", m, "MiB")
    def boot(self):
        """start an engine: gpumem .84 first, .78 after an earlier failure; each config tried once."""
        cfgs = [GPUMEM[0], GPUMEM[1]] if self.fails == 0 else [GPUMEM[1], GPUMEM[0]]
        last = None
        for gm in cfgs:
            if self.boots >= MAX_BOOTS: break
            try:
                self.start(gm); r = self.wait_ready(); self.dead = False; return r
            except Exception as e:
                last = e; log("worker boot failed", self.g, gm, repr(e)); sh(f"tail -40 {self.logf}"); self.kill()
        self.dead = True
        raise WorkerError(f"worker {self.g}: no working engine config ({last})")
    def restart(self):
        self.fails += 1; log("restarting worker", self.g, "fails", self.fails); self.kill(); return self.boot()

workers = [Worker(0), Worker(1)]
boot_res = {}
def boot_bg(g):
    try: boot_res[g] = workers[g].boot()
    except Exception as e: log("boot error", g, repr(e))
tb0 = threading.Thread(target=boot_bg, args=(0,)); tb0.start()
if MODE == "smoke":   # validate the fp8 merge against two shards of the published Baphomet on GPU1 while GPU0 boots
    try:
        dl_retry(REPOS["baphomet"], DIRS["baphomet"], patterns=["model-00001-of-00010.safetensors", "model-00006-of-00010.safetensors",
                                                                "model.safetensors.index.json", "config.json"])
        sh(f"CUDA_VISIBLE_DEVICES=1 python {DATA}/valmerge.py {DIRS['base']} {TVDIR} {DIRS['baphomet']} {W}/valmerge.json", t=1800)
    except Exception as e:
        log("valmerge skipped", repr(e))
    shutil.rmtree(DIRS["baphomet"], ignore_errors=True)
else:
    time.sleep(120)   # stagger the two engine boots (host RAM)
boot_bg(1); tb0.join()
if not boot_res:
    status(stage="boot_failed"); raise SystemExit("no worker booted")
status(stage="workers_ready", boot=boot_res)

# ---------------------------------------------------------------- items / scoring
import numpy as np, torch
torch.set_num_threads(1)   # deterministic CPU generator training (replayable)
import fitscore
FIT = json.load(open(f"{DATA}/fitness_items.json"))
FITBY = {i["id"]: i for i in FIT}
TASKS = ["mmlupro", "gpqa", "bfcl"]
NT = {t: sum(1 for i in FIT if i["bench"] == t) for t in TASKS}
BF = [i for i in FIT if i["bench"] == "bfcl"]
MC = [i for i in FIT if i["bench"] == "gpqa"] + [i for i in FIT if i["bench"] == "mmlupro"]   # longest first
json.dump(BF, open("/tmp/fit_bfcl.json", "w")); json.dump(MC, open("/tmp/fit_mc.json", "w"))
os.makedirs(f"{W}/fit_outputs", exist_ok=True)
LOG = f"{W}/evo_log.jsonl"
def lkey(x): return ",".join(f"{float(v):.6f}" for v in x)
EVAL_TIMES = []; THR = []
cache = {}
# resume: every evo_log.jsonl in the inputs (smoke run, crashed runs) is merged into the cache
for prev in find_all("/kaggle/input/**/evo_log.jsonl"):
    n0 = len(cache)
    for l in open(prev):
        try: r = json.loads(l)
        except Exception: continue
        if r["lkey"] not in cache: cache[r["lkey"]] = r
    log("resume: read", prev, len(cache) - n0, "new records")
with open(LOG, "w") as f:
    for r in cache.values(): f.write(json.dumps(r) + "\n")
for r in cache.values():
    if not r.get("rejected") and not r.get("failed"): EVAL_TIMES.append(r["t_total"])
    if r.get("mc_secs"): THR.append(r["mc_tokens"] / max(r["mc_secs"], 1))
log("cached evaluations", len(cache), "eval times", EVAL_TIMES)
log_lock = threading.Lock()
def safe_score(it, out, fr):
    try: return fitscore.score(it, out, fr)
    except Exception as e: return dict(correct=False, truncated=fr == "length", error=repr(e)[:200])
def evaluate(lams, w, meta):
    t0 = time.time()
    r = w.call(T_LOAD, op="load_merge", tvs=TV, lams=[float(v) for v in lams])
    t_swap = r["t"]; tag = hashlib.md5(lkey(lams).encode()).hexdigest()[:10]
    outs = []; t_gen = 0.0; gen_tok = 0; mc_secs = mc_tok = None
    rb = w.call(T_GEN_FIT, op="gen", items="/tmp/fit_bfcl.json", max_tokens=FIT_MAXTOK, out=f"/tmp/o_{w.g}.json")
    outs += json.load(open(f"/tmp/o_{w.g}.json")); t_gen += rb["secs"]; gen_tok += rb["gen_tokens"]
    sc = {o["id"]: safe_score(FITBY[o["id"]], o["output"], o["finish_reason"]) for o in outs}
    bacc = sum(sc[i["id"]]["correct"] and not sc[i["id"]]["truncated"] for i in BF) / NT["bfcl"]
    rejected = bacc < REJECT_BFCL
    if not rejected:
        rm = w.call(T_GEN_FIT, op="gen", items="/tmp/fit_mc.json", max_tokens=FIT_MAXTOK, out=f"/tmp/o_{w.g}.json")
        mo = json.load(open(f"/tmp/o_{w.g}.json")); outs += mo; t_gen += rm["secs"]; gen_tok += rm["gen_tokens"]
        mc_secs, mc_tok = rm["secs"], rm["gen_tokens"]; THR.append(mc_tok / max(mc_secs, 1))
        for o in mo: sc[o["id"]] = safe_score(FITBY[o["id"]], o["output"], o["finish_reason"])
    acc, trunc, tok = {}, {}, {}
    for t in TASKS:
        its = [i for i in FIT if i["bench"] == t]
        acc[t] = sum(1 for i in its if i["id"] in sc and sc[i["id"]]["correct"] and not sc[i["id"]]["truncated"]) / NT[t]
        done = [o for o in outs if FITBY[o["id"]]["bench"] == t]
        trunc[t] = round(sum(o["finish_reason"] == "length" for o in done) / max(1, len(done)), 4)
        tok[t] = round(sum(o["completion_tokens"] for o in done) / max(1, len(done)), 1)
    fit = float(np.mean([acc[t] for t in TASKS]))
    with gzip.open(f"{W}/fit_outputs/{tag}.json.gz", "wt") as f:
        json.dump(dict(lams=list(map(float, lams)), outputs=outs, scores={k: dict(correct=v["correct"], truncated=v["truncated"]) for k, v in sc.items()}), f)
    rec = dict(lkey=lkey(lams), lams=[float(v) for v in lams], tag=tag, fitness=fit, acc=acc, trunc=trunc, mean_tokens=tok,
               rejected=rejected, failed=False, gpu=w.g, t_swap=round(t_swap, 1), t_gen=round(t_gen, 1), gen_tokens=gen_tok,
               mc_secs=mc_secs, mc_tokens=mc_tok, t_total=round(time.time() - t0, 1), elapsed_h=round((time.time() - T00) / 3600, 3), **meta)
    if not rejected: EVAL_TIMES.append(rec["t_total"])
    return rec

# ---------------------------------------------------------------- budget
FINAL_TOK = 572000   # Baphomet's generated tokens on the 180-item test at 8192 (bench raw_baphomet)
def t_eval_est():
    return 1.1 * float(np.mean(EVAL_TIMES)) if EVAL_TIMES else 1500.0
def final_est():
    thr = float(np.mean(THR)) if THR else 200.0
    return 1.35 * FINAL_TOK / (2 * thr) + 1200
def evo_deadline(): return T00 + HARD_H * 3600 - final_est()
def n_alive(): return sum(1 for w in workers if not w.dead and w.alive())
def evals_affordable():
    return int(max(1, n_alive()) * max(0.0, evo_deadline() - time.time()) / t_eval_est())

STOP = {"flag": False}
def eval_batch(X, meta_list):
    """Evaluate effective-lambda vectors on the live GPUs; returns fitness array (nan = not evaluated: time budget / no engine).
    A failing candidate restarts its engine and is retried once; a second failure records fitness 0 (failed=True)."""
    q = queue.Queue()
    for j in range(len(X)): q.put(j)
    F = np.full(len(X), np.nan)
    def lane(w):
        while True:
            if STOP.get("abort"): return
            try: j = q.get_nowait()
            except queue.Empty: return
            k = lkey(X[j])
            if k in cache: F[j] = cache[k]["fitness"]; continue
            if MODE != "smoke" and time.time() + t_eval_est() > evo_deadline():
                STOP["flag"] = True; log("time budget: skipping", meta_list[j].get("kind")); continue
            rec = None; errs = []
            for attempt in range(2):
                try:
                    rec = evaluate(X[j], w, meta_list[j]); break
                except Exception as e:
                    errs.append(repr(e)[:1500]); log("EVAL ERROR gpu", w.g, "attempt", attempt, repr(e)[:2000])
                    try: w.restart()
                    except Exception as e2:
                        log("worker", w.g, "could not restart:", repr(e2)[:500], "-> lane stops, candidate requeued")
                        q.put(j); return
            if rec is None:
                rec = dict(lkey=k, lams=[float(v) for v in X[j]], tag=None, fitness=0.0, acc={t: 0.0 for t in TASKS}, failed=True,
                           rejected=False, errors=errs, gpu=w.g, t_total=None, elapsed_h=round((time.time() - T00) / 3600, 3), **meta_list[j])
            with log_lock:
                nf = sum(1 for r in cache.values() if r.get("failed")) + int(rec["failed"])
                if rec["failed"] and nf >= 4 and not any(not r.get("failed") for r in cache.values()):
                    STOP["abort"] = True   # systematic: the first 4 evaluations all failed -> do not burn the budget on zeros
                cache[rec["lkey"]] = rec; F[j] = rec["fitness"]
                with open(LOG, "a") as f: f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
                log(f"[{rec.get('round')}/{rec.get('iter')}] n={len(cache)} gpu{w.g} fit={rec['fitness']:.4f} acc={rec['acc']} "
                    f"rej={rec['rejected']} failed={rec['failed']} swap {rec.get('t_swap')}s total {rec.get('t_total')}s tok {rec.get('gen_tokens')}")
                status(stage="evolve", n_evals=len(cache), last=dict(fit=rec["fitness"], acc=rec["acc"]))
    for w in workers:   # an engine that died between batches is brought back (bounded by MAX_BOOTS)
        if not w.alive() and w.boots < MAX_BOOTS and (time.time() + 900 + t_eval_est() < evo_deadline()):
            log("worker", w.g, "not running at batch start -> restart")
            try: w.restart()
            except Exception as e: log("restart failed:", repr(e)[:300])
    lanes = [w for w in workers if not w.dead and w.alive()]
    if not lanes: STOP["flag"] = True; log("no live engine -> stop evolution"); return F
    ths = [threading.Thread(target=lane, args=(w,)) for w in lanes]; [t.start() for t in ths]; [t.join() for t in ths]
    if STOP.get("abort"):
        status(stage="aborted_systematic_failures"); log("ABORT: first evaluations all failed"); [w.kill() for w in workers]; raise SystemExit("systematic eval failure")
    if not q.empty() and n_alive() > 0:   # candidates requeued by a lane that died after the others finished
        ths = [threading.Thread(target=lane, args=(w,)) for w in workers if not w.dead and w.alive()]; [t.start() for t in ths]; [t.join() for t in ths]
    return F

# ---------------------------------------------------------------- EvoGM
np.random.seed(42); torch.manual_seed(42)
NE = linmerge.N
HID = (128, 256, 256, 256, 128); GEN_LR = 1e-4; GEN_EPOCHS = 200; CYCLE = 100.0; OPT = 1.0; WIN_PORTION = 0.3
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self, dim):
        super().__init__(); L = []; p = dim
        for h in HID: L += [nn.Linear(p, h), nn.ReLU()]; p = h
        L += [nn.Linear(p, dim), nn.Tanh()]; self.net = nn.Sequential(*L)
    def forward(self, x): return self.net(x)
class Dual(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.lose2win = Generator(dim); self.win2lose = Generator(dim); self.mse = nn.MSELoss(reduction="none")
    def forward(self, win, lose):
        fw = self.lose2win(lose); fl = self.win2lose(win)
        cl = self.win2lose(fw); cw = self.lose2win(fl)
        return CYCLE * (self.mse(cl, lose).mean() + self.mse(cw, win).mean()), fw, fl
def train_epoch(g, opt, winners, losers):
    g.train(); wt = torch.tensor(winners, dtype=torch.float32); lt = torch.tensor(losers, dtype=torch.float32)
    m = min(len(wt), len(lt))
    if len(wt) > m: wt = wt[torch.randperm(len(wt))[:m]]
    if len(lt) > m: lt = lt[torch.randperm(len(lt))[:m]]
    opt.zero_grad(); cyc, fw, _ = g(wt, lt)
    loss = cyc + OPT * nn.functional.mse_loss(fw, wt.mean(dim=0, keepdim=True).expand_as(fw))
    loss.backward(); opt.step(); return float(loss)
def init_population(P, include_baph):
    pop = [np.eye(NE)[i] for i in range(NE)] + [np.full(NE, 1.0 / NE)]
    if include_baph: pop.append(np.array(linmerge.BAPHOMET_LAMBDA, dtype=float))
    while len(pop) < P: pop.append(np.random.uniform(-1, 1, NE))
    return np.array(pop[:P])

# ---- checkpoint state: everything needed to continue the search at a step boundary
STATE = f"{W}/evo_state.pt"
def save_state(S):
    S = dict(S); S["np_rng"] = np.random.get_state(); S["torch_rng"] = torch.get_rng_state()
    torch.save(S, STATE + ".tmp"); os.replace(STATE + ".tmp", STATE)
    log("state saved: next step", S["next"])
def load_state():
    best = None
    for p in find_all("/kaggle/input/**/evo_state.pt"):
        try:
            S = torch.load(p, weights_only=False)
            if best is None or tuple(S["next"]) > tuple(best["next"]): best = S
            log("resume: found state", p, "next", S["next"])
        except Exception as e: log("resume: unreadable state", p, repr(e)[:300])
    return best
basis = np.eye(NE, dtype=np.float32)
def run_eval(pop, rnd, it, kinds):
    E = np.asarray(pop, dtype=np.float32) @ basis
    meta = [dict(round=rnd, iter=it, kind=kinds[j], coords=[float(v) for v in pop[j]]) for j in range(len(pop))]
    return eval_batch(E, meta)

status(stage="evolve_start")
P12 = init_population(12, True)
kinds12 = [f"onehot_{s}" for s in linmerge.SHORT] + ["average", "baphomet"] + ["random"] * 2
first = [9, 8]   # baphomet, average (one per GPU) -> measured cost decides the plan

if MODE == "smoke":
    F_first = run_eval(P12[first], 1, 0, [kinds12[j] for j in first])
    log("SMOKE evals", F_first.tolist())
    # generator + checkpoint round trip on the two real records
    hp = P12[first]; hf = F_first.copy()
    dual = Dual(NE); opt = torch.optim.Adam(dual.parameters(), lr=GEN_LR)
    idx = np.argsort(-hf, kind="stable"); nw = max(1, int(len(hp) * WIN_PORTION))
    for ep in range(GEN_EPOCHS): loss = train_epoch(dual, opt, hp[idx[:nw]], hp[idx[nw:]])
    with torch.no_grad(): new = np.clip(dual.lose2win(torch.tensor(hp, dtype=torch.float32)).numpy(), -1, 1)
    S = dict(next=(1, 1), plan=None, dual=dual.state_dict(), opt=opt.state_dict(), new=new)
    STATE = "/tmp/smoke_state.pt"; save_state(S)
    S2 = torch.load(STATE, weights_only=False); d2 = Dual(NE); d2.load_state_dict(S2["dual"])
    with torch.no_grad(): new2 = np.clip(d2.lose2win(torch.tensor(hp, dtype=torch.float32)).numpy(), -1, 1)
    log("SMOKE generator loss", loss, "state round trip identical:", bool(np.array_equal(new, new2)), "new", new.round(3).tolist())
    plan_info = dict(E_total_if_full=evals_affordable(), t_eval_est_s=t_eval_est(), thr_tok_s=float(np.mean(THR)) if THR else None,
                     final_est_s=final_est())
    json.dump(plan_info, open(f"{W}/plan.json", "w"), indent=1, default=str); log("SMOKE plan info", plan_info)
    best = dict(lams=list(map(float, linmerge.BAPHOMET_LAMBDA)), fitness=float(F_first[0]))
else:
    S = load_state()
    if S is not None:
        plan = S["plan"]; basis = S["basis"]; HIST = S["HIST"]; np.random.set_state(S["np_rng"]); torch.set_rng_state(S["torch_rng"])
        start_rnd, start_it = S["next"]
        log("RESUMING at round", start_rnd, "iter", start_it, "plan", plan)
    else:
        HIST = []
        F_first = run_eval(P12[first], 1, 0, [kinds12[j] for j in first])
        E_total = len([r for r in cache.values() if r.get("round") == 1]) + evals_affordable()
        MAX_ITER = 3   # repo default max_iter; iterations run while the measured budget allows
        if E_total >= 64: plan = dict(P=12, rounds=2, max_iter=MAX_ITER)       # 12 + 2*12 | 4 + 2*12 minimum
        elif E_total >= 36: plan = dict(P=12, rounds=1, max_iter=MAX_ITER)
        elif E_total >= 20: plan = dict(P=10, rounds=1, max_iter=MAX_ITER)
        else: plan = None
        plan_info = dict(plan=plan, E_total=E_total, t_eval_est_s=t_eval_est(), thr_tok_s=float(np.mean(THR)) if THR else None,
                         final_est_s=final_est(), evo_deadline_h=(evo_deadline() - T00) / 3600)
        json.dump(plan_info, open(f"{W}/plan.json", "w"), indent=1, default=str)
        log("PLAN", plan_info)
        if plan is None:
            status(stage="stopped_budget", plan=plan_info); [w.kill() for w in workers]; sys.exit(0)
        start_rnd, start_it = 1, 0
    P = plan["P"]
    for rnd in range(start_rnd, plan["rounds"] + 1):
        if rnd == start_rnd and start_it > 0:   # resume inside a round
            pop, scores_fit, hist_pop, hist_fit = S["pop"], S["scores_fit"], S["hist_pop"], S["hist_fit"]
            dual = Dual(NE); dual.load_state_dict(S["dual"]); opt = torch.optim.Adam(dual.parameters(), lr=GEN_LR); opt.load_state_dict(S["opt"])
            torch.set_rng_state(S["torch_rng"])   # Dual() init above consumed RNG; restore the saved stream
            it0 = start_it
        else:
            if rnd == 1:
                pop = P12[:P]; kinds = kinds12[:P]
            else:
                pop = init_population(P, False); kinds = [f"onehot_prev_top{i+1}" for i in range(NE)] + ["average"] + ["random"] * (P - NE - 1)
            scores_fit = run_eval(pop, rnd, 0, kinds)
            hist_pop = pop.copy(); hist_fit = scores_fit.copy()
            ok = ~np.isnan(hist_fit)
            HIST.append(dict(round=rnd, iter=0, best=float(np.nanmax(hist_fit)) if ok.any() else None,
                             mean=float(np.nanmean(hist_fit)) if ok.any() else None, n=int(ok.sum()), elapsed_h=(time.time() - T00) / 3600))
            json.dump(dict(plan=plan, hist=HIST), open(f"{W}/history.json", "w"), indent=1)
            log("HIST", HIST[-1])
            pop, scores_fit = pop[ok], scores_fit[ok]
            order = np.argsort(-scores_fit, kind="stable"); pop, scores_fit = pop[order][:P], scores_fit[order][:P]
            dual = Dual(NE); opt = torch.optim.Adam(dual.parameters(), lr=GEN_LR)
            it0 = 1
            if not STOP["flag"]:
                save_state(dict(next=(rnd, 1), plan=plan, basis=basis, HIST=HIST, pop=pop, scores_fit=scores_fit, hist_pop=hist_pop,
                                hist_fit=hist_fit, dual=dual.state_dict(), opt=opt.state_dict()))
        for it in range(it0, plan["max_iter"] + 1):
            if STOP["flag"] or len(pop) == 0: break
            last_round = rnd == plan["rounds"]
            reserve = 0 if last_round else (P - NE) + 2 * P
            aff = evals_affordable()
            if aff - reserve < (P // 2 if last_round else P):
                log(f"budget: ending round {rnd} before iter {it} (affordable {aff}, reserve {reserve})"); break
            hm = ~np.isnan(hist_fit); hp, hf = hist_pop[hm], hist_fit[hm]
            idx = np.argsort(-hf, kind="stable"); nw = max(1, int(len(hp) * WIN_PORTION))
            winners, losers = hp[idx[:nw]], hp[idx[nw:]]
            for ep in range(GEN_EPOCHS): loss = train_epoch(dual, opt, winners, losers)
            with torch.no_grad(): new = dual.lose2win(torch.tensor(pop, dtype=torch.float32)).numpy()
            new = np.clip(new, -1, 1).astype(np.float64)
            log(f"round {rnd} iter {it}: generator loss {loss:.4f}, winners {nw}/{len(hp)}")
            new_fit = run_eval(new, rnd, it, ["generated"] * len(new))
            hist_pop = np.vstack([hist_pop, new]); hist_fit = np.concatenate([hist_fit, new_fit])
            ok = ~np.isnan(new_fit)
            allp = np.vstack([pop, new[ok]]); allf = np.concatenate([scores_fit, new_fit[ok]])
            order = np.argsort(-allf, kind="stable")[:P]; pop, scores_fit = allp[order], allf[order]
            HIST.append(dict(round=rnd, iter=it, best=float(scores_fit[0]), pop_mean=float(scores_fit.mean()),
                             new_mean=float(np.nanmean(new_fit)) if ok.any() else None, new_best=float(np.nanmax(new_fit)) if ok.any() else None,
                             n_new=int(ok.sum()), gen_loss=loss, elapsed_h=(time.time() - T00) / 3600))
            json.dump(dict(plan=plan, hist=HIST), open(f"{W}/history.json", "w"), indent=1)
            log("HIST", HIST[-1])
            if not STOP["flag"]:
                save_state(dict(next=(rnd, it + 1), plan=plan, basis=basis, HIST=HIST, pop=pop, scores_fit=scores_fit, hist_pop=hist_pop,
                                hist_fit=hist_fit, dual=dual.state_dict(), opt=opt.state_dict()))
        if STOP["flag"]: break
        if rnd < plan["rounds"]:
            hm = ~np.isnan(hist_fit); hp, hf = hist_pop[hm], hist_fit[hm]
            top = hp[np.argsort(-hf, kind="stable")[:NE]]
            basis = (top.astype(np.float32) @ basis).astype(np.float32)
            np.save(f"{W}/basis_round{rnd+1}.npy", basis)
            log("basis shift ->", basis.round(3).tolist())
            save_state(dict(next=(rnd + 1, 0), plan=plan, basis=basis, HIST=HIST))
    recs = [r for r in cache.values() if not r.get("failed")]
    best = max(recs, key=lambda r: (r["fitness"], -recs.index(r)))
    json.dump(dict(plan=plan, n_evaluated=len(cache), n_failed=sum(1 for r in cache.values() if r.get("failed")), best=best, hist=HIST,
                   stopped_early=STOP["flag"], selection_rule="max fitness (mean of 3 task accuracies) over all evaluated", hours=(time.time() - T00) / 3600),
              open(f"{W}/evo_result.json", "w"), indent=1)
    log("BEST", best["fitness"], best["lams"], best["acc"])
status(stage="final_start", best=dict(fit=best["fitness"], lams=best["lams"]))

# ---------------------------------------------------------------- Final: winner on the 180-item test set (8192 cap)
TEST = {json.loads(l)["id"]: json.loads(l) for l in open(f"{DATA}/test_items.jsonl")}
ids = json.load(open(f"{DATA}/subset_ids.json"))
if MODE == "smoke":   # mini final: 2 items per bench, exercising the restart path on GPU1
    ids = [i for b in ("mmlupro", "gpqa", "bfcl") for i in [x for x in ids if TEST[x]["bench"] == b][:2]]
halves = [ids[0::2], ids[1::2]]
fres = {}
def final_lane(g, w, restart_first=False):
    items = [TEST[i] for i in halves[g]]
    items = [i for i in items if i["bench"] == "gpqa"] + [i for i in items if i["bench"] == "mmlupro"] + [i for i in items if i["bench"] == "bfcl"]
    json.dump(items, open(f"/tmp/test_{g}.json", "w"))
    for attempt in range(3):
        try:
            if restart_first or w.dead or not w.alive():
                restart_first = False; w.restart()
            w.call(T_LOAD, op="load_merge", tvs=TV, lams=best["lams"])
            t0 = time.time(); r = w.call(T_GEN_FINAL, op="gen", items=f"/tmp/test_{g}.json", max_tokens=8192, out=f"{W}/raw_evogm_{g}.json")
            fres[g] = dict(secs=time.time() - t0, gen_tokens=r["gen_tokens"], gpu=w.g); log("final half", g, "done on gpu", w.g, fres[g]); return
        except Exception as e:
            log("FINAL ERROR half", g, "gpu", w.g, "attempt", attempt, repr(e)[:1500])
            try: w.kill()
            except Exception: pass
            w.dead = True
            if attempt == 2: return
ths = [threading.Thread(target=final_lane, args=(g, workers[g], MODE == "smoke" and g == 1)) for g in (0, 1)]
[t.start() for t in ths]; [t.join() for t in ths]
for g in (0, 1):   # if one half failed, run it on the other GPU
    if g not in fres and (1 - g) in fres:
        log("running half", g, "on gpu", 1 - g); final_lane(g, workers[1 - g])
with open(f"{W}/raw_evogm.jsonl", "w") as f:
    for g in (0, 1):
        p = f"{W}/raw_evogm_{g}.json"
        if not os.path.exists(p): continue
        for o in json.load(open(p)):
            it = TEST[o["id"]]
            f.write(json.dumps(dict(id=o["id"], bench=it["bench"], category=it["category"], model="evogm", prompt_tokens=o["prompt_tokens"],
                                    completion_tokens=o["completion_tokens"], finish_reason=o["finish_reason"], output=o["output"])) + "\n")
n = sum(1 for _ in open(f"{W}/raw_evogm.jsonl"))
quick = {}
for l in open(f"{W}/raw_evogm.jsonl"):
    o = json.loads(l); s = safe_score(TEST[o["id"]], o["output"], o["finish_reason"])
    quick.setdefault(o["bench"], []).append(bool(s["correct"]))
log("FINAL raw lines", n, {k: (sum(v), len(v)) for k, v in quick.items()}, fres)
for w in workers: w.kill()
status(stage="done", final=dict(n=n, quick={k: [sum(v), len(v)] for k, v in quick.items()}, fres=fres), hours=(time.time() - T00) / 3600)
log("ALL DONE hours", (time.time() - T00) / 3600)
