# EvoGM v2 (2026-11-15) for Baphomet: evolutionary linear merge of the 8 Baphomet donors on Qwen3.5-4B, Kaggle T4x2.
#   lambda = 16 dims = per donor x {attention group, MLP group}: lams16 = [attn_0..attn_7, mlp_0..mlp_7] (donor order =
#   linmerge.DONORS). MLP group = tensors whose key has a ".mlp." component (layers.*.mlp.{gate,up,down}_proj, mtp mlp, vision
#   mlp); everything else (self_attn, linear_attn, embeddings/tied lm_head, norms, mtp fc/norms/attn) = attention group.
#   Search set 200 (MMLU-Pro 110 / GPQA 70 / BFCL 20 mid-difficulty), validation 150 (75/45/30), test 180 only for the winner.
#   Fitness = mean of per-task accuracy on the search set (chat template, thinking, temp .6 top_p .95 top_k 20 seed 0,
#   max_tokens 4096, truncated = wrong).
#   Round 1: the initial population (10, see INIT) is evaluated on all 200 items -> per-task 2PL IRT fit in-kernel ->
#   ~60-item Fisher subset whose task split is chosen by leave-one-out on the round-1 data. Generator iterations (EvoGM dual
#   MLP generators, as v1) use the p-IRT subset estimate + racing early-cut + completion guard (full 200) exactly as evogm-r2;
#   basis shift between rounds (16 experts).
#   Selection: top candidates by full search fitness (+ each round's best + Baphomet-1.1) are evaluated on the validation set;
#   a candidate whose validation BFCL accuracy is below Qwen3.5-4B base's (measured here, once) is not eligible. Winner = best
#   validation fitness among the eligible; final 180-item test @8192 only for the winner (skipped if it is Baphomet-1.1).
# Setup / engines / worker protocol / checkpointing = hardened evogm v3 + r2 (see evogm/NOTES.md).
import os, sys, json, time, glob, gzip, shutil, subprocess, threading, queue, hashlib, math, traceback, base64
from concurrent.futures import ThreadPoolExecutor
T00 = time.time()
MOCK = json.loads(os.environ.get("EVOGM2_MOCK", "{}"))   # mock harness only (path prefix, time scale, fault injection); unset on Kaggle
KR = MOCK.get("root", "")                               # path prefix ("" on Kaggle)
TS = float(MOCK.get("ts", 1.0))                         # time scale of priors/margins (1.0 on Kaggle)
MODE = "full"
HARD_H = float(MOCK.get("hard_h", 10.5))                 # whole-kernel ceiling for this session
FIT_MAXTOK = 4096
GPUMEM = [0.84, 0.78]; MAXSEQS = 48                     # eager engines
DISK_CAP_GB = 70.0
T_LOAD, T_GEN_FULL, T_GEN_SUB, T_GEN_FINAL, T_BOOT = MOCK.get("timeouts", (1500, 5400, 3600, 4 * 3600, 1800))
MAX_BOOTS = 6
W = f"{KR}/kaggle/working"; TMP = f"{KR}/tmp"; INP = f"{KR}/kaggle/input"
def find_all(pat): return sorted(glob.glob(pat, recursive=True))
def find(pat):
    g = find_all(pat); return g[0] if g else None
DATA = os.path.dirname(find(f"{INP}/**/linmerge.py"))
if os.path.exists(f"{DATA}/bfcl_eval.tar.gz"):
    os.makedirs(f"{TMP}/pylib", exist_ok=True); subprocess.run(f"tar -xzf {DATA}/bfcl_eval.tar.gz -C {TMP}/pylib", shell=True, check=True); PYLIB = f"{TMP}/pylib"
else:
    PYLIB = f"{DATA}/bfcl_eval"
assert os.path.exists(f"{PYLIB}/bfcl_eval/constants/enums.py"), PYLIB
sys.path.insert(0, DATA); sys.path.insert(1, PYLIB)
os.environ["PYTHONPATH"] = f"{DATA}:{PYLIB}:" + os.environ.get("PYTHONPATH", "")
os.makedirs(W, exist_ok=True)
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
def mock_kill(tag):   # fault injection for the mock harness: hard exit at a named point (simulates a killed session)
    if MOCK.get("kill_at") == tag:
        log("MOCK KILL at", tag); os._exit(9)
log("EvoGM v2 MODE", MODE, "HARD_H", HARD_H, "mock" if MOCK else "")
if not MOCK: sh("nvidia-smi -L; df -h /tmp /kaggle/working; free -g; nproc", t=60)
import linmerge
linmerge.log = log
def disk_used_gb():
    tot = 0
    for d in (f"{TMP}/m", f"{TMP}/tv"):
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
if not MOCK: threading.Thread(target=_monitor, daemon=True).start()
STATUS = dict(stage="start", mode=MODE)
def status(**kw):
    STATUS.update(kw); STATUS["elapsed_h"] = round((time.time() - T00) / 3600, 3)
    try:
        json.dump(STATUS, open(f"{W}/status.tmp", "w"), indent=1, default=str); os.replace(f"{W}/status.tmp", f"{W}/status.json")
    except Exception as e: log("status write err", repr(e))
status()

# ---------------------------------------------------------------- setup: pip || downloads -> fp8 TV extraction (bounded disk)
DIRS = {"base": f"{TMP}/m/base", **{s: f"{TMP}/m/{s}" for s in linmerge.SHORT}}
REPOS = {"base": linmerge.BASE_ID, **{d[0]: d[1] for d in linmerge.DONORS}}
REMAP = {d[0]: d[2] for d in linmerge.DONORS}
TVDIR = f"{TMP}/tv"; os.makedirs(TVDIR, exist_ok=True)
TV = [f"{TVDIR}/{s}.safetensors" for s in linmerge.SHORT]
log("models+tv disk at start", disk_used_gb())
pip_ok = threading.Event(); pip_res = {}
def pip_thread():
    t = time.time()
    if MOCK:
        pip_res.update(versions="0.33.0 mock", secs=0); pip_ok.set(); return
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
            time.sleep(30 * TS)
def disk_guard(need_gb, tag):
    t = time.time()
    while disk_used_gb() + need_gb > DISK_CAP_GB:
        if time.time() - t > 2400: raise RuntimeError(f"disk guard timeout before {tag}: {disk_used_gb()} GB used")
        log("disk guard: waiting before", tag, disk_used_gb(), "GB used"); time.sleep(20)

dl_done = {}; dl_err = {}; slots = threading.Semaphore(3); ready_q = queue.Queue()   # <=3 donors on disk
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
open(f"{TMP}/ext1.py", "w").write(
    "import sys, json\n"
    "def main():\n"
    "    sys.path.insert(0, sys.argv[1]); import linmerge\n"
    "    st = linmerge.extract_tv(sys.argv[2], sys.argv[3], sys.argv[4] == '1', sys.argv[5], device='cuda')\n"
    "    json.dump(st, open(sys.argv[5] + '.json', 'w'))\n"
    "if __name__ == '__main__':\n"
    "    main()\n")
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
            time.sleep(3 * min(1.0, TS * 50)); continue
        k, ok = item
        if ok:
            for a in range(2):
                t = time.time()
                try:
                    r = subprocess.run(f"CUDA_VISIBLE_DEVICES={g} {sys.executable if MOCK else 'python'} {TMP}/ext1.py {DATA} {DIRS['base']} {DIRS[k]} {int(REMAP[k])} {TVDIR}/{k}.safetensors",
                                       shell=True, capture_output=True, text=True, timeout=1800)
                    rc, err = r.returncode, r.stderr[-1500:]
                except Exception as e: rc, err = -1, repr(e)
                if rc == 0:
                    st = json.load(open(f"{TVDIR}/{k}.safetensors.json")); st["secs"] = round(time.time() - t)
                    with ext_lock: tv_stats[k] = st; ext_fail.pop(k, None)
                    log("tv", k, {x: st.get(x) for x in ("n", "zero", "missing", "rel_qerr", "secs")}); break
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

# ---------------------------------------------------------------- workers (persistent vLLM engines, file protocol)
WORKER_B64 = "__WORKER_B64__"
WORKER_PY = f"{TMP}/gpuworker4.py"
if MOCK.get("worker"): shutil.copy(MOCK["worker"], WORKER_PY)
else: open(WORKER_PY, "wb").write(base64.b64decode(WORKER_B64))
class WorkerError(Exception): pass
def gpu_mem_used(g):
    if MOCK: return 0
    try:
        r = subprocess.run(f"timeout 20 nvidia-smi -i {g} --query-gpu=memory.used --format=csv,noheader,nounits", shell=True, capture_output=True, text=True, timeout=30)
        return int(r.stdout.strip().split()[0])
    except Exception: return -1
class Worker:
    def __init__(self, g):
        self.g = g; self.boots = 0; self.fails = 0; self.p = None; self.dead = False; self.gpumem = None
    def alive(self): return self.p is not None and self.p.poll() is None
    def start(self, gpumem):
        self.n = 0; self.gpumem = gpumem
        self.dir = f"{TMP}/w{self.g}_{self.boots}"; shutil.rmtree(self.dir, ignore_errors=True); os.makedirs(self.dir)
        self.logf = f"{W}/worker_{self.g}_{self.boots}.log"
        py = sys.executable if MOCK else "python"
        self.p = subprocess.Popen(f"CUDA_VISIBLE_DEVICES={self.g} exec {py} {WORKER_PY} {DIRS['base']} {self.dir} {gpumem} 1 {MAXSEQS} "
                                  f"> {self.logf} 2>&1", shell=True)
        self.boots += 1
    def wait_ready(self):
        t = time.time()
        while not os.path.exists(f"{self.dir}/ready.json"):
            if self.p.poll() is not None: raise WorkerError(f"worker {self.g} died during init (boot {self.boots})")
            if time.time() - t > T_BOOT: raise WorkerError(f"worker {self.g} init timeout")
            time.sleep(2 * min(1.0, TS * 50))
        r = json.load(open(f"{self.dir}/ready.json")); log("worker", self.g, "ready", r); return r
    def call(self, timeout, poll=None, **cmd):
        if not self.alive(): raise WorkerError(f"worker {self.g} not running")
        n = self.n; self.n += 1; t = time.time(); tp = time.time()
        json.dump(cmd, open(f"{self.dir}/cmd_{n}.tmp", "w")); os.rename(f"{self.dir}/cmd_{n}.tmp", f"{self.dir}/cmd_{n}.json")
        while not os.path.exists(f"{self.dir}/res_{n}.json"):
            if poll is not None and time.time() - tp > 10 * min(1.0, TS * 5):
                tp = time.time()
                try: poll()
                except Exception as e: log("poll error", repr(e)[:300])
            if self.p.poll() is not None: raise WorkerError(f"worker {self.g} died during {cmd.get('op')}; see {self.logf}")
            if time.time() - t > timeout: raise WorkerError(f"worker {self.g} {cmd.get('op')} timeout {timeout}s")
            time.sleep(0.5 * min(1.0, TS * 50))
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
        if self.p is not None: subprocess.run(f"pkill -9 -f '[g]puworker4.py \\S+ {self.dir} '", shell=True)
        t = time.time(); m = gpu_mem_used(self.g)
        while m > 700 and time.time() - t < 180:
            time.sleep(5); m = gpu_mem_used(self.g)
        if m > 700:
            sh(f"nvidia-smi -i {self.g} --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9; sleep 10")
            m = gpu_mem_used(self.g)
        log("worker", self.g, "killed; GPU mem used now", m, "MiB")
    def boot(self):
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
time.sleep(120 * TS)   # stagger the two engine boots (host RAM)
boot_bg(1); tb0.join()
if not boot_res:
    status(stage="boot_failed"); raise SystemExit("no worker booted")
status(stage="workers_ready", boot=boot_res)

# ---------------------------------------------------------------- items / scoring
import numpy as np, torch
torch.set_num_threads(1)   # deterministic CPU generator training / IRT fits (replayable)
import fitscore
TASKS = ["mmlupro", "gpqa", "bfcl"]
SEARCH = json.load(open(f"{DATA}/search_items.json")); VALI = json.load(open(f"{DATA}/val_items.json"))
FITBY = {i["id"]: i for i in SEARCH}; VALBY = {i["id"]: i for i in VALI}
assert len(FITBY) == len(SEARCH) and len(VALBY) == len(VALI) and not (set(FITBY) & set(VALBY))
NT = {t: sum(1 for i in SEARCH if i["bench"] == t) for t in TASKS}
NV = {t: sum(1 for i in VALI if i["bench"] == t) for t in TASKS}
SIDS = {t: [i["id"] for i in SEARCH if i["bench"] == t] for t in TASKS}          # IRT item order per task
IPOS = {t: {iid: j for j, iid in enumerate(SIDS[t])} for t in TASKS}
log("search", NT, "val", NV)
def ordered(items):   # longest first (GPQA, MMLU-Pro, BFCL)
    return [i for i in items if i["bench"] == "gpqa"] + [i for i in items if i["bench"] == "mmlupro"] + [i for i in items if i["bench"] == "bfcl"]
json.dump(ordered(SEARCH), open(f"{TMP}/fit_full.json", "w")); json.dump(ordered(VALI), open(f"{TMP}/val.json", "w"))
N8 = linmerge.N; NE = 2 * N8                     # 16 coordinates: [attn x 8 donors, mlp x 8 donors]
B11 = [0.0116, 0.2039, 0.5901, 0.0013, 0.2972, 0.0323, 0.0030, -0.0007]   # Baphomet-1.1 (evogm-r2 winner), 8-dim
B10 = [0.1375] * N8                                                        # Baphomet-1.0 effective lambda
def ex(a, m): return [float(v) for v in a] + [float(v) for v in m]
Z8 = [0.0] * N8
_rs = np.random.RandomState(20261115)
INIT = [   # round-1 population in priority order (the budget may cut the tail; >= MIN_CAL rows needed for the IRT fit)
    ("baphomet11", ex(B11, B11)),                                   # incumbent
    ("baphomet10", ex(B10, B10)),                                   # uniform 0.1375
    ("b11attn_b10mlp", ex(B11, B10)),                               # group crossovers of the two known-good merges
    ("b10attn_b11mlp", ex(B10, B11)),
    ("b11_attn125_mlp075", ex(1.25 * np.array(B11), 0.75 * np.array(B11))),   # group re-weighting of 1.1
    ("b11_attn075_mlp125", ex(0.75 * np.array(B11), 1.25 * np.array(B11))),
    ("onehot_q38distill", ex(np.eye(N8)[2], np.eye(N8)[2])),        # best single donor (v1 fit .686); cheap (short outputs)
    ("b11_attn_only", ex(B11, Z8)),                                 # group ablations: low-ability anchors for the IRT + which group carries the gain
    ("b11_mlp_only", ex(Z8, B11)),
    ("b11_local1", list(np.clip(np.array(ex(B11, B11)) + _rs.normal(0, 0.1, NE), -1, 1))),   # local random around 1.1
]   # (the reference's "average" and U(-1,1) randoms are left out: v1 average .592 / randoms are long-output, expensive full-200 rows)
MIN_CAL = 8
B11_KEY = None   # set below (lkey of ex(B11,B11))
BASE16 = [0.0] * NE
def safe_score(it, out, fr):
    try: return fitscore.score(it, out, fr)
    except Exception as e: return dict(correct=False, truncated=fr == "length", error=repr(e)[:200])
def ok_item(s): return bool(s["correct"]) and not s["truncated"]
def task_stats(outs, BY):
    trunc, tok = {}, {}
    for t in TASKS:
        done = [o for o in outs if BY[o["id"]]["bench"] == t and o["finish_reason"] != "aborted"]
        trunc[t] = round(sum(o["finish_reason"] == "length" for o in done) / max(1, len(done)), 4)
        tok[t] = round(sum(o["completion_tokens"] for o in done) / max(1, len(done)), 1)
    return trunc, tok
def grp(l): return dict(attn_sum=round(float(np.sum(l[:N8])), 4), mlp_sum=round(float(np.sum(l[N8:])), 4))

# ---------------------------------------------------------------- caches (resume: every evo_log / val_log in the inputs)
for d in ("fit_outputs", "val_outputs"): os.makedirs(f"{W}/{d}", exist_ok=True)
LOG = f"{W}/evo_log.jsonl"; VLOG = f"{W}/val_log.jsonl"
def lkey(x): return ",".join(f"{float(v):.6f}" for v in x)
B11_KEY = lkey(ex(B11, B11)); BASE_KEY = lkey(BASE16)
cache = {}; vcache = {}
def _read(pat, dst):
    for prev in find_all(pat):
        n0 = len(dst)
        for l in open(prev):
            try: r = json.loads(l)
            except Exception: continue
            if r["lkey"] in dst: dst[r["lkey"]].update(r)
            else: dst[r["lkey"]] = r
        log("resume: read", prev, len(dst) - n0, "new records")
_read(f"{INP}/**/evo_log.jsonl", cache); _read(f"{INP}/**/val_log.jsonl", vcache)
for sub in ("fit_outputs", "val_outputs"):
    for p in find_all(f"{INP}/**/{sub}/*.json.gz"):
        d = f"{W}/{sub}/{os.path.basename(p)}"
        if not os.path.exists(d): shutil.copy(p, d)
with open(LOG, "w") as f:
    for r in cache.values(): f.write(json.dumps(r) + "\n")
with open(VLOG, "w") as f:
    for r in vcache.values(): f.write(json.dumps(r) + "\n")
log("cached evaluations", len(cache), "val records", len(vcache))
def value(r):
    if r is None: return np.nan
    if r.get("failed"): return 0.0
    return r["full_fitness"] if r.get("full_fitness") is not None else r["fitness"]
log_lock = threading.Lock()
def append(path, r):
    with open(path, "a") as f: f.write(json.dumps(r) + "\n"); f.flush(); os.fsync(f.fileno())
FULL_T = []; SUB_T = []; COMP_T = []; VAL_T = []
SESSION = {"n": 0, "fail": 0}
RACE = {"on": True, "up": 1.5, "min_frac": 0.5}

# ---------------------------------------------------------------- IRT (per-task 2PL, fit in-kernel on the round-1 full rows)
IRT = {}
GRID = np.linspace(-5, 5, 2001)
def sig(x): return 1.0 / (1.0 + np.exp(-x))
def fit2pl(Y, iters=600):
    """joint MAP: theta~N(0,1), log a~N(0,.5^2), b~N(0,2^2) (as evogm irt.py)."""
    Y = torch.tensor(Y, dtype=torch.float64); n, m = Y.shape
    th = torch.zeros(n, requires_grad=True, dtype=torch.float64); la = torch.zeros(m, requires_grad=True, dtype=torch.float64)
    b = torch.zeros(m, requires_grad=True, dtype=torch.float64)
    opt = torch.optim.LBFGS([th, la, b], max_iter=iters, line_search_fn="strong_wolfe")
    def cl():
        opt.zero_grad(); z = torch.exp(la)[None] * (th[:, None] - b[None])
        nll = torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction="sum")
        l = nll + 0.5 * (th ** 2).sum() + 0.5 * (la ** 2).sum() / 0.5 ** 2 + 0.5 * (b ** 2).sum() / 2.0 ** 2
        l.backward(); return l
    opt.step(cl)
    return th.detach().numpy(), np.exp(la.detach().numpy()), b.detach().numpy()
def theta_map(y, a, b):
    if len(y) == 0: return 0.0, 1.0
    z = a[None] * (GRID[:, None] - b[None])
    ll = (y[None] * -np.logaddexp(0, -z) + (1 - y[None]) * -np.logaddexp(0, z)).sum(1) - 0.5 * GRID ** 2
    w = np.exp(ll - ll.max()); w /= w.sum(); mu = float((w * GRID).sum())
    return float(GRID[int(np.argmax(ll))]), float(np.sqrt((w * (GRID - mu) ** 2).sum()))
def est_from(obs, a_t, b_t, up=0.0):
    """p-IRT estimate of the full-search fitness: per task (observed correct + 2PL p for the unobserved items)/n_task, mean."""
    det = {}; ests = []
    for t in TASKS:
        o = [(IPOS[t][i], v) for i, v in obs.items() if i in IPOS[t]]
        js = np.array([j for j, _ in o], dtype=int); y = np.array([v for _, v in o], dtype=float)
        th, sd = theta_map(y, a_t[t][js], b_t[t][js]); thu = th + up * sd
        rest = np.setdiff1d(np.arange(NT[t]), js)
        e = (y.sum() + sig(a_t[t][rest] * (thu - b_t[t][rest])).sum()) / NT[t]
        det[t] = dict(est=round(float(e), 4), theta=round(th, 3), sd=round(sd, 3), n_obs=int(len(js)), obs_acc=round(float(y.mean()), 4) if len(y) else None)
        ests.append(e)
    return float(np.mean(ests)), det
def irt_fitness(obs, up=0.0): return est_from(obs, IRT["a"], IRT["b"], up)
SPLITS = [(30, 20, 10), (32, 22, 6), (28, 24, 8), (34, 18, 8), (26, 26, 8), (36, 16, 8), (30, 26, 4), (24, 24, 12), (28, 20, 12)]
ALPHA = 0.5   # Fisher information / mean_tokens^ALPHA (cost weighting, as r2)
def full_rows():
    rows = []
    for r in cache.values():
        if r.get("full_fitness") is None or r.get("failed"): continue
        d = json.load(gzip.open(f"{W}/fit_outputs/{r['tag']}_full.json.gz", "rt"))
        sc = d["scores"]; tk = {o["id"]: o["completion_tokens"] for o in d["outputs"]}
        rows.append((r, {t: np.array([int(bool(sc[i]["correct"]) and not sc[i]["truncated"]) for i in SIDS[t]]) for t in TASKS},
                     {t: np.array([tk[i] for i in SIDS[t]], dtype=float) for t in TASKS}))
    rows.sort(key=lambda x: x[0]["lkey"])   # deterministic order
    return rows
def irt_stage(prev=None, out="irt_v2.json", rnd=1):
    """per-task 2PL fit + LOO split choice + Fisher subset on ALL full-200 rows in the cache (round 1: the init rows; at a round
    boundary: round-1 rows + completion rows). prev = the fit in use (refit: its time priors are kept, since completion rows'
    t_gen is a subset-run time, not a full-run time)."""
    t0 = time.time()
    rows = full_rows(); n = len(rows)
    F = np.array([r["full_fitness"] for r, _, _ in rows])
    C = {t: np.stack([c[t] for _, c, _ in rows]).astype(float) for t in TASKS}
    Tk = {t: np.stack([k[t] for _, _, k in rows]) for t in TASKS}
    fits = {t: fit2pl(C[t]) for t in TASKS}
    a = {t: fits[t][1] for t in TASKS}; b = {t: fits[t][2] for t in TASKS}
    top = np.argsort(-F, kind="stable")[:max(3, n // 2)]
    cost = {t: Tk[t].mean(0) + 1.0 for t in TASKS}
    def select(K):
        out = {}
        for t, k in zip(TASKS, K):
            th = fits[t][0][top]; P = sig(a[t][None] * (th[:, None] - b[t][None])); I = (a[t] ** 2 * P * (1 - P)).mean(0)
            out[t] = np.sort(np.argsort(-(I / np.power(cost[t], ALPHA)), kind="stable")[:k])
        return out
    # leave-one-out item parameters (independent of the split) -> estimate each held-out candidate from a subset
    loo = []
    for c in range(n):
        m = np.arange(n) != c; ft = {t: fit2pl(C[t][m], iters=200) for t in TASKS}
        loo.append(({t: ft[t][1] for t in TASKS}, {t: ft[t][2] for t in TASKS}))
    table = []
    for K in SPLITS:
        sel = select(K); est = []
        for c in range(n):
            obs = {SIDS[t][j]: int(C[t][c, j]) for t in TASKS for j in sel[t]}
            est.append(est_from(obs, loo[c][0], loo[c][1])[0])
        est = np.array(est); err = np.abs(est - F)
        naive = np.mean([C[t][:, sel[t]].mean(1) for t in TASKS], 0)
        rho = float(np.corrcoef(np.argsort(np.argsort(est)), np.argsort(np.argsort(F)))[0, 1]) if n > 2 else float("nan")
        tok = float(np.mean([sum(Tk[t][c, sel[t]].sum() for t in TASKS) for c in range(n)]))
        table.append(dict(split=K, mae=round(float(err.mean()), 4), mae_top=round(float(err[top].mean()), 4), max_err=round(float(err.max()), 4),
                          rank_corr=round(rho, 3), naive_mae=round(float(np.abs(naive - F).mean()), 4), mean_tokens=round(tok),
                          score=round(float(0.5 * err.mean() + 0.5 * err[top].mean()), 5)))
        log("IRT LOO", table[-1])
    best = min(table, key=lambda d: (d["score"], d["mean_tokens"]))
    sel = select(best["split"])
    tot_tok = np.array([sum(Tk[t][c].sum() for t in TASKS) for c in range(n)])
    sub_tok = np.array([sum(Tk[t][c, sel[t]].sum() for t in TASKS) for c in range(n)])
    tg = np.array([r["t_gen"] for r, _, _ in rows]); tsw = np.array([r.get("t_swap", 0.0) for r, _, _ in rows])
    frac = sub_tok / np.maximum(tot_tok, 1)
    res = dict(n_cal=n, cal_lkeys=[r["lkey"] for r, _, _ in rows], a={t: a[t].tolist() for t in TASKS}, b={t: b[t].tolist() for t in TASKS},
               theta={t: fits[t][0].tolist() for t in TASKS}, loo_table=table, chosen=best, alpha=ALPHA,
               subset={t: [int(j) for j in sel[t]] for t in TASKS}, subset_ids=[SIDS[t][j] for t in TASKS for j in sel[t]],
               const_items={t: dict(all_right=int((C[t].min(0) == 1).sum()), all_wrong=int((C[t].max(0) == 0).sum())) for t in TASKS},
               # time priors from the top-half rows (generated candidates resemble the good, short-output merges, not the verbose ablations)
               t_sub_prior=float(np.mean((tg * frac * 1.15 + tsw)[top])), t_comp_prior=float(np.mean((tg * (1 - frac) * 1.05 + tsw)[top])),
               secs=round(time.time() - t0, 1), fit_round=rnd)
    if prev is not None:
        res["t_sub_prior"], res["t_comp_prior"] = prev["t_sub_prior"], prev["t_comp_prior"]
        res["prev_n_cal"] = prev.get("n_cal"); res["prev_subset_ids"] = prev.get("subset_ids")
        res["refits"] = list(prev.get("refits") or []) + [dict(round=rnd, n_cal=n, chosen=best, subset_overlap=len(set(res["subset_ids"]) & set(prev.get("subset_ids") or [])))]
    for t in TASKS:
        assert np.all(np.isfinite(a[t])) and np.all(np.isfinite(b[t])) and np.all(np.isfinite(fits[t][0])), f"non-finite 2PL fit ({t})"
    log("IRT chosen split", best, "subset", len(res["subset_ids"]), "const", res["const_items"], "t_sub_prior", round(res["t_sub_prior"]), "t_comp_prior", round(res["t_comp_prior"]), "fit secs", res["secs"])
    json.dump(res, open(f"{W}/{out}", "w"), indent=0)
    return res
SUB_ITEMS = []; IRT_RES = {}
def set_irt(res):
    IRT_RES.clear(); IRT_RES.update(res)
    IRT.clear(); IRT.update(res); IRT["a"] = {t: np.array(res["a"][t]) for t in TASKS}; IRT["b"] = {t: np.array(res["b"][t]) for t in TASKS}
    SUB_ITEMS[:] = ordered([FITBY[i] for i in res["subset_ids"]])
    json.dump(SUB_ITEMS, open(f"{TMP}/fit_sub.json", "w"))
def rescore_subset_records(tagv):
    """re-estimate every subset (IRT) record from its saved observed items with the current IRT fit, so all estimates are on one
    scale (racing thresholds, population ranking, basis shift). Deterministic and idempotent; updated records go to evo_log
    (on resume the later line wins)."""
    n = 0; dmax = 0.0
    for k in sorted(cache):
        r = cache[k]
        if r.get("fitness_kind") != "irt" or r.get("failed") or not r.get("tag"): continue
        d = json.load(gzip.open(f"{W}/fit_outputs/{r['tag']}.json.gz", "rt"))
        obs = {i: int(bool(s["correct"]) and not s["truncated"]) for i, s in d["scores"].items()}
        fit, det = irt_fitness(obs)
        if r.get("irt_version") == tagv and abs(fit - r["fitness"]) < 1e-12: continue
        upd = dict(lkey=k, fitness=fit, irt=det, irt_version=tagv)
        if "fitness_r1irt" not in r: upd["fitness_r1irt"] = r["fitness"]
        dmax = max(dmax, abs(fit - r["fitness"])); r.update(upd); append(LOG, r); n += 1
    log(f"rescored {n} subset records with IRT {tagv} (max |change| {dmax:.4f})")
def refit_irt(rnd):
    """round boundary: refit the 2PL on all full rows (round 1 + completions) and reselect the subset; on any failure keep the
    previous fit. Then rescore the subset records with the fit in use."""
    prev = dict(IRT_RES)
    if prev.get("fit_round") == rnd + 1 or prev.get("refit_failed_round") == rnd:
        log(f"IRT refit for the round {rnd} boundary already in use"); rescore_subset_records(IRT_RES.get("fit_round", 1)); return
    try:
        res = irt_stage(prev=prev, out=f"irt_v2_refit_r{rnd}.json", rnd=rnd + 1)
        if MOCK.get("refit_fail") == rnd: raise RuntimeError("mock refit failure")
        set_irt(res)
        log(f"IRT REFIT at round {rnd} boundary: {prev.get('n_cal')} -> {res['n_cal']} rows, split {prev['chosen']['split']} -> {res['chosen']['split']}, "
            f"subset overlap {res['refits'][-1]['subset_overlap']}/{len(res['subset_ids'])}")
    except Exception as e:
        log("IRT REFIT FAILED -> keeping the previous fit:", repr(e)[:1500]); traceback.print_exc()
        prev["refit_failed_round"] = rnd; prev.setdefault("refit_failures", []).append(dict(round=rnd, error=repr(e)[:500]))
        set_irt(prev)
    rescore_subset_records(IRT_RES.get("fit_round", 1))

# ---------------------------------------------------------------- evaluations
def _gen(w, items_path, timeout, out, **kw):
    return w.call(timeout, op="gen", items=items_path, max_tokens=FIT_MAXTOK, out=out, **kw)
def evaluate_full(lams, w, meta):
    """all 200 search items (round-1 definition of fitness)."""
    t0 = time.time()
    r = w.call(T_LOAD, op="load_merge", tvs=TV, lams=[float(v) for v in lams]); t_swap = r["t"]
    rg = _gen(w, f"{TMP}/fit_full.json", T_GEN_FULL, f"{TMP}/of_{w.g}.json")
    outs = json.load(open(f"{TMP}/of_{w.g}.json"))
    assert sorted(o["id"] for o in outs) == sorted(FITBY)
    sc = {o["id"]: safe_score(FITBY[o["id"]], o["output"], o["finish_reason"]) for o in outs}
    acc = {t: sum(1 for i in SIDS[t] if ok_item(sc[i])) / NT[t] for t in TASKS}
    trunc, tok = task_stats(outs, FITBY)
    tag = hashlib.md5(lkey(lams).encode()).hexdigest()[:10]
    with gzip.open(f"{W}/fit_outputs/{tag}_full.json.gz", "wt") as f:
        json.dump(dict(lams=list(map(float, lams)), kind="full", outputs=outs, scores={k: dict(correct=v["correct"], truncated=v["truncated"]) for k, v in sc.items()}), f)
    fit = float(np.mean([acc[t] for t in TASKS]))
    rec = dict(lkey=lkey(lams), lams=[float(v) for v in lams], tag=tag, fitness=fit, fitness_kind="full", full_fitness=fit, full_acc=acc,
               full_trunc=trunc, full_mean_tokens=tok, full_gen_tokens=rg["gen_tokens"], acc=acc, raced=False, failed=False, gpu=w.g,
               t_swap=round(t_swap, 1), t_gen=round(rg["secs"], 1), gen_tokens=rg["gen_tokens"], t_total=round(time.time() - t0, 1),
               elapsed_h=round((time.time() - T00) / 3600, 3), groups=grp(lams), **meta)
    FULL_T.append(rec["t_total"])
    return rec
def evaluate_sub(lams, w, meta, thr):
    """IRT-subset evaluation with racing: stop when the optimistic estimate (theta + 1.5 sd) < thr (as evogm-r2)."""
    t0 = time.time()
    r = w.call(T_LOAD, op="load_merge", tvs=TV, lams=[float(v) for v in lams]); t_swap = r["t"]
    tag = hashlib.md5(lkey(lams).encode()).hexdigest()[:10]
    race_on = RACE["on"] and thr is not None and np.isfinite(thr)
    partial = f"{TMP}/part_{w.g}.json"; stopf = f"{TMP}/stop_{w.g}"
    for p in (partial, stopf, partial + ".seen"):
        try: os.remove(p)
        except OSError: pass
    st = dict(stopped=False, ub=None, scored={})
    def poll():
        if not race_on: return
        try: P = json.load(open(partial))
        except Exception: return
        for o in (P if not st["stopped"] else []):
            if o["id"] not in st["scored"]: st["scored"][o["id"]] = int(ok_item(safe_score(FITBY[o["id"]], o["output"], o["finish_reason"])))
        if not st["stopped"] and len(st["scored"]) >= RACE["min_frac"] * len(SUB_ITEMS) and len(st["scored"]) < len(SUB_ITEMS):
            ub, _ = irt_fitness(st["scored"], up=RACE["up"])
            if ub < thr:
                open(stopf, "w").write("stop"); st["stopped"] = True; st["ub"] = ub
                log(f"RACE gpu{w.g}: stop after {len(st['scored'])}/{len(SUB_ITEMS)} items, optimistic {ub:.4f} < top-P threshold {thr:.4f}")
        with open(partial + ".seen.tmp", "w") as f: f.write(str(len(P)))   # ack (lets the mock worker wait for the decision; ignored by the real worker)
        os.replace(partial + ".seen.tmp", partial + ".seen")
    kw = dict(partial=partial, stop=stopf) if race_on else {}
    rg = w.call(T_GEN_SUB, poll=poll, op="gen", items=f"{TMP}/fit_sub.json", max_tokens=FIT_MAXTOK, out=f"{TMP}/o_{w.g}.json", **kw)
    if rg.get("race_error"):
        RACE["on"] = False; log("RACE path failed -> racing disabled, plain generate used:", rg["race_error"][:1500])
    outs = json.load(open(f"{TMP}/o_{w.g}.json"))
    sc = {o["id"]: safe_score(FITBY[o["id"]], o["output"], o["finish_reason"]) for o in outs if o["finish_reason"] != "aborted"}
    obs = {i: int(ok_item(s)) for i, s in sc.items()}
    fit, det = irt_fitness(obs)
    n_ab = sum(1 for o in outs if o["finish_reason"] == "aborted")
    trunc, tok = task_stats(outs, FITBY)
    with gzip.open(f"{W}/fit_outputs/{tag}.json.gz", "wt") as f:
        json.dump(dict(lams=list(map(float, lams)), kind="sub", outputs=outs, scores={k: dict(correct=v["correct"], truncated=v["truncated"]) for k, v in sc.items()}), f)
    rec = dict(lkey=lkey(lams), lams=[float(v) for v in lams], tag=tag, fitness=fit, fitness_kind="irt", irt=det,
               acc={t: det[t]["obs_acc"] for t in TASKS}, raced=n_ab > 0, n_aborted=n_ab, race_ub=st["ub"], race_thr=thr,
               full_fitness=None, trunc=trunc, mean_tokens=tok, failed=False, gpu=w.g, t_swap=round(t_swap, 1),
               t_gen=round(rg["secs"], 1), gen_tokens=rg["gen_tokens"], t_total=round(time.time() - t0, 1),
               elapsed_h=round((time.time() - T00) / 3600, 3), groups=grp(lams), **meta)
    if not rec["raced"]: SUB_T.append(rec["t_total"])
    return rec
def complete_full(rec, w):
    """Generate the search items not observed in the subset run -> exact 200-item fitness."""
    t0 = time.time()
    d = json.load(gzip.open(f"{W}/fit_outputs/{rec['tag']}.json.gz", "rt"))
    have = {o["id"]: o for o in d["outputs"] if o["finish_reason"] != "aborted"}
    todo = ordered([i for i in SEARCH if i["id"] not in have])
    json.dump(todo, open(f"{TMP}/comp_{w.g}.json", "w"))
    r = w.call(T_LOAD, op="load_merge", tvs=TV, lams=rec["lams"]); t_swap = r["t"]
    rg = _gen(w, f"{TMP}/comp_{w.g}.json", T_GEN_FULL, f"{TMP}/oc_{w.g}.json")
    outs = list(have.values()) + json.load(open(f"{TMP}/oc_{w.g}.json"))
    assert sorted(o["id"] for o in outs) == sorted(FITBY), "completion does not cover the search items"
    sc = {o["id"]: safe_score(FITBY[o["id"]], o["output"], o["finish_reason"]) for o in outs}
    acc = {t: sum(1 for i in SIDS[t] if ok_item(sc[i])) / NT[t] for t in TASKS}
    trunc, tok = task_stats(outs, FITBY)
    with gzip.open(f"{W}/fit_outputs/{rec['tag']}_full.json.gz", "wt") as f:
        json.dump(dict(lams=rec["lams"], kind="full", outputs=outs, scores={k: dict(correct=v["correct"], truncated=v["truncated"]) for k, v in sc.items()}), f)
    upd = dict(lkey=rec["lkey"], full_fitness=float(np.mean([acc[t] for t in TASKS])), full_acc=acc, full_trunc=trunc, full_mean_tokens=tok,
               full_gen_tokens=sum(o["completion_tokens"] for o in outs), t_complete=round(time.time() - t0, 1), t_complete_swap=round(t_swap, 1),
               completed_h=round((time.time() - T00) / 3600, 3))
    COMP_T.append(upd["t_complete"])
    return upd
def evaluate_val(lams, w, meta):
    """validation set (150 items), plain accuracy per task; val_fitness = mean of the 3 task accuracies."""
    t0 = time.time()
    r = w.call(T_LOAD, op="load_merge", tvs=TV, lams=[float(v) for v in lams]); t_swap = r["t"]
    rg = _gen(w, f"{TMP}/val.json", T_GEN_FULL, f"{TMP}/ov_{w.g}.json")
    outs = json.load(open(f"{TMP}/ov_{w.g}.json"))
    assert sorted(o["id"] for o in outs) == sorted(VALBY)
    sc = {o["id"]: safe_score(VALBY[o["id"]], o["output"], o["finish_reason"]) for o in outs}
    acc = {t: sum(1 for i in VALI if i["bench"] == t and ok_item(sc[i["id"]])) / NV[t] for t in TASKS}
    trunc, tok = task_stats(outs, VALBY)
    tag = hashlib.md5(lkey(lams).encode()).hexdigest()[:10]
    with gzip.open(f"{W}/val_outputs/val_{tag}.json.gz", "wt") as f:
        json.dump(dict(lams=list(map(float, lams)), kind="val", outputs=outs, scores={k: dict(correct=v["correct"], truncated=v["truncated"]) for k, v in sc.items()}), f)
    rec = dict(lkey=lkey(lams), lams=[float(v) for v in lams], tag=tag, val_fitness=float(np.mean([acc[t] for t in TASKS])), val_acc=acc,
               val_trunc=trunc, val_mean_tokens=tok, val_gen_tokens=rg["gen_tokens"], t_total=round(time.time() - t0, 1), t_swap=round(t_swap, 1),
               gpu=w.g, failed=False, elapsed_h=round((time.time() - T00) / 3600, 3), **meta)
    if any(float(v) != 0.0 for v in lams): VAL_T.append(rec["t_total"])   # base (verbose) excluded from the val time model
    return rec

# ---------------------------------------------------------------- budget
SAVE_MARGIN = 420 * TS                       # output saving / container overhead
T_FULL_PRIOR = 2400.0 * TS                   # 200 items @4096 + swap (simt time model on v1/r2 per-item tokens: 1.1-like ~1650 s gen, Bap-1.0-like ~2700 s, round-1 mix ~2300 s)
FINAL_TOKS_PER_S = 150.0                     # per GPU, eager fp16 T4 (r2 final: 165k tokens / 1110 s)
K_VAL = 4; VAL_MAX = 6                       # validation: top-K by full fitness (+ B11 + per-round bests), at most VAL_MAX
P = 10; MAX_ITER = 3; MAX_ROUND = int(MOCK.get("max_round", 4)); COMP_MARGIN = 0.02
def t_full(): return 1.1 * (float(np.mean(FULL_T[-6:])) if FULL_T else T_FULL_PRIOR)
def t_val():
    """validation eval time: measured (non-base) if any, else 0.72 x the full-200 cost of the current top-K (val candidates are the
    short-output good merges; the verbose base / ablations would make t_full() far too pessimistic)."""
    if VAL_T: return 1.1 * float(np.mean(VAL_T[-4:]))
    c = [r["t_total"] + (r.get("t_complete") or 0.0) for r in fulls()[:K_VAL] if r.get("t_total")]
    return 1.1 * 0.72 * (float(np.mean(c)) if c else T_FULL_PRIOR)
def t_sub(): return 1.1 * (float(np.mean(SUB_T[-12:])) if len(SUB_T) >= 2 else IRT.get("t_sub_prior", 0.33 * T_FULL_PRIOR))
def t_comp(): return 1.1 * (float(np.mean(COMP_T[-6:])) if COMP_T else IRT.get("t_comp_prior", 0.75 * T_FULL_PRIOR))
def fulls(): return sorted([r for r in cache.values() if r.get("full_fitness") is not None and not r.get("failed")], key=lambda r: (-r["full_fitness"], r["lkey"]))
def final_est():
    toks = [r.get("full_gen_tokens") or 0 for r in fulls()[:K_VAL]] or [0]
    tok = max(toks) if max(toks) > 0 else 330000
    return (1.3 * tok / 2 / FINAL_TOKS_PER_S + 450) * TS   # 180 items @8192 ~1.1-1.3x the 200-item search tokens (1.1: 330k vs 297k)
def n_alive(): return sum(1 for w in workers if not w.dead and w.alive())
def lanes(): return max(1, n_alive())
def val_reserve():
    n = (0 if BASE_KEY in vcache else 1) + K_VAL + 1
    return math.ceil(n / lanes()) * t_val()
def end_deadline(): return T00 + HARD_H * 3600 - SAVE_MARGIN - max(final_est(), 1900 * TS)
def evo_deadline(): return end_deadline() - val_reserve()
def remaining(): return max(0.0, evo_deadline() - time.time())
def iter_cost(P): return math.ceil(P / lanes()) * t_sub() + t_comp()
def round_cost(P, T): return math.ceil(1 / lanes()) * t_sub() + T * iter_cost(P)   # round >= 2 init: one new eval (average)

STOP = {"flag": False}
def run_jobs(jobs, do, on_fail, est, deadline=evo_deadline):
    """jobs: payloads, do(payload, w) -> record, on_fail(payload, errs, w) -> record or None; est(j) seconds for job j.
    Runs on the live GPUs; a failing job restarts its engine and is retried once."""
    if not jobs: return {}
    q = queue.Queue()
    for j in range(len(jobs)): q.put(j)
    out = {}
    def lane(w):
        while True:
            if STOP.get("abort"): return
            try: j = q.get_nowait()
            except queue.Empty: return
            if time.time() + est(j) > deadline():
                STOP["flag"] = True; log("time budget: skipping job", j, jobs[j] if isinstance(jobs[j], (str, int)) else ""); continue
            rec = None; errs = []
            for attempt in range(2):
                try:
                    rec = do(jobs[j], w); break
                except Exception as e:
                    errs.append(repr(e)[:1500]); log("JOB ERROR gpu", w.g, "attempt", attempt, repr(e)[:2000])
                    if RACE["on"]:
                        RACE["on"] = False; log("racing disabled after a job error (safety)")
                    try: w.restart()
                    except Exception as e2:
                        log("worker", w.g, "could not restart:", repr(e2)[:500], "-> lane stops, job requeued")
                        q.put(j); return
            if rec is None: rec = on_fail(jobs[j], errs, w)
            with log_lock:
                SESSION["n"] += 1; SESSION["fail"] += int(rec is None or rec.get("failed", False))
                if SESSION["n"] >= 4 and SESSION["fail"] == SESSION["n"]:
                    STOP["abort"] = True
                if rec is not None: out[j] = rec
    for w in workers:   # an engine that died between batches is brought back (bounded by MAX_BOOTS)
        if not w.alive() and w.boots < MAX_BOOTS and (time.time() + 900 * TS + est(0) < deadline()):
            log("worker", w.g, "not running at batch start -> restart")
            try: w.restart()
            except Exception as e: log("restart failed:", repr(e)[:300])
    live = [w for w in workers if not w.dead and w.alive()]
    if not live: STOP["flag"] = True; log("no live engine"); return out
    ths = [threading.Thread(target=lane, args=(w,)) for w in live]; [t.start() for t in ths]; [t.join() for t in ths]
    if STOP.get("abort"):
        status(stage="aborted_systematic_failures"); log("ABORT: first evaluations all failed"); [w.kill() for w in workers]; raise SystemExit("systematic eval failure")
    if not q.empty() and n_alive() > 0:
        ths = [threading.Thread(target=lane, args=(w,)) for w in workers if not w.dead and w.alive()]; [t.start() for t in ths]; [t.join() for t in ths]
    return out
def _fail_rec(lams, errs, w, meta):
    return dict(lkey=lkey(lams), lams=[float(v) for v in lams], tag=None, fitness=0.0, fitness_kind="failed", full_fitness=None, failed=True,
                errors=errs, gpu=w.g, elapsed_h=round((time.time() - T00) / 3600, 3), **meta)
def _store(rec, msg):
    with log_lock:
        cache[rec["lkey"]] = rec; append(LOG, rec); log(msg(rec))
        status(stage=STATUS.get("stage"), n_evals=len(cache))
def eval_batch(X, meta_list, thr=None):
    """subset (IRT) evaluations; effective-lambda vectors -> value array (full fitness if known, else IRT; nan = not evaluated)."""
    F = np.full(len(X), np.nan); todo = []
    for j in range(len(X)):
        k = lkey(X[j])
        if k in cache: F[j] = value(cache[k])
        elif not any(lkey(X[i]) == k for i in todo): todo.append(j)
    def do(j, w):
        rec = evaluate_sub(X[j], w, meta_list[j], thr)
        _store(rec, lambda r: f"[{r.get('round')}/{r.get('iter')}] n={len(cache)} gpu{w.g} irt_fit={r['fitness']:.4f} theta={ {t: r['irt'][t]['theta'] for t in TASKS} } "
                              f"raced={r['raced']}({r['n_aborted']}) {r['groups']} swap {r['t_swap']}s total {r['t_total']}s tok {r['gen_tokens']}")
        return rec
    def fail(j, errs, w):
        rec = _fail_rec(X[j], errs, w, meta_list[j])
        with log_lock: cache[rec["lkey"]] = rec; append(LOG, rec)
        return rec
    run_jobs(todo, do, fail, lambda j: t_sub())
    for j in range(len(X)):
        k = lkey(X[j])
        if k in cache: F[j] = value(cache[k])
    return F
def store_val(rec):
    with log_lock:
        vcache[rec["lkey"]] = rec; append(VLOG, rec)
        log(f"VAL {rec.get('kind')} gpu{rec.get('gpu')} val_fit={rec.get('val_fitness')} acc={rec.get('val_acc')} trunc={rec.get('val_trunc')} {rec.get('t_total')}s")
    mock_kill(f"val:{len(vcache)}")
def complete_batch(recs):
    def do(r, w):
        upd = complete_full(r, w)
        with log_lock:
            cache[r["lkey"]].update(upd); append(LOG, cache[r["lkey"]])
            log(f"COMPLETE gpu{w.g} irt_est {r['fitness']:.4f} -> full {upd['full_fitness']:.4f} acc {upd['full_acc']} tok {upd['full_gen_tokens']} {upd['t_complete']}s")
        return upd
    return run_jobs(recs, do, lambda r, errs, w: None, lambda j: t_comp())

# ---------------------------------------------------------------- EvoGM generator (port of the reference, as v1/r2; dim 16)
np.random.seed(42); torch.manual_seed(42)
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
STATE = f"{W}/evo_state.pt"
def save_state(S):
    S = dict(S); S["v"] = 3; S["np_rng"] = np.random.get_state(); S["torch_rng"] = torch.get_rng_state()
    S["irt"] = dict(IRT_RES) if IRT_RES else None
    torch.save(S, STATE + ".tmp"); os.replace(STATE + ".tmp", STATE)
    log("state saved: next step", S["next"], "evo_done", S.get("evo_done", False))
    mock_kill(f"state:{S['next'][0]},{S['next'][1]}" + (":done" if S.get("evo_done") else ""))
def load_state():
    best = None
    for p in find_all(f"{INP}/**/evo_state.pt"):
        try:
            S = torch.load(p, weights_only=False)
            if S.get("v") != 3: log("resume: ignoring state of another version", p); continue
            if best is None or (bool(S.get("evo_done")), tuple(S["next"])) > (bool(best.get("evo_done")), tuple(best["next"])): best, bp = S, p
            log("resume: found state", p, "next", S["next"], "evo_done", S.get("evo_done"))
        except Exception as e: log("resume: unreadable state", p, repr(e)[:300])
    if best is not None: shutil.copy(bp, STATE)   # keep the state in this session's output too (chained resumes)
    return best
def eff(coords, basis): return np.asarray(coords, dtype=np.float32) @ basis
def vals(coords, basis): return np.array([value(cache.get(lkey(e))) for e in eff(coords, basis)], dtype=float)
def run_eval(pop, basis, rnd, it, kinds, thr=None):
    meta = [dict(round=rnd, iter=it, kind=kinds[j], coords=[float(v) for v in pop[j]]) for j in range(len(pop))]
    return eval_batch(eff(pop, basis), meta, thr)
def best_full(): return fulls()[0]
def maybe_complete(pop, basis, tag):
    """Guard against IRT error / winner's curse (as r2): complete (all 200 items) the best un-completed population members when
    the best IRT estimate is within COMP_MARGIN of the best full fitness so far (up to one per GPU)."""
    bf = best_full()["full_fitness"]
    cands = []
    for e in eff(pop, basis):
        r = cache.get(lkey(e))
        if r is None or r.get("failed") or r.get("full_fitness") is not None or r.get("raced"): continue
        if all(c["lkey"] != r["lkey"] for c in cands): cands.append(r)
    cands.sort(key=lambda r: (-r["fitness"], r["lkey"]))
    if not cands or cands[0]["fitness"] < bf - COMP_MARGIN:
        log(f"completion {tag}: none (best estimate {cands[0]['fitness'] if cands else None} vs best full {bf:.4f})"); return
    pick = cands[:max(1, min(2, lanes()))]
    log(f"completion {tag}:", [round(r["fitness"], 4) for r in pick], "best full", round(bf, 4))
    complete_batch(pick)

# ---------------------------------------------------------------- resume state
S = load_state()
basis = np.eye(NE, dtype=np.float32); HIST = []; evo_done = False; rnd, it0 = 1, 0; ctx = None
if S is not None:
    if S.get("irt"): set_irt(S["irt"])
    basis = S["basis"]; rnd, it0 = S["next"]; HIST = list(S["HIST"]); evo_done = bool(S.get("evo_done"))
    np.random.set_state(S["np_rng"]); torch.set_rng_state(S["torch_rng"])
    log("RESUMING at round", rnd, "iter", it0, "evo_done", evo_done, "irt subset", len(SUB_ITEMS))
X1 = np.array([l for _, l in INIT], dtype=np.float32)          # round-1 coordinates (basis = identity -> effective lambda)
assert lkey(eff(X1[:1], np.eye(NE, dtype=np.float32))[0]) == B11_KEY
plan = dict(HARD_H=HARD_H, P=P, max_iter=MAX_ITER, max_round=MAX_ROUND, init=[k for k, _ in INIT], min_cal=MIN_CAL, k_val=K_VAL, val_max=VAL_MAX,
            t_full_prior=T_FULL_PRIOR, setup_h=(time.time() - T00) / 3600, lanes=lanes(), evo_remaining_h=remaining() / 3600,
            val_reserve_h=val_reserve() / 3600, final_reserve_h=max(final_est(), 1900 * TS) / 3600, race=RACE, resume=S is not None)
json.dump(plan, open(f"{W}/plan.json", "w"), indent=1, default=str); log("PLAN", plan)

# ---------------------------------------------------------------- round 1: initial population on all 200 items (+ base validation, once)
if S is None:
    status(stage="round1")
    jobs = [("full", k) for k in range(len(INIT))]; jobs.insert(2, ("val_base", -1))
    todo = [j for j in jobs if (j[0] == "full" and lkey(X1[j[1]]) not in cache) or (j[0] == "val_base" and (BASE_KEY not in vcache or vcache[BASE_KEY].get("failed")))]
    log("round 1 jobs", len(todo), "of", len(jobs), "(rest cached)")
    def est1(j):
        kind, k = todo[j]
        if kind == "val_base": return t_val()
        return t_full() + (iter_cost(P) if k >= MIN_CAL else 0.0)   # beyond the calibration minimum: keep room for >= 1 iteration
    def do1(job, w):
        kind, k = job
        if kind == "val_base":
            rec = evaluate_val(BASE16, w, dict(kind="qwen35_base", round=0)); store_val(rec); return rec
        rec = evaluate_full(X1[k], w, dict(round=1, iter=0, kind=INIT[k][0], coords=[float(v) for v in X1[k]]))
        _store(rec, lambda r: f"[1/0 {r['kind']}] n={len(cache)} gpu{w.g} full={r['full_fitness']:.4f} acc={ {t: round(r['full_acc'][t], 3) for t in TASKS} } "
                              f"{r['groups']} trunc={r['full_trunc']} tok={r['full_mean_tokens']} swap {r['t_swap']}s total {r['t_total']}s")
        mock_kill(f"r1:{sum(1 for r in cache.values() if r.get('round') == 1)}")
        return rec
    def fail1(job, errs, w):
        kind, k = job
        if kind == "val_base":
            rec = dict(lkey=BASE_KEY, lams=BASE16, kind="qwen35_base", failed=True, errors=errs, gpu=w.g); store_val(rec); return rec
        rec = _fail_rec(X1[k], errs, w, dict(round=1, iter=0, kind=INIT[k][0], coords=[float(v) for v in X1[k]]))
        with log_lock: cache[rec["lkey"]] = rec; append(LOG, rec)
        return rec
    run_jobs(todo, do1, fail1, est1)
    v1 = vals(X1, basis); ok = ~np.isnan(v1) & np.array([not (cache.get(lkey(e)) or {}).get("failed", True) for e in eff(X1, basis)])
    HIST.append(dict(round=1, iter=0, n_full=int(ok.sum()), best=float(np.nanmax(v1)) if ok.any() else None,
                     values={INIT[k][0]: (round(float(v1[k]), 4) if ok[k] else None) for k in range(len(INIT))}, elapsed_h=(time.time() - T00) / 3600))
    json.dump(dict(hist=HIST), open(f"{W}/history.json", "w"), indent=1); log("HIST", HIST[-1])
    if ok.sum() < MIN_CAL:
        log(f"round 1: only {int(ok.sum())} full rows (< {MIN_CAL}) -> no IRT/evolution; going to validation"); evo_done = True
    else:
        status(stage="irt")
        set_irt(irt_stage())
        STOP["flag"] = False   # a skipped round-1 tail job only means no more round-1 rows; the iteration check below decides
        pop0 = X1[ok]; hist_pop = pop0.copy(); pv = vals(pop0, basis)
        pop = pop0[np.argsort(-pv, kind="stable")[:P]]
        dual = Dual(NE); opt = torch.optim.Adam(dual.parameters(), lr=GEN_LR)
        ctx = dict(pop=pop, hist_pop=hist_pop, dual=dual, opt=opt); rnd, it0 = 1, 1
        save_state(dict(next=(1, 1), basis=basis, HIST=HIST, pop=pop, hist_pop=hist_pop, dual=dual.state_dict(), opt=opt.state_dict()))

# ---------------------------------------------------------------- EvoGM iterations (subset IRT + racing + completion), rounds with basis shift
status(stage="evolve")
END_IT = MAX_ITER + 1   # state next=(rnd, END_IT): the round's iterations are finished, its boundary (IRT refit + basis shift) is next
while not evo_done and rnd <= MAX_ROUND and not STOP["flag"]:
    if ctx is None and it0 > 0:   # resume inside a round
        dual = Dual(NE); dual.load_state_dict(S["dual"]); opt = torch.optim.Adam(dual.parameters(), lr=GEN_LR); opt.load_state_dict(S["opt"])
        torch.set_rng_state(S["torch_rng"])
        ctx = dict(pop=S["pop"], hist_pop=S["hist_pop"], dual=dual, opt=opt)
    elif ctx is None:   # round >= 2 init: the 16 experts (one-hots of the new basis, cached) + their average (one new subset eval)
        if remaining() < round_cost(P, 1):
            log(f"budget: no time for round {rnd} (remaining {remaining()/3600:.2f} h, min round {round_cost(P, 1)/3600:.2f} h)"); break
        init = np.vstack([np.eye(NE), np.full((1, NE), 1.0 / NE)]).astype(np.float32)
        f0 = run_eval(init, basis, rnd, 0, [f"onehot_prev_top{i+1}" for i in range(NE)] + ["average"])
        ok = ~np.isnan(f0); hist_pop = init[ok]
        pop = hist_pop[np.argsort(-f0[ok], kind="stable")[:P]]
        HIST.append(dict(round=rnd, iter=0, best=float(np.nanmax(f0)) if ok.any() else None, n=int(ok.sum()),
                         average=round(float(f0[-1]), 4) if ok[-1] else None, elapsed_h=(time.time() - T00) / 3600))
        json.dump(dict(hist=HIST), open(f"{W}/history.json", "w"), indent=1); log("HIST", HIST[-1])
        dual = Dual(NE); opt = torch.optim.Adam(dual.parameters(), lr=GEN_LR)
        ctx = dict(pop=pop, hist_pop=hist_pop, dual=dual, opt=opt); it0 = 1
        if STOP["flag"]: break
        save_state(dict(next=(rnd, 1), basis=basis, HIST=HIST, pop=pop, hist_pop=hist_pop, dual=dual.state_dict(), opt=opt.state_dict()))
    pop, hist_pop, dual, opt = ctx["pop"], ctx["hist_pop"], ctx["dual"], ctx["opt"]
    for it in range(it0, MAX_ITER + 1):
        if STOP["flag"] or len(pop) == 0: break
        rem, ic, rc2 = remaining(), iter_cost(P), round_cost(P, 2)
        if rem < ic or (it >= 3 and rem - ic < rc2 <= rem and rnd < MAX_ROUND):
            log(f"budget: ending round {rnd} before iter {it} (remaining {rem/3600:.2f} h, iter {ic/3600:.2f} h, min next round {rc2/3600:.2f} h)"); break
        hv = vals(hist_pop, basis); hm = ~np.isnan(hv); hp, hf = hist_pop[hm], hv[hm]
        pv = vals(pop, basis); order = np.argsort(-pv, kind="stable")[:P]; pop, pv = pop[order], pv[order]
        idx = np.argsort(-hf, kind="stable"); nw = max(1, int(len(hp) * WIN_PORTION))
        winners, losers = hp[idx[:nw]], hp[idx[nw:]]
        for ep in range(GEN_EPOCHS): loss = train_epoch(dual, opt, winners, losers)
        with torch.no_grad(): new = dual.lose2win(torch.tensor(pop, dtype=torch.float32)).numpy()
        new = np.clip(new, -1, 1).astype(np.float32)
        thr = float(pv[-1]) if len(pv) >= P else None   # racing threshold: the current P-th best value
        log(f"round {rnd} iter {it}: generator loss {loss:.4f}, winners {nw}/{len(hp)}, race thr {thr}")
        nf = run_eval(new, basis, rnd, it, ["generated"] * len(new), thr)
        hist_pop = np.vstack([hist_pop, new])
        ok = ~np.isnan(nf)
        allp = np.vstack([pop, new[ok]]); av = vals(allp, basis)
        pop = allp[np.argsort(-av, kind="stable")[:P]]
        if not STOP["flag"]: maybe_complete(pop, basis, f"r{rnd}i{it}")
        pv = vals(pop, basis); order = np.argsort(-pv, kind="stable"); pop, pv = pop[order], pv[order]
        bfr = best_full()
        HIST.append(dict(round=rnd, iter=it, best_value=float(pv[0]), pop_mean=float(np.nanmean(pv)), new_mean=float(np.nanmean(nf)) if ok.any() else None,
                         new_best=float(np.nanmax(nf)) if ok.any() else None, n_new=int(ok.sum()),
                         n_raced=int(sum(1 for e in eff(new, basis) if (cache.get(lkey(e)) or {}).get("raced"))),
                         best_full=bfr["full_fitness"], best_full_round=bfr.get("round"), best_full_kind=bfr.get("kind"), gen_loss=loss,
                         elapsed_h=(time.time() - T00) / 3600))
        json.dump(dict(hist=HIST), open(f"{W}/history.json", "w"), indent=1); log("HIST", HIST[-1])
        ctx.update(pop=pop, hist_pop=hist_pop)
        if not STOP["flag"]:
            save_state(dict(next=(rnd, it + 1), basis=basis, HIST=HIST, pop=pop, hist_pop=hist_pop, dual=dual.state_dict(), opt=opt.state_dict()))
    if STOP["flag"]: break
    # basis shift: top-16 of this round's history (best available value) become the next round's experts
    hv = vals(hist_pop, basis); hm = ~np.isnan(hv)
    if hm.sum() < NE:
        log(f"round {rnd}: only {int(hm.sum())} evaluated members (< {NE}) -> no basis shift possible, evolution ends"); break
    if rnd < MAX_ROUND:
        if it0 != END_IT:   # mark "iterations of this round finished" (resume point before the refit)
            save_state(dict(next=(rnd, END_IT), basis=basis, HIST=HIST, pop=pop, hist_pop=hist_pop, dual=dual.state_dict(), opt=opt.state_dict()))
        # round boundary: refit the 2PL on all full rows (round 1 + completions), reselect the subset, rescore the subset records
        status(stage=f"irt_refit_r{rnd}")
        refit_irt(rnd)
        mock_kill(f"refit:{rnd}")
        hv = vals(hist_pop, basis); hm = ~np.isnan(hv)
    top = hist_pop[hm][np.argsort(-hv[hm], kind="stable")[:NE]]
    basis = (top.astype(np.float32) @ basis).astype(np.float32)
    rnd += 1; it0 = 0; ctx = None
    np.save(f"{W}/basis_round{rnd}.npy", basis); log(f"basis shift -> round {rnd}", basis.round(3).tolist())
    save_state(dict(next=(rnd, 0), basis=basis, HIST=HIST))
if not evo_done:
    evo_done = True
    save_state(dict(next=(rnd, it0), basis=basis, HIST=HIST, evo_done=True))
STOP["flag"] = False

# ---------------------------------------------------------------- validation: top candidates by full search fitness (+ B11 + per-round bests); BFCL floor = Qwen3.5-4B base
status(stage="validation")
FL = fulls()
sel = []
def _add(r):
    if r is not None and r.get("full_fitness") is not None and all(x["lkey"] != r["lkey"] for x in sel): sel.append(r)
for r in FL[:K_VAL]: _add(r)
_add(cache.get(B11_KEY))
by_round = {}
for r in FL: by_round.setdefault(r.get("round"), r)
for rd in sorted(by_round, key=lambda x: (x is None, x)): _add(by_round[rd])
sel = sel[:VAL_MAX]
log("validation candidates", [(r.get("kind"), r.get("round"), r.get("iter"), round(r["full_fitness"], 4)) for r in sel])
vjobs = ([dict(lkey=BASE_KEY, lams=BASE16, kind="qwen35_base", round=0)] if (BASE_KEY not in vcache or vcache[BASE_KEY].get("failed")) else []) + \
        [r for r in sel if r["lkey"] not in vcache or vcache[r["lkey"]].get("failed")]
def dov(r, w):
    meta = dict(kind=r.get("kind"), round=r.get("round"), iter=r.get("iter"), full_fitness=r.get("full_fitness"))
    rec = evaluate_val(r["lams"], w, meta); store_val(rec); return rec
def failv(r, errs, w):
    rec = dict(lkey=r["lkey"], lams=r["lams"], kind=r.get("kind"), failed=True, errors=errs, gpu=w.g); store_val(rec); return rec
run_jobs(vjobs, dov, failv, lambda j: t_val(), deadline=end_deadline)
base = vcache.get(BASE_KEY)
floor = base["val_acc"]["bfcl"] if base and not base.get("failed") else None
rows = []
for r in sel:
    v = vcache.get(r["lkey"])
    if not v or v.get("failed"): continue
    elig = floor is None or v["val_acc"]["bfcl"] >= floor - 1e-9
    rows.append(dict(lkey=r["lkey"], lams=r["lams"], kind=r.get("kind"), round=r.get("round"), iter=r.get("iter"), groups=grp(r["lams"]),
                     full_fitness=r["full_fitness"], full_acc=r.get("full_acc"), val_fitness=v["val_fitness"], val_acc=v["val_acc"],
                     val_trunc=v.get("val_trunc"), val_mean_tokens=v.get("val_mean_tokens"), eligible=elig))
    if not elig: log(f"BFCL FLOOR: {r.get('kind')} r{r.get('round')}i{r.get('iter')} val BFCL {v['val_acc']['bfcl']:.4f} < base {floor:.4f} -> not eligible")
er = [x for x in rows if x["eligible"]]
winner = max(er, key=lambda x: (x["val_fitness"], x["full_fitness"], x["lkey"])) if er else None
vres = dict(base=base, bfcl_floor=floor, floor_verified=floor is not None, rows=sorted(rows, key=lambda x: -x["val_fitness"]), winner=winner,
            rule="winner = max validation fitness (mean of 3 task accuracies, truncated = wrong) among candidates whose val BFCL >= Qwen3.5-4B base val BFCL; ties -> higher full search fitness")
json.dump(vres, open(f"{W}/val_result.json", "w"), indent=1, default=str)
log("VALIDATION", [(x["kind"], x["round"], x["iter"], round(x["full_fitness"], 4), round(x["val_fitness"], 4), x["val_acc"], x["eligible"]) for x in vres["rows"]],
    "floor", floor, "WINNER", (winner["kind"], winner["round"], winner["iter"], winner["val_fitness"]) if winner else None)
recs_irt = [r for r in cache.values() if r.get("fitness_kind") == "irt"]
evo_result = dict(n_evals=len(cache), n_full_round1=sum(1 for r in cache.values() if r.get("round") == 1 and r.get("full_fitness") is not None),
                  n_subset=len(recs_irt), n_raced=sum(1 for r in recs_irt if r.get("raced")), n_failed=sum(1 for r in cache.values() if r.get("failed")),
                  completed=[dict(round=r.get("round"), iter=r.get("iter"), irt=r["fitness"], full=r["full_fitness"]) for r in recs_irt if r.get("full_fitness") is not None],
                  irt=dict(chosen=IRT_RES.get("chosen"), loo_table=IRT_RES.get("loo_table"), n_cal=IRT_RES.get("n_cal"), fit_round=IRT_RES.get("fit_round"),
                           refits=IRT_RES.get("refits"), refit_failures=IRT_RES.get("refit_failures")) if IRT_RES else None,
                  best_full=FL[0] if FL else None, hist=HIST, val=vres, winner=winner, winner_is_b11=bool(winner and winner["lkey"] == B11_KEY),
                  hours=(time.time() - T00) / 3600)
json.dump(evo_result, open(f"{W}/evo_result.json", "w"), indent=1, default=str)
if winner is None:
    for w in workers: w.kill()
    status(stage="done", final="skipped: no eligible winner (see val_result.json)", hours=(time.time() - T00) / 3600)
    log("ALL DONE (no eligible winner) hours", (time.time() - T00) / 3600); sys.exit(0)
if winner["lkey"] == B11_KEY:
    for w in workers: w.kill()
    status(stage="done", final="skipped: winner is Baphomet-1.1 (its 180-item result exists: evogm-r2)", hours=(time.time() - T00) / 3600)
    log("ALL DONE (winner = Baphomet-1.1, final skipped) hours", (time.time() - T00) / 3600); sys.exit(0)

# ---------------------------------------------------------------- final: winner on the 180-item test set (8192 cap)
status(stage="final", winner=dict(kind=winner["kind"], round=winner["round"], val=winner["val_fitness"], lams=winner["lams"]))
prev = [p for p in find_all(f"{INP}/**/final_meta.json") if json.load(open(p)).get("lkey") == winner["lkey"]]
if prev and os.path.exists(os.path.join(os.path.dirname(prev[0]), "raw_evogm.jsonl")) and \
        sum(1 for _ in open(os.path.join(os.path.dirname(prev[0]), "raw_evogm.jsonl"))) == 180:
    shutil.copy(os.path.join(os.path.dirname(prev[0]), "raw_evogm.jsonl"), f"{W}/raw_evogm.jsonl"); shutil.copy(prev[0], f"{W}/final_meta.json")
    for w in workers: w.kill()
    status(stage="done", final="reused from a previous session"); log("final reused from", prev[0]); sys.exit(0)
TEST = {json.loads(l)["id"]: json.loads(l) for l in open(f"{DATA}/test_items.jsonl")}
ids = json.load(open(f"{DATA}/subset_ids.json"))
halves = [ids[0::2], ids[1::2]]
fres = {}
def final_lane(g, w):
    items = ordered([TEST[i] for i in halves[g]])
    json.dump(items, open(f"{TMP}/test_{g}.json", "w"))
    for attempt in range(3):
        try:
            if w.dead or not w.alive(): w.restart()
            w.call(T_LOAD, op="load_merge", tvs=TV, lams=winner["lams"])
            t0 = time.time(); r = w.call(T_GEN_FINAL, op="gen", items=f"{TMP}/test_{g}.json", max_tokens=8192, out=f"{W}/raw_evogm_{g}.json")
            fres[g] = dict(secs=time.time() - t0, gen_tokens=r["gen_tokens"], gpu=w.g); log("final half", g, "done on gpu", w.g, fres[g]); return
        except Exception as e:
            log("FINAL ERROR half", g, "gpu", w.g, "attempt", attempt, repr(e)[:1500])
            try: w.kill()
            except Exception: pass
            w.dead = True
            if attempt == 2: return
ths = [threading.Thread(target=final_lane, args=(g, workers[g])) for g in (0, 1)]
[t.start() for t in ths]; [t.join() for t in ths]
for g in (0, 1):
    if g not in fres and (1 - g) in fres:
        log("running half", g, "on gpu", 1 - g); final_lane(g, workers[1 - g])
with open(f"{W}/raw_evogm.jsonl", "w") as f:
    for g in (0, 1):
        p = f"{W}/raw_evogm_{g}.json"
        if not os.path.exists(p): continue
        for o in json.load(open(p)):
            it = TEST[o["id"]]
            f.write(json.dumps(dict(id=o["id"], bench=it["bench"], category=it["category"], model="evogm2", prompt_tokens=o["prompt_tokens"],
                                    completion_tokens=o["completion_tokens"], finish_reason=o["finish_reason"], output=o["output"])) + "\n")
n = sum(1 for _ in open(f"{W}/raw_evogm.jsonl"))
quick = {}
for l in open(f"{W}/raw_evogm.jsonl"):
    o = json.loads(l); s = safe_score(TEST[o["id"]], o["output"], o["finish_reason"])
    quick.setdefault(o["bench"], []).append(bool(s["correct"]))
json.dump(dict(lkey=winner["lkey"], lams=winner["lams"], n=n, quick={k: [sum(v), len(v)] for k, v in quick.items()}, fres=fres), open(f"{W}/final_meta.json", "w"), indent=1)
log("FINAL raw lines", n, {k: (sum(v), len(v)) for k, v in quick.items()}, fres)
for w in workers: w.kill()
status(stage="done", final=dict(n=n, quick={k: [sum(v), len(v)] for k, v in quick.items()}, fres=fres), hours=(time.time() - T00) / 3600)
log("ALL DONE hours", (time.time() - T00) / 3600)
