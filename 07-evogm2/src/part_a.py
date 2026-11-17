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
