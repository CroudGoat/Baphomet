# EvoGM-DFS (2026-11-17): Data Flow Space layer-path merge search (Sakana "Evolutionary Model Merge" style) over three Qwen3.5-4B
# merges, on the EvoGM v2 evaluator.  Kaggle T4x2.
#   Donors: r1i1 (EvoGM v2 val-best), b11 (Baphomet-1.1), b10 (Baphomet-1.0); each = Qwen3.5 base + sum lambda*TV (the v2 linmerge
#   fp8-TV merges, bit-identical).  Genotype (57) = 48 binary slot indicators (2 repetitions x 8 blocks of 4 layers x 3 donors,
#   slot order rep -> block -> donor) + 3x3 scaling matrix W (residual-stream scale at every block boundary, folded exactly into
#   the output projections).  Path 6..10 blocks; the engine has 40 layers, unused slots = identity layers (zeroed outputs), so a
#   candidate is an in-place weight swap (see dfslib.py for the exact definitions).  Embedding / final norm / tied lm_head and all
#   other non-layer tensors = B11's.
#   Evaluator = EvoGM v2: IRT 2PL (v2 fit reused, refit at round boundaries with v2 + DFS full rows), ~60-item Fisher subset,
#   racing, completion guard (full 200), validation 150 with BFCL floor = Qwen3.5 base val BFCL - 2/30, final 180 @8192.  New: a
#   16-item smoke screen (subset items with short reference outputs, max_tokens 1024) before each subset run, so broken layer
#   paths are rejected in ~1-2 min instead of ~30 min; finished smoke outputs count as subset observations.
#   Optimizer: EvoGM dual generator (lose2win) on the 57 continuous coordinates, bit = score > 0.
import os, sys, json, time, glob, gzip, shutil, subprocess, threading, queue, hashlib, math, traceback, base64
from concurrent.futures import ThreadPoolExecutor
T00 = time.time()
MOCK = json.loads(os.environ.get("EVOGM_DFS_MOCK", "{}"))   # mock harness only (path prefix, time scale, fault injection); unset on Kaggle
KR = MOCK.get("root", "")                               # path prefix ("" on Kaggle)
TS = float(MOCK.get("ts", 1.0))                         # time scale of priors/margins (1.0 on Kaggle)
MODE = "full"
HARD_H = float(MOCK.get("hard_h", 10.5))                 # whole-kernel ceiling for this session
FIT_MAXTOK = 4096
GPUMEM = [0.90, 0.86]; MAXSEQS = 48                     # eager engines; 40 layers: +1.7 GiB weights vs v2 (0.84: KV 2.89 GiB) -> 0.90 keeps ~2.1 GiB KV
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
DFSDATA = os.path.dirname(find(f"{INP}/**/v2_prev.json"))
os.makedirs(TMP, exist_ok=True)
DFSLIB_B64 = "__DFSLIB_B64__"
if MOCK.get("dfslib"): shutil.copy(MOCK["dfslib"], f"{TMP}/dfslib.py")
else: open(f"{TMP}/dfslib.py", "wb").write(base64.b64decode(DFSLIB_B64))
sys.path.insert(0, DATA); sys.path.insert(1, PYLIB); sys.path.insert(0, TMP)
os.environ["PYTHONPATH"] = f"{TMP}:{DATA}:{PYLIB}:" + os.environ.get("PYTHONPATH", "")
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
log("EvoGM-DFS MODE", MODE, "HARD_H", HARD_H, "mock" if MOCK else "")
if not MOCK: sh("nvidia-smi -L; df -h /tmp /kaggle/working; free -g; nproc", t=60)
import linmerge, dfslib
linmerge.log = log
def disk_used_gb():
    tot = 0
    for d in (f"{TMP}/m", f"{TMP}/tv", f"{TMP}/sh", f"{TMP}/boot"):
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

# ---------------------------------------------------------------- donor block shards (3 donors x 8 blocks + B11 non-layer) -> boot dir
PREV = json.load(open(f"{DFSDATA}/v2_prev.json"))
DONOR_LAMS = [PREV["donors"][n]["lams"] for n in dfslib.DONORS]
SH = f"{TMP}/sh"; BOOT = f"{TMP}/boot"; os.makedirs(SH, exist_ok=True)
open(f"{TMP}/sh1.py", "w").write(
    "import sys, json\n"
    "def main():\n"
    "    sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); import linmerge, dfslib\n"
    "    a = json.load(open(sys.argv[3]))\n"
    "    M = linmerge.Merger(a['base'], a['tvs'])\n"
    "    dfslib.build_shards(M, a['lams'], a['blocks'], a['out'], nonlayer_lams=a.get('nonlayer_lams'), device='cuda')\n"
    "if __name__ == '__main__':\n"
    "    main()\n")
def shards_ok():
    need = [dfslib.shard_name(d, b) for d in range(dfslib.ND) for b in range(dfslib.NB)] + [dfslib.NONLAYER]
    return all(os.path.exists(f"{SH}/{n}") for n in need)
status(stage="shards")
t_sh = time.time()
if MOCK:
    time.sleep(300 * TS)
    for d in range(dfslib.ND):
        for b in range(dfslib.NB): open(f"{SH}/{dfslib.shard_name(d, b)}", "w").write("mock")
    open(f"{SH}/{dfslib.NONLAYER}", "w").write("mock")
else:
    # peak during the build: base 9.3 + 8 TVs 35.2 + shards ~24 (3 donors x 7.2 GB layers + 2.5 GB B11 non-layer) ~ 69 GB (< the ~94 GB
    # where earlier runs hit a read-only filesystem); nothing else is downloading at this point.  TVs + base weights are deleted after.
    sh("df -h /tmp", t=60)
    def sh_lane(g, blocks, nonlayer):
        a = dict(base=DIRS["base"], tvs=TV, lams=DONOR_LAMS, blocks=blocks, out=SH, nonlayer_lams=DONOR_LAMS[dfslib.PAD_DONOR] if nonlayer else None)
        json.dump(a, open(f"{TMP}/sh_{g}.json", "w"))
        for att in range(2):
            r = subprocess.run(f"CUDA_VISIBLE_DEVICES={g} python {TMP}/sh1.py {DATA} {TMP} {TMP}/sh_{g}.json", shell=True, capture_output=True, text=True, timeout=3600)
            log("shard lane", g, "rc", r.returncode, r.stdout[-3000:], r.stderr[-2000:])
            if r.returncode == 0: return
    ths = [threading.Thread(target=sh_lane, args=(0, [0, 1, 2, 3], True)), threading.Thread(target=sh_lane, args=(1, [4, 5, 6, 7], False))]
    [t.start() for t in ths]; [t.join() for t in ths]
if not shards_ok():
    status(stage="shards_failed"); raise SystemExit("donor shard build failed")
log("donor shards built in", round(time.time() - t_sh), "s; disk", disk_used_gb())
if not MOCK:   # free the disk: the TVs and the base weights are not needed any more (tokenizer/config files are kept)
    shutil.rmtree(TVDIR, ignore_errors=True)
    for f in glob.glob(f"{DIRS['base']}/*.safetensors"): os.remove(f)
    nk = dfslib.write_boot_dir(DIRS["base"], SH, BOOT, dfslib.L_MAX)
    log("boot dir", BOOT, nk, "tensors; disk", disk_used_gb()); sh(f"ls -la {BOOT}; df -h /tmp", t=60)
else:
    os.makedirs(BOOT, exist_ok=True)
status(stage="shards_done", shard_secs=round(time.time() - t_sh))

# ---------------------------------------------------------------- workers (persistent vLLM engines, file protocol)
WORKER_B64 = "__WORKER_B64__"
WORKER_PY = f"{TMP}/gpuworker_dfs.py"
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
        self.p = subprocess.Popen(f"CUDA_VISIBLE_DEVICES={self.g} exec {py} {WORKER_PY} {BOOT} {self.dir} {gpumem} 1 {MAXSEQS} "
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
        if self.p is not None: subprocess.run(f"pkill -9 -f '[g]puworker_dfs.py \\S+ {self.dir} '", shell=True)
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
