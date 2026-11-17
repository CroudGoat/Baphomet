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

# ---------------------------------------------------------------- items / scoring (as v2)
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
DIM = dfslib.DIM
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

# ---------------------------------------------------------------- v2 results reused (dataset evogm-dfs-data/v2_prev.json)
V2ROWS = PREV["rows"]                                   # v2 full-200 rows (IRT calibration data; item parameters are model-independent)
BASE_VAL = PREV["base_val"]                              # Qwen3.5-4B base on the 150 validation items (BFCL floor), measured in v2
V2R1 = PREV["v2_records"]["r1i1"]; V2B11 = PREV["v2_records"]["b11"]
BFCL_TOL = 2.0 / 30                                      # eligible iff val BFCL >= base val BFCL - 2/30 (user, 2026-11-16)
PH_B11 = dfslib.decode(dfslib.geno_plain(1)); PH_R1 = dfslib.decode(dfslib.geno_plain(0))
B11_KEY = PH_B11["key"]
log("v2 reuse: rows", len(V2ROWS), "base val BFCL", BASE_VAL["val_acc"]["bfcl"], "B11 full", V2B11["evo"]["full_fitness"], "val", V2B11["val"]["val_fitness"],
    "v2-r1i1 full", V2R1["evo"]["full_fitness"], "val", V2R1["val"]["val_fitness"])

# ---------------------------------------------------------------- caches (resume: every dfs_log / dfs_val_log in the inputs)
FITD = f"{W}/dfs_fit_outputs"; VALD = f"{W}/dfs_val_outputs"
for d in (FITD, VALD): os.makedirs(d, exist_ok=True)
LOG = f"{W}/dfs_log.jsonl"; VLOG = f"{W}/dfs_val_log.jsonl"
def gkey(x): return ",".join(f"{float(v):.6f}" for v in x)
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
_read(f"{INP}/**/dfs_log.jsonl", cache); _read(f"{INP}/**/dfs_val_log.jsonl", vcache)
for sub, dst in (("dfs_fit_outputs", FITD), ("dfs_val_outputs", VALD)):
    for p in find_all(f"{INP}/**/{sub}/*.json.gz"):
        d = f"{dst}/{os.path.basename(p)}"
        if not os.path.exists(d): shutil.copy(p, d)
# plain B11 = exactly the v2 Baphomet-1.1 model (B11 layers + B11 non-layer tensors; tensors bit-identical) -> v2 search/val results
if B11_KEY not in cache:
    e = V2B11["evo"]
    cache[B11_KEY] = dict(lkey=B11_KEY, geno=[float(v) for v in dfslib.geno_plain(1)], path=PH_B11["path"], C=PH_B11["C"], summary=dfslib.summary(PH_B11),
                          n_blocks=8, kind="plain_b11", round=0, iter=0, tag=None, fitness=e["full_fitness"], fitness_kind="full", full_fitness=e["full_fitness"],
                          full_acc=e["full_acc"], full_gen_tokens=e.get("full_gen_tokens"), failed=False, from_v2=True, v2_lkey=e["lkey"])
if B11_KEY not in vcache:
    v = V2B11["val"]
    vcache[B11_KEY] = dict(v, lkey=B11_KEY, kind="plain_b11", from_v2=True, v2_lkey=v["lkey"], failed=False)
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
SUB_T = []; COMP_T = []; VAL_T = []; SCR_T = []
SESSION = {"n": 0, "fail": 0}
RACE = {"on": True, "up": 1.5, "min_frac": 0.5}

# ---------------------------------------------------------------- IRT (per-task 2PL; v2 fit reused, refit at round boundaries)
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
    """IRT calibration rows: the 16 v2 full-200 rows + every DFS candidate with a full-200 result (completions)."""
    rows = []
    for r in V2ROWS:
        rows.append((dict(lkey="v2:" + r["lkey"], full_fitness=r["full_fitness"], t_gen=r.get("t_gen") or 0.0, t_swap=r.get("t_swap") or 0.0),
                     {t: np.array([int(r["correct"][i]) for i in SIDS[t]]) for t in TASKS},
                     {t: np.array([float(r["tokens"][i]) for i in SIDS[t]]) for t in TASKS}))
    for r in cache.values():
        if r.get("full_fitness") is None or r.get("failed") or r.get("from_v2"): continue
        d = json.load(gzip.open(f"{FITD}/{r['tag']}_full.json.gz", "rt"))
        sc = d["scores"]; tk = {o["id"]: o["completion_tokens"] for o in d["outputs"]}
        rows.append((r, {t: np.array([int(bool(sc[i]["correct"]) and not sc[i]["truncated"]) for i in SIDS[t]]) for t in TASKS},
                     {t: np.array([tk[i] for i in SIDS[t]], dtype=float) for t in TASKS}))
    rows.sort(key=lambda x: x[0]["lkey"])   # deterministic order
    return rows
def irt_stage(prev=None, out="dfs_irt.json", rnd=1):
    """per-task 2PL fit + LOO split choice + Fisher subset on all full rows (as v2 irt_stage). prev = the fit in use (its time
    priors are kept)."""
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
            th = fits[t][0][top]; P_ = sig(a[t][None] * (th[:, None] - b[t][None])); I = (a[t] ** 2 * P_ * (1 - P_)).mean(0)
            out[t] = np.sort(np.argsort(-(I / np.power(cost[t], ALPHA)), kind="stable")[:k])
        return out
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
               t_sub_prior=float(np.mean((tg * frac * 1.15 + tsw)[top])), t_comp_prior=float(np.mean((tg * (1 - frac) * 1.05 + tsw)[top])),
               secs=round(time.time() - t0, 1), fit_round=rnd, source="dfs_fit")
    if prev is not None:
        res["t_sub_prior"], res["t_comp_prior"] = prev["t_sub_prior"], prev["t_comp_prior"]
        res["prev_n_cal"] = prev.get("n_cal"); res["prev_subset_ids"] = prev.get("subset_ids")
        res["refits"] = list(prev.get("refits") or []) + [dict(round=rnd, n_cal=n, chosen=best, subset_overlap=len(set(res["subset_ids"]) & set(prev.get("subset_ids") or [])))]
    for t in TASKS:
        assert np.all(np.isfinite(a[t])) and np.all(np.isfinite(b[t])) and np.all(np.isfinite(fits[t][0])), f"non-finite 2PL fit ({t})"
    log("IRT chosen split", best, "subset", len(res["subset_ids"]), "const", res["const_items"], "fit secs", res["secs"])
    json.dump(res, open(f"{W}/{out}", "w"), indent=0)
    return res
SUB_ITEMS = []; SMOKE = []; IRT_RES = {}
SMOKE_N = 16; SMOKE_MAXTOK = 1024; SMOKE_MIN_CORRECT = 2; SMOKE_MIN_FIN = 5
V2_TOP = [r for r in V2ROWS if r["full_fitness"] >= float(np.median([x["full_fitness"] for x in V2ROWS]))]
def smoke_items(sub_ids):
    """screen items: every BFCL subset item + the subset's other (MMLU-Pro/GPQA) items with the shortest reference outputs (v2 top-half rows),
    SMOKE_N in total.  The reference models finish ~15/16 within 1024 tokens; a broken layer path finishes almost none."""
    tok = {i: float(np.mean([r["tokens"][i] for r in V2_TOP])) for i in sub_ids}
    bf = [i for i in sub_ids if FITBY[i]["bench"] == "bfcl"]
    mm = sorted([i for i in sub_ids if FITBY[i]["bench"] != "bfcl"], key=lambda i: (tok[i], i))
    return (bf + mm)[:SMOKE_N]
def set_irt(res):
    IRT_RES.clear(); IRT_RES.update(res)
    IRT.clear(); IRT.update(res); IRT["a"] = {t: np.array(res["a"][t]) for t in TASKS}; IRT["b"] = {t: np.array(res["b"][t]) for t in TASKS}
    SUB_ITEMS[:] = ordered([FITBY[i] for i in res["subset_ids"]])
    json.dump(SUB_ITEMS, open(f"{TMP}/fit_sub.json", "w"))
    SMOKE[:] = ordered([FITBY[i] for i in smoke_items(res["subset_ids"])])
    json.dump(SMOKE, open(f"{TMP}/fit_smoke.json", "w"))
    rb = [r for r in V2ROWS if r["lkey"] == V2B11["evo"]["lkey"]][0]
    IRT["smoke_ref_b11"] = dict(correct=sum(int(rb["correct"][i["id"]] and rb["tokens"][i["id"]] < SMOKE_MAXTOK) for i in SMOKE),
                                finished=sum(int(rb["tokens"][i["id"]] < SMOKE_MAXTOK and rb["finish"][i["id"]] == "stop") for i in SMOKE))
    log("IRT in use: fit_round", res.get("fit_round"), "subset", len(SUB_ITEMS), "smoke", len(SMOKE), [i["bench"][0] for i in SMOKE], "B11 v2 on smoke", IRT["smoke_ref_b11"])
def rescore_subset_records(tagv):
    """re-estimate every subset/screened record from its saved observations with the current IRT fit (one scale)."""
    n = 0; dmax = 0.0
    for k in sorted(cache):
        r = cache[k]
        if r.get("fitness_kind") not in ("irt", "screened") or r.get("failed") or not r.get("tag"): continue
        d = json.load(gzip.open(f"{FITD}/{r['tag']}.json.gz", "rt"))
        fit, det = irt_fitness({i: int(v) for i, v in d["obs"].items()})
        if r.get("irt_version") == tagv and abs(fit - r["fitness"]) < 1e-12: continue
        upd = dict(lkey=k, fitness=fit, irt=det, irt_version=tagv)
        if "fitness_r1irt" not in r: upd["fitness_r1irt"] = r["fitness"]
        dmax = max(dmax, abs(fit - r["fitness"])); r.update(upd); append(LOG, r); n += 1
    log(f"rescored {n} subset records with IRT {tagv} (max |change| {dmax:.4f})")
def refit_irt(rnd):
    """round boundary: refit on all full rows (v2 + DFS completions) and reselect the subset, only if new DFS full rows exist; on
    failure keep the previous fit.  Then rescore the subset records with the fit in use."""
    prev = dict(IRT_RES)
    n_now = len(full_rows())
    if prev.get("fit_round") == rnd + 1 or prev.get("refit_failed_round") == rnd:
        log(f"IRT refit for the round {rnd} boundary already in use"); rescore_subset_records(IRT_RES.get("fit_round", 1)); return
    if n_now <= (prev.get("n_cal") or 0):
        log(f"IRT refit at round {rnd} boundary skipped: no new full rows ({n_now})"); return
    try:
        res = irt_stage(prev=prev, out=f"dfs_irt_refit_r{rnd}.json", rnd=rnd + 1)
        if MOCK.get("refit_fail") == rnd: raise RuntimeError("mock refit failure")
        set_irt(res)
        log(f"IRT REFIT at round {rnd} boundary: {prev.get('n_cal')} -> {res['n_cal']} rows, split {prev['chosen']['split']} -> {res['chosen']['split']}, "
            f"subset overlap {res['refits'][-1]['subset_overlap']}/{len(res['subset_ids'])}")
    except Exception as e:
        log("IRT REFIT FAILED -> keeping the previous fit:", repr(e)[:1500]); traceback.print_exc()
        prev["refit_failed_round"] = rnd; prev.setdefault("refit_failures", []).append(dict(round=rnd, error=repr(e)[:500]))
        set_irt(prev)
    rescore_subset_records(IRT_RES.get("fit_round", 1))

# ---------------------------------------------------------------- evaluations (every candidate = one in-place load_dfs swap)
def load(w, ph):
    return w.call(T_LOAD, op="load_dfs", shards=SH, plan=dfslib.plan_of(ph))["t"]
def _gen(w, items_path, timeout, out, max_tokens=FIT_MAXTOK, **kw):
    return w.call(timeout, op="gen", items=items_path, max_tokens=max_tokens, out=out, **kw)
def phmeta(ph): return dict(path=ph["path"], C=ph["C"], summary=dfslib.summary(ph), n_blocks=ph["n_blocks"], repaired=ph["repaired"])
def tag_of(key): return hashlib.md5(key.encode()).hexdigest()[:10]
def smoke(w, tagn):
    """16 short subset items @1024 tokens. -> (outputs, obs of finished items, n_correct, n_finished, secs)"""
    t0 = time.time()
    rg = _gen(w, f"{TMP}/fit_smoke.json", T_GEN_SUB, f"{TMP}/os_{w.g}.json", max_tokens=SMOKE_MAXTOK)
    outs = json.load(open(f"{TMP}/os_{w.g}.json"))
    fin = [o for o in outs if o["finish_reason"] == "stop"]
    obs = {o["id"]: int(ok_item(safe_score(FITBY[o["id"]], o["output"], o["finish_reason"]))) for o in fin}
    return outs, obs, sum(obs.values()), len(fin), time.time() - t0, rg["gen_tokens"]
def evaluate_sub(ph, geno, w, meta, thr):
    """smoke screen, then the IRT subset with racing (stop when theta+1.5sd estimate < thr), as v2.  Finished smoke outputs are
    subset observations (not regenerated)."""
    t0 = time.time(); key = ph["key"]; tag = tag_of(key)
    t_swap = load(w, ph)
    s_outs, s_obs, s_cor, s_fin, t_smoke, s_tok = smoke(w, tag)
    SCR_T.append(t_smoke)
    base = dict(lkey=key, geno=[float(v) for v in geno], tag=tag, failed=False, gpu=w.g, t_swap=round(t_swap, 1), t_smoke=round(t_smoke, 1),
                smoke=dict(correct=s_cor, finished=s_fin, n=len(SMOKE), tokens=s_tok), **phmeta(ph), **meta)
    if SCREEN["on"] and (s_cor < SMOKE_MIN_CORRECT or s_fin < SMOKE_MIN_FIN):
        obs = {o["id"]: (s_obs.get(o["id"], 0)) for o in s_outs}          # unfinished at 1024 counted wrong (screen estimate only)
        fit, det = irt_fitness(obs)
        with gzip.open(f"{FITD}/{tag}.json.gz", "wt") as f:
            json.dump(dict(key=key, kind="screened", outputs=s_outs, obs=obs), f)
        return dict(base, fitness=fit, fitness_kind="screened", screened=True, irt=det, full_fitness=None, raced=False, n_aborted=0,
                    t_gen=0.0, gen_tokens=s_tok, t_total=round(time.time() - t0, 1), elapsed_h=round((time.time() - T00) / 3600, 3))
    have = {o["id"]: o for o in s_outs if o["finish_reason"] == "stop"}
    todo = [i for i in SUB_ITEMS if i["id"] not in have]
    json.dump(todo, open(f"{TMP}/sub_{w.g}.json", "w"))
    race_on = RACE["on"] and thr is not None and np.isfinite(thr)
    partial = f"{TMP}/part_{w.g}.json"; stopf = f"{TMP}/stop_{w.g}"
    for p in (partial, stopf, partial + ".seen"):
        try: os.remove(p)
        except OSError: pass
    st = dict(stopped=False, ub=None, scored={i: v for i, v in s_obs.items()})
    def poll():
        if not race_on: return
        try: P_ = json.load(open(partial))
        except Exception: return
        for o in (P_ if not st["stopped"] else []):
            if o["id"] not in st["scored"]: st["scored"][o["id"]] = int(ok_item(safe_score(FITBY[o["id"]], o["output"], o["finish_reason"])))
        if not st["stopped"] and len(st["scored"]) >= RACE["min_frac"] * len(SUB_ITEMS) and len(st["scored"]) < len(SUB_ITEMS):
            ub, _ = irt_fitness(st["scored"], up=RACE["up"])
            if ub < thr:
                open(stopf, "w").write("stop"); st["stopped"] = True; st["ub"] = ub
                log(f"RACE gpu{w.g}: stop after {len(st['scored'])}/{len(SUB_ITEMS)} items, optimistic {ub:.4f} < threshold {thr:.4f}")
        with open(partial + ".seen.tmp", "w") as f: f.write(str(len(P_)))
        os.replace(partial + ".seen.tmp", partial + ".seen")
    kw = dict(partial=partial, stop=stopf) if race_on else {}
    rg = w.call(T_GEN_SUB, poll=poll, op="gen", items=f"{TMP}/sub_{w.g}.json", max_tokens=FIT_MAXTOK, out=f"{TMP}/o_{w.g}.json", **kw)
    if rg.get("race_error"):
        RACE["on"] = False; log("RACE path failed -> racing disabled, plain generate used:", rg["race_error"][:1500])
    outs = list(have.values()) + json.load(open(f"{TMP}/o_{w.g}.json"))
    sc = {o["id"]: safe_score(FITBY[o["id"]], o["output"], o["finish_reason"]) for o in outs if o["finish_reason"] != "aborted"}
    obs = {i: int(ok_item(s)) for i, s in sc.items()}
    fit, det = irt_fitness(obs)
    n_ab = sum(1 for o in outs if o["finish_reason"] == "aborted")
    trunc, tok = task_stats(outs, FITBY)
    with gzip.open(f"{FITD}/{tag}.json.gz", "wt") as f:
        json.dump(dict(key=key, kind="sub", outputs=outs, obs=obs, scores={k: dict(correct=v["correct"], truncated=v["truncated"]) for k, v in sc.items()}), f)
    rec = dict(base, fitness=fit, fitness_kind="irt", screened=False, irt=det, acc={t: det[t]["obs_acc"] for t in TASKS}, raced=n_ab > 0, n_aborted=n_ab,
               race_ub=st["ub"], race_thr=thr, full_fitness=None, trunc=trunc, mean_tokens=tok, t_gen=round(rg["secs"], 1),
               gen_tokens=rg["gen_tokens"] + s_tok, t_total=round(time.time() - t0, 1), elapsed_h=round((time.time() - T00) / 3600, 3))
    if not rec["raced"]: SUB_T.append(rec["t_total"])
    return rec
def complete_full(rec, w):
    """generate the search items not observed in the subset run -> exact 200-item fitness."""
    t0 = time.time()
    d = json.load(gzip.open(f"{FITD}/{rec['tag']}.json.gz", "rt"))
    have = {o["id"]: o for o in d["outputs"] if o["finish_reason"] != "aborted"}
    todo = ordered([i for i in SEARCH if i["id"] not in have])
    json.dump(todo, open(f"{TMP}/comp_{w.g}.json", "w"))
    ph = dfslib.decode(rec["geno"]); assert ph["key"] == rec["lkey"]
    t_swap = load(w, ph)
    rg = _gen(w, f"{TMP}/comp_{w.g}.json", T_GEN_FULL, f"{TMP}/oc_{w.g}.json")
    outs = list(have.values()) + json.load(open(f"{TMP}/oc_{w.g}.json"))
    assert sorted(o["id"] for o in outs) == sorted(FITBY), "completion does not cover the search items"
    sc = {o["id"]: safe_score(FITBY[o["id"]], o["output"], o["finish_reason"]) for o in outs}
    acc = {t: sum(1 for i in SIDS[t] if ok_item(sc[i])) / NT[t] for t in TASKS}
    trunc, tok = task_stats(outs, FITBY)
    with gzip.open(f"{FITD}/{rec['tag']}_full.json.gz", "wt") as f:
        json.dump(dict(key=rec["lkey"], kind="full", outputs=outs, scores={k: dict(correct=v["correct"], truncated=v["truncated"]) for k, v in sc.items()}), f)
    upd = dict(lkey=rec["lkey"], full_fitness=float(np.mean([acc[t] for t in TASKS])), full_acc=acc, full_trunc=trunc, full_mean_tokens=tok,
               full_gen_tokens=sum(o["completion_tokens"] for o in outs), t_complete=round(time.time() - t0, 1), t_complete_swap=round(t_swap, 1),
               completed_h=round((time.time() - T00) / 3600, 3))
    COMP_T.append(upd["t_complete"])
    return upd
def evaluate_val(ph, w, meta):
    """validation set (150 items); val_fitness = mean of the 3 task accuracies."""
    t0 = time.time()
    t_swap = load(w, ph)
    rg = _gen(w, f"{TMP}/val.json", T_GEN_FULL, f"{TMP}/ov_{w.g}.json")
    outs = json.load(open(f"{TMP}/ov_{w.g}.json"))
    assert sorted(o["id"] for o in outs) == sorted(VALBY)
    sc = {o["id"]: safe_score(VALBY[o["id"]], o["output"], o["finish_reason"]) for o in outs}
    acc = {t: sum(1 for i in VALI if i["bench"] == t and ok_item(sc[i["id"]])) / NV[t] for t in TASKS}
    trunc, tok = task_stats(outs, VALBY)
    tag = tag_of(ph["key"])
    with gzip.open(f"{VALD}/val_{tag}.json.gz", "wt") as f:
        json.dump(dict(key=ph["key"], kind="val", outputs=outs, scores={k: dict(correct=v["correct"], truncated=v["truncated"]) for k, v in sc.items()}), f)
    rec = dict(lkey=ph["key"], tag=tag, val_fitness=float(np.mean([acc[t] for t in TASKS])), val_acc=acc, val_trunc=trunc, val_mean_tokens=tok,
               val_gen_tokens=rg["gen_tokens"], t_total=round(time.time() - t0, 1), t_swap=round(t_swap, 1), gpu=w.g, failed=False,
               elapsed_h=round((time.time() - T00) / 3600, 3), summary=dfslib.summary(ph), **meta)
    VAL_T.append(rec["t_total"])
    return rec

# ---------------------------------------------------------------- budget (priors = v2 measurements x SLOW: 40-layer engine, ~25% more compute + smaller KV cache)
SLOW = 1.3
SAVE_MARGIN = 420 * TS
T_FULL_PRIOR = 2400.0 * TS * SLOW
FINAL_TOKS_PER_S = 150.0 / SLOW              # per GPU (v2/r2 final: 150 tok/s on the 32-layer engine)
K_VAL = 4; VAL_MAX = 6
P = 10; ROUND_LEN = 3; MAX_ITER = int(MOCK.get("max_iter", 30)); COMP_MARGIN = 0.02
SCREEN = {"on": True}
SMOKE_PRIOR = 120.0   # s, smoke screen (16 items @1024) per candidate, 40-layer engine
def t_sub():
    """subset eval incl. swap + smoke: measured (non-raced) if >= 2, else the v2 prior (real seconds, 32-layer engine) x SLOW."""
    if len(SUB_T) >= 2: return 1.1 * float(np.mean(SUB_T[-12:]))
    return 1.1 * (IRT.get("t_sub_prior", 700.0) * SLOW + SMOKE_PRIOR) * TS
def t_comp():
    if COMP_T: return 1.1 * float(np.mean(COMP_T[-6:]))
    return 1.1 * IRT.get("t_comp_prior", 1150.0) * SLOW * TS
def fulls(dfs_only=False):
    return sorted([r for r in cache.values() if r.get("full_fitness") is not None and not r.get("failed") and not (dfs_only and r.get("from_v2"))],
                  key=lambda r: (-r["full_fitness"], r["lkey"]))
def t_val():
    """val time: measured if any, else 0.72 x the measured full-200 cost (subset + completion) of the current DFS top-K."""
    if VAL_T: return 1.1 * float(np.mean(VAL_T[-4:]))
    c = [r["t_total"] + (r.get("t_complete") or 0.0) for r in fulls(True)[:K_VAL] if r.get("t_total")]
    return 1.1 * 0.72 * (float(np.mean(c)) if c else T_FULL_PRIOR)
def final_est():
    toks = [r.get("full_gen_tokens") or 0 for r in fulls(True)[:K_VAL]] or [0]
    tok = max(toks) if max(toks) > 0 else 330000
    return (1.3 * tok / 2 / FINAL_TOKS_PER_S + 450) * TS
def n_alive(): return sum(1 for w in workers if not w.dead and w.alive())
def lanes(): return max(1, n_alive())
def n_val_pending(): return min(K_VAL, max(1, len([r for r in fulls(True) if r["lkey"] not in vcache])))
def val_reserve(): return math.ceil(n_val_pending() / lanes()) * t_val()
def end_deadline(): return T00 + HARD_H * 3600 - SAVE_MARGIN - max(final_est(), 1900 * TS * SLOW)
def evo_deadline(): return end_deadline() - val_reserve()
def remaining(): return max(0.0, evo_deadline() - time.time())
def iter_cost(P_): return math.ceil(P_ / lanes()) * t_sub() + t_comp()

STOP = {"flag": False}
def run_jobs(jobs, do, on_fail, est, deadline=evo_deadline):
    """jobs: payloads, do(payload, w) -> record, on_fail(payload, errs, w) -> record or None; est(j) seconds for job j.
    Runs on the live GPUs; a failing job restarts its engine and is retried once.  (as v2)"""
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
                STOP["flag"] = True; log("time budget: skipping job", j); continue
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
    for w in workers:
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
def _fail_rec(ph, geno, errs, w, meta):
    return dict(lkey=ph["key"], geno=[float(v) for v in geno], tag=None, fitness=0.0, fitness_kind="failed", full_fitness=None, failed=True,
                errors=errs, gpu=w.g, elapsed_h=round((time.time() - T00) / 3600, 3), **phmeta(ph), **meta)
def _store(rec, msg):
    with log_lock:
        cache[rec["lkey"]] = rec; append(LOG, rec); log(msg(rec))
        status(stage=STATUS.get("stage"), n_evals=len(cache))
def eval_batch(X, meta_list, thr=None):
    """genotypes -> value array (full fitness if known, else IRT estimate; screened = their 16-item estimate; nan = not evaluated).
    Identical phenotypes are evaluated once (cache by phenotype key)."""
    PH = [dfslib.decode(x) for x in X]
    F = np.full(len(X), np.nan); todo = []
    for j in range(len(X)):
        k = PH[j]["key"]
        if k in cache: F[j] = value(cache[k])
        elif not any(PH[i]["key"] == k for i in todo): todo.append(j)
    def do(j, w):
        rec = evaluate_sub(PH[j], X[j], w, meta_list[j], thr)
        _store(rec, lambda r: f"[{r.get('round')}/{r.get('iter')} {r.get('kind')}] n={len(cache)} gpu{w.g} {r['fitness_kind']}={r['fitness']:.4f} "
                              f"{r['summary']} smoke={r['smoke']['correct']}/{r['smoke']['finished']}/{r['smoke']['n']} raced={r.get('raced')}({r.get('n_aborted')}) "
                              f"swap {r['t_swap']}s smoke {r['t_smoke']}s total {r['t_total']}s tok {r['gen_tokens']}")
        mock_kill(f"eval:{len(cache)}")
        return rec
    def fail(j, errs, w):
        rec = _fail_rec(PH[j], X[j], errs, w, meta_list[j])
        with log_lock: cache[rec["lkey"]] = rec; append(LOG, rec)
        return rec
    run_jobs(todo, do, fail, lambda j: t_sub())
    for j in range(len(X)):
        k = PH[j]["key"]
        if k in cache: F[j] = value(cache[k])
    return F
def store_val(rec):
    with log_lock:
        vcache[rec["lkey"]] = rec; append(VLOG, rec)
        log(f"VAL {rec.get('kind')} gpu{rec.get('gpu')} val_fit={rec.get('val_fitness')} acc={rec.get('val_acc')} trunc={rec.get('val_trunc')} {rec.get('t_total')}s {rec.get('summary')}")
    mock_kill(f"val:{len(vcache)}")
def complete_batch(recs):
    def do(r, w):
        upd = complete_full(r, w); upd["completion_tag"] = r.get("completion_tag")
        with log_lock:
            cache[r["lkey"]].update(upd); append(LOG, cache[r["lkey"]])
            log(f"COMPLETE gpu{w.g} {r.get('summary')} irt_est {r['fitness']:.4f} -> full {upd['full_fitness']:.4f} acc {upd['full_acc']} tok {upd['full_gen_tokens']} {upd['t_complete']}s")
        return upd
    return run_jobs(recs, do, lambda r, errs, w: None, lambda j: t_comp())

# ---------------------------------------------------------------- EvoGM generator (as v1/v2; dim 57)
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
def novel(new, taken, seen):
    """make generated phenotypes new: a genotype whose phenotype was evaluated before this iteration (seen; records of the current
    iteration from a killed session do not count, so a resumed iteration regenerates the same batch) or is taken in this batch gets its least
    confident not-yet-flipped indicator flipped (score -> -0.05*sign), repeated up to 6 times; deterministic.
    -> (genotypes, number of flips, number of genuinely new phenotypes)"""
    out = []; nrep = 0; nnew = 0
    for x in new:
        x = np.array(x, dtype=np.float32); flipped = set()
        for k in range(7):
            key = dfslib.decode(x)["key"]
            if key not in seen and key not in taken: nnew += 1; break
            if k == 6: break
            s = x[:dfslib.NBITS]
            t = next(int(i) for i in np.argsort(np.abs(s), kind="stable") if int(i) not in flipped)
            x[t] = -0.05 if s[t] > 0 else 0.05; flipped.add(t); nrep += 1
        taken.add(dfslib.decode(x)["key"]); out.append(x)
    return np.array(out, dtype=np.float32), nrep, nnew
STATE = f"{W}/dfs_state.pt"
def save_state(S):
    S = dict(S); S["v"] = "dfs1"; S["np_rng"] = np.random.get_state(); S["torch_rng"] = torch.get_rng_state()
    S["irt"] = dict(IRT_RES) if IRT_RES else None
    torch.save(S, STATE + ".tmp"); os.replace(STATE + ".tmp", STATE)
    log("state saved: next", S["next"], "evo_done", S.get("evo_done", False))
    mock_kill(f"state:{S['next']}" + (":done" if S.get("evo_done") else ""))
def load_state():
    best = None
    for p in find_all(f"{INP}/**/dfs_state.pt"):
        try:
            S = torch.load(p, weights_only=False)
            if S.get("v") != "dfs1": log("resume: ignoring state of another version", p); continue
            if best is None or (bool(S.get("evo_done")), S["next"]) > (bool(best.get("evo_done")), best["next"]): best, bp = S, p
            log("resume: found state", p, "next", S["next"], "evo_done", S.get("evo_done"))
        except Exception as e: log("resume: unreadable state", p, repr(e)[:300])
    if best is not None: shutil.copy(bp, STATE)
    return best
def vals(G): return np.array([value(cache.get(dfslib.decode(g)["key"])) for g in G], dtype=float)
def run_eval(G, it, kinds, thr=None):
    meta = [dict(round=1 if it == 0 else (it - 1) // ROUND_LEN + 1, iter=it, kind=kinds[j]) for j in range(len(G))]
    return eval_batch(G, meta, thr)
def best_full(): return fulls()[0]
def maybe_complete(pop, tag):
    """completion guard (as v2): complete (all 200 items) up to one candidate per GPU whose IRT estimate is within COMP_MARGIN of the
    best full fitness.  Candidates already completed under this tag count as picked (so a resumed session repeats the same picks)."""
    here = [r for r in cache.values() if r.get("completion_tag") == tag]
    other = [r for r in fulls() if r.get("completion_tag") != tag]
    bf = other[0]["full_fitness"] if other else best_full()["full_fitness"]
    cands = list(here)
    for g in pop:
        r = cache.get(dfslib.decode(g)["key"])
        if r is None or r.get("failed") or r.get("screened") or r.get("full_fitness") is not None or r.get("raced"): continue
        if all(c["lkey"] != r["lkey"] for c in cands): cands.append(r)
    cands.sort(key=lambda r: (-r["fitness"], r["lkey"]))
    if not cands or cands[0]["fitness"] < bf - COMP_MARGIN:
        log(f"completion {tag}: none (best estimate {cands[0]['fitness'] if cands else None} vs best full {bf:.4f})"); return
    pick = cands[:max(1, min(2, lanes()))]
    todo = [r for r in pick if r.get("full_fitness") is None]
    log(f"completion {tag}:", [(r["summary"], round(r["fitness"], 4)) for r in pick], "best full", round(bf, 4), "to run", len(todo))
    for r in todo: r["completion_tag"] = tag
    complete_batch(todo)

# ---------------------------------------------------------------- resume state / IRT in use
S = load_state()
if S is not None and S.get("irt"):
    set_irt(S["irt"])
elif PREV.get("irt") and not MOCK.get("fit_irt"):
    irt0 = dict(PREV["irt"]); irt0["source"] = "v2_irt_refit_r1"; irt0["v2_fit_round"] = irt0.get("fit_round"); irt0["v2_refits"] = irt0.pop("refits", None)
    irt0["fit_round"] = 1; irt0["refits"] = []
    json.dump(irt0, open(f"{W}/dfs_irt.json", "w"), indent=0); set_irt(irt0)
else:
    status(stage="irt"); set_irt(irt_stage(rnd=1))
HIST = []; evo_done = False; it0 = 1; ctx = None
if S is not None:
    it0 = S["next"]; HIST = list(S["HIST"]); evo_done = bool(S.get("evo_done"))
    np.random.set_state(S["np_rng"]); torch.set_rng_state(S["torch_rng"])
    log("RESUMING at iteration", it0, "evo_done", evo_done, "irt fit_round", IRT_RES.get("fit_round"), "subset", len(SUB_ITEMS))

# ---------------------------------------------------------------- session sanity of the padded engine (every session, ~3 min)
#   (a) plain B11 (pads at the end) smoke vs the v2 B11 outputs' correctness on the same items  -> engine / shard / padding sanity
#   (b) plain B11 with the 2 pad blocks in the MIDDLE of the path: must give the same outputs as (a) (identity layers add exactly 0)
status(stage="calibration")
def plan_mid():
    p = dfslib.plan_of(PH_B11); n = len(PH_B11["path"]) * dfslib.BL; h = 4 * dfslib.BL
    pads = [[dfslib.PAD_DONOR, j % dfslib.NL, 0.0] for j in range(h, h + (dfslib.L_MAX - n))]
    mid = p[:h] + pads + p[h:n]
    for j, (d, src, f) in enumerate(mid): assert src % dfslib.BL == j % dfslib.BL
    return mid
CAL = {}
def do_cal(job, w):
    name, plan = job
    t = w.call(T_LOAD, op="load_dfs", shards=SH, plan=plan)["t"]
    outs, obs, cor, fin, secs, tok = smoke(w, name)
    CAL[name] = dict(outputs={o["id"]: o["output"] for o in outs}, correct=cor, finished=fin, t_swap=round(t, 1), t_smoke=round(secs, 1), tokens=tok, gpu=w.g)
    log("CAL", name, "gpu", w.g, "smoke correct", cor, "finished", fin, "swap", round(t, 1), "s smoke", round(secs, 1), "s tokens", tok)
    return CAL[name]
run_jobs([("b11_end", dfslib.plan_of(PH_B11)), ("b11_mid", plan_mid())], do_cal, lambda j, e, w: None, lambda j: 600 * TS, deadline=end_deadline)
ref = IRT["smoke_ref_b11"]
cal_res = dict(ref_v2_b11=ref, session_start_h=round((time.time() - T00) / 3600, 3))
for k in ("b11_end", "b11_mid"):
    if k in CAL: cal_res[k] = {x: CAL[k][x] for x in ("correct", "finished", "t_swap", "t_smoke", "tokens", "gpu")}
ok_engine = "b11_end" in CAL and CAL["b11_end"]["correct"] >= ref["correct"] - 4 and CAL["b11_end"]["finished"] >= ref["finished"] - 5 \
    and CAL["b11_end"]["finished"] >= len(SMOKE) // 2 and CAL["b11_end"]["correct"] >= SMOKE_MIN_CORRECT
if "b11_end" in CAL and "b11_mid" in CAL:
    same = sum(1 for i in CAL["b11_end"]["outputs"] if CAL["b11_end"]["outputs"][i] == CAL["b11_mid"]["outputs"].get(i))
    cal_res["mid_vs_end_identical_outputs"] = same
    ok_pad = same >= len(SMOKE) // 2 or abs(CAL["b11_mid"]["correct"] - CAL["b11_end"]["correct"]) <= 3
else:
    ok_pad = False
cal_res.update(ok_engine=ok_engine, ok_pad=ok_pad)
prevcal = []
for p in find_all(f"{INP}/**/dfs_calib.json"):
    try: prevcal = json.load(open(p))
    except Exception: pass
json.dump(prevcal + [cal_res], open(f"{W}/dfs_calib.json", "w"), indent=1)
log("CALIBRATION", cal_res)
if not (ok_engine and ok_pad):
    for w in workers: w.kill()
    status(stage="aborted_sanity", calib=cal_res, hours=(time.time() - T00) / 3600)
    log("ABORT: padded-engine sanity failed (see dfs_calib.json) -> no search, quota saved"); sys.exit(0)

# ---------------------------------------------------------------- round 1: initial population (subset IRT + smoke screen; plain B11 = v2 result)
G = dfslib.geno_from_slots
R1, B11d, B10d = 0, 1, 2
def wlog(d1, d2, w):   # W coords with W[d1,d2] = w (others 1)
    x = np.zeros((dfslib.ND, dfslib.ND)); x[d1, d2] = math.log2(w) / dfslib.W_LOG2; return x
_rs = np.random.RandomState(20261117)
def local():
    x = G([(0, b, R1) for b in range(8)]) + np.concatenate([_rs.normal(0, 0.3, dfslib.NBITS), _rs.normal(0, 0.1, dfslib.NW)])
    return np.clip(x, -1, 1)
INIT = [   # priority order; plain_b11 is cached (v2), the rest are new models
    ("plain_r1i1", G([(0, b, R1) for b in range(8)])),                                   # r1i1 layers + B11 embedding/norm (not the v2 r1i1 model)
    ("plain_b11", G([(0, b, B11d) for b in range(8)])),
    ("r1i1_lo_b11_hi", G([(0, b, R1) for b in range(4)] + [(0, b, B11d) for b in range(4, 8)])),
    ("b11_lo_r1i1_hi", G([(0, b, B11d) for b in range(4)] + [(0, b, R1) for b in range(4, 8)])),
    ("r1i1_b11_even", G([(0, b, B11d if b % 2 == 0 else R1) for b in range(8)])),
    ("r1i1_b11_odd", G([(0, b, B11d if b % 2 == 1 else R1) for b in range(8)])),
    ("r1i1_plus_b11blk4", G([(0, b, R1) for b in range(8)] + [(0, 4, B11d)])),             # 9 blocks: r1i1 0-4, b11 4, r1i1 5-7
    ("r1i1_dup45", G([(0, b, R1) for b in range(6)] + [(1, b, R1) for b in range(4, 8)])),  # 10 blocks: r1i1 0-5, 4-7
    ("r1i1_drop6", G([(0, b, R1) for b in range(8) if b != 6])),                          # 7 blocks
    ("r1i1_b10_tail", G([(0, b, R1) for b in range(6)] + [(0, b, B10d) for b in (6, 7)])),
    ("r1i1_W0.92", G([(0, b, R1) for b in range(8)], w=wlog(R1, R1, 2 ** -0.125))),       # stream x0.917 at every boundary
    ("r1i1_local1", local()),                                                              # random local moves (few bit flips, small W)
    ("r1i1_local2", local()),
]
X1 = np.array([g for _, g in INIT], dtype=np.float32)
assert dfslib.decode(X1[1])["key"] == B11_KEY and dfslib.is_plain(dfslib.decode(X1[0]), 0)
plan = dict(HARD_H=HARD_H, P=P, round_len=ROUND_LEN, max_iter=MAX_ITER, init=[(k, dfslib.summary(dfslib.decode(g))) for k, g in INIT], k_val=K_VAL,
            l_max=dfslib.L_MAX, blocks=[dfslib.MIN_BLOCKS, dfslib.MAX_BLOCKS], w_range=[2 ** -dfslib.W_LOG2, 2 ** dfslib.W_LOG2], c_clamp=[dfslib.C_MIN, dfslib.C_MAX],
            slot_order="rep -> block -> donor (r1i1, b11, b10)", smoke=dict(n=len(SMOKE), max_tokens=SMOKE_MAXTOK, min_correct=SMOKE_MIN_CORRECT, min_finished=SMOKE_MIN_FIN),
            setup_h=(time.time() - T00) / 3600, lanes=lanes(), evo_remaining_h=remaining() / 3600, t_sub_est=t_sub(), t_comp_est=t_comp(),
            val_reserve_h=val_reserve() / 3600, final_reserve_h=max(final_est(), 1900 * TS * SLOW) / 3600, race=RACE, resume=S is not None,
            bfcl_floor=BASE_VAL["val_acc"]["bfcl"] - BFCL_TOL)
json.dump(plan, open(f"{W}/plan.json", "w"), indent=1, default=str); log("PLAN", plan)
if S is None:
    status(stage="round1")
    thr0 = V2B11["evo"]["full_fitness"] - 0.05   # init racing: stop candidates clearly below Baphomet-1.1
    f1 = run_eval(X1, 0, [k for k, _ in INIT], thr=thr0)
    ok = ~np.isnan(f1)
    HIST.append(dict(round=1, iter=0, n=int(ok.sum()), best=float(np.nanmax(f1)) if ok.any() else None,
                     values={INIT[k][0]: (round(float(f1[k]), 4) if ok[k] else None) for k in range(len(INIT))}, elapsed_h=(time.time() - T00) / 3600))
    json.dump(dict(hist=HIST), open(f"{W}/history.json", "w"), indent=1); log("HIST", HIST[-1])
    if ok.sum() < 4:
        log(f"round 1: only {int(ok.sum())} evaluated members -> no evolution; going to validation"); evo_done = True
    else:
        STOP["flag"] = False
        hist_pop = X1[ok]; pv = f1[ok]
        pop = hist_pop[np.argsort(-pv, kind="stable")[:P]]
        if not STOP["flag"]: maybe_complete(pop, "init")
        dual = Dual(DIM); opt = torch.optim.Adam(dual.parameters(), lr=GEN_LR)
        ctx = dict(pop=pop, hist_pop=hist_pop, dual=dual, opt=opt)
        save_state(dict(next=1, HIST=HIST, pop=pop, hist_pop=hist_pop, dual=dual.state_dict(), opt=opt.state_dict()))

# ---------------------------------------------------------------- EvoGM iterations (subset IRT + smoke + racing + completion); IRT refit every ROUND_LEN iterations
status(stage="evolve")
if not evo_done and ctx is None and S is not None and "pop" in S:
    dual = Dual(DIM); dual.load_state_dict(S["dual"]); opt = torch.optim.Adam(dual.parameters(), lr=GEN_LR); opt.load_state_dict(S["opt"])
    torch.set_rng_state(S["torch_rng"])
    ctx = dict(pop=S["pop"], hist_pop=S["hist_pop"], dual=dual, opt=opt)
it = it0
while not evo_done and ctx is not None and it <= MAX_ITER and not STOP["flag"]:
    pop, hist_pop, dual, opt = ctx["pop"], ctx["hist_pop"], ctx["dual"], ctx["opt"]
    if it > 1 and (it - 1) % ROUND_LEN == 0:   # round boundary: IRT refit (v2 rows + DFS completions), rescore subset records
        status(stage=f"irt_refit_r{(it - 1) // ROUND_LEN}")
        refit_irt((it - 1) // ROUND_LEN)
        mock_kill(f"refit:{(it - 1) // ROUND_LEN}")
    rem, ic = remaining(), iter_cost(P)
    if rem < ic:
        log(f"budget: evolution ends before iteration {it} (remaining {rem/3600:.2f} h, iteration {ic/3600:.2f} h)"); break
    hv = vals(hist_pop); hm = ~np.isnan(hv); hp, hf = hist_pop[hm], hv[hm]
    pv = vals(pop); order = np.argsort(-pv, kind="stable")[:P]; pop, pv = pop[order], pv[order]
    idx = np.argsort(-hf, kind="stable"); nw = max(1, int(len(hp) * WIN_PORTION))
    winners, losers = hp[idx[:nw]], hp[idx[nw:]]
    for ep in range(GEN_EPOCHS): loss = train_epoch(dual, opt, winners, losers)
    with torch.no_grad(): new = dual.lose2win(torch.tensor(pop, dtype=torch.float32)).numpy()
    new = np.clip(new, -1, 1).astype(np.float32)
    seen = {k for k, r in cache.items() if r.get("from_v2") or (r.get("iter") is not None and r["iter"] < it)}
    new, nrep, nnew = novel(new, set(), seen)
    thr = float(pv[-1]) if len(pv) >= P else float(np.nanmin(pv))
    log(f"iteration {it}: generator loss {loss:.4f}, winners {nw}/{len(hp)}, novelty flips {nrep}, new phenotypes {nnew}/{len(new)}, race thr {thr:.4f}, new",
        [dfslib.summary(dfslib.decode(x)) for x in new])
    nf = run_eval(new, it, ["generated"] * len(new), thr)
    hist_pop = np.vstack([hist_pop, new])
    ok = ~np.isnan(nf)
    allp = np.vstack([pop, new[ok]]); av = vals(allp)
    pop = allp[np.argsort(-av, kind="stable")[:P]]
    if not STOP["flag"]: maybe_complete(pop, f"i{it}")
    pv = vals(pop); order = np.argsort(-pv, kind="stable"); pop, pv = pop[order], pv[order]
    bfr = best_full()
    HIST.append(dict(round=(it - 1) // ROUND_LEN + 1, iter=it, best_value=float(pv[0]), best_summary=dfslib.summary(dfslib.decode(pop[0])), pop_mean=float(np.nanmean(pv)),
                     new_mean=float(np.nanmean(nf)) if ok.any() else None, new_best=float(np.nanmax(nf)) if ok.any() else None, n_new=int(ok.sum()),
                     n_screened=int(sum(1 for x in new if (cache.get(dfslib.decode(x)["key"]) or {}).get("screened"))),
                     n_raced=int(sum(1 for x in new if (cache.get(dfslib.decode(x)["key"]) or {}).get("raced"))), novelty_flips=nrep, n_novel=nnew,
                     best_full=bfr["full_fitness"], best_full_summary=bfr.get("summary"), gen_loss=loss, elapsed_h=(time.time() - T00) / 3600))
    json.dump(dict(hist=HIST), open(f"{W}/history.json", "w"), indent=1); log("HIST", HIST[-1])
    ctx.update(pop=pop, hist_pop=hist_pop)
    if STOP["flag"]: break
    it += 1
    save_state(dict(next=it, HIST=HIST, pop=pop, hist_pop=hist_pop, dual=dual.state_dict(), opt=opt.state_dict()))
if not evo_done:
    evo_done = True
    save_state(dict(next=it, HIST=HIST, evo_done=True))
STOP["flag"] = False

# ---------------------------------------------------------------- validation: DFS top-K by full search fitness; plain B11 + the v2 r1i1 model from v2 (cached)
status(stage="validation")
FL = [r for r in fulls(True)]
sel = FL[:K_VAL]
log("validation candidates", [(r.get("kind"), r.get("summary"), round(r["full_fitness"], 4)) for r in sel])
vjobs = [r for r in sel if r["lkey"] not in vcache or vcache[r["lkey"]].get("failed")]
def dov(r, w):
    meta = dict(kind=r.get("kind"), round=r.get("round"), iter=r.get("iter"), full_fitness=r.get("full_fitness"))
    rec = evaluate_val(dfslib.decode(r["geno"]), w, meta); store_val(rec); return rec
def failv(r, errs, w):
    rec = dict(lkey=r["lkey"], kind=r.get("kind"), failed=True, errors=errs, gpu=w.g); store_val(rec); return rec
run_jobs(vjobs, dov, failv, lambda j: t_val(), deadline=end_deadline)
floor = BASE_VAL["val_acc"]["bfcl"]
rows = []
def vrow(name, r, v, full, src):
    elig = v["val_acc"]["bfcl"] >= floor - BFCL_TOL - 1e-9
    return dict(lkey=name, kind=src, summary=r.get("summary") if r else None, geno=r.get("geno") if r else None, full_fitness=full, full_acc=(r or {}).get("full_acc"),
                val_fitness=v["val_fitness"], val_acc=v["val_acc"], val_trunc=v.get("val_trunc"), val_mean_tokens=v.get("val_mean_tokens"), eligible=elig)
for r in sel + [cache[B11_KEY]]:
    v = vcache.get(r["lkey"])
    if not v or v.get("failed"): continue
    rows.append(vrow(r["lkey"], r, v, r["full_fitness"], r.get("kind")))
rows.append(vrow("v2:" + V2R1["evo"]["lkey"], dict(summary="v2 r1i1 model (own embedding; not in the DFS space)"), V2R1["val"], V2R1["evo"]["full_fitness"], "v2_r1i1"))
for x in rows:
    if not x["eligible"]: log(f"BFCL FLOOR: {x['kind']} {x['summary']} val BFCL {x['val_acc']['bfcl']:.4f} < base {floor:.4f} - {BFCL_TOL:.4f} -> not eligible")
er = [x for x in rows if x["eligible"]]
winner = max(er, key=lambda x: (x["val_fitness"], x["full_fitness"], x["lkey"])) if er else None
vres = dict(base=BASE_VAL, bfcl_floor=floor, bfcl_tol=BFCL_TOL, rows=sorted(rows, key=lambda x: -x["val_fitness"]), winner=winner,
            rule="winner = max validation fitness among DFS top-K + plain B11 + the v2 r1i1 model whose val BFCL >= base - 2/30; ties -> full search fitness")
json.dump(vres, open(f"{W}/val_result.json", "w"), indent=1, default=str)
log("VALIDATION", [(x["kind"], x["summary"], round(x["full_fitness"], 4), round(x["val_fitness"], 4), x["val_acc"], x["eligible"]) for x in vres["rows"]],
    "WINNER", (winner["kind"], winner["summary"], winner["val_fitness"]) if winner else None)
recs = [r for r in cache.values() if not r.get("from_v2")]
evo_result = dict(n_evals=len(recs), n_screened=sum(1 for r in recs if r.get("screened")), n_subset=sum(1 for r in recs if r.get("fitness_kind") == "irt"),
                  n_raced=sum(1 for r in recs if r.get("raced")), n_failed=sum(1 for r in recs if r.get("failed")),
                  completed=[dict(summary=r.get("summary"), kind=r.get("kind"), iter=r.get("iter"), irt=r["fitness"], full=r["full_fitness"]) for r in recs if r.get("full_fitness") is not None],
                  irt=dict(source=IRT_RES.get("source"), chosen=IRT_RES.get("chosen"), n_cal=IRT_RES.get("n_cal"), fit_round=IRT_RES.get("fit_round"),
                           refits=IRT_RES.get("refits"), refit_failures=IRT_RES.get("refit_failures")),
                  best_full=FL[0] if FL else None, hist=HIST, val=vres, winner=winner, calibration=cal_res, hours=(time.time() - T00) / 3600)
json.dump(evo_result, open(f"{W}/evo_result.json", "w"), indent=1, default=str)
if winner is None or winner["kind"] in ("plain_b11", "v2_r1i1"):
    for w in workers: w.kill()
    why = "no eligible winner" if winner is None else f"winner is {winner['kind']} (its 180-item result exists)"
    status(stage="done", final=f"skipped: {why}", hours=(time.time() - T00) / 3600)
    log("ALL DONE (final skipped:", why, ") hours", (time.time() - T00) / 3600); sys.exit(0)

# ---------------------------------------------------------------- final: winner on the 180-item test set (8192 cap)
WPH = dfslib.decode(winner["geno"]); assert WPH["key"] == winner["lkey"]
status(stage="final", winner=dict(kind=winner["kind"], summary=winner["summary"], val=winner["val_fitness"]))
prev = [p for p in find_all(f"{INP}/**/final_meta.json") if json.load(open(p)).get("lkey") == winner["lkey"]]
if prev and os.path.exists(os.path.join(os.path.dirname(prev[0]), "raw_evogm_dfs.jsonl")) and \
        sum(1 for _ in open(os.path.join(os.path.dirname(prev[0]), "raw_evogm_dfs.jsonl"))) == 180:
    shutil.copy(os.path.join(os.path.dirname(prev[0]), "raw_evogm_dfs.jsonl"), f"{W}/raw_evogm_dfs.jsonl"); shutil.copy(prev[0], f"{W}/final_meta.json")
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
            load(w, WPH)
            t0 = time.time(); r = w.call(T_GEN_FINAL, op="gen", items=f"{TMP}/test_{g}.json", max_tokens=8192, out=f"{W}/raw_dfs_{g}.json")
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
with open(f"{W}/raw_evogm_dfs.jsonl", "w") as f:
    for g in (0, 1):
        p = f"{W}/raw_dfs_{g}.json"
        if not os.path.exists(p): continue
        for o in json.load(open(p)):
            it_ = TEST[o["id"]]
            f.write(json.dumps(dict(id=o["id"], bench=it_["bench"], category=it_["category"], model="evogm_dfs", prompt_tokens=o["prompt_tokens"],
                                    completion_tokens=o["completion_tokens"], finish_reason=o["finish_reason"], output=o["output"])) + "\n")
n = sum(1 for _ in open(f"{W}/raw_evogm_dfs.jsonl"))
quick = {}
for l in open(f"{W}/raw_evogm_dfs.jsonl"):
    o = json.loads(l); s_ = safe_score(TEST[o["id"]], o["output"], o["finish_reason"])
    quick.setdefault(o["bench"], []).append(bool(s_["correct"]))
json.dump(dict(lkey=winner["lkey"], geno=winner["geno"], summary=winner["summary"], plan=dfslib.plan_of(WPH), n=n,
               quick={k: [sum(v), len(v)] for k, v in quick.items()}, fres=fres), open(f"{W}/final_meta.json", "w"), indent=1)
log("FINAL raw lines", n, {k: (sum(v), len(v)) for k, v in quick.items()}, fres)
for w in workers: w.kill()
status(stage="done", final=dict(n=n, quick={k: [sum(v), len(v)] for k, v in quick.items()}, fres=fres), hours=(time.time() - T00) / 3600)
log("ALL DONE hours", (time.time() - T00) / 3600)
