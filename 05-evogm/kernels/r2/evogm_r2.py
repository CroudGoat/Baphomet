# EvoGM rounds 2+ (2026-11-13): resume evogm-full round 1 (evo_state.pt/evo_log.jsonl via kernel_sources), basis shift,
# IRT-subset fitness (40 items, per-task 2PL p-IRT estimate of the 110-item fitness) + racing (abort a candidate's generation
# when its optimistic estimate cannot reach the population's top-P), completion of promising candidates on all 110 items,
# winner = max full-110 fitness; final 180-item test @8192 only if the winner changed. Setup/engine/merge code = hardened v3.
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
import os, sys, json, time, glob, gzip, shutil, subprocess, threading, queue, hashlib, math, traceback, base64
from concurrent.futures import ThreadPoolExecutor
T00 = time.time()
MODE = "full"                                           # "smoke" | "full"  (set per kernel at push time)
HARD_H = 7.7                                             # whole-kernel ceiling for this session
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
WORKER_B64 = "IyBQZXJzaXN0ZW50IHBlci1HUFUgZ2VuZXJhdGlvbiB3b3JrZXIgKGdwdXdvcmtlcjIgKyByYWNpbmcpLiBPbmUgaW4tcHJvY2VzcyB2TExNIDAuMzMuMCBlbmdpbmUsIHdlaWdodHMgc3dhcHBlZCBpbiBwbGFjZQojIChXb3JrZXIucmVsb2FkX3dlaWdodHMgd2l0aCBhIHN0cmVhbWVkIGV4YWN0IGxpbmVhciBtZXJnZSkuIEZpbGUtYmFzZWQgY29tbWFuZCBwcm90b2NvbCAoc2VlIHJ1biBzY3JpcHQpLgojIG9wIGdlbiB3aXRoICJwYXJ0aWFsIisic3RvcCI6IHJlcXVlc3RzIGFyZSBmZWQgdGhyb3VnaCBsbG0ubGxtX2VuZ2luZSAoYWRkX3JlcXVlc3Qvc3RlcCk7IGZpbmlzaGVkIG91dHB1dHMgYXJlIHdyaXR0ZW4gdG8KIyBgcGFydGlhbGAgZXZlcnkgfjggczsgaWYgdGhlIGNvbnRyb2xsZXIgY3JlYXRlcyBgc3RvcGAsIHRoZSB1bmZpbmlzaGVkIHJlcXVlc3RzIGFyZSBhYm9ydGVkIChmaW5pc2hfcmVhc29uICJhYm9ydGVkIikuCiMgQW55IGZhaWx1cmUgb2YgdGhhdCBwYXRoIGZhbGxzIGJhY2sgdG8gcGxhaW4gbGxtLmdlbmVyYXRlIGZvciB0aGUgc2FtZSBjYWxsIChyZXNbInJhY2VfZXJyb3IiXSBzYXlzIHdoeSkuCiMgdXNhZ2U6IENVREFfVklTSUJMRV9ERVZJQ0VTPWcgcHl0aG9uIGdwdXdvcmtlcjMucHkgQkFTRV9ESVIgQ01EX0RJUiBHUFVNRU0gRUFHRVIoMC8xKSBNQVhTRVFTCmltcG9ydCBvcywgc3lzLCBqc29uLCB0aW1lLCB0cmFjZWJhY2sKb3MuZW52aXJvblsiVkxMTV9FTkFCTEVfVjFfTVVMVElQUk9DRVNTSU5HIl0gPSAiMCIKb3MuZW52aXJvbi5zZXRkZWZhdWx0KCJWTExNX0FUVEVOVElPTl9CQUNLRU5EIiwgIlRSSVRPTl9BVFROIikKb3MuZW52aXJvbi5zZXRkZWZhdWx0KCJQWVRPUkNIX0NVREFfQUxMT0NfQ09ORiIsICJleHBhbmRhYmxlX3NlZ21lbnRzOlRydWUiKQpzeXMucGF0aC5pbnNlcnQoMCwgb3MucGF0aC5kaXJuYW1lKG9zLnBhdGguYWJzcGF0aChfX2ZpbGVfXykpKQoKTUFYTEVOID0gMTIyODgKCgpkZWYgX3JlYyhpdCwgbyk6CiAgICBjID0gby5vdXRwdXRzWzBdCiAgICByZXR1cm4gZGljdChpZD1pdFsiaWQiXSwgcHJvbXB0X3Rva2Vucz1sZW4oby5wcm9tcHRfdG9rZW5faWRzIG9yIFtdKSwgY29tcGxldGlvbl90b2tlbnM9bGVuKGMudG9rZW5faWRzKSwKICAgICAgICAgICAgICAgIGZpbmlzaF9yZWFzb249Yy5maW5pc2hfcmVhc29uLCBvdXRwdXQ9Yy50ZXh0KQoKCmRlZiByYWNlX2dlbmVyYXRlKGxsbSwgcHJvbXB0cywgaXRlbXMsIHNwLCBwYXJ0aWFsLCBzdG9wLCB0YWcpOgogICAgZW5nID0gbGxtLmxsbV9lbmdpbmUKICAgIHRyeToKICAgICAgICBmcm9tIHZsbG0uc2FtcGxpbmdfcGFyYW1zIGltcG9ydCBSZXF1ZXN0T3V0cHV0S2luZAogICAgICAgIHNwID0gc3AuY2xvbmUoKTsgc3Aub3V0cHV0X2tpbmQgPSBSZXF1ZXN0T3V0cHV0S2luZC5GSU5BTF9PTkxZCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHBhc3MKICAgIGV4dCA9IFtmInt0YWd9X3trfSIgZm9yIGsgaW4gcmFuZ2UobGVuKHByb21wdHMpKV0KICAgIGlkbWFwID0ge30KICAgIGZvciBrLCAocmlkLCBwKSBpbiBlbnVtZXJhdGUoemlwKGV4dCwgcHJvbXB0cykpOgogICAgICAgIHIgPSBlbmcuYWRkX3JlcXVlc3QocmlkLCBwLCBzcCkKICAgICAgICBpZG1hcFtyaWRdID0gawogICAgICAgIGlmIGlzaW5zdGFuY2Uociwgc3RyKTogaWRtYXBbcl0gPSBrCiAgICBkZWYga2lkeChyaWQpOgogICAgICAgIGlmIHJpZCBpbiBpZG1hcDogcmV0dXJuIGlkbWFwW3JpZF0KICAgICAgICBmb3IgZSBpbiBleHQ6CiAgICAgICAgICAgIGlmIHJpZC5zdGFydHN3aXRoKGUgKyAiLSIpOiByZXR1cm4gaWRtYXBbZV0KICAgICAgICByZXR1cm4gTm9uZQogICAgZG9uZSA9IHt9OyBsYXN0ID0gdGltZS50aW1lKCk7IGFib3J0ZWQgPSBbXQogICAgd2hpbGUgZW5nLmhhc191bmZpbmlzaGVkX3JlcXVlc3RzKCk6CiAgICAgICAgZm9yIG8gaW4gZW5nLnN0ZXAoKToKICAgICAgICAgICAgaWYgZ2V0YXR0cihvLCAiZmluaXNoZWQiLCBGYWxzZSk6CiAgICAgICAgICAgICAgICBrID0ga2lkeChvLnJlcXVlc3RfaWQpCiAgICAgICAgICAgICAgICBpZiBrIGlzIG5vdCBOb25lIGFuZCBrIG5vdCBpbiBhYm9ydGVkOiBkb25lW2tdID0gX3JlYyhpdGVtc1trXSwgbykKICAgICAgICBpZiB0aW1lLnRpbWUoKSAtIGxhc3QgPiA4IG9yIG5vdCBlbmcuaGFzX3VuZmluaXNoZWRfcmVxdWVzdHMoKToKICAgICAgICAgICAgbGFzdCA9IHRpbWUudGltZSgpCiAgICAgICAgICAgIGpzb24uZHVtcChbZG9uZVtrXSBmb3IgayBpbiBzb3J0ZWQoZG9uZSldLCBvcGVuKHBhcnRpYWwgKyAiLnRtcCIsICJ3IikpOyBvcy5yZXBsYWNlKHBhcnRpYWwgKyAiLnRtcCIsIHBhcnRpYWwpCiAgICAgICAgICAgIGlmIG9zLnBhdGguZXhpc3RzKHN0b3ApIGFuZCBlbmcuaGFzX3VuZmluaXNoZWRfcmVxdWVzdHMoKToKICAgICAgICAgICAgICAgIGFib3J0ZWQgPSBbayBmb3IgayBpbiByYW5nZShsZW4oZXh0KSkgaWYgayBub3QgaW4gZG9uZV0KICAgICAgICAgICAgICAgIGVuZy5hYm9ydF9yZXF1ZXN0KFtleHRba10gZm9yIGsgaW4gYWJvcnRlZF0pCiAgICAgICAgICAgICAgICBmb3IgXyBpbiByYW5nZSgyMDApOiAgICMgZHJhaW4gd2hhdGV2ZXIgdGhlIGVuZ2luZSBzdGlsbCByZXBvcnRzCiAgICAgICAgICAgICAgICAgICAgaWYgbm90IGVuZy5oYXNfdW5maW5pc2hlZF9yZXF1ZXN0cygpOiBicmVhawogICAgICAgICAgICAgICAgICAgIGVuZy5zdGVwKCkKICAgICAgICAgICAgICAgIGJyZWFrCiAgICByZWNzID0gW10KICAgIGZvciBrLCBpdCBpbiBlbnVtZXJhdGUoaXRlbXMpOgogICAgICAgIGlmIGsgaW4gZG9uZTogcmVjcy5hcHBlbmQoZG9uZVtrXSkKICAgICAgICBlbHNlOiByZWNzLmFwcGVuZChkaWN0KGlkPWl0WyJpZCJdLCBwcm9tcHRfdG9rZW5zPTAsIGNvbXBsZXRpb25fdG9rZW5zPTAsIGZpbmlzaF9yZWFzb249ImFib3J0ZWQiLCBvdXRwdXQ9IiIpKQogICAgcmV0dXJuIHJlY3MsIGxlbihhYm9ydGVkKQoKCmRlZiBtYWluKCk6CiAgICBiYXNlLCBjZGlyLCBncHVtZW0sIGVhZ2VyLCBtYXhzZXFzID0gc3lzLmFyZ3ZbMV0sIHN5cy5hcmd2WzJdLCBmbG9hdChzeXMuYXJndlszXSksIHN5cy5hcmd2WzRdID09ICIxIiwgaW50KHN5cy5hcmd2WzVdKQogICAgZnJvbSB2bGxtIGltcG9ydCBMTE0sIFNhbXBsaW5nUGFyYW1zCiAgICBmcm9tIHRyYW5zZm9ybWVycyBpbXBvcnQgQXV0b1Rva2VuaXplcgogICAgaW1wb3J0IHRvcmNoCiAgICB0MCA9IHRpbWUudGltZSgpCiAgICBsbG0gPSBMTE0oYmFzZSwgZHR5cGU9ImZsb2F0MTYiLCBtYXhfbW9kZWxfbGVuPU1BWExFTiwgZ3B1X21lbW9yeV91dGlsaXphdGlvbj1ncHVtZW0sIGVuZm9yY2VfZWFnZXI9ZWFnZXIsCiAgICAgICAgICAgICAgZW5hYmxlX3ByZWZpeF9jYWNoaW5nPUZhbHNlLCBzZWVkPTAsIG1heF9udW1fc2Vxcz1tYXhzZXFzLCBsaW1pdF9tbV9wZXJfcHJvbXB0PXsiaW1hZ2UiOiAwLCAidmlkZW8iOiAwfSkKICAgIHRvayA9IEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKGJhc2UpCiAgICB0X2luaXQgPSB0aW1lLnRpbWUoKSAtIHQwCgogICAgZGVmIHB0cnMobW9kZWwpOgogICAgICAgIHJldHVybiB7bjogdC5kYXRhX3B0cigpIGZvciBuLCB0IGluIGxpc3QobW9kZWwubmFtZWRfcGFyYW1ldGVycygpKSArIGxpc3QobW9kZWwubmFtZWRfYnVmZmVycygpKX0KICAgIGt2ID0gTm9uZQogICAgdHJ5OgogICAgICAgIGt2ID0gbGxtLmxsbV9lbmdpbmUudmxsbV9jb25maWcuY2FjaGVfY29uZmlnLm51bV9ncHVfYmxvY2tzCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHBhc3MKICAgIGpzb24uZHVtcChkaWN0KG9rPVRydWUsIHRfaW5pdD10X2luaXQsIGVhZ2VyPWVhZ2VyLCBncHVtZW09Z3B1bWVtLCBudW1fZ3B1X2Jsb2Nrcz1rdiwKICAgICAgICAgICAgICAgICAgIGZyZWVfZ2I9dG9yY2guY3VkYS5tZW1fZ2V0X2luZm8oKVswXSAvIDIqKjMwKSwKICAgICAgICAgICAgICBvcGVuKGYie2NkaXJ9L3JlYWR5LnRtcCIsICJ3IikpCiAgICBvcy5yZW5hbWUoZiJ7Y2Rpcn0vcmVhZHkudG1wIiwgZiJ7Y2Rpcn0vcmVhZHkuanNvbiIpCiAgICBuID0gMAogICAgd2hpbGUgVHJ1ZToKICAgICAgICBwID0gZiJ7Y2Rpcn0vY21kX3tufS5qc29uIgogICAgICAgIGlmIG5vdCBvcy5wYXRoLmV4aXN0cyhwKToKICAgICAgICAgICAgdGltZS5zbGVlcCgwLjMpOyBjb250aW51ZQogICAgICAgIGNtZCA9IGpzb24ubG9hZChvcGVuKHApKTsgdDEgPSB0aW1lLnRpbWUoKTsgcmVzID0gZGljdChvaz1UcnVlKQogICAgICAgIHRyeToKICAgICAgICAgICAgb3AgPSBjbWRbIm9wIl0KICAgICAgICAgICAgaWYgb3AgPT0gImV4aXQiOgogICAgICAgICAgICAgICAgcmVzWyJieWUiXSA9IFRydWUKICAgICAgICAgICAgZWxpZiBvcCA9PSAibG9hZF9tZXJnZSI6CiAgICAgICAgICAgICAgICBiZWZvcmUgPSBsbG0uYXBwbHlfbW9kZWwocHRycylbMF0KICAgICAgICAgICAgICAgIGRlZiBfZm4od29ya2VyLCBiYXNlLCB0dnMsIGxhbXMpOgogICAgICAgICAgICAgICAgICAgIGltcG9ydCB0b3JjaCwgbGlubWVyZ2UKICAgICAgICAgICAgICAgICAgICBrZXkgPSAoYmFzZSwgdHVwbGUodHZzKSkKICAgICAgICAgICAgICAgICAgICBpZiBnZXRhdHRyKHdvcmtlciwgIl9tZXJnZXJfa2V5IiwgTm9uZSkgIT0ga2V5OgogICAgICAgICAgICAgICAgICAgICAgICB3b3JrZXIuX21lcmdlciA9IGxpbm1lcmdlLk1lcmdlcihiYXNlLCB0dnMpOyB3b3JrZXIuX21lcmdlcl9rZXkgPSBrZXkKICAgICAgICAgICAgICAgICAgICBnZW4gPSB3b3JrZXIuX21lcmdlci5pdGVyX21lcmdlZChsYW1zLCBkZXZpY2U9d29ya2VyLmRldmljZSwga2VlcF9vbl9kZXZpY2U9VHJ1ZSkKICAgICAgICAgICAgICAgICAgICB3b3JrZXIucmVsb2FkX3dlaWdodHMod2VpZ2h0c19pdGVyYXRvcj1nZW4sIGlzX2NoZWNrcG9pbnRfZm9ybWF0PVRydWUpCiAgICAgICAgICAgICAgICAgICAgdG9yY2guY3VkYS5zeW5jaHJvbml6ZSgpOyB0b3JjaC5jdWRhLmVtcHR5X2NhY2hlKCkKICAgICAgICAgICAgICAgICAgICByZXR1cm4gVHJ1ZQogICAgICAgICAgICAgICAgbGxtLmNvbGxlY3RpdmVfcnBjKF9mbiwgYXJncz0oYmFzZSwgY21kWyJ0dnMiXSwgY21kWyJsYW1zIl0pKQogICAgICAgICAgICAgICAgYWZ0ZXIgPSBsbG0uYXBwbHlfbW9kZWwocHRycylbMF0KICAgICAgICAgICAgICAgIHJlc1sicHRyX2NoYW5nZWQiXSA9IHN1bSgxIGZvciBrIGluIGJlZm9yZSBpZiBiZWZvcmVba10gIT0gYWZ0ZXIuZ2V0KGspKQogICAgICAgICAgICAgICAgcmVzWyJ0Il0gPSB0aW1lLnRpbWUoKSAtIHQxCiAgICAgICAgICAgICAgICByZXNbImZyZWVfZ2IiXSA9IHRvcmNoLmN1ZGEubWVtX2dldF9pbmZvKClbMF0gLyAyKiozMAogICAgICAgICAgICBlbGlmIG9wID09ICJnZW4iOgogICAgICAgICAgICAgICAgaXRlbXMgPSBqc29uLmxvYWQob3BlbihjbWRbIml0ZW1zIl0pKQogICAgICAgICAgICAgICAgcHJvbXB0cyA9IFtdCiAgICAgICAgICAgICAgICBmb3IgaXQgaW4gaXRlbXM6CiAgICAgICAgICAgICAgICAgICAga3cgPSBkaWN0KHRva2VuaXplPUZhbHNlLCBhZGRfZ2VuZXJhdGlvbl9wcm9tcHQ9VHJ1ZSwgZW5hYmxlX3RoaW5raW5nPVRydWUpCiAgICAgICAgICAgICAgICAgICAgaWYgaXQuZ2V0KCJ0b29scyIpOiBrd1sidG9vbHMiXSA9IGl0WyJ0b29scyJdCiAgICAgICAgICAgICAgICAgICAgcHJvbXB0cy5hcHBlbmQodG9rLmFwcGx5X2NoYXRfdGVtcGxhdGUoaXRbIm1lc3NhZ2VzIl0sICoqa3cpKQogICAgICAgICAgICAgICAgc3AgPSBTYW1wbGluZ1BhcmFtcyh0ZW1wZXJhdHVyZT0wLjYsIHRvcF9wPTAuOTUsIHRvcF9rPTIwLCBzZWVkPTAsIG1heF90b2tlbnM9aW50KGNtZFsibWF4X3Rva2VucyJdKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgc2tpcF9zcGVjaWFsX3Rva2Vucz1GYWxzZSkKICAgICAgICAgICAgICAgIHJlY3MgPSBOb25lCiAgICAgICAgICAgICAgICBpZiBjbWQuZ2V0KCJwYXJ0aWFsIik6CiAgICAgICAgICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgICAgICAgICByZWNzLCByZXNbImFib3J0ZWQiXSA9IHJhY2VfZ2VuZXJhdGUobGxtLCBwcm9tcHRzLCBpdGVtcywgc3AsIGNtZFsicGFydGlhbCJdLCBjbWRbInN0b3AiXSwgZiJje259IikKICAgICAgICAgICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6CiAgICAgICAgICAgICAgICAgICAgICAgIHJlc1sicmFjZV9lcnJvciJdID0gcmVwcihlKVs6NjAwXSArICIgfCAiICsgdHJhY2ViYWNrLmZvcm1hdF9leGMoKVstMTUwMDpdCiAgICAgICAgICAgICAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGVuZyA9IGxsbS5sbG1fZW5naW5lCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBlbmcuYWJvcnRfcmVxdWVzdChbZiJje259X3trfSIgZm9yIGsgaW4gcmFuZ2UobGVuKHByb21wdHMpKV0pCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBmb3IgXyBpbiByYW5nZSgyMDApOgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIG5vdCBlbmcuaGFzX3VuZmluaXNoZWRfcmVxdWVzdHMoKTogYnJlYWsKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBlbmcuc3RlcCgpCiAgICAgICAgICAgICAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICAgICAgICAgICAgICAgICBwYXNzCiAgICAgICAgICAgICAgICAgICAgICAgIHJlY3MgPSBOb25lCiAgICAgICAgICAgICAgICBpZiByZWNzIGlzIE5vbmU6CiAgICAgICAgICAgICAgICAgICAgb3V0cyA9IGxsbS5nZW5lcmF0ZShwcm9tcHRzLCBzcCwgdXNlX3RxZG09RmFsc2UpCiAgICAgICAgICAgICAgICAgICAgcmVjcyA9IFtfcmVjKGl0LCBvKSBmb3IgaXQsIG8gaW4gemlwKGl0ZW1zLCBvdXRzKV0KICAgICAgICAgICAgICAgIGpzb24uZHVtcChyZWNzLCBvcGVuKGNtZFsib3V0Il0sICJ3IikpCiAgICAgICAgICAgICAgICByZXNbIm4iXSA9IGxlbihyZWNzKTsgcmVzWyJnZW5fdG9rZW5zIl0gPSBzdW0oclsiY29tcGxldGlvbl90b2tlbnMiXSBmb3IgciBpbiByZWNzKQogICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihvcCkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6CiAgICAgICAgICAgIHJlcyA9IGRpY3Qob2s9RmFsc2UsIGVycm9yPXJlcHIoZSksIHRiPXRyYWNlYmFjay5mb3JtYXRfZXhjKClbLTQwMDA6XSkKICAgICAgICByZXNbInNlY3MiXSA9IHRpbWUudGltZSgpIC0gdDEKICAgICAgICBqc29uLmR1bXAocmVzLCBvcGVuKGYie2NkaXJ9L3Jlc197bn0udG1wIiwgInciKSk7IG9zLnJlbmFtZShmIntjZGlyfS9yZXNfe259LnRtcCIsIGYie2NkaXJ9L3Jlc197bn0uanNvbiIpCiAgICAgICAgcHJpbnQoZiJbZ3B1d29ya2VyM10gY21kIHtufSB7Y21kLmdldCgnb3AnKX0gb2s9e3Jlc1snb2snXX0ge3Jlc1snc2VjcyddOi4xZn1zIGFib3J0ZWQ9e3Jlcy5nZXQoJ2Fib3J0ZWQnKX0gcmFjZV9lcnI9e3Jlcy5nZXQoJ3JhY2VfZXJyb3InLCcnKVs6MjAwXX0iLCBmbHVzaD1UcnVlKQogICAgICAgIGlmIGNtZC5nZXQoIm9wIikgPT0gImV4aXQiOgogICAgICAgICAgICBkZWwgbGxtCiAgICAgICAgICAgIHJldHVybgogICAgICAgIG4gKz0gMQoKCmlmIF9fbmFtZV9fID09ICJfX21haW5fXyI6CiAgICBtYWluKCkK"
IRT_B64 = "ewoidGFza3MiOiBbCiJtbWx1cHJvIiwKImdwcWEiLAoiYmZjbCIKXSwKIm5fdGFzayI6IHsKIm1tbHVwcm8iOiA0MCwKImdwcWEiOiAzMCwKImJmY2wiOiA0MAp9LAoiY3JlYXRlZCI6ICIyMDI2LTExLTEzIGZyb20gZXZvZ20tZnVsbCB2MSByb3VuZC0xICg0MCBjYW5kaWRhdGVzIHggMTEwIGl0ZW1zKSwgcGVyLXRhc2sgMlBMIGpvaW50IE1BUCAodGhldGF+TigwLDEpLCBsb2cgYX5OKDAsLjVeMiksIGJ+TigwLDJeMikpOyBzdWJzZXQgPSB0b3AgRmlzaGVyIGluZm8gYXQgdGhlIHRvcC0xNSBjYW5kaWRhdGVzJyB0aGV0YSAvIHNxcnQobWVhbiB0b2tlbnMpLCBwZXIgdGFzayAxNiBNTUxVLVBybyAvIDEwIEdQUUEgLyAxNCBCRkNMIiwKIml0ZW1zIjogewoibW1sdXBybyI6IFsKImZpdF9tbWx1cHJvXzM0MzciLAoiZml0X21tbHVwcm9fMzQ0NSIsCiJmaXRfbW1sdXByb18zNjgiLAoiZml0X21tbHVwcm9fMjMzIiwKImZpdF9tbWx1cHJvXzgyMSIsCiJmaXRfbW1sdXByb180NDE3IiwKImZpdF9tbWx1cHJvXzQzMjYiLAoiZml0X21tbHVwcm9fMzY2NiIsCiJmaXRfbW1sdXByb180NjM4IiwKImZpdF9tbWx1cHJvXzEwNTE0IiwKImZpdF9tbWx1cHJvXzY4OTkiLAoiZml0X21tbHVwcm9fNjkwNSIsCiJmaXRfbW1sdXByb183MjUwIiwKImZpdF9tbWx1cHJvXzEyMDk1IiwKImZpdF9tbWx1cHJvXzExOTMxIiwKImZpdF9tbWx1cHJvXzExODc1IiwKImZpdF9tbWx1cHJvXzY3NzAiLAoiZml0X21tbHVwcm9fNjEwMyIsCiJmaXRfbW1sdXByb182MDI5IiwKImZpdF9tbWx1cHJvXzQ4NzEiLAoiZml0X21tbHVwcm9fMTk0NCIsCiJmaXRfbW1sdXByb18xOTQzIiwKImZpdF9tbWx1cHJvXzkxNyIsCiJmaXRfbW1sdXByb18xNTMwIiwKImZpdF9tbWx1cHJvXzc4OTgiLAoiZml0X21tbHVwcm9fODE0OSIsCiJmaXRfbW1sdXByb184MTIxIiwKImZpdF9tbWx1cHJvXzgzNjMiLAoiZml0X21tbHVwcm9fNTIzNyIsCiJmaXRfbW1sdXByb181MjQ2IiwKImZpdF9tbWx1cHJvXzU1MzQiLAoiZml0X21tbHVwcm9fMTExNzgiLAoiZml0X21tbHVwcm9fMTEyMzAiLAoiZml0X21tbHVwcm9fOTE4NSIsCiJmaXRfbW1sdXByb185NDI2IiwKImZpdF9tbWx1cHJvXzk5MTkiLAoiZml0X21tbHVwcm9fOTczMSIsCiJmaXRfbW1sdXByb18yMDE5IiwKImZpdF9tbWx1cHJvXzE5OTMiLAoiZml0X21tbHVwcm9fMjE2NSIKXSwKImdwcWEiOiBbCiJmaXRfZ3BxYV9tYWluXzQxMyIsCiJmaXRfZ3BxYV9tYWluXzMzIiwKImZpdF9ncHFhX21haW5fMzM3IiwKImZpdF9ncHFhX21haW5fNzQiLAoiZml0X2dwcWFfbWFpbl8xOTciLAoiZml0X2dwcWFfbWFpbl80NDAiLAoiZml0X2dwcWFfbWFpbl80NDYiLAoiZml0X2dwcWFfbWFpbl8xMDciLAoiZml0X2dwcWFfbWFpbl8xNDMiLAoiZml0X2dwcWFfZXh0ZW5kZWRfMjk5IiwKImZpdF9ncHFhX2V4dGVuZGVkXzQwNiIsCiJmaXRfZ3BxYV9tYWluXzM3MiIsCiJmaXRfZ3BxYV9tYWluXzM3NSIsCiJmaXRfZ3BxYV9tYWluXzE4NSIsCiJmaXRfZ3BxYV9tYWluXzY2IiwKImZpdF9ncHFhX21haW5fMzA3IiwKImZpdF9ncHFhX21haW5fNTIiLAoiZml0X2dwcWFfZXh0ZW5kZWRfMTU4IiwKImZpdF9ncHFhX2V4dGVuZGVkXzQxNyIsCiJmaXRfZ3BxYV9tYWluXzM5NyIsCiJmaXRfZ3BxYV9tYWluXzIxIiwKImZpdF9ncHFhX21haW5fNDE0IiwKImZpdF9ncHFhX2V4dGVuZGVkXzQ1OCIsCiJmaXRfZ3BxYV9tYWluXzIzNSIsCiJmaXRfZ3BxYV9tYWluXzE4OSIsCiJmaXRfZ3BxYV9tYWluXzE5MCIsCiJmaXRfZ3BxYV9leHRlbmRlZF81MzAiLAoiZml0X2dwcWFfZXh0ZW5kZWRfMTM5IiwKImZpdF9ncHFhX2V4dGVuZGVkXzE3NiIsCiJmaXRfZ3BxYV9tYWluXzMyMyIKXSwKImJmY2wiOiBbCiJmaXRfc2ltcGxlX3B5dGhvbl8yNDAiLAoiZml0X3NpbXBsZV9weXRob25fODMiLAoiZml0X3NpbXBsZV9weXRob25fMTkiLAoiZml0X3NpbXBsZV9weXRob25fMzYyIiwKImZpdF9zaW1wbGVfcHl0aG9uXzIyNCIsCiJmaXRfc2ltcGxlX3B5dGhvbl8yMTkiLAoiZml0X3NpbXBsZV9weXRob25fMTg5IiwKImZpdF9zaW1wbGVfcHl0aG9uXzE2MyIsCiJmaXRfc2ltcGxlX3B5dGhvbl8xMTIiLAoiZml0X3NpbXBsZV9weXRob25fMzAwIiwKImZpdF9tdWx0aXBsZV8xNjYiLAoiZml0X211bHRpcGxlXzEyIiwKImZpdF9tdWx0aXBsZV8xNTgiLAoiZml0X211bHRpcGxlXzU2IiwKImZpdF9tdWx0aXBsZV81NSIsCiJmaXRfbXVsdGlwbGVfMTkzIiwKImZpdF9tdWx0aXBsZV8xNjAiLAoiZml0X211bHRpcGxlXzQ0IiwKImZpdF9tdWx0aXBsZV80NyIsCiJmaXRfbXVsdGlwbGVfMTcyIiwKImZpdF9wYXJhbGxlbF81MiIsCiJmaXRfcGFyYWxsZWxfNjQiLAoiZml0X3BhcmFsbGVsXzEwOSIsCiJmaXRfcGFyYWxsZWxfNDUiLAoiZml0X3BhcmFsbGVsXzU5IiwKImZpdF9wYXJhbGxlbF84MSIsCiJmaXRfcGFyYWxsZWxfMTkyIiwKImZpdF9wYXJhbGxlbF84MCIsCiJmaXRfcGFyYWxsZWxfMTUwIiwKImZpdF9wYXJhbGxlbF8yNyIsCiJmaXRfcGFyYWxsZWxfbXVsdGlwbGVfOTMiLAoiZml0X3BhcmFsbGVsX211bHRpcGxlXzEwOSIsCiJmaXRfcGFyYWxsZWxfbXVsdGlwbGVfMiIsCiJmaXRfcGFyYWxsZWxfbXVsdGlwbGVfMTA0IiwKImZpdF9wYXJhbGxlbF9tdWx0aXBsZV81OSIsCiJmaXRfcGFyYWxsZWxfbXVsdGlwbGVfMTc1IiwKImZpdF9wYXJhbGxlbF9tdWx0aXBsZV8xMTUiLAoiZml0X3BhcmFsbGVsX211bHRpcGxlXzEzMCIsCiJmaXRfcGFyYWxsZWxfbXVsdGlwbGVfMTM0IiwKImZpdF9wYXJhbGxlbF9tdWx0aXBsZV8xMDUiCl0KfSwKImEiOiB7CiJtbWx1cHJvIjogWwoxLjc0MDcyODA0MDIyNzY4MTUsCjEuNzMyOTU5MTkxNDUwMzQ1MywKMS45MzAzNzMzODA5Njc2MzczLAoxLjU0ODE2MjA0NzUwNTYwODQsCjEuNTQ4MTYyMDQ3NTA1NjA4NCwKMS45NTQ3NjA1MTg3NTg3NTk4LAoxLjYzMjQxMDcxODQ0OTg4ODYsCjEuNTQ4MTYyMDQ3NTA1NjA4NCwKMi4yMzM1NDc3MDc3MTQ2Mzk1LAoxLjgxMDU2NTQwOTE0MzgwODEsCjEuODEwNTY1NDA5MTQzODA4MSwKMS42MzI0MTA3MTg0NDk4ODg2LAoyLjAyNTc4ODY4NzY3MzY5NywKMS42MzI0MTA3MTg0NDk4ODg2LAozLjM2MzgyMTc3MDYyNzU4MjQsCjEuNjMyNDEwNzE4NDQ5ODg4NiwKMS42MzI0MTA3MTg0NDk4ODg2LAoxLjU0ODE2MjA0NzUwNTYwODQsCjAuNzM0NzMzMzkyNjI1MDQ5NywKMC43MjMxMDUxNTQ1MTI0MDgyLAowLjc5NDQ3Mjk0MDEwMDIyNDIsCjEuNDI3MDg1NTMzNTQ2NzE2LAoxLjQ2MTk5NDc4NzYzNjA0MzMsCjUuNDk0NTcwOTY4OTIyMDg5LAoxLjMyOTkyNDMxODk1NzM4MiwKMS4xODUyNzI4Nzk0MDM0Mjk1LAoxLjU0ODE2MjA0NzUwNTYwODQsCjEuNTQ4MTYyMDQ3NTA1NjA4NCwKMS42NTc5NzU5MDI5NDcxNjgsCjEuNjMyNDEwNzE4NDQ5ODg4NiwKMC44NzIwNTE5MDQ1NzYyNzgsCjEuMDQyMzI1NDg2OTE3Mjg5OCwKMC41MTc2NzIwMjI5NTkxMzU0LAoxLjgxMDU2NTQwOTE0MzgwOSwKMS44MTA1NjU0MDkxNDM4MDksCjEuNjU3OTc1OTAyOTQ3MTY0NiwKMS4yMjUxMTE5MjM5NzU3MDAyLAowLjkxMTkyMjQ4NjEyMjg1MjUsCjEuMzA0MTUxMDA2NDQyOTg3NSwKMS41NDgxNjIwNDc1MDU2MDg4Cl0sCiJncHFhIjogWwoxLjYwOTM0MzkxMzEzNDIyOTgsCjIuMjk1MzQ3MjYzNTQ5MDE3NywKMS44MTkwODM5NDI4MzgwNjksCjAuOTI1OTk0MzA4Mjg2MjUyNCwKMS42MzM1NjA2MzYxMDUxNTY0LAoxLjQwNzA4Mjg3MTg5NDQwMDYsCjEuNjA5MzQzOTEzMTM0MjI5OCwKMS4wOTIxMzU3NDM0NzMwMDM1LAoxLjc2NjMzMzUzMDQ1MDE3MDQsCjEuMjQyOTM2MzA4MTM1NDcyLAoxLjYwOTM0MzkxMzEzNDIyOTgsCjEuNjA5MzQzOTEzMTM0MjI5OCwKMS4xNjkwOTY2MTMyNzE3MzksCjIuMTA4MDE4MTcxMjg2MDA5LAoxLjU2MTM3OTczMjA4NDEwOTksCjIuNTE1ODM5MTA2MDQzNDMsCjEuNjQ2Mjc2Mzg1NzcwNTIxLAoxLjYwOTM0MzkxMzEzNDIzLAozLjA1OTY3NjczOTkyNTU4ODQsCjEuMTA5MDI5NzE2MDQ4OTMwNCwKMS4xNjM5NzU3MzA5MDcyMDYsCjEuNTY4MDIyMDY5NjU1ODE3NiwKMS4wNDExMDM3NTg1ODAzMjg5LAoxLjQ4NDg3MzcwNDU1NDU0ODQsCjEuNjA5MzQzOTEzMTM0MjMsCjEuNjA5MzQzOTEzMTM0MjMsCjEuMjkyODI1MTM1NDc2MTU4NiwKMS40Mzc4MjY2ODgxNzU2Njk4LAoxLjYwOTM0MzkxMzEzNDIzLAoxLjYyMTExNDU4Njg3MzI4NzcKXSwKImJmY2wiOiBbCjEuNTU2ODk5NTQ2NDQwMjI2LAoxLjU1Njg5OTU0NjQ0MDIyNiwKMS43MzM5MzQ4OTU1MDA1NzQ3LAoxLjIxNTI0NDAzMzQxODE5NDYsCjEuMDk3NzQ4NDk4NDQwMjU2NiwKMS45MTA0MzQ0MjA2NzU3MDIsCjEuNTU2ODk5NTQ2NDQwMjI2LAowLjcyOTgwMDg5NzI1MzUyMDIsCjEuNDUxMTU5MTIwMTEyMjYwNywKMS41NTY4OTk1NDY0NDAyMjYsCjEuNTU2ODk5NTQ2NDQwMjI2LAoxLjU1Njg5OTU0NjQ0MDIyNiwKMS41NTY4OTk1NDY0NDAyMjYsCjEuNTU2ODk5NTQ2NDQwMjI2LAoxLjU1Njg5OTU0NjQ0MDIyNiwKMS4wNzI3MDY1NDg4NDI3ODM2LAoxLjU1Njg5OTU0NjQ0MDIyNiwKMS41NTY4OTk1NDY0NDAyMjYsCjEuMDQ4NTYzMDcyMTIxMDkwMSwKMS41NTY4OTk1NDY0NDAyMjYsCjAuOTA4NjAyNTkzMjU5NzE1MiwKMS45MTA0MzQ0MjA2NzU3MDIsCjEuOTEwNDM0NDIwNjc1NzAyLAowLjk5MjgyNDEwMjE1NDM0NTUsCjAuNjkyNDQ4NjM4OTUwNjA3OSwKNS4xODcxMjE4NDE4OTYwNjEsCjEuNTU2ODk5NTQ2NDQwMjI2LAo3LjAzNjk3OTQ2MzQ3OTI2MywKMS41NTY4OTk1NDY0NDAyMjYsCjEuNTU2ODk5NTQ2NDQwMjI2LAoxLjU1Njg5OTU0NjQ0MDIyNiwKMS41NTY4OTk1NDY0NDAyMjYsCjEuMjIxMjMzMTI1MTg4OTIxNywKMS4zMzg2MDI1Njg3NTM3ODc1LAoxLjMzODYwMjU2ODc1Mzc4NzUsCjEuOTEwNDM0NDIwNjc1NzAyLAoxLjIzODEzNjgyNzA2NzE5ODIsCjEuNTU2ODk5NTQ2NDQwMjI2NCwKMS41NTY4OTk1NDY0NDAyMjY0LAoxLjkxMDQzNDQyMDY3NTcwMgpdCn0sCiJiIjogewoibW1sdXBybyI6IFsKMC4xNDY3NzAxMTQ0MjU0Mjg2NCwKLTEuMTQxNzczMzYwODY4MjQzLAowLjQ2ODY4NTQwNzA2MDkzNTM3LAotMy4wNTQ2NjcyODQ0MTk0NDAyLAotMy4wNTQ2NjcyODQ0MTk0NDAyLAotMS40MDYwMzE5Nzk1OTM1MzMyLAozLjEwNTMyNzI5Nzc5MTA1OTMsCi0zLjA1NDY2NzI4NDQxOTQ0MDIsCi0xLjU1MDA2MDQ3MTA1OTcyNywKLTIuMTI1Mzc1MTExMjk4OTg0LAotMi4xMjUzNzUxMTEyOTg5ODQsCjMuMTA1MzI3Mjk3NzkxMDU5MywKLTEuMTczNTk1MTcyMDU5MjUxLAozLjEwNTMyNzI5Nzc5MTA1OTMsCjAuMDE3NjM5NTM1NDUxMzY0ODA0LAozLjEwNTMyNzI5Nzc5MTA1OTMsCjMuMTA1MzI3Mjk3NzkxMDU5MywKLTMuMDU0NjY3Mjg0NDE5NDQwMiwKLTIuMzM3Mjc1NDA5MjU4NDU2LAotMC42NzAzNTI5NTU2MzQwODMsCjEuNTQ4NDg5MDYzNjcwMzk0NywKMS44MjAxNDE2NjgzOTg5NjksCjEuMjM2MzkyOTM2NjcyOTEwNywKMC4xNTc3MzMxMjE2NTQyODgxMywKLTEuMjQ3OTE1Mjc5OTY4NDcsCi0yLjc2NTQ4OTAxMTY2MDkxODUsCi0zLjA1NDY2NzI4NDQxOTQ0MDIsCi0zLjA1NDY2NzI4NDQxOTQ0MDIsCi0yLjIzNDY4OTI3NzQ5NjE3ODQsCjMuMTA1MzI3Mjk3NzkxMDU5MywKLTIuMjc4MzYxMjA2MTAxODMsCi0xLjIwODAwNjU4MDE1ODU4MDIsCjAuMzQ3ODE4MjAwMjI2MjcyMjQsCi0yLjEyNTM3NTExMTI5ODk4NCwKLTIuMTI1Mzc1MTExMjk4OTg0LAotMi4yMzQ2ODkyNzc0OTYxNzc1LAotMi4yODA0MzA4NTU5MzMzNTI2LAotMS4wNTI0ODE5MjI4MDg0OTc2LAowLjcxNzA0MzcyMDQyNjc2NywKLTMuMDU0NjY3Mjg0NDE5NDQwNwpdLAoiZ3BxYSI6IFsKMi45MzIwNTA3NzE2ODcxMDUsCi0wLjk4NzYwODY5NDczMjkyNjEsCi0wLjI1NzY5ODUwMzUzNDQyNDI1LAowLjEzNzc4MDExMzI3NTg1MzQsCi0xLjMxNjE1NzY1NTQ2MDQ5ODEsCjAuODE5MDE2MTQyNDA0NDU0LAoyLjkzMjA1MDc3MTY4NzEwNSwKMi41MjgyNzE1NjcxMDk1MjgsCi0wLjUwODI5OTk1MjEwOTkxNjQsCjEuNDIwNTY2MDk1Nzc2NjMxLAoyLjkzMjA1MDc3MTY4NzEwNSwKMi45MzIwNTA3NzE2ODcxMDUsCjAuODI3OTQyNzQzNzcyODkzLAowLjQyMDIyNDg0NTU4NjMxNjcsCi0zLjQ2MjUyMDU4NzgxODIwODQsCjAuNzgwMTYzMzM4OTk2NjEzNCwKLTAuOTMyODMxNTkyNzIxMzU5NiwKMi45MzIwNTA3NzE2ODcxMDUsCi0xLjAwNTk4OTg4NzM1NTY0NjMsCjEuOTQzODY5NzM0NzE1Njk1NSwKMC43MTk5MzU3MDIzOTczOTI1LAotMS40OTkxNjIxMDgyMDk4ODc0LAotMi4xMTk5NDMwNTM0ODUwMDcyLAotMS4zODgwMzQxODc3NDE4MjYsCjIuOTMyMDUwNzcxNjg3MTA1LAoyLjkzMjA1MDc3MTY4NzEwNSwKMS4xMDQ5MjY2NjgxNDI4NjQsCi0wLjQ5Nzk0ODUzNTE5ODEzNzc1LAoyLjkzMjA1MDc3MTY4NzEwNSwKMC43NDA0Nzk2MTIzODM0NDk0Cl0sCiJiZmNsIjogWwotMi42NTkwMTAxNjU0NDk3MDMsCi0yLjY1OTAxMDE2NTQ0OTcwMywKLTEuMzcxOTg3Mjc1NTMyMjIxLAotMi4zNjY3NTMxODAxMDk1MywKMy4zMjI2MDYwNzQ3NzA5MjM2LAotMS41OTE2MTM2NTE1ODcyOTYsCi0yLjY1OTAxMDE2NTQ0OTcwMywKLTEuNDA0MDIzMzA2ODc1Njc0LAoyLjQyNjYxMzc5NDQyNjA4NCwKLTIuNjU5MDEwMTY1NDQ5NzAzLAotMi42NTkwMTAxNjU0NDk3MDMsCi0yLjY1OTAxMDE2NTQ0OTcwMywKLTIuNjU5MDEwMTY1NDQ5NzAzLAotMi42NTkwMTAxNjU0NDk3MDMsCi0yLjY1OTAxMDE2NTQ0OTcwMywKLTIuMTg0NzE1NzY5NDYwMTg1LAotMi42NTkwMTAxNjU0NDk3MDMsCi0yLjY1OTAxMDE2NTQ0OTcwMywKLTAuNTE0NDMzODQwNzQyNTU2MiwKLTIuNjU5MDEwMTY1NDQ5NzAzLAotMS45MDIyOTMwMDE0MTgyMTE2LAotMS41OTE2MTM2NTE1ODcyOTYsCi0xLjU5MTYxMzY1MTU4NzI5NiwKMi4zMjYzMDMyNTI2MjU1NTksCi0wLjYwNjAwMTMwMjU4MDgxNzksCjAuMTI5MjM5MjE2MTQ2NzE2ODgsCi0yLjY1OTAxMDE2NTQ0OTcwMywKMC40MTcyNzc1NDEyODIxMDA3NiwKLTIuNjU5MDEwMTY1NDQ5NzAzLAotMi42NTkwMTAxNjU0NDk3MDMsCi0yLjY1OTAxMDE2NTQ0OTcwMywKLTIuNjU5MDEwMTY1NDQ5NzAzLAotMi4zNTY4NjkxNDMzOTY4OCwKLTIuMTc3ODc1ODI1NDAyNjc1LAotMi4xNzc4NzU4MjU0MDI2NzUsCi0xLjU5MTYxMzY1MTU4NzI5NiwKLTIuMzI5MzgzOTczOTIxODUyLAotMi42NTkwMTAxNjU0NDk3MDMsCi0yLjY1OTAxMDE2NTQ0OTcwMywKLTEuNTkxNjEzNjUxNTg3Mjk2Cl0KfSwKInN1YnNldCI6IHsKIm1tbHVwcm8iOiBbCjAsCjEsCjIsCjUsCjEyLAoxNCwKMTksCjIwLAoyMSwKMjIsCjIzLAoyNCwKMjUsCjMxLAozNywKMzgKXSwKImdwcWEiOiBbCjEsCjIsCjUsCjgsCjEzLAoxNSwKMTYsCjI2LAoyNywKMjkKXSwKImJmY2wiOiBbCjIsCjQsCjUsCjcsCjgsCjE1LAoxOCwKMjAsCjIyLAoyMywKMjQsCjI1LAoyNywKMzUKXQp9LAoic3Vic2V0X2lkcyI6IFsKImZpdF9tbWx1cHJvXzM0MzciLAoiZml0X21tbHVwcm9fMzQ0NSIsCiJmaXRfbW1sdXByb18zNjgiLAoiZml0X21tbHVwcm9fNDQxNyIsCiJmaXRfbW1sdXByb183MjUwIiwKImZpdF9tbWx1cHJvXzExOTMxIiwKImZpdF9tbWx1cHJvXzQ4NzEiLAoiZml0X21tbHVwcm9fMTk0NCIsCiJmaXRfbW1sdXByb18xOTQzIiwKImZpdF9tbWx1cHJvXzkxNyIsCiJmaXRfbW1sdXByb18xNTMwIiwKImZpdF9tbWx1cHJvXzc4OTgiLAoiZml0X21tbHVwcm9fODE0OSIsCiJmaXRfbW1sdXByb18xMTE3OCIsCiJmaXRfbW1sdXByb18yMDE5IiwKImZpdF9tbWx1cHJvXzE5OTMiLAoiZml0X2dwcWFfbWFpbl8zMyIsCiJmaXRfZ3BxYV9tYWluXzMzNyIsCiJmaXRfZ3BxYV9tYWluXzQ0MCIsCiJmaXRfZ3BxYV9tYWluXzE0MyIsCiJmaXRfZ3BxYV9tYWluXzE4NSIsCiJmaXRfZ3BxYV9tYWluXzMwNyIsCiJmaXRfZ3BxYV9tYWluXzUyIiwKImZpdF9ncHFhX2V4dGVuZGVkXzUzMCIsCiJmaXRfZ3BxYV9leHRlbmRlZF8xMzkiLAoiZml0X2dwcWFfbWFpbl8zMjMiLAoiZml0X3NpbXBsZV9weXRob25fMTkiLAoiZml0X3NpbXBsZV9weXRob25fMjI0IiwKImZpdF9zaW1wbGVfcHl0aG9uXzIxOSIsCiJmaXRfc2ltcGxlX3B5dGhvbl8xNjMiLAoiZml0X3NpbXBsZV9weXRob25fMTEyIiwKImZpdF9tdWx0aXBsZV8xOTMiLAoiZml0X211bHRpcGxlXzQ3IiwKImZpdF9wYXJhbGxlbF81MiIsCiJmaXRfcGFyYWxsZWxfMTA5IiwKImZpdF9wYXJhbGxlbF80NSIsCiJmaXRfcGFyYWxsZWxfNTkiLAoiZml0X3BhcmFsbGVsXzgxIiwKImZpdF9wYXJhbGxlbF84MCIsCiJmaXRfcGFyYWxsZWxfbXVsdGlwbGVfMTc1IgpdCn0="
open("/tmp/gpuworker3.py", "wb").write(base64.b64decode(WORKER_B64))
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
        self.p = subprocess.Popen(f"CUDA_VISIBLE_DEVICES={self.g} exec python /tmp/gpuworker3.py {DIRS['base']} {self.dir} {gpumem} 1 {MAXSEQS} "
                                  f"> {self.logf} 2>&1", shell=True)
        self.boots += 1
    def wait_ready(self):
        t = time.time()
        while not os.path.exists(f"{self.dir}/ready.json"):
            if self.p.poll() is not None: raise WorkerError(f"worker {self.g} died during init (boot {self.boots})")
            if time.time() - t > T_BOOT: raise WorkerError(f"worker {self.g} init timeout")
            time.sleep(2)
        r = json.load(open(f"{self.dir}/ready.json")); log("worker", self.g, "ready", r); return r
    def call(self, timeout, poll=None, **cmd):
        if not self.alive(): raise WorkerError(f"worker {self.g} not running")
        n = self.n; self.n += 1; t = time.time(); tp = time.time()
        json.dump(cmd, open(f"{self.dir}/cmd_{n}.tmp", "w")); os.rename(f"{self.dir}/cmd_{n}.tmp", f"{self.dir}/cmd_{n}.json")
        while not os.path.exists(f"{self.dir}/res_{n}.json"):
            if poll is not None and time.time() - tp > 10:
                tp = time.time()
                try: poll()
                except Exception as e: log("poll error", repr(e)[:300])
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
        if self.p is not None: subprocess.run(f"pkill -9 -f '[g]puworker3.py \\S+ {self.dir} '", shell=True)
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

# ---------------------------------------------------------------- items / scoring / IRT fitness
import numpy as np, torch
torch.set_num_threads(1)   # deterministic CPU generator training (replayable)
import fitscore
FIT = json.load(open(f"{DATA}/fitness_items.json"))
FITBY = {i["id"]: i for i in FIT}
TASKS = ["mmlupro", "gpqa", "bfcl"]
NT = {t: sum(1 for i in FIT if i["bench"] == t) for t in TASKS}
IRT = json.loads(base64.b64decode(IRT_B64).decode())
for t in TASKS: assert IRT["items"][t] == [i["id"] for i in FIT if i["bench"] == t], t
IA = {t: np.array(IRT["a"][t]) for t in TASKS}; IB = {t: np.array(IRT["b"][t]) for t in TASKS}
IPOS = {t: {iid: j for j, iid in enumerate(IRT["items"][t])} for t in TASKS}
SUBSET = set(IRT["subset_ids"])
def ordered(items):   # longest first (GPQA, MMLU-Pro, BFCL)
    return [i for i in items if i["bench"] == "gpqa"] + [i for i in items if i["bench"] == "mmlupro"] + [i for i in items if i["bench"] == "bfcl"]
SUB_ITEMS = ordered([i for i in FIT if i["id"] in SUBSET])
json.dump(SUB_ITEMS, open("/tmp/fit_sub.json", "w"))
log("IRT subset", len(SUB_ITEMS), {t: sum(1 for i in SUB_ITEMS if i["bench"] == t) for t in TASKS})
GRID = np.linspace(-5, 5, 2001)
def sig(x): return 1.0 / (1.0 + np.exp(-x))
def theta_map(y, a, b):
    """MAP theta (N(0,1) prior) on a grid + posterior sd."""
    if len(y) == 0: return 0.0, 1.0
    z = a[None] * (GRID[:, None] - b[None])
    ll = (y[None] * -np.logaddexp(0, -z) + (1 - y[None]) * -np.logaddexp(0, z)).sum(1) - 0.5 * GRID ** 2
    w = np.exp(ll - ll.max()); w /= w.sum(); mu = float((w * GRID).sum())
    return float(GRID[int(np.argmax(ll))]), float(np.sqrt((w * (GRID - mu) ** 2).sum()))
def irt_fitness(obs, up=0.0):
    """p-IRT estimate of the full-110 fitness from observed items {id: 0/1}: per task, observed correct + 2PL-predicted p
    (MAP theta from the observed items, shifted by up*sd for an optimistic bound) for every unobserved item, / n_task."""
    det = {}; ests = []
    for t in TASKS:
        o = [(IPOS[t][i], v) for i, v in obs.items() if i in IPOS[t]]
        js = np.array([j for j, _ in o], dtype=int); y = np.array([v for _, v in o], dtype=float)
        th, sd = theta_map(y, IA[t][js], IB[t][js])
        thu = th + up * sd
        rest = np.setdiff1d(np.arange(NT[t]), js)
        e = (y.sum() + sig(IA[t][rest] * (thu - IB[t][rest])).sum()) / NT[t]
        det[t] = dict(est=round(float(e), 4), theta=round(th, 3), sd=round(sd, 3), n_obs=int(len(js)), obs_acc=round(float(y.mean()), 4) if len(y) else None)
        ests.append(e)
    return float(np.mean(ests)), det
os.makedirs(f"{W}/fit_outputs", exist_ok=True)
LOG = f"{W}/evo_log.jsonl"
def lkey(x): return ",".join(f"{float(v):.6f}" for v in x)
cache = {}
# resume: every evo_log.jsonl in the inputs; a later line for the same lkey updates the record (completions)
for prev in find_all("/kaggle/input/**/evo_log.jsonl"):
    n0 = len(cache)
    for l in open(prev):
        try: r = json.loads(l)
        except Exception: continue
        if "fitness_kind" not in r:   # round-1 record: plain accuracy on all 110 items
            r["fitness_kind"] = "full"
            if not r.get("failed"): r["full_fitness"] = r["fitness"]; r["full_acc"] = r["acc"]; r["full_gen_tokens"] = r.get("gen_tokens")
        if r["lkey"] in cache: cache[r["lkey"]].update(r)
        else: cache[r["lkey"]] = r
    log("resume: read", prev, len(cache) - n0, "new records")
for p in find_all("/kaggle/input/**/fit_outputs/*.json.gz"):   # keep per-item outputs together (needed by completions)
    d = f"{W}/fit_outputs/{os.path.basename(p)}"
    if not os.path.exists(d): shutil.copy(p, d)
with open(LOG, "w") as f:
    for r in cache.values(): f.write(json.dumps(r) + "\n")
R1_BEST = max((r for r in cache.values() if r.get("full_fitness") is not None), key=lambda r: r["full_fitness"])
log("cached evaluations", len(cache), "best full", R1_BEST["full_fitness"], R1_BEST["lams"])
def val(r):
    if r is None: return np.nan
    if r.get("failed"): return 0.0
    return r["full_fitness"] if r.get("full_fitness") is not None else r["fitness"]
SUB_T = []; COMP_T = []; THR = []
SESSION = {"n": 0, "fail": 0}
RACE = {"on": True, "up": 1.5, "min_frac": 0.5}
log_lock = threading.Lock()
def safe_score(it, out, fr):
    try: return fitscore.score(it, out, fr)
    except Exception as e: return dict(correct=False, truncated=fr == "length", error=repr(e)[:200])
def ok_item(s): return bool(s["correct"]) and not s["truncated"]
def task_stats(outs):
    trunc, tok = {}, {}
    for t in TASKS:
        done = [o for o in outs if FITBY[o["id"]]["bench"] == t and o["finish_reason"] != "aborted"]
        trunc[t] = round(sum(o["finish_reason"] == "length" for o in done) / max(1, len(done)), 4)
        tok[t] = round(sum(o["completion_tokens"] for o in done) / max(1, len(done)), 1)
    return trunc, tok
def evaluate_sub(lams, w, meta, thr):
    """IRT-subset evaluation (40 items) with racing: stop when the optimistic estimate (theta + 1.5 sd) < thr."""
    t0 = time.time()
    r = w.call(T_LOAD, op="load_merge", tvs=TV, lams=[float(v) for v in lams]); t_swap = r["t"]
    tag = hashlib.md5(lkey(lams).encode()).hexdigest()[:10]
    race_on = RACE["on"] and thr is not None and np.isfinite(thr)
    partial = f"/tmp/part_{w.g}.json"; stopf = f"/tmp/stop_{w.g}"
    for p in (partial, stopf):
        try: os.remove(p)
        except OSError: pass
    st = dict(stopped=False, ub=None, scored={})
    def poll():
        if not race_on or st["stopped"]: return
        try: P = json.load(open(partial))
        except Exception: return
        for o in P:
            if o["id"] not in st["scored"]: st["scored"][o["id"]] = int(ok_item(safe_score(FITBY[o["id"]], o["output"], o["finish_reason"])))
        if len(st["scored"]) < RACE["min_frac"] * len(SUB_ITEMS): return
        ub, _ = irt_fitness(st["scored"], up=RACE["up"])
        if ub < thr:
            open(stopf, "w").write("stop"); st["stopped"] = True; st["ub"] = ub
            log(f"RACE gpu{w.g}: stop after {len(st['scored'])}/{len(SUB_ITEMS)} items, optimistic {ub:.4f} < top-P threshold {thr:.4f}")
    kw = dict(partial=partial, stop=stopf) if race_on else {}
    rg = w.call(T_GEN_FIT, poll=poll, op="gen", items="/tmp/fit_sub.json", max_tokens=FIT_MAXTOK, out=f"/tmp/o_{w.g}.json", **kw)
    if rg.get("race_error"):
        RACE["on"] = False; log("RACE path failed -> racing disabled, plain generate used:", rg["race_error"][:1500])
    outs = json.load(open(f"/tmp/o_{w.g}.json"))
    sc = {o["id"]: safe_score(FITBY[o["id"]], o["output"], o["finish_reason"]) for o in outs if o["finish_reason"] != "aborted"}
    obs = {i: int(ok_item(s)) for i, s in sc.items()}
    fit, det = irt_fitness(obs)
    n_ab = sum(1 for o in outs if o["finish_reason"] == "aborted")
    trunc, tok = task_stats(outs)
    secs_gen = rg["secs"]; THR.append(rg["gen_tokens"] / max(secs_gen, 1))
    with gzip.open(f"{W}/fit_outputs/{tag}.json.gz", "wt") as f:
        json.dump(dict(lams=list(map(float, lams)), kind="sub", outputs=outs, scores={k: dict(correct=v["correct"], truncated=v["truncated"]) for k, v in sc.items()}), f)
    rec = dict(lkey=lkey(lams), lams=[float(v) for v in lams], tag=tag, fitness=fit, fitness_kind="irt", irt=det,
               acc={t: det[t]["obs_acc"] for t in TASKS}, raced=n_ab > 0, n_aborted=n_ab, race_ub=st["ub"], race_thr=thr,
               full_fitness=None, trunc=trunc, mean_tokens=tok, rejected=False, failed=False, gpu=w.g, t_swap=round(t_swap, 1),
               t_gen=round(secs_gen, 1), gen_tokens=rg["gen_tokens"], t_total=round(time.time() - t0, 1),
               elapsed_h=round((time.time() - T00) / 3600, 3), **meta)
    SUB_T.append(rec["t_total"])
    return rec
def complete_full(rec, w):
    """Generate the fitness items not observed in the subset run -> plain accuracy on all 110 items (round-1 definition)."""
    t0 = time.time()
    d = json.load(gzip.open(f"{W}/fit_outputs/{rec['tag']}.json.gz", "rt"))
    have = {o["id"]: o for o in d["outputs"] if o["finish_reason"] != "aborted"}
    todo = ordered([i for i in FIT if i["id"] not in have])
    json.dump(todo, open(f"/tmp/comp_{w.g}.json", "w"))
    r = w.call(T_LOAD, op="load_merge", tvs=TV, lams=rec["lams"]); t_swap = r["t"]
    rg = w.call(T_GEN_FIT * 2, op="gen", items=f"/tmp/comp_{w.g}.json", max_tokens=FIT_MAXTOK, out=f"/tmp/oc_{w.g}.json")
    outs = list(have.values()) + json.load(open(f"/tmp/oc_{w.g}.json"))
    assert sorted(o["id"] for o in outs) == sorted(FITBY), "completion does not cover the 110 items"
    sc = {o["id"]: safe_score(FITBY[o["id"]], o["output"], o["finish_reason"]) for o in outs}
    acc = {t: sum(1 for i in FIT if i["bench"] == t and ok_item(sc[i["id"]])) / NT[t] for t in TASKS}
    trunc, tok = task_stats(outs)
    with gzip.open(f"{W}/fit_outputs/{rec['tag']}_full.json.gz", "wt") as f:
        json.dump(dict(lams=rec["lams"], kind="full", outputs=outs, scores={k: dict(correct=v["correct"], truncated=v["truncated"]) for k, v in sc.items()}), f)
    upd = dict(lkey=rec["lkey"], full_fitness=float(np.mean([acc[t] for t in TASKS])), full_acc=acc, full_trunc=trunc, full_mean_tokens=tok,
               full_gen_tokens=sum(o["completion_tokens"] for o in outs), t_complete=round(time.time() - t0, 1), t_complete_swap=round(t_swap, 1),
               completed_h=round((time.time() - T00) / 3600, 3))
    COMP_T.append(upd["t_complete"])
    return upd

# ---------------------------------------------------------------- budget
T_SUB_PRIOR, T_COMP_PRIOR = 640.0, 900.0     # offline time model from round-1 per-item tokens (gen 440 s + swap ~150 s; completion 70 items)
SAVE_MARGIN = 420                            # output saving / container overhead
def t_sub(): return 1.1 * (float(np.mean(SUB_T[-12:])) if len(SUB_T) >= 2 else T_SUB_PRIOR)
def t_comp(): return 1.1 * (float(np.mean(COMP_T[-6:])) if len(COMP_T) >= 1 else T_COMP_PRIOR)
def best_full():
    return max((r for r in cache.values() if r.get("full_fitness") is not None and not r.get("failed")), key=lambda r: r["full_fitness"])
def final_est():
    b = best_full()
    if b["lkey"] == R1_BEST["lkey"]: return 0.0          # winner unchanged -> the 180-item test already exists (evogm-full)
    tok110 = b.get("full_gen_tokens") or 160000
    return 2.8 * tok110 / 2 / 150.0 + 450            # 180 items @8192 ~ 2.8x the 110-item tokens (round 1: 2.2-2.7x), 2 GPUs ~150 tok/s each
def evo_deadline(): return T00 + HARD_H * 3600 - SAVE_MARGIN - max(final_est(), 1900)   # always keep room for a final test
def remaining(): return max(0.0, evo_deadline() - time.time())
def n_alive(): return sum(1 for w in workers if not w.dead and w.alive())
def lanes(): return max(1, n_alive())
def iter_cost(P): return math.ceil(P / lanes()) * t_sub() + t_comp()
def round_cost(P, T): return math.ceil((P - NE) / lanes()) * t_sub() + T * iter_cost(P)

STOP = {"flag": False}
def run_jobs(jobs, do, on_fail, est):
    """jobs: list of payloads, do(payload, w) -> record, on_fail(payload, errs) -> record or None; est() seconds per job.
    Runs on the live GPUs; a failing job restarts its engine and is retried once. Returns {index: record}."""
    q = queue.Queue()
    for j in range(len(jobs)): q.put(j)
    out = {}
    def lane(w):
        while True:
            if STOP.get("abort"): return
            try: j = q.get_nowait()
            except queue.Empty: return
            if time.time() + est() > evo_deadline():
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
                    STOP["abort"] = True   # systematic: the first evaluations of this session all failed
                if rec is not None: out[j] = rec
    for w in workers:   # an engine that died between batches is brought back (bounded by MAX_BOOTS)
        if not w.alive() and w.boots < MAX_BOOTS and (time.time() + 900 + est() < evo_deadline()):
            log("worker", w.g, "not running at batch start -> restart")
            try: w.restart()
            except Exception as e: log("restart failed:", repr(e)[:300])
    live = [w for w in workers if not w.dead and w.alive()]
    if not live: STOP["flag"] = True; log("no live engine -> stop evolution"); return out
    ths = [threading.Thread(target=lane, args=(w,)) for w in live]; [t.start() for t in ths]; [t.join() for t in ths]
    if STOP.get("abort"):
        status(stage="aborted_systematic_failures"); log("ABORT: first evaluations all failed"); [w.kill() for w in workers]; raise SystemExit("systematic eval failure")
    if not q.empty() and n_alive() > 0:
        ths = [threading.Thread(target=lane, args=(w,)) for w in workers if not w.dead and w.alive()]; [t.start() for t in ths]; [t.join() for t in ths]
    return out
def append_log(r):
    with open(LOG, "a") as f: f.write(json.dumps(r) + "\n"); f.flush(); os.fsync(f.fileno())
def eval_batch(X, meta_list, thr=None):
    """Effective-lambda vectors -> value array (full fitness if known, else IRT estimate; nan = not evaluated)."""
    F = np.full(len(X), np.nan); todo = []
    for j in range(len(X)):
        k = lkey(X[j])
        if k in cache: F[j] = val(cache[k])
        elif not any(lkey(X[i]) == k for i in todo): todo.append(j)
    def do(j, w):
        rec = evaluate_sub(X[j], w, meta_list[j], thr)
        with log_lock:
            cache[rec["lkey"]] = rec; append_log(rec)
            log(f"[{rec.get('round')}/{rec.get('iter')}] n={len(cache)} gpu{w.g} irt_fit={rec['fitness']:.4f} "
                f"theta={ {t: rec['irt'][t]['theta'] for t in TASKS} } raced={rec['raced']}({rec['n_aborted']}) swap {rec['t_swap']}s total {rec['t_total']}s tok {rec['gen_tokens']}")
            status(stage="evolve", n_evals=len(cache), last=dict(fit=rec["fitness"], raced=rec["raced"]))
        return rec
    def fail(j, errs, w):
        rec = dict(lkey=lkey(X[j]), lams=[float(v) for v in X[j]], tag=None, fitness=0.0, fitness_kind="failed", full_fitness=None, failed=True,
                   errors=errs, gpu=w.g, elapsed_h=round((time.time() - T00) / 3600, 3), **meta_list[j])
        with log_lock: cache[rec["lkey"]] = rec; append_log(rec)
        return rec
    run_jobs(todo, do, fail, t_sub)
    for j in range(len(X)):
        k = lkey(X[j])
        if k in cache: F[j] = val(cache[k])
    return F
def complete_batch(recs):
    def do(r, w):
        upd = complete_full(r, w)
        with log_lock:
            cache[r["lkey"]].update(upd); append_log(cache[r["lkey"]])
            log(f"COMPLETE gpu{w.g} irt_est {r['fitness']:.4f} -> full {upd['full_fitness']:.4f} acc {upd['full_acc']} tok {upd['full_gen_tokens']} {upd['t_complete']}s")
        return upd
    return run_jobs(recs, do, lambda r, errs, w: None, t_comp)

# ---------------------------------------------------------------- EvoGM (rounds 2.., basis shift)
np.random.seed(42); torch.manual_seed(42)
NE = linmerge.N
HID = (128, 256, 256, 256, 128); GEN_LR = 1e-4; GEN_EPOCHS = 200; CYCLE = 100.0; OPT = 1.0; WIN_PORTION = 0.3
P = 10; MAX_ITER = 3; MAX_ROUND = 5; COMP_MARGIN = 0.02
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
def init_population(P):
    pop = [np.eye(NE)[i] for i in range(NE)] + [np.full(NE, 1.0 / NE)]
    while len(pop) < P: pop.append(np.random.uniform(-1, 1, NE))
    return np.array(pop[:P])
STATE = f"{W}/evo_state.pt"
def save_state(S):
    S = dict(S); S["v"] = 2; S["np_rng"] = np.random.get_state(); S["torch_rng"] = torch.get_rng_state()
    torch.save(S, STATE + ".tmp"); os.replace(STATE + ".tmp", STATE)
    log("state saved: next step", S["next"])
def load_state():
    best = None
    for p in find_all("/kaggle/input/**/evo_state.pt"):
        try:
            S = torch.load(p, weights_only=False); S.setdefault("v", 1)
            if best is None or (S["v"], tuple(S["next"])) > (best["v"], tuple(best["next"])): best = S
            log("resume: found state", p, "v", S["v"], "next", S["next"])
        except Exception as e: log("resume: unreadable state", p, repr(e)[:300])
    return best
def eff(coords, basis): return np.asarray(coords, dtype=np.float32) @ basis
def vals(coords, basis): return np.array([val(cache.get(lkey(e))) for e in eff(coords, basis)], dtype=float)
def run_eval(pop, basis, rnd, it, kinds, thr=None):
    meta = [dict(round=rnd, iter=it, kind=kinds[j], coords=[float(v) for v in pop[j]]) for j in range(len(pop))]
    return eval_batch(eff(pop, basis), meta, thr)
def maybe_complete(pop, basis, tag):
    """Guard against IRT error / winner's curse: complete (full 110 items) the best un-completed population members when the
    best IRT estimate is within COMP_MARGIN of the best full fitness so far (up to one per GPU)."""
    bf = best_full()["full_fitness"]
    cands = []
    for e in eff(pop, basis):
        r = cache.get(lkey(e))
        if r is None or r.get("failed") or r.get("full_fitness") is not None or r.get("raced"): continue
        if all(c["lkey"] != r["lkey"] for c in cands): cands.append(r)
    cands.sort(key=lambda r: -r["fitness"])
    if not cands or cands[0]["fitness"] < bf - COMP_MARGIN:
        log(f"completion {tag}: none (best estimate {cands[0]['fitness'] if cands else None} vs best full {bf:.4f})"); return
    pick = cands[:max(1, min(2, lanes()))]
    log(f"completion {tag}:", [round(r["fitness"], 4) for r in pick], "best full", round(bf, 4))
    complete_batch(pick)

status(stage="evolve_start")
S = load_state()
assert S is not None, "no evo_state.pt in the inputs"
HIST = list(S["HIST"])
basis = np.eye(NE, dtype=np.float32)
if S["v"] < 2:
    # round 1 (evogm-full) finished: basis shift exactly as the reference (top-N of the round history become the experts)
    np.random.set_state(S["np_rng"]); torch.set_rng_state(S["torch_rng"])
    hp = np.asarray(S["hist_pop"]); hv = vals(hp, basis)
    top = hp[np.argsort(-hv, kind="stable")[:NE]]
    log("round-1 top-8 values", np.sort(hv)[::-1][:NE].round(4).tolist())
    basis = (top.astype(np.float32) @ basis).astype(np.float32)
    rnd, it0 = 2, 0
    np.save(f"{W}/basis_round2.npy", basis); log("basis shift -> round 2", basis.round(3).tolist())
    save_state(dict(next=(2, 0), basis=basis, HIST=HIST))
else:
    basis = S["basis"]; rnd, it0 = S["next"]; np.random.set_state(S["np_rng"]); torch.set_rng_state(S["torch_rng"])
    log("RESUMING at round", rnd, "iter", it0)
plan_info = dict(P=P, max_iter=MAX_ITER, max_round=MAX_ROUND, t_sub_prior=T_SUB_PRIOR, t_comp_prior=T_COMP_PRIOR, setup_h=(time.time() - T00) / 3600,
                 remaining_h=remaining() / 3600, round_cost_T2_h=round_cost(P, 2) / 3600, round_cost_T3_h=round_cost(P, 3) / 3600,
                 min_rounds_affordable=int(remaining() // round_cost(P, 2)), irt_subset=len(SUB_ITEMS), race=RACE)
json.dump(plan_info, open(f"{W}/plan.json", "w"), indent=1, default=str); log("PLAN", plan_info)
while rnd <= MAX_ROUND and not STOP["flag"]:
    if it0 > 0:   # resume inside a round
        pop = S["pop"]; hist_pop = S["hist_pop"]
        dual = Dual(NE); dual.load_state_dict(S["dual"]); opt = torch.optim.Adam(dual.parameters(), lr=GEN_LR); opt.load_state_dict(S["opt"])
        torch.set_rng_state(S["torch_rng"])
    else:
        if remaining() < round_cost(P, 1):
            log(f"budget: no time for round {rnd} (remaining {remaining()/3600:.2f} h, min round {round_cost(P, 1)/3600:.2f} h)"); break
        pop = init_population(P); kinds = [f"onehot_prev_top{i+1}" for i in range(NE)] + ["average"] + ["random"] * (P - NE - 1)
        f0 = run_eval(pop, basis, rnd, 0, kinds)
        hist_pop = pop.copy()
        ok = ~np.isnan(f0)
        HIST.append(dict(round=rnd, iter=0, best=float(np.nanmax(f0)) if ok.any() else None, mean=float(np.nanmean(f0)) if ok.any() else None,
                         n=int(ok.sum()), new=[round(float(v), 4) for v in f0[NE:]], elapsed_h=(time.time() - T00) / 3600))
        json.dump(dict(hist=HIST), open(f"{W}/history.json", "w"), indent=1); log("HIST", HIST[-1])
        pop = pop[ok]
        dual = Dual(NE); opt = torch.optim.Adam(dual.parameters(), lr=GEN_LR)
        it0 = 1
        if STOP["flag"]: break
        save_state(dict(next=(rnd, 1), basis=basis, HIST=HIST, pop=pop, hist_pop=hist_pop, dual=dual.state_dict(), opt=opt.state_dict()))
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
        new = np.clip(new, -1, 1).astype(np.float64)
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
                         new_best=float(np.nanmax(nf)) if ok.any() else None, n_new=int(ok.sum()), n_raced=int(sum(1 for e in eff(new, basis) if (cache.get(lkey(e)) or {}).get("raced"))),
                         best_full=bfr["full_fitness"], best_full_round=bfr.get("round"), gen_loss=loss, elapsed_h=(time.time() - T00) / 3600))
        json.dump(dict(hist=HIST), open(f"{W}/history.json", "w"), indent=1); log("HIST", HIST[-1])
        if not STOP["flag"]:
            save_state(dict(next=(rnd, it + 1), basis=basis, HIST=HIST, pop=pop, hist_pop=hist_pop, dual=dual.state_dict(), opt=opt.state_dict()))
    if STOP["flag"]: break
    # basis shift: top-N of this round's history (best available value) become the next round's experts
    hv = vals(hist_pop, basis); hm = ~np.isnan(hv)
    top = hist_pop[hm][np.argsort(-hv[hm], kind="stable")[:NE]]
    basis = (top.astype(np.float32) @ basis).astype(np.float32)
    rnd += 1; it0 = 0
    np.save(f"{W}/basis_round{rnd}.npy", basis); log(f"basis shift -> round {rnd}", basis.round(3).tolist())
    save_state(dict(next=(rnd, 0), basis=basis, HIST=HIST))
best = best_full()
recs_new = [r for r in cache.values() if r.get("fitness_kind") == "irt"]
json.dump(dict(n_cached_total=len(cache), n_new=len(recs_new), n_raced=sum(1 for r in recs_new if r.get("raced")),
               n_failed=sum(1 for r in cache.values() if r.get("failed")), completed=[dict(round=r.get("round"), iter=r.get("iter"), irt=r["fitness"], full=r["full_fitness"])
                                                                                   for r in recs_new if r.get("full_fitness") is not None],
               best=best, round1_best_lkey=R1_BEST["lkey"], winner_changed=best["lkey"] != R1_BEST["lkey"], hist=HIST, stopped_early=STOP["flag"],
               selection_rule="max full-110 fitness (mean of 3 task accuracies, truncated=wrong) over all fully evaluated candidates (round 1 + completions)",
               search_fitness="rounds 2+: p-IRT estimate from a 40-item subset (per-task 2PL fit on round 1), full value once completed",
               hours=(time.time() - T00) / 3600), open(f"{W}/evo_result.json", "w"), indent=1, default=str)
log("BEST", best["full_fitness"], best["lams"], best.get("full_acc"), "round", best.get("round"), "changed", best["lkey"] != R1_BEST["lkey"])
status(stage="final_start", best=dict(fit=best["full_fitness"], lams=best["lams"]))
if best["lkey"] == R1_BEST["lkey"]:
    for w in workers: w.kill()
    status(stage="done", final="skipped: winner unchanged from round 1 (its 180-item result exists from evogm-full)", hours=(time.time() - T00) / 3600)
    log("ALL DONE (winner unchanged, final skipped) hours", (time.time() - T00) / 3600); sys.exit(0)

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

