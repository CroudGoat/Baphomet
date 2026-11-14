# EvoGM-style evolutionary linear merge over the 8 Baphomet donors (Qwen3.5-4B base) on Kaggle T4x2.
# Search algorithm ported from github.com/JiangTao97/evogm (fusion_bench/method/evogm.py, MIT):
#   coefficients in [-1,1]^N (generator output tanh + clip), init = one-hot experts + average + random U(-1,1)
#   (+ Baphomet's effective lambda, user request), winners = top 30% of the round history, dual MLP generators
#   lose2win / win2lose (hidden 128-256-256-256-128, ReLU, Tanh), loss = 100*(cycle MSE both ways) + 1*MSE(fake_win, winner
#   centroid), Adam lr 1e-4, 200 full-batch epochs per iteration (generator + optimizer persist within a round),
#   new = clip(lose2win(population)), population = top-P of (population U new); rounds with basis shift
#   (top-N of the round history become the next round's experts; coefficients map through the basis).
# Fitness = mean accuracy over {MMLU-Pro, GPQA, BFCL non-live} by GENERATION (own chat template, thinking on, temp .6,
#   top_p .95, top_k 20, seed 0, max_tokens 4096); truncated = wrong. Final: winner on the 180-item test set at 8192.
import os, sys, json, time, glob, gzip, shutil, subprocess, threading, queue, hashlib, math
T00 = time.time()
HARD_H = float(os.environ.get("HARD_H", "8.55"))      # whole-kernel ceiling (user budget 10 GPU-h incl. final)
FIT_MAXTOK = 4096
REJECT_BFCL = 0.25                                       # early stop: BFCL acc < 25% -> MC items not generated (counted wrong)
GPUMEM = [0.84, 0.78]; MAXSEQS = 48   # v2: eager engines (v1 with torch.compile+cudagraphs died silently during boot)
W = "/kaggle/working"
def find(pat):
    g = sorted(glob.glob(pat, recursive=True)); return g[0] if g else None
DATA = os.path.dirname(find("/kaggle/input/**/linmerge.py"))
if os.path.exists(f"{DATA}/bfcl_eval.tar.gz"):
    os.makedirs("/tmp/pylib", exist_ok=True); subprocess.run(f"tar -xzf {DATA}/bfcl_eval.tar.gz -C /tmp/pylib", shell=True, check=True); PYLIB = "/tmp/pylib"
else:
    PYLIB = f"{DATA}/bfcl_eval"   # Kaggle auto-extracts archives
assert os.path.exists(f"{PYLIB}/bfcl_eval/constants/enums.py"), PYLIB
sys.path.insert(0, DATA); sys.path.insert(1, PYLIB)
os.environ["PYTHONPATH"] = f"{DATA}:{PYLIB}:" + os.environ.get("PYTHONPATH", "")
def sh(c, t=None):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
    try: log("$", c, "->", r.stdout[-1500:], r.stderr[-1500:], "rc", r.returncode)
    except NameError: print("$", c, r.stdout[-1500:], r.stderr[-1500:], r.returncode, flush=True)
    return r
sh("nvidia-smi -L; df -h /tmp /kaggle/working; free -g; nproc")
import linmerge
_PL = open(f"{W}/progress.log", "a")
def log(*a):
    line = time.strftime("%H:%M:%S") + f" [{(time.time()-T00)/3600:.3f}h] " + " ".join(str(x) for x in a)
    print(line, flush=True); _PL.write(line + "\n"); _PL.flush(); os.fsync(_PL.fileno())
linmerge.log = log
def _monitor():
    while True:
        try:
            m = subprocess.run("free -m | awk 'NR==2{print $3\"/\"$2\" MB used, avail \"$7}'; timeout 20 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\\n' ' '; df -h /tmp | awk 'NR==2{print $3\" used /tmp\"}'",
                               shell=True, capture_output=True, text=True, timeout=30).stdout.replace("\n", " | ")
            log("MON", m)
        except Exception as e: log("MON err", repr(e))
        time.sleep(120)
threading.Thread(target=_monitor, daemon=True).start()
STATUS = dict(stage="start")
def status(**kw):
    STATUS.update(kw); STATUS["elapsed_h"] = round((time.time() - T00) / 3600, 3)
    json.dump(STATUS, open(f"{W}/status.json", "w"), indent=1)
status()

# ---------------------------------------------------------------- setup: pip || downloads -> fp8 TV extraction (bounded disk)
DIRS = {"base": "/tmp/m/base", **{s: f"/tmp/m/{s}" for s in linmerge.SHORT}, "baphomet": "/tmp/m/baphomet"}
REPOS = {"base": linmerge.BASE_ID, **{d[0]: d[1] for d in linmerge.DONORS}, "baphomet": "CloudGoat/Baphomet-1.0-4B"}
REMAP = {d[0]: d[2] for d in linmerge.DONORS}
TVDIR = "/tmp/tv"; os.makedirs(TVDIR, exist_ok=True)
TV = [f"{TVDIR}/{s}.safetensors" for s in linmerge.SHORT]
def disk_used_gb():
    tot = 0
    for d in ("/tmp/m", "/tmp/tv"):
        for r, _, fs in os.walk(d):
            for f in fs:
                try: tot += os.path.getsize(os.path.join(r, f))
                except OSError: pass
    return round(tot / 1e9, 1)
log("models+tv disk at start", disk_used_gb())
pip_ok = threading.Event(); pip_res = {}
def pip_thread():
    t = time.time()
    r = subprocess.run(f"pip install --no-cache-dir vllm==0.33.0 > {W}/pip.log 2>&1", shell=True)
    c = subprocess.run("python -c 'import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)'",
                       shell=True, capture_output=True, text=True)
    pip_res.update(rc=r.returncode, versions=c.stdout.strip(), err=c.stderr[-800:], secs=round(time.time() - t))
    log("pip finished", pip_res); pip_ok.set()
threading.Thread(target=pip_thread, daemon=True).start()
dl_done = {}; slots = threading.Semaphore(2); ready_q = queue.Queue()
def dl_all():
    t = time.time(); linmerge.download(REPOS["base"], DIRS["base"]); dl_done["base"] = round(time.time() - t); log("downloaded base", dl_done["base"], "disk", disk_used_gb())
    def one(k):
        slots.acquire()
        t = time.time(); linmerge.download(REPOS[k], DIRS[k]); dl_done[k] = round(time.time() - t)
        log("downloaded", k, dl_done[k], "disk", disk_used_gb()); ready_q.put(k)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex: list(ex.map(one, linmerge.SHORT))
th = threading.Thread(target=dl_all); th.start()
pip_ok.wait()
if not pip_res.get("versions", "").startswith("0.33.0"):
    sh(f"tail -30 {W}/pip.log"); status(stage="pip_failed", pip=pip_res); raise SystemExit("pip install vllm failed")
open("/tmp/ext1.py", "w").write(
    "import sys, json; sys.path.insert(0, sys.argv[1]); import linmerge\n"
    "st = linmerge.extract_tv(sys.argv[2], sys.argv[3], sys.argv[4] == '1', sys.argv[5], device='cuda')\n"
    "json.dump(st, open(sys.argv[5] + '.json', 'w'))\n")
tv_stats = {}; ext_err = []
def ext_lane(g):
    while len(tv_stats) + len(ext_err) < linmerge.N:
        try: k = ready_q.get(timeout=5)
        except queue.Empty:
            if not th.is_alive() and ready_q.empty(): ext_err.append(("download", "download thread ended early")); break
            continue
        t = time.time()
        r = subprocess.run(f"CUDA_VISIBLE_DEVICES={g} python /tmp/ext1.py {DATA} {DIRS['base']} {DIRS[k]} {int(REMAP[k])} {TVDIR}/{k}.safetensors",
                           shell=True, capture_output=True, text=True)
        if r.returncode != 0: ext_err.append((k, r.stderr[-1500:])); log("EXTRACT FAILED", k, r.stderr[-1500:])
        else:
            tv_stats[k] = json.load(open(f"{TVDIR}/{k}.safetensors.json")); tv_stats[k]["secs"] = round(time.time() - t)
            log("tv", k, {x: tv_stats[k][x] for x in ("n", "zero", "missing", "rel_qerr", "secs")})
        shutil.rmtree(DIRS[k], ignore_errors=True); slots.release()
        log("disk", disk_used_gb())
els = [threading.Thread(target=ext_lane, args=(g,)) for g in (0, 1)]; [t.start() for t in els]; [t.join() for t in els]; th.join()
json.dump(tv_stats, open(f"{W}/tv_stats.json", "w"), indent=1)
if ext_err: status(stage="extract_failed", err=ext_err); raise SystemExit("TV extraction failed")
status(stage="tvs_done", dl=dl_done, pip=pip_res)

# ---------------------------------------------------------------- workers
class Worker:
    def __init__(self, g):
        self.g = g; self.attempt = 0; self.eager = False; self.p = None
    def start(self, gpumem, eager):
        self.n = 0; self.eager = eager; self.gpumem = gpumem
        self.dir = f"/tmp/w{self.g}_{self.attempt}"; shutil.rmtree(self.dir, ignore_errors=True); os.makedirs(self.dir)
        self.p = subprocess.Popen(f"CUDA_VISIBLE_DEVICES={self.g} python {DATA}/gpuworker2.py {DIRS['base']} {self.dir} {gpumem} {int(eager)} {MAXSEQS} "
                                  f"> {W}/worker_{self.g}_{self.attempt}.log 2>&1", shell=True)
        self.attempt += 1
    def wait_ready(self):
        while not os.path.exists(f"{self.dir}/ready.json"):
            if self.p.poll() is not None: raise RuntimeError(f"worker {self.g} died during init (attempt {self.attempt})")
            time.sleep(2)
        r = json.load(open(f"{self.dir}/ready.json")); log("worker", self.g, "ready", r); return r
    def call(self, **cmd):
        n = self.n; self.n += 1
        json.dump(cmd, open(f"{self.dir}/cmd_{n}.tmp", "w")); os.rename(f"{self.dir}/cmd_{n}.tmp", f"{self.dir}/cmd_{n}.json")
        while not os.path.exists(f"{self.dir}/res_{n}.json"):
            if self.p.poll() is not None: raise RuntimeError(f"worker {self.g} died; see worker log")
            time.sleep(0.3)
        r = json.load(open(f"{self.dir}/res_{n}.json"))
        if not r["ok"]: raise RuntimeError(f"worker {self.g} {cmd.get('op')}: {r['error']}\n{r.get('tb')}")
        return r
    def kill(self):
        if self.p and self.p.poll() is None:
            try:
                n = self.n; self.n += 1
                json.dump(dict(op="exit"), open(f"{self.dir}/cmd_{n}.tmp", "w")); os.rename(f"{self.dir}/cmd_{n}.tmp", f"{self.dir}/cmd_{n}.json")
                self.p.wait(timeout=90)
            except Exception: pass
        b = os.path.basename(self.dir)
        sh(f"pkill -9 -f '[g]puworker2.py .*{b} ' ; sleep 8; nvidia-smi --query-gpu=index,memory.used --format=csv")
    def boot(self):
        """(0.84, eager) then fallback (0.78, eager). At most 2 attempts."""
        cfgs = [(GPUMEM[0], True), (GPUMEM[1], True)]
        for gm, eg in cfgs[self.attempt:]:
            try:
                self.start(gm, eg); return self.wait_ready()
            except Exception as e:
                log("worker boot failed", self.g, gm, eg, repr(e)); sh(f"tail -40 {W}/worker_{self.g}_{self.attempt-1}.log"); self.kill()
        raise RuntimeError(f"worker {self.g}: no working engine config")
    def fallback(self):
        self.kill()
        if self.attempt >= 2: raise RuntimeError(f"worker {self.g}: fallback exhausted")
        self.attempt = 1; return self.boot()

workers = [Worker(0), Worker(1)]
boot_res = {}
def boot0(): boot_res[0] = workers[0].boot()
tb0 = threading.Thread(target=boot0); tb0.start()
# validate the merge against two shards of the published Baphomet on GPU1 while GPU0 boots
try:
    linmerge.download(REPOS["baphomet"], DIRS["baphomet"], patterns=["model-00001-of-00010.safetensors", "model-00006-of-00010.safetensors",
                                                                     "model.safetensors.index.json", "config.json"])
    sh(f"CUDA_VISIBLE_DEVICES=1 python {DATA}/valmerge.py {DIRS['base']} {TVDIR} {DIRS['baphomet']} {W}/valmerge.json", t=1800)
except Exception as e:
    log("valmerge skipped", repr(e))
shutil.rmtree(DIRS["baphomet"], ignore_errors=True)
boot_res[1] = workers[1].boot()
tb0.join()
if 0 not in boot_res: raise RuntimeError("worker 0 failed to boot")
status(stage="workers_ready", boot=boot_res)

# ---------------------------------------------------------------- items / scoring
import numpy as np, torch
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
cache = {}
prev = find("/kaggle/input/**/evo_log.jsonl")
if prev:
    for l in open(prev):
        r = json.loads(l); cache[r["lkey"]] = r
    shutil.copy(prev, LOG)
log("cached evaluations", len(cache))
def lkey(x): return ",".join(f"{float(v):.6f}" for v in x)
log_lock = threading.Lock()
EVAL_TIMES = []; THR = []
def evaluate(lams, w, meta):
    t0 = time.time()
    r = w.call(op="load_merge", tvs=TV, lams=[float(v) for v in lams])
    if r["ptr_changed"] and not w.eager: raise PtrChanged(r["ptr_changed"])
    t_swap = r["t"]; tag = hashlib.md5(lkey(lams).encode()).hexdigest()[:10]
    outs = []; t_gen = 0.0; gen_tok = 0
    rb = w.call(op="gen", items="/tmp/fit_bfcl.json", max_tokens=FIT_MAXTOK, out=f"/tmp/o_{w.g}.json")
    outs += json.load(open(f"/tmp/o_{w.g}.json")); t_gen += rb["secs"]; gen_tok += rb["gen_tokens"]
    sc = {o["id"]: fitscore.score(FITBY[o["id"]], o["output"], o["finish_reason"]) for o in outs}
    bacc = sum(sc[i["id"]]["correct"] and not sc[i["id"]]["truncated"] for i in BF) / NT["bfcl"]
    rejected = bacc < REJECT_BFCL
    if not rejected:
        rm = w.call(op="gen", items="/tmp/fit_mc.json", max_tokens=FIT_MAXTOK, out=f"/tmp/o_{w.g}.json")
        mo = json.load(open(f"/tmp/o_{w.g}.json")); outs += mo; t_gen += rm["secs"]; gen_tok += rm["gen_tokens"]
        THR.append(rm["gen_tokens"] / max(rm["secs"], 1))
        for o in mo: sc[o["id"]] = fitscore.score(FITBY[o["id"]], o["output"], o["finish_reason"])
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
               rejected=rejected, gpu=w.g, t_swap=round(t_swap, 1), t_gen=round(t_gen, 1), gen_tokens=gen_tok,
               t_total=round(time.time() - t0, 1), elapsed_h=round((time.time() - T00) / 3600, 3), **meta)
    if not rejected: EVAL_TIMES.append(rec["t_total"])
    return rec
class PtrChanged(Exception): pass

# ---------------------------------------------------------------- budget
FINAL_TOK = 572000   # Baphomet's generated tokens on the 180-item test at 8192 (bench raw_baphomet)
def t_eval_est():
    return 1.1 * float(np.mean(EVAL_TIMES)) if EVAL_TIMES else 1500.0
def final_est():
    thr = float(np.mean(THR)) if THR else 200.0
    return 1.35 * FINAL_TOK / (2 * thr) + 900
def evo_deadline(): return T00 + HARD_H * 3600 - final_est()
def evals_affordable():
    return int(2 * max(0.0, evo_deadline() - time.time()) / t_eval_est())

STOP = {"flag": False}
def eval_batch(X, meta_list):
    """Evaluate effective-lambda vectors on both GPUs; returns fitness array (nan = not evaluated for lack of time)."""
    q = queue.Queue()
    for j, x in enumerate(X): q.put(j)
    F = np.full(len(X), np.nan); err = []
    def lane(w):
        while not err:
            try: j = q.get_nowait()
            except queue.Empty: return
            k = lkey(X[j])
            if k in cache: F[j] = cache[k]["fitness"]; continue
            if time.time() + t_eval_est() > evo_deadline():
                STOP["flag"] = True; log("time budget: skipping", meta_list[j]); continue
            for attempt in range(2):
                try:
                    rec = evaluate(X[j], w, meta_list[j]); break
                except Exception as e:
                    log("eval error gpu", w.g, repr(e)[:2000])
                    if attempt == 1: err.append(repr(e)); return
                    try: w.fallback()
                    except Exception as e2: err.append(repr(e2)); return
            with log_lock:
                cache[rec["lkey"]] = rec; F[j] = rec["fitness"]
                with open(LOG, "a") as f: f.write(json.dumps(rec) + "\n")
                log(f"[{rec.get('round')}/{rec.get('iter')}] n={len(cache)} gpu{w.g} fit={rec['fitness']:.4f} acc={rec['acc']} "
                    f"rej={rec['rejected']} swap {rec['t_swap']}s total {rec['t_total']}s tok {rec['gen_tokens']}")
                status(stage="evolve", n_evals=len(cache), last=dict(fit=rec["fitness"], acc=rec["acc"]))
    alive = [w for w in workers if w.p is not None and w.p.poll() is None]
    ths = [threading.Thread(target=lane, args=(w,)) for w in alive]; [t.start() for t in ths]; [t.join() for t in ths]
    if err: raise RuntimeError(err)
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

HIST = []   # per-iteration summaries
plan = None
basis = np.eye(NE, dtype=np.float32)
def eff(pop): return np.asarray(pop, dtype=np.float32) @ basis
def run_eval(pop, rnd, it, kinds):
    E = eff(pop); meta = [dict(round=rnd, iter=it, kind=kinds[j], coords=[float(v) for v in pop[j]]) for j in range(len(pop))]
    return eval_batch(E, meta)

status(stage="evolve_start")
# round-1 measurement: Baphomet-lambda and the average first (one per GPU), then choose the plan from measured cost
P12 = init_population(12, True)
kinds12 = [f"onehot_{s}" for s in linmerge.SHORT] + ["average", "baphomet"] + ["random"] * 2
first = [9, 8]   # baphomet, average
F_first = run_eval(P12[first], 1, 0, [kinds12[j] for j in first])
E_total = len([r for r in cache.values() if r.get("round") == 1]) + evals_affordable()
MAX_ITER = 3   # repo default max_iter; iterations run while the measured budget allows
if E_total >= 64: plan = dict(P=12, rounds=2, max_iter=MAX_ITER)       # 12 + 2*12 | 4 + 2*12 minimum
elif E_total >= 36: plan = dict(P=12, rounds=1, max_iter=MAX_ITER)
elif E_total >= 20: plan = dict(P=10, rounds=1, max_iter=MAX_ITER)
else: plan = None
plan_info = dict(plan=plan, E_total=E_total, t_eval_est_s=t_eval_est(), thr_tok_s=float(np.mean(THR)) if THR else None,
                 final_est_s=final_est(), evo_deadline_h=(evo_deadline() - T00) / 3600, measured=[r for r in cache.values()])
json.dump(plan_info, open(f"{W}/plan.json", "w"), indent=1, default=str)
log("PLAN", {k: v for k, v in plan_info.items() if k != "measured"})
if plan is None:
    status(stage="stopped_budget", plan=plan_info); [w.kill() for w in workers]; sys.exit(0)
P = plan["P"]
global_best = None
for rnd in range(1, plan["rounds"] + 1):
    if rnd == 1:
        pop = P12[:P]; kinds = kinds12[:P]
    else:
        pop = init_population(P, False); kinds = [f"onehot_prev_top{i+1}" for i in range(NE)] + ["average"] + ["random"] * (P - NE - 1)
    scores_fit = run_eval(pop, rnd, 0, kinds)
    hist_pop = pop.copy(); hist_fit = scores_fit.copy()
    ok = ~np.isnan(hist_fit)
    HIST.append(dict(round=rnd, iter=0, best=float(np.nanmax(hist_fit)), mean=float(np.nanmean(hist_fit)), n=int(ok.sum()), elapsed_h=(time.time() - T00) / 3600))
    json.dump(dict(plan=plan, hist=HIST), open(f"{W}/history.json", "w"), indent=1)
    pop, scores_fit = pop[ok], scores_fit[ok]
    order = np.argsort(-scores_fit, kind="stable"); pop, scores_fit = pop[order][:P], scores_fit[order][:P]
    dual = Dual(NE); opt = torch.optim.Adam(dual.parameters(), lr=GEN_LR)
    for it in range(1, plan["max_iter"] + 1):
        if STOP["flag"]: break
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
    if STOP["flag"]: break
    if rnd < plan["rounds"]:
        hm = ~np.isnan(hist_fit); hp, hf = hist_pop[hm], hist_fit[hm]
        top = hp[np.argsort(-hf, kind="stable")[:NE]]
        basis = (top.astype(np.float32) @ basis).astype(np.float32)
        np.save(f"{W}/basis_round{rnd+1}.npy", basis)
        log("basis shift ->", basis.round(3).tolist())

recs = list(cache.values())
best = max(recs, key=lambda r: (r["fitness"], -recs.index(r)))
json.dump(dict(plan=plan, n_evaluated=len(recs), best=best, hist=HIST, stopped_early=STOP["flag"],
               selection_rule="max fitness (mean of 3 task accuracies) over all evaluated", hours=(time.time() - T00) / 3600),
          open(f"{W}/evo_result.json", "w"), indent=1)
log("BEST", best["fitness"], best["lams"], best["acc"])
status(stage="final_start", best=dict(fit=best["fitness"], lams=best["lams"]))

# ---------------------------------------------------------------- Final: winner on the 180-item test set (8192 cap)
TEST = {json.loads(l)["id"]: json.loads(l) for l in open(f"{DATA}/test_items.jsonl")}
ids = json.load(open(f"{DATA}/subset_ids.json"))
halves = [ids[0::2], ids[1::2]]
fres = {}
def final_lane(g):
    w = workers[g]
    items = [TEST[i] for i in halves[g]]
    items = [i for i in items if i["bench"] == "gpqa"] + [i for i in items if i["bench"] == "mmlupro"] + [i for i in items if i["bench"] == "bfcl"]
    json.dump(items, open(f"/tmp/test_{g}.json", "w"))
    for attempt in range(2):
        try:
            w.call(op="load_merge", tvs=TV, lams=best["lams"])
            t0 = time.time(); r = w.call(op="gen", items=f"/tmp/test_{g}.json", max_tokens=8192, out=f"{W}/raw_evogm_{g}.json")
            fres[g] = dict(secs=time.time() - t0, gen_tokens=r["gen_tokens"]); return
        except Exception as e:
            log("final error", g, repr(e)[:1500])
            if attempt == 1: return
            w.fallback()
ths = [threading.Thread(target=final_lane, args=(g,)) for g in (0, 1)]; [t.start() for t in ths]; [t.join() for t in ths]
# if one GPU failed, run its half on the other
for g in (0, 1):
    if g not in fres and (1 - g) in fres:
        log("running half", g, "on gpu", 1 - g); workers[g] = workers[1 - g]; final_lane(g)
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
    o = json.loads(l); s = fitscore.score(TEST[o["id"]], o["output"], o["finish_reason"])
    quick.setdefault(o["bench"], []).append(s["correct"])
log("FINAL raw lines", n, {k: (sum(v), len(v)) for k, v in quick.items()}, fres)
for w in workers: w.kill()
status(stage="done", final=dict(n=n, quick={k: [sum(v), len(v)] for k, v in quick.items()}, fres=fres), hours=(time.time() - T00) / 3600)
log("ALL DONE hours", (time.time() - T00) / 3600)
