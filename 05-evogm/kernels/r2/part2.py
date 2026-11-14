
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
