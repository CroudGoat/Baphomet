
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
