
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
def irt_stage():
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
               secs=round(time.time() - t0, 1))
    log("IRT chosen split", best, "subset", len(res["subset_ids"]), "const", res["const_items"], "t_sub_prior", round(res["t_sub_prior"]), "t_comp_prior", round(res["t_comp_prior"]), "fit secs", res["secs"])
    json.dump(res, open(f"{W}/irt_v2.json", "w"), indent=0)
    return res
SUB_ITEMS = []; IRT_RES = {}
def set_irt(res):
    IRT_RES.clear(); IRT_RES.update(res)
    IRT.clear(); IRT.update(res); IRT["a"] = {t: np.array(res["a"][t]) for t in TASKS}; IRT["b"] = {t: np.array(res["b"][t]) for t in TASKS}
    SUB_ITEMS[:] = ordered([FITBY[i] for i in res["subset_ids"]])
    json.dump(SUB_ITEMS, open(f"{TMP}/fit_sub.json", "w"))

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
