
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
