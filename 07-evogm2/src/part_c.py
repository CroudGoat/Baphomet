
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
