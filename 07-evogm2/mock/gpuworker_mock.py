# mock GPU worker for EvoGM v2: deterministic 2PL-like correctness per (item, lambda), token lengths, simulated time
# (simt-style cost model scaled by MOCK_TS), racing via partial/stop files, fault injection (MOCK_CRASH g:boot:ngen:kind, MOCK_RACEERR).
import os, sys, json, time, hashlib, math
def main():
    base, cdir = sys.argv[1], sys.argv[2]
    g = os.environ.get("CUDA_VISIBLE_DEVICES"); boot = cdir.rsplit("_", 1)[1]
    TS = float(os.environ.get("MOCK_TS", "0.002"))
    time.sleep(400 * TS)
    json.dump(dict(ok=True, mock=True), open(cdir + "/ready.tmp", "w")); os.rename(cdir + "/ready.tmp", cdir + "/ready.json")
    lams = None; n = 0; ngen = 0
    CRASH = os.environ.get("MOCK_CRASH", ""); RACEERR = os.environ.get("MOCK_RACEERR", "")
    BASEBFCL = os.environ.get("MOCK_BASEBFCL", "")
    B11 = [0.0116, 0.2039, 0.5901, 0.0013, 0.2972, 0.0323, 0.0030, -0.0007]
    SH = float(os.environ.get("MOCK_SHIFT", "1.0")); SHARP = float(os.environ.get("MOCK_SHARP", "2.5"))
    TGT = B11 + B11; TGT[2] += 0.15 * SH; TGT[8 + 4] += 0.12 * SH; TGT[8 + 1] -= 0.10 * SH
    def H(s): return int(hashlib.md5(s.encode()).hexdigest(), 16)
    def U(s): return (H(s) % 10**9) / 1e9
    def theta(l):
        if len(l) == 8: l = l + l
        return 1.0 - SHARP * sum((x - t) ** 2 for x, t in zip(l, TGT))
    def item_p(it, l):
        iid = it["id"]; bench = it["bench"]
        a = 0.7 + 1.5 * U("a" + iid)
        lo, hi = dict(mmlupro=(-2.5, 1.5), gpqa=(-1.0, 2.5), bfcl=(-2.0, 1.0))[bench]
        b = lo + (hi - lo) * U("b" + iid)
        return 1 / (1 + math.exp(-a * (theta(l) - b)))
    def length(it, l, maxtok):
        base_len = dict(mmlupro=1100, gpqa=2500, bfcl=220)[it["bench"]]
        f = 1 + 0.4 * max(0.0, 1.0 - theta(l))
        L = int(base_len * f * math.exp(0.6 * (U("L" + it["id"] + ",".join(f"{x:.6f}" for x in l)) - 0.5) * 2))
        return min(L, maxtok)
    def simsecs(lens):   # simt: continuous batching FCFS, 0.034 s/step + 0.0055 s per active seq-step
        lens = list(lens); q = 0; act = []; steps = 0; sa = 0
        while q < len(lens) or act:
            while len(act) < 48 and q < len(lens): act.append(lens[q]); q += 1
            st = min(act); steps += st; sa += st * len(act)
            act = [x - st for x in act if x - st > 0]
        return 0.034 * steps + 0.0055 * sa
    while True:
        p = f"{cdir}/cmd_{n}.json"
        if not os.path.exists(p): time.sleep(0.02); continue
        cmd = json.load(open(p)); res = dict(ok=True); t = time.time()
        if cmd["op"] == "exit": res["bye"] = True
        elif cmd["op"] == "load_merge":
            lams = [float(x) for x in cmd["lams"]]; assert len(lams) in (8, 16); time.sleep(100 * TS); res.update(t=100 * TS, ptr_changed=0)
        elif cmd["op"] == "gen":
            ngen += 1
            if CRASH:
                for spec in CRASH.split(";"):
                    cg, cb, cn, kind = spec.split(":")
                    if cg == g and cb == boot and int(cn) == ngen:
                        if kind == "exit": os._exit(3)
                        if kind == "hang": time.sleep(10**6)
                        if kind == "err": res = dict(ok=False, error="mock CUDA OOM")
            if res["ok"]:
                items = json.load(open(cmd["items"])); recs = []; mt = int(cmd["max_tokens"])
                key = ",".join(f"{x:.6f}" for x in lams)
                for it in items:
                    L = length(it, lams, mt); pc = item_p(it, lams)
                    if BASEBFCL and it["bench"] == "bfcl" and all(x == 0 for x in lams): pc = float(BASEBFCL)
                    ok = U("c" + it["id"] + key) < pc
                    recs.append(dict(id=it["id"], prompt_tokens=10, completion_tokens=L, finish_reason="length" if L >= mt else "stop", output="OK" if ok else "no"))
                total = simsecs([r["completion_tokens"] for r in recs]) * TS
                if cmd.get("partial") and not RACEERR:
                    order = sorted(range(len(recs)), key=lambda k: (recs[k]["completion_tokens"], recs[k]["id"])); done = []; ab = 0
                    for s, k in enumerate(order):
                        done.append(recs[k])
                        if s % 5 == 4 or s == len(order) - 1:
                            time.sleep(total * 5 / len(order))
                            json.dump(done, open(cmd["partial"] + ".tmp", "w")); os.replace(cmd["partial"] + ".tmp", cmd["partial"])
                            t_ack = time.time()   # wait for the controller's ack of this partial (deterministic racing in the mock)
                            while time.time() - t_ack < 20:
                                try:
                                    if int(open(cmd["partial"] + ".seen").read()) >= len(done): break
                                except Exception: pass
                                time.sleep(0.01)
                            if os.path.exists(cmd["stop"]) and s < len(order) - 1:
                                ids = {d["id"] for d in done}
                                for r in recs:
                                    if r["id"] not in ids: r.update(finish_reason="aborted", output="", completion_tokens=0); ab += 1
                                break
                    res["aborted"] = ab
                else:
                    if cmd.get("partial"): res["race_error"] = "mock race error"
                    time.sleep(total)
                json.dump(recs, open(cmd["out"], "w")); res.update(n=len(recs), gen_tokens=sum(r["completion_tokens"] for r in recs))
        res["secs"] = time.time() - t
        json.dump(res, open(f"{cdir}/res_{n}.tmp", "w")); os.rename(f"{cdir}/res_{n}.tmp", f"{cdir}/res_{n}.json")
        if cmd["op"] == "exit": sys.exit(0)
        n += 1
if __name__ == "__main__":
    main()
