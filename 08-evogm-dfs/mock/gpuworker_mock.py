# mock GPU worker for EvoGM-DFS: deterministic 2PL-like correctness per (item, layer path) from a synthetic DFS landscape, token
# lengths, simulated time (simt cost model x 1.25 for the 40-layer engine, concurrency capped at 24 by the smaller KV cache, scaled by
# MOCK_TS), racing via partial/stop files, fault injection (MOCK_CRASH g:boot:ngen:kind, MOCK_RACEERR, MOCK_BROKEN).
# Landscape: theta = mean donor quality (r1i1 1.15, b11 .95, b10 .30) - 0.5/missing block index - 0.15/duplicate (+0.05 for blocks
# 4,5) - 0.6/out-of-order step + 0.12 (b11 at block 2) + 0.10 (b11 at block 5) - 1.5 * mean(log2 fold)^2 ; theta < -0.5 = broken
# (every output runs to max_tokens, wrong).  Hidden optimum: r1i1 with b11 blocks 2 and 5 (theta ~1.32 > plain r1i1 1.15).
import os, sys, json, time, hashlib, math
Q = [1.15, 0.95, 0.30]
def H(s): return int(hashlib.md5(s.encode()).hexdigest(), 16)
def U(s): return (H(s) % 10**9) / 1e9
def blocks_of(plan):
    out = []
    for j in range(0, len(plan), 4):
        d, src, f = plan[j]
        if float(f) == 0.0: continue
        out.append((int(d), int(src) // 4, float(f)))
    return out
def pkey(plan): return ";".join(f"{d},{b},{f:.4f}" for d, b, f in blocks_of(plan))
def theta_of(plan):
    if os.environ.get("MOCK_BROKEN"): return -3.0
    B = blocks_of(plan)
    if not B: return -3.0
    th = sum(Q[d] for d, _, _ in B) / len(B)
    idx = [b for _, b, _ in B]
    th -= 0.5 * sum(1 for k in range(8) if k not in idx)
    for k in range(8):
        c = idx.count(k)
        if c > 1: th += (c - 1) * (0.05 if k in (4, 5) else -0.15)
    th -= 0.6 * sum(1 for a, b in zip(idx, idx[1:]) if b < a)
    th += 0.12 * any(d == 1 and b == 2 for d, b, _ in B) + 0.10 * any(d == 1 and b == 5 for d, b, _ in B)
    th -= 1.5 * sum(math.log2(f) ** 2 for _, _, f in B) / len(B)
    return th
def item_p(it, th):
    iid = it["id"]; a = 0.7 + 1.5 * U("a" + iid)
    lo, hi = dict(mmlupro=(-2.5, 1.5), gpqa=(-1.0, 2.5), bfcl=(-2.0, 1.0))[it["bench"]]
    b = lo + (hi - lo) * U("b" + iid)
    return 1 / (1 + math.exp(-a * (th - b)))
def length(it, th, key, maxtok):
    if th < -0.5: return maxtok
    base_len = dict(mmlupro=1100, gpqa=2500, bfcl=220)[it["bench"]]
    f = 1 + 0.4 * max(0.0, 1.0 - th)
    return min(int(base_len * f * math.exp(0.6 * (U("L" + it["id"] + key) - 0.5) * 2)), maxtok)
def gen_recs(items, th, key, maxtok):
    recs = []
    for it in items:
        L = length(it, th, key, maxtok)
        ok = th >= -0.5 and U("c" + it["id"] + key) < item_p(it, th) and L < maxtok
        recs.append(dict(id=it["id"], prompt_tokens=10, completion_tokens=L, finish_reason="length" if L >= maxtok else "stop", output="OK" if ok else "no"))
    return recs
def simsecs(lens, cap=24, slow=1.25):   # simt: continuous batching FCFS, 0.034 s/step + 0.0055 s per active seq-step
    lens = list(lens); q = 0; act = []; steps = 0; sa = 0
    while q < len(lens) or act:
        while len(act) < cap and q < len(lens): act.append(lens[q]); q += 1
        st = min(act); steps += st; sa += st * len(act)
        act = [x - st for x in act if x - st > 0]
    return (0.034 * steps + 0.0055 * sa) * slow
def main():
    base, cdir = sys.argv[1], sys.argv[2]
    g = os.environ.get("CUDA_VISIBLE_DEVICES"); boot = cdir.rsplit("_", 1)[1]
    TS = float(os.environ.get("MOCK_TS", "0.002"))
    time.sleep(450 * TS)
    json.dump(dict(ok=True, mock=True), open(cdir + "/ready.tmp", "w")); os.rename(cdir + "/ready.tmp", cdir + "/ready.json")
    plan = None; n = 0; ngen = 0
    CRASH = os.environ.get("MOCK_CRASH", ""); RACEERR = os.environ.get("MOCK_RACEERR", "")
    while True:
        p = f"{cdir}/cmd_{n}.json"
        if not os.path.exists(p): time.sleep(0.02); continue
        cmd = json.load(open(p)); res = dict(ok=True); t = time.time()
        if cmd["op"] == "exit": res["bye"] = True
        elif cmd["op"] == "load_dfs":
            plan = cmd["plan"]; assert len(plan) == 40 and all(int(s) % 4 == j % 4 for j, (d, s, f) in enumerate(plan)), "bad plan"
            assert os.path.isdir(cmd["shards"])
            time.sleep(45 * TS); res.update(t=45 * TS, ptr_changed=0)
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
                items = json.load(open(cmd["items"])); mt = int(cmd["max_tokens"])
                th = theta_of(plan); key = pkey(plan)
                recs = gen_recs(items, th, key, mt)
                total = simsecs([r["completion_tokens"] for r in recs]) * TS
                if cmd.get("partial") and not RACEERR:
                    order = sorted(range(len(recs)), key=lambda k: (recs[k]["completion_tokens"], recs[k]["id"])); done = []; ab = 0
                    for s, k in enumerate(order):
                        done.append(recs[k])
                        if s % 5 == 4 or s == len(order) - 1:
                            time.sleep(total * 5 / len(order))
                            json.dump(done, open(cmd["partial"] + ".tmp", "w")); os.replace(cmd["partial"] + ".tmp", cmd["partial"])
                            t_ack = time.time()
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
