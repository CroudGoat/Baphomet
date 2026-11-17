# compare two EvoGM-DFS mock runs' final outputs (dfs_log by phenotype key, dfs_val_log, winner, history, final)
import json, sys, os
def load(d):
    W = d + "/kaggle/working"
    ev = {}
    for l in open(W + "/dfs_log.jsonl"):
        r = json.loads(l); ev.setdefault(r["lkey"], {}).update(r)
    vl = {}
    for l in open(W + "/dfs_val_log.jsonl"):
        r = json.loads(l); vl[r["lkey"]] = r
    res = json.load(open(W + "/evo_result.json"))
    fin = json.load(open(W + "/final_meta.json")) if os.path.exists(W + "/final_meta.json") else None
    return ev, vl, res, fin
A, B = load(sys.argv[1]), load(sys.argv[2])
keys = ("fitness", "full_fitness", "raced", "n_aborted", "round", "iter", "kind", "failed", "screened", "completion_tag")
ev_diff = [k for k in set(A[0]) | set(B[0]) if k not in A[0] or k not in B[0] or any(A[0][k].get(x) != B[0][k].get(x) for x in keys)]
vl_diff = [k for k in set(A[1]) | set(B[1]) if k not in A[1] or k not in B[1] or A[1][k].get("val_fitness") != B[1][k].get("val_fitness")]
print("evo records", len(A[0]), len(B[0]), "differing", len(ev_diff))
for k in ev_diff[:5]: print("  diff", k[:60], {x: (A[0].get(k, {}).get(x), B[0].get(k, {}).get(x)) for x in keys})
print("val records", len(A[1]), len(B[1]), "differing", len(vl_diff))
wa, wb = A[2]["winner"], B[2]["winner"]
print("winner same:", (wa or {}).get("lkey") == (wb or {}).get("lkey"), (wa or {}).get("summary"), (wa or {}).get("val_fitness"), (wb or {}).get("val_fitness"))
hk = lambda H: [(h.get("round"), h.get("iter"), h.get("best_value", h.get("best")), h.get("n_new", h.get("n"))) for h in H]
print("hist same:", hk(A[2]["hist"]) == hk(B[2]["hist"]))
print("final same:", (A[3] or {}).get("quick") == (B[3] or {}).get("quick"), (A[3] or {}).get("quick"))
print("REPRODUCED" if not ev_diff and not vl_diff and (wa or {}).get("lkey") == (wb or {}).get("lkey") and hk(A[2]["hist"]) == hk(B[2]["hist"]) else "DIFFERENT")
