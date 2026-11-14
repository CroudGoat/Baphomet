# Persistent per-GPU worker: one vLLM engine (base tokenizer) kept alive for the whole run; weights swapped in place.
# usage: CUDA_VISIBLE_DEVICES=g python gpuworker.py BASE_DIR CMD_DIR
# Protocol: the orchestrator writes CMD_DIR/cmd_<n>.json; the worker executes commands in order and writes
# CMD_DIR/res_<n>.json (atomic rename). Commands:
#   {"op": "load_dir", "path": ...}                                           in-place reload of a full checkpoint
#   {"op": "load_merge", "tv": [...], "w": [...], "d": [...], "seed": int}   dare_ties merge streamed into the engine
#   {"op": "eval", "items": path, "drop_shared": bool}                        loglik MC eval -> results
#   {"op": "checksum"}  {"op": "exit"}
import os, sys, json, time, traceback, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import persist

def main():
    base, cdir = sys.argv[1], sys.argv[2]
    E = persist.Engine(base, tokenizer=base)
    json.dump(dict(ok=True, t_init=E.t_init), open(f"{cdir}/ready.tmp", "w")); os.rename(f"{cdir}/ready.tmp", f"{cdir}/ready.json")
    n = 0
    while True:
        p = f"{cdir}/cmd_{n}.json"
        if not os.path.exists(p):
            time.sleep(0.5); continue
        cmd = json.load(open(p)); t0 = time.time(); res = dict(ok=True)
        try:
            op = cmd["op"]
            if op == "exit":
                res["bye"] = True
            elif op == "load_dir":
                res["t"] = E.load_dir(cmd["path"])
            elif op == "load_merge":
                res["t"] = E.load_merge(base, cmd["tv"], cmd["w"], cmd["d"], cmd["seed"])
            elif op == "eval":
                r, st = E.evaluate(json.load(open(cmd["items"])), drop_shared=cmd.get("drop_shared", False))
                res.update(results=r, stats=st)
            elif op == "checksum":
                res["ck"] = E.checksum()[0]
        except Exception as e:
            res = dict(ok=False, error=repr(e), tb=traceback.format_exc()[-4000:])
        res["secs"] = time.time() - t0
        json.dump(res, open(f"{cdir}/res_{n}.tmp", "w")); os.rename(f"{cdir}/res_{n}.tmp", f"{cdir}/res_{n}.json")
        print(f"[gpuworker] cmd {n} {cmd.get('op')} ok={res['ok']} {res['secs']:.1f}s", flush=True)
        if cmd.get("op") == "exit": return
        n += 1

if __name__ == "__main__":
    main()
