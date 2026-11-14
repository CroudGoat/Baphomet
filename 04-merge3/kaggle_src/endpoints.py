# Kernel E (MERGE^3 "Estimate" inputs): evaluate base + 5 donors (+ Baphomet as reference) on the FULL fitness pool
# (Open LLM Leaderboard v2 MMLU-Pro + GPQA(diamond/main/extended), test items excluded), loglik MC, no chat template.
import os, sys, json, subprocess, time, threading, shutil, glob
DATA = glob.glob("/kaggle/input/*merge3-data*")[0] if glob.glob("/kaggle/input/*merge3-data*") else "/kaggle/input/merge3-data"
sys.path.insert(0, DATA)
W = "/kaggle/working"
def sh(c, t=None):
    print("$", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
    print(r.stdout[-3000:], r.stderr[-3000:], "rc", r.returncode, flush=True); return r
t00 = time.time()
sh("nvidia-smi -L; df -h /tmp /kaggle/working; ls " + DATA)
sh("pip install -q vllm==0.33.0 2>&1 | tail -2", t=2400)
import m3lib
pool = json.load(open(f"{DATA}/fitness_pool.json"))
items = [dict(key=it["key"], ctx=it["ctx"], conts=it["conts"], gold=it["gold"]) for it in pool]
json.dump(items, open("/tmp/pool_items.json", "w"))
print("pool", len(items), flush=True)
MODELS = [("base", m3lib.BASE_ID)] + list(zip(m3lib.SHORT, m3lib.DONORS)) + [("baphomet", "CloudGoat/Baphomet-1.0-4B")]
lanes = {0: MODELS[0::2], 1: MODELS[1::2]}
PC = os.environ.get("PREFIX_CACHE", "1")
def lane(gpu):
    for short, repo in lanes[gpu]:
        out = f"{W}/endpoint_{short}.json"
        if os.path.exists(out): continue
        t0 = time.time(); d = f"/tmp/m_{short}"
        m3lib.download(repo, d); td = time.time() - t0
        r = subprocess.run(f"CUDA_VISIBLE_DEVICES={gpu} python {DATA}/evalworker.py {d} /tmp/pool_items.json {out}.tmp {PC} > {W}/log_{short}.txt 2>&1", shell=True)
        if r.returncode == 0:
            os.rename(out + ".tmp", out)
        m3lib.log(f"lane{gpu} {short} rc={r.returncode} download {td:.0f}s eval {time.time()-t0-td:.0f}s")
        if short != "base": shutil.rmtree(d, ignore_errors=True)
ths = [threading.Thread(target=lane, args=(g,)) for g in (0, 1)]
[t.start() for t in ths]; [t.join() for t in ths]
summ = {}
for f in sorted(glob.glob(f"{W}/endpoint_*.json")):
    d = json.load(open(f)); R = d["results"]; short = os.path.basename(f)[9:-5]
    by = {}
    for it in pool:
        r = R[it["key"]]; by.setdefault(it["task"], []).append(r["correct"])
    summ[short] = {t: round(sum(v) / len(v), 4) for t, v in by.items()} | dict(t_load=round(d["t_load"]), t_total=round(d["t_total"]),
               nmiss=sum(r["nmiss"] > 0 for r in R.values()))
    print(short, summ[short], flush=True)
json.dump(summ, open(f"{W}/endpoint_summary.json", "w"), indent=1)
print("TOTAL s", time.time() - t00)
