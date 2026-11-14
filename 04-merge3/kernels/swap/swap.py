import subprocess, time, os, json, glob, threading, shutil, sys
def sh(c, t=None):
    print(f"\n$ {c}", flush=True); t0 = time.time()
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
    print(r.stdout[-6000:], r.stderr[-4000:], f"[rc={r.returncode} {time.time()-t0:.0f}s]", flush=True); return r
T0 = time.time()
DATA = os.path.dirname(glob.glob("/kaggle/input/**/persist.py", recursive=True)[0]); print("DATA", DATA, os.listdir(DATA))
sys.path.insert(0, DATA)
sh("nvidia-smi -L; free -g; df -h /tmp")
def dl():
    import m3lib
    from concurrent.futures import ThreadPoolExecutor
    jobs = [("Qwen/Qwen3.5-4B", "/tmp/base")] + [(r, f"/tmp/donor_{s}") for s, r in zip(m3lib.SHORT, m3lib.DONORS)]
    with ThreadPoolExecutor(3) as ex: list(ex.map(lambda j: m3lib.download(*j), jobs))
    print("downloads done", time.time() - T0, flush=True)
th = threading.Thread(target=dl); th.start()
sh("pip install -q vllm==0.33.0 2>&1 | tail -2", t=2400)
th.join()
sh("du -sh /tmp/base /tmp/donor_*")
ext = r'''
import sys, json, time, threading, shutil
sys.path.insert(0, sys.argv[1]); import m3lib, torch, os
os.makedirs("/tmp/tv", exist_ok=True); st = {}
def lane(g, idx):
    torch.cuda.set_device(g)
    for i in idx:
        s = m3lib.SHORT[i]; t0 = time.time()
        st[s] = m3lib.extract_tv("/tmp/base", f"/tmp/donor_{s}", f"/tmp/tv/{s}.safetensors", device=f"cuda:{g}"); st[s]["secs"] = time.time() - t0
        if s != "arex_turbo": shutil.rmtree(f"/tmp/donor_{s}")
        print(s, st[s], flush=True)
ts = [threading.Thread(target=lane, args=(g, list(range(g, 5, 2)))) for g in (0, 1)]
[t.start() for t in ts]; [t.join() for t in ts]
json.dump(st, open("/kaggle/working/tv_stats.json", "w"), indent=1)
'''
open("/tmp/ext.py", "w").write(ext)
sh(f"python /tmp/ext.py {DATA}", t=3600)
sh("du -sh /tmp/tv/*; free -g")
mon = subprocess.Popen("while true; do nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> /kaggle/working/gpumon.txt; free -g | sed -n 2p >> /kaggle/working/gpumon.txt; sleep 20; done", shell=True)
ps = [subprocess.Popen(f"CUDA_VISIBLE_DEVICES={g} python {DATA}/swaptest.py {r} > /kaggle/working/log_{r}.txt 2>&1", shell=True)
      for g, r in ((0, "inplace"), (1, "disk"))]
for p in ps: p.wait()
mon.kill()
for r in ("inplace", "disk"):
    sh(f"grep -v -E 'it/s|s/it' /kaggle/working/log_{r}.txt | grep -E '\\[{r}|Error|error|Traceback|WARN.*not loaded' | tail -40")
    sh(f"grep -A30 Traceback /kaggle/working/log_{r}.txt | tail -60")
try:
    a = json.load(open("/kaggle/working/swap_inplace.json")); b = json.load(open("/kaggle/working/swap_disk.json"))
    summ = {}
    summ["ck_equal_inplace_vs_disk"] = a.get("ck_M") == b.get("ck_M")
    if a.get("ck_M_params") and b.get("ck_M_params"):
        diff = [k for k in set(a["ck_M_params"]) | set(b["ck_M_params"]) if a["ck_M_params"].get(k) != b["ck_M_params"].get(k)]
        summ["n_params"] = len(a["ck_M_params"]); summ["param_mismatch"] = diff[:20]; summ["n_mismatch"] = len(diff)
    summ["ck_base_roundtrip"] = a.get("ck_base") == a.get("ck_base_after"); summ["ck_M_again"] = a.get("ck_M") == a.get("ck_M_again")
    def cmp(x, y):
        X, Y = x["results"], y["results"]
        return dict(n=len(X), pred_agree=sum(X[k]["pred"] == Y[k]["pred"] for k in X),
                    exact_scores=sum(X[k]["scores"] == Y[k]["scores"] for k in X),
                    max_abs=max(abs(p - q) for k in X for p, q in zip(X[k]["scores"], Y[k]["scores"])))
    summ["errors"] = dict(inplace=a.get("errors"), disk=b.get("errors"))
    for nm, x, y in (("S_M_inplace_vs_disk", a.get("S_M"), b.get("S_M")), ("S_base_vs_base_after", a.get("S_base"), a.get("S_base_after")),
                     ("S_base_vs_M", a.get("S_base"), a.get("S_M"))):
        if x and y: summ[nm] = cmp(x, y)
    if "GPQA_drop_M" in b:
        X, Y = b["GPQA_full_M"]["results"], b["GPQA_drop_M"]["results"]
        import statistics
        offs = [max(p - q for p, q in zip(X[k]["scores"], Y[k]["scores"])) - min(p - q for p, q in zip(X[k]["scores"], Y[k]["scores"])) for k in X]
        summ["gpqa_drop"] = dict(n=len(X), pred_agree=sum(X[k]["pred"] == Y[k]["pred"] for k in X), max_offset_spread=max(offs),
                                 acc_full=sum(v["correct"] for v in X.values()) / len(X), acc_drop=sum(v["correct"] for v in Y.values()) / len(Y),
                                 dropped=sorted(set(v["dropped"] for v in Y.values())))
    for k in ("t_init", "t_swap_merge", "t_swap_base_dir", "t_swap_endpoint_dir", "t_swap_merge_3", "t_merge_to_disk"):
        summ[k] = a.get(k, b.get(k))
    for nm, src in (("A40_M", a), ("MM1000_M", a), ("S_M", a), ("GPQA_full_M", b), ("GPQA_drop_M", b)):
        if nm in src: summ["stats_" + nm] = src[nm]["stats"]
    json.dump(summ, open("/kaggle/working/swap_summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))
except Exception as e:
    import traceback; traceback.print_exc()
print("TOTAL", time.time() - T0)
