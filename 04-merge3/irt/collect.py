import pandas as pd, numpy as np, requests, json, os, sys, time, random, threading
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
from rangefile import RangeFile
N = int(sys.argv[1]) if len(sys.argv)>1 else 1000
c = pd.read_parquet("contents.parquet")
c = c[~c["Flagged"].fillna(False)]
names = sorted(set(c["fullname"]))
random.Random(42).shuffle(names)
names = names[:N]
TASKS = {"leaderboard_mmlu_pro":"acc","leaderboard_gpqa_diamond":"acc_norm","leaderboard_gpqa_main":"acc_norm","leaderboard_gpqa_extended":"acc_norm"}
os.makedirs("resp", exist_ok=True)
lock = threading.Lock()
def one(name):
    out = f"resp/{name.replace('/','__')}.json"
    if os.path.exists(out): return "cached"
    R = f"open-llm-leaderboard/{name.replace('/','__')}-details"
    s = requests.Session()
    try:
        s.post(f"https://huggingface.co/datasets/{R}/ask-access", json={}, allow_redirects=False, timeout=60)
        pj = s.get(f"https://huggingface.co/api/datasets/{R}/parquet", timeout=60).json()
        res = {}
        for t, m in TASKS.items():
            key = [k for k in pj if k.endswith("__"+t)]
            if not key: return f"missing {t}"
            url = pj[key[0]]["latest"][0]
            pf = pq.ParquetFile(RangeFile(url, s), pre_buffer=True)
            cols = ["doc_id", m] + (["doc_hash"] if "gpqa" in t else [])
            tb = pf.read(columns=cols).to_pydict()
            res[t] = {"doc_id": tb["doc_id"], "y": tb[m], "doc_hash": tb.get("doc_hash")}
        json.dump(res, open(out, "w"))
        return "ok"
    except Exception as e:
        return "err " + repr(e)[:150]
t0 = time.time(); stats = {}
with ThreadPoolExecutor(24) as ex:
    for i, (n, r) in enumerate(zip(names, ex.map(one, names))):
        k = r.split()[0]; stats[k] = stats.get(k, 0) + 1
        if i % 50 == 0: print(i, n, r, stats, f"{time.time()-t0:.0f}s", flush=True)
print("DONE", stats, flush=True)
json.dump(names, open("sampled_models.json","w"))
