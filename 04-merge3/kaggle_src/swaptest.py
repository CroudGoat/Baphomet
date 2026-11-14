# Debug: validate in-place weight reloading in a persistent vLLM engine + measure swap time and throughput.
# usage: CUDA_VISIBLE_DEVICES=g python swaptest.py ROLE  (ROLE = inplace | disk)
import os, sys, json, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import persist  # sets VLLM env first
import m3lib, numpy as np
ROLE = sys.argv[1]
DATA = os.path.dirname(os.path.abspath(__file__))
W = "/kaggle/working"
TV = [f"/tmp/tv/{s}.safetensors" for s in m3lib.SHORT]
M = dict(weights=[0.3, 0.2, 0.2, 0.15, 0.15], densities=[0.6, 0.7, 0.5, 0.8, 0.9], seed=12345)
pool = json.load(open(f"{DATA}/fitness_pool.json"))
def strip(p): return dict(key=p["key"], ctx=p["ctx"], conts=p["conts"], gold=p["gold"])
mm = [p for p in pool if p["task"] == "mmlu_pro"]; gp = [p for p in pool if p["task"] == "gpqa"]
rnd = random.Random(7)
S = [strip(p) for p in rnd.sample(mm, 120) + rnd.sample(gp, 80)]
anch = json.load(open(f"{DATA}/anchors.json")); A40 = [strip(a) for t in anch for a in anch[t]]
sub_idx = np.random.RandomState(420).choice(len(mm), 1000, replace=False)
MM1000 = [strip(mm[i]) for i in sorted(sub_idx)]
GP = [strip(p) for p in gp]
out = dict(role=ROLE); T0 = time.time()
def save(): json.dump(out, open(f"{W}/swap_{ROLE}.json", "w"), indent=1)
def lg(*a): print(f"[{ROLE} {time.time()-T0:6.0f}s]", *a, flush=True)
def ev(name, items, **kw):
    r, st = E.evaluate(items, **kw); out[name] = dict(stats=st, results=r); lg(name, st); save(); return r, st

import traceback
def stage(name, f):
    try:
        f()
    except Exception as e:
        out.setdefault("errors", []).append(f"{name}: {e!r}"[:2000]); lg("STAGE FAILED", name, traceback.format_exc()[-3000:]); save()

if ROLE == "inplace":
    E = persist.Engine("/tmp/base"); out["t_init"] = E.t_init; lg("init", E.t_init)
    def s1(): out["ck_base"] = E.checksum()[0]
    stage("ck_base", s1)
    stage("S_base", lambda: ev("S_base", S))
    def s2():
        ts = []
        for _ in range(2):
            ts.append(E.load_merge("/tmp/base", TV, **M)); lg("merge swap", ts[-1])
        out["t_swap_merge"] = ts
        out["ck_M"], out["ck_M_params"] = E.checksum()
        ev("S_M", S)
    stage("merge_swap", s2)
    def s3():
        out["t_swap_base_dir"] = E.load_dir("/tmp/base"); out["ck_base_after"] = E.checksum()[0]
        ev("S_base_after", S)
    stage("base_dir_swap", s3)
    def s4(): out["t_swap_endpoint_dir"] = E.load_dir("/tmp/donor_arex_turbo"); lg("endpoint dir swap", out["t_swap_endpoint_dir"])
    stage("endpoint_dir_swap", s4)
    def s5():
        out["t_swap_merge_3"] = E.load_merge("/tmp/base", TV, **M); out["ck_M_again"] = E.checksum()[0]
        ev("A40_M", A40)
    stage("merge_again", s5)
    stage("MM1000", lambda: ev("MM1000_M", MM1000))
else:
    def d1():
        t0 = time.time(); m3lib.merge_dare_ties("/tmp/base", TV, M["weights"], M["densities"], "/tmp/mdisk", seed=M["seed"], device="cuda:0")
        out["t_merge_to_disk"] = time.time() - t0; lg("merge to disk", out["t_merge_to_disk"])
        import torch; torch.cuda.empty_cache()
    d1()
    E = persist.Engine("/tmp/mdisk", tokenizer="/tmp/base"); out["t_init"] = E.t_init; lg("init", E.t_init)
    def d2(): out["ck_M"], out["ck_M_params"] = E.checksum()
    stage("ck_M", d2)
    stage("S_M", lambda: ev("S_M", S))
    stage("GPQA_full", lambda: ev("GPQA_full_M", GP))
    stage("GPQA_drop", lambda: ev("GPQA_drop_M", GP, drop_shared=True))
save(); lg("DONE")
