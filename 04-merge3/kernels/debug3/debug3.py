import subprocess, time, os, json, glob, random
def sh(c, t=None):
    print(f"\n$ {c}", flush=True); t0=time.time()
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
    print(r.stdout[-8000:], r.stderr[-3000:], f"[rc={r.returncode} {time.time()-t0:.0f}s]", flush=True); return r
DATA = os.path.dirname(glob.glob("/kaggle/input/**/m3lib.py", recursive=True)[0]); print("DATA", DATA)
sh("pip install -q vllm==0.33.0 2>&1 | tail -1", t=1800)
sh("python -c \"from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.5-4B', local_dir='/tmp/base', allow_patterns=['*.safetensors','*.json','*.jinja','*.txt'])\"", t=1800)
pool = json.load(open(f"{DATA}/fitness_pool.json"))
rnd = random.Random(0)
mm = [p for p in pool if p["task"]=="mmlu_pro"]; gp = [p for p in pool if p["task"]=="gpqa"]
samp = rnd.sample(mm, 400) + rnd.sample(gp, 200)
items = [dict(key=p["key"], ctx=p["ctx"], conts=p["conts"], gold=p["gold"]) for p in samp]
json.dump(items, open("/tmp/s.json","w"))
sh(f"CUDA_VISIBLE_DEVICES=0 python {DATA}/evalworker.py /tmp/base /tmp/s.json /tmp/r_pc1.json 1 > /tmp/l0.txt 2>&1 & CUDA_VISIBLE_DEVICES=1 python {DATA}/evalworker.py /tmp/base /tmp/s.json /tmp/r_pc0.json 0 > /tmp/l1.txt 2>&1; wait; grep -v -E 'it/s|s/it' /tmp/l0.txt | tail -40; echo =====; grep -v -E 'it/s|s/it' /tmp/l1.txt | tail -15", t=3600)
ref = r'''
import json, sys, os
sys.path.insert(0, sys.argv[1])
from evalworker import encode_pair
def main():
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    items = json.load(open("/tmp/s.json"))[400:440]
    llm = LLM("/tmp/base", dtype="float16", max_model_len=8192, gpu_memory_utilization=0.88, enforce_eager=True, enable_prefix_caching=False, seed=0, limit_mm_per_prompt={"image":0,"video":0})
    tok = llm.get_tokenizer(); reqs=[]; meta=[]
    for it in items:
        for ci,c in enumerate(it["conts"]):
            a,b = encode_pair(tok, it["ctx"], c); reqs.append(TokensPrompt(prompt_token_ids=a+b)); meta.append((it["key"],ci,len(b)))
    outs = llm.generate(reqs, SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0), use_tqdm=False)
    S={}
    for (k,ci,nb),o in zip(meta,outs):
        ids=o.prompt_token_ids; pl=o.prompt_logprobs; S.setdefault(k,[]).append(sum(pl[j][ids[j]].logprob for j in range(len(ids)-nb,len(ids))))
    R=json.load(open("/tmp/r_pc1.json"))["results"]
    agree=sum(int(max(range(4),key=lambda i:S[k][i])==R[k]["pred"]) for k in S); md=max(abs(a-b) for k in S for a,b in zip(S[k],R[k]["scores"]))
    print("GPQA ref agree", agree, "/", len(S), "max |score diff|", md, flush=True)
if __name__=="__main__": main()
'''
open("/tmp/ref.py","w").write(ref)
sh(f"CUDA_VISIBLE_DEVICES=0 python /tmp/ref.py {DATA} 2>&1 | grep -v -E 'it/s|s/it' | tail -5", t=1800)
for f in ["/tmp/r_pc1.json","/tmp/r_pc0.json"]:
    if os.path.exists(f):
        d=json.load(open(f)); R=d["results"]
        acc_m=sum(R[i["key"]]["correct"] for i in items[:400])/400; acc_g=sum(R[i["key"]]["correct"] for i in items[400:])/200
        print(f, {k:v for k,v in d.items() if k!="results"}, "acc mmlu", acc_m, "gpqa", acc_g, "nmiss items", sum(r["nmiss"]>0 for r in R.values()), "max ctx tok", max(r["ntok"] for r in R.values()))
a=json.load(open("/tmp/r_pc1.json"))["results"]; b=json.load(open("/tmp/r_pc0.json"))["results"]
print("pred agree pc1 vs pc0", sum(a[k]["pred"]==b[k]["pred"] for k in a), "/", len(a), "max score diff", max(abs(x-y) for k in a for x,y in zip(a[k]["scores"],b[k]["scores"])))
