import subprocess, time, os, json, sys
def sh(c, t=None):
    print(f"\n$ {c}", flush=True); t0=time.time()
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
    print(r.stdout[-6000:], r.stderr[-4000:], f"[rc={r.returncode} {time.time()-t0:.0f}s]", flush=True); return r
sh("nvidia-smi -L; python -V; pip list 2>/dev/null | grep -i -E '^(torch|vllm|transformers|triton|numpy|accelerate|datasets|safetensors) '")
sh("pip install -q vllm==0.33.0 2>&1 | tail -2", t=1800)
sh("pip list 2>/dev/null | grep -i -E '^(torch|vllm|transformers|triton|flashinfer|numpy) '")
sh("python -c 'import vllm,torch;print(vllm.__version__,torch.__version__,torch.cuda.device_count())'")
t0=time.time()
sh("cd /tmp && HF_HUB_ENABLE_HF_TRANSFER=0 python -c \"from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-4B', local_dir='/tmp/base'))\"", t=1800)
print("download s", time.time()-t0)
sh("du -sh /tmp/base; df -h /tmp")
test = r'''
import os, sys
def main():
    
    import os, time, json, sys
    os.environ.setdefault("VLLM_USE_V1","1")
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    import string
    gpu = sys.argv[1]
    t0=time.time()
    llm = LLM("/tmp/base", dtype="float16", max_model_len=8192, gpu_memory_utilization=0.90, enforce_eager=True,
              enable_prefix_caching=True, max_logprobs=100, limit_mm_per_prompt={"image":0,"video":0}, seed=0)
    print("load s", time.time()-t0, flush=True)
    tok = llm.get_tokenizer()
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    val = ds["validation"]; test = ds["test"]
    def d2t(doc):
        s = f"{doc['question']}\n"
        for i,o in enumerate(doc["options"]): s += f"{string.ascii_uppercase[i]}. {o}\n"
        return s + "Answer:"
    shots = "".join(d2t(val[i]) + " " + val[i]["answer"] + "\n\n" for i in range(5))
    docs = [test[i] for i in range(0, 12000, 40)][:300]
    ctxs = [shots + d2t(d) for d in docs]
    L = [len(tok(c).input_ids) for c in ctxs]; print("ctx tokens mean", sum(L)/len(L), "max", max(L), flush=True)
    letters = {string.ascii_uppercase[i]: tok(" "+string.ascii_uppercase[i], add_special_tokens=False).input_ids for i in range(10)}
    print("letter toks", letters, flush=True)
    sp = SamplingParams(max_tokens=1, temperature=0, logprobs=100)
    t0=time.time(); outs = llm.generate(ctxs, sp); dt=time.time()-t0
    print(f"MMLU-Pro 300 items gen-logprob: {dt:.1f}s, {sum(L)/dt:.0f} prompt tok/s", flush=True)
    corr=0; miss=0
    for d,o in zip(docs,outs):
        lp = o.outputs[0].logprobs[0]
        best=None; bv=-1e9
        for i in range(len(d["options"])):
            t = letters[string.ascii_uppercase[i]][0]
            if t in lp and lp[t].logprob > bv: bv=lp[t].logprob; best=string.ascii_uppercase[i]
        if best is None: miss+=1
        corr += best == d["answer"]
    print("acc", corr/len(docs), "miss", miss, flush=True)
    # prompt_logprobs path on 20 items x choices to cross-check
    t0=time.time(); agree=0
    reqs=[]; 
    for d,c in zip(docs[:20],ctxs[:20]):
        for i in range(len(d["options"])): reqs.append(c+" "+string.ascii_uppercase[i])
    outs2 = llm.generate(reqs, SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0))
    print("prompt_logprobs 20 items s", time.time()-t0, flush=True)
    k=0
    for d,o in zip(docs[:20],outs[:20]):
        sc=[]
        for i in range(len(d["options"])):
            pl = outs2[k].prompt_logprobs[-1]; tid = outs2[k].prompt_token_ids[-1]; sc.append(pl[tid].logprob); k+=1
        b1 = string.ascii_uppercase[max(range(len(sc)), key=lambda i: sc[i])]
        lp = o.outputs[0].logprobs[0]
        b2 = max([string.ascii_uppercase[i] for i in range(len(d["options"])) if letters[string.ascii_uppercase[i]][0] in lp], key=lambda L_: lp[letters[L_][0]].logprob)
        agree += b1==b2
    print("agree gen-vs-prompt_logprobs", agree, "/20", flush=True)
    
if __name__=='__main__':
    main()
'''
open("/tmp/t.py","w").write(test)
sh("cd /tmp && CUDA_VISIBLE_DEVICES=0 python t.py 0 > /tmp/t0.log 2>&1 & cd /tmp && sleep 20 && CUDA_VISIBLE_DEVICES=1 python t.py 1 > /tmp/t1.log 2>&1; wait; tail -c 5000 /tmp/t0.log; echo ======; tail -c 5000 /tmp/t1.log", t=3000)
sh("nvidia-smi --query-gpu=memory.used --format=csv")
