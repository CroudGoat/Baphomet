import json,struct,requests,sys
def hdr(repo,fn):
    u=f"https://huggingface.co/{repo}/resolve/main/{fn}"
    r=requests.get(u,headers={"Range":"bytes=0-7"},allow_redirects=True); n=struct.unpack("<Q",r.content)[0]
    r=requests.get(u,headers={"Range":f"bytes=8-{7+n}"},allow_redirects=True); h=json.loads(r.content); h.pop("__metadata__",None); return h
def keys(repo):
    try:
        idx=requests.get(f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json").json(); files=sorted(set(idx["weight_map"].values()))
    except Exception: files=["model.safetensors"]
    H={}
    for f in files: H.update(hdr(repo,f))
    return {k:(v["dtype"],tuple(v["shape"])) for k,v in H.items()}
repos=["Qwen/Qwen3.5-4B","InternScience/Agents-A1-4B","Jackrong/DeepSeek-V4-Pro-Qwen3.5-4B","empero-ai/Qwen3.8-4B-Distill","Jackrong/Qwopus3.5-4B-v3","BAAI/AREX-Turbo"]
K={r:keys(r) for r in repos}
json.dump({r:{k:list(map(str,v)) for k,v in K[r].items()} for r in K},open("donor_keys.json","w"))
b=K[repos[0]]
import math
print("base",len(b),sum(math.prod(s) for d,s in b.values())/1e9,"B params", {d for d,s in b.values()})
for r in repos[1:]:
    d=K[r]; miss=set(b)-set(d); extra=set(d)-set(b); sh=[k for k in set(b)&set(d) if b[k][1]!=d[k][1]]
    print(r,len(d),"missing",len(miss),list(miss)[:3],"extra",len(extra),list(extra)[:3],"shape-diff",sh[:5],{x for x,s in d.values()})
