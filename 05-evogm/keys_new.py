import json,struct,requests,math,sys,os
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
old=json.load(open('/workspace/work/merge3/donor_keys.json'))
K={r:{k:(v[0],v[1]) for k,v in d.items()} for r,d in old.items()}
for r in ["TokenRhythm/NeoHorse-1-4B","allenai/tmax-4b","Jackrong/Qwopus3.5-4B-Coder","CloudGoat/Baphomet-1.0-4B"]:
    K[r]={k:(v[0],str(v[1])) for k,v in keys(r).items()}
json.dump(K,open("donor_keys.json","w"))
b=K["Qwen/Qwen3.5-4B"]
for r,d in K.items():
    miss=set(b)-set(d); extra=set(d)-set(b); sh=[k for k in set(b)&set(d) if str(b[k][1])!=str(d[k][1])]
    print(r,len(d),"missing",len(miss),sorted(miss)[:2],"extra",len(extra),sorted(extra)[:3],"shape-diff",sh[:3],{x for x,s in d.values()})
