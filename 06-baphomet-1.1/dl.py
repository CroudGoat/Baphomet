import os, sys, time
os.environ.setdefault("HF_HOME", "/workspace/work/baphomet11/hf_home")
from huggingface_hub import snapshot_download
REV = {"Qwen/Qwen3.5-4B":"851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a","InternScience/Agents-A1-4B":"945c40a4aa6f534d434a353207b8d42ecf7a5293",
"Jackrong/DeepSeek-V4-Pro-Qwen3.5-4B":"5b0d0fb0445ccda45587b93a522c49b093d0c146","empero-ai/Qwen3.8-4B-Distill":"c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e",
"Jackrong/Qwopus3.5-4B-Coder":"e726694f407b1c95d037264ba61456d7182e18f4","Jackrong/Qwopus3.5-4B-v3":"99e41a88ff18e15cf49e30b054046513a676c9ba",
"BAAI/AREX-Turbo":"e455a798e6780f7a1e815c16d6041e495fa2f9e8","TokenRhythm/NeoHorse-1-4B":"56f0584bb40578a2c33b1b40a08ccd17243ad710",
"allenai/tmax-4b":"c733e09b808d837ae7450eccc8cb8dace07316d6"}
repo, d = sys.argv[1], sys.argv[2]
pat = None if len(sys.argv) > 3 and sys.argv[3] == "all" else ["*.safetensors", "*.json", "*.jinja", "*.txt"]
for a in range(4):
    try:
        t = time.time(); snapshot_download(repo, revision=REV[repo], local_dir=d, allow_patterns=pat, max_workers=8)
        print("done", repo, round(time.time() - t), flush=True); break
    except Exception as e:
        print("err", a, repr(e)[:300], flush=True); time.sleep(20)
