from huggingface_hub import hf_hub_download, list_repo_files
import json, hashlib, os
repos = {"base":"Qwen/Qwen3.5-4B","agents_a1":"InternScience/Agents-A1-4B","dsv4pro":"Jackrong/DeepSeek-V4-Pro-Qwen3.5-4B",
 "qwen38distill":"empero-ai/Qwen3.8-4B-Distill","qwopus_v3":"Jackrong/Qwopus3.5-4B-v3","arex_turbo":"BAAI/AREX-Turbo"}
want = ["tokenizer.json","tokenizer_config.json","vocab.json","merges.txt","special_tokens_map.json","added_tokens.json","chat_template.jinja","config.json"]
info = {}
for s, r in repos.items():
    fs = list_repo_files(r); d = f"toks/{s}"; info[s] = {}
    for f in want:
        if f in fs:
            p = hf_hub_download(r, f, local_dir=d); info[s][f] = hashlib.sha256(open(p,"rb").read()).hexdigest()[:16]
print(json.dumps(info, indent=1))
