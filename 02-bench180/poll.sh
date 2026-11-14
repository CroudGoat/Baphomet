#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
cd /workspace/work/bench
if ! command -v colab >/dev/null; then pip install --user -q google-colab-cli >/dev/null 2>&1; fi
mkdir -p ~/.config/gcloud ~/.config/colab-cli
[ -f ~/.config/gcloud/application_default_credentials.json ] || cp .adc/application_default_credentials.json ~/.config/gcloud/
[ -f ~/.config/colab-cli/sessions.json ] || cp .adc/sessions.json ~/.config/colab-cli/
[ -f /tmp/st.py ] || cat > /tmp/st.py <<'PY'
import subprocess
print(subprocess.run("date; tail -4 /content/orch.log; for f in /content/client_qwen35.log /content/client_baphomet.log; do [ -f $f ] && (echo $f; tail -1 $f); done; for f in /content/vllm_qwen2.log /content/vllm_baphomet.log; do [ -f $f ] && (echo $f; grep -a 'throughput' $f | tail -1 | sed -E 's/.*Engine 000: //' | cut -c1-180); done; wc -l /content/raw_*.jsonl", shell=True, capture_output=True, text=True).stdout)
PY
timeout 60 colab --auth=adc exec -s bench -f /tmp/st.py 2>&1 | grep -av "FA version" | tail -12
for m in qwen35 baphomet; do timeout 40 colab --auth=adc download -s bench /content/raw_$m.jsonl raw_$m.jsonl.tmp >/dev/null 2>&1 && mv raw_$m.jsonl.tmp raw_$m.jsonl; done
rm -f raw_*.tmp; wc -l raw_*.jsonl 2>/dev/null
python3 -c "
import json
s=set(json.load(open('subset_ids.json')))
for m in ['qwen35','baphomet']:
    try: d=set(json.loads(l)['id'] for l in open(f'raw_{m}.jsonl'))
    except FileNotFoundError: d=set()
    print(m,'subset done',len(s&d),'/',len(s))
"
