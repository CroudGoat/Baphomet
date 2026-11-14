#!/bin/bash
cd /content
pkill -f orchestrate.sh
echo "$(date) orch2 start" >> /content/orch.log
# wait for qwen subset completion
while true; do
  n=$(python3 -c "
import json
s=set(json.load(open('/content/subset_ids.json')))
d=set()
for l in open('/content/raw_qwen35.jsonl'):
    try: d.add(json.loads(l)['id'])
    except: pass
print(len(s-d))")
  [ "$n" = "0" ] && break
  sleep 30
done
echo "$(date) qwen subset complete" >> /content/orch.log
pkill -f client.py; sleep 3; pkill -f 'vllm serve'; sleep 10; pkill -9 -f VLLM; pkill -9 -f 'vllm serve'; sleep 5
nohup vllm serve /content/models/baphomet --served-model-name m --dtype float16 --max-model-len 12288 \
    --gpu-memory-utilization 0.95 --max-num-seqs 28 --limit-mm-per-prompt '{"image":0,"video":0}' --port 8000 > /content/vllm_baphomet.log 2>&1 &
for i in $(seq 1 180); do curl -s localhost:8000/health >/dev/null && break; sleep 10; done
echo "$(date) baphomet server up" >> /content/orch.log
SUBSET=/content/subset_ids.json CONC=28 MAXPHASE=1 python client.py /content/models/baphomet baphomet /content/raw_baphomet.jsonl >> /content/client_baphomet.log 2>&1
echo "$(date) baphomet subset done" >> /content/orch.log
