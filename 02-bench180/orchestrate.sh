#!/bin/bash
cd /content
serve() { # $1 model dir name
  pkill -f client.py; pkill -f 'vllm serve'; sleep 10; pkill -9 -f VLLM; pkill -9 -f 'vllm serve'; sleep 5
  nohup vllm serve /content/models/$1 --served-model-name m --dtype float16 --max-model-len 12288 \
    --gpu-memory-utilization 0.95 --max-num-seqs 28 --limit-mm-per-prompt '{"image":0,"video":0}' --port 8000 > /content/vllm_$1.log 2>&1 &
  for i in $(seq 1 180); do curl -s localhost:8000/health >/dev/null && break; sleep 10; done
  echo "$(date) server $1 up" >> /content/orch.log
}
run() { # $1 model $2 maxphase
  echo "$(date) start $1 phase<=$2" >> /content/orch.log
  CONC=28 MAXPHASE=$2 python client.py /content/models/$1 $1 /content/raw_$1.jsonl >> /content/client_$1.log 2>&1
  echo "$(date) done $1 phase<=$2" >> /content/orch.log
}
# qwen35 server may already be starting
for i in $(seq 1 180); do curl -s localhost:8000/health >/dev/null && break; sleep 10; done
echo "$(date) initial server up" >> /content/orch.log
run qwen35 1
serve baphomet; run baphomet 1; run baphomet 2
serve qwen35; run qwen35 2; run qwen35 3
serve baphomet; run baphomet 3
pkill -f 'vllm serve'
echo "$(date) ALL FINISHED" >> /content/orch.log
