#!/bin/bash
# usage: poll.sh KERNEL SECONDS  -> waits up to SECONDS (<=110) printing status when not RUNNING
cd /workspace/work/merge3 && source kenv.sh
end=$((SECONDS+${2:-100}))
while [ $SECONDS -lt $end ]; do
  s=$(timeout 15 kaggle kernels status cloudgoat/$1 2>&1)
  case "$s" in *RUNNING*|*QUEUED*) sleep 25;; *) echo "$s"; exit 0;; esac
done
echo "$(date -u +%H:%M) $s"
