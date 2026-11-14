#!/bin/bash
# usage: poll.sh KERNEL SECONDS -> waits up to SECONDS printing status only when not RUNNING/QUEUED
source /workspace/work/merge3/kenv.sh
end=$((SECONDS+${2:-100}))
while [ $SECONDS -lt $end ]; do
  s=$(timeout 15 kaggle kernels status cloudgoat/$1 2>&1)
  case "$s" in *RUNNING*|*QUEUED*) sleep 20;; *) echo "$s"; exit 0;; esac
done
echo "$(date -u +%H:%M) $(echo $s | grep -o 'RUNNING\|QUEUED')"
