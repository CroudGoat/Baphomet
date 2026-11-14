#!/bin/bash
# kw.sh KERNEL N : last N progress lines (live log stream, ~60 s)
source /workspace/work/merge3/kenv.sh
timeout 100 python /workspace/work/evogm/klogs.py $1 55 >/dev/null 2>&1
grep -o '"data":"[0-9][0-9]:[0-9][0-9]:[0-9][0-9] \[[^"]*' /tmp/klive.txt | sed 's/^"data":"//; s/\\n$//' | grep -v "MON\|\] disk" | tail -n ${2:-10} | cut -c1-260
date -u +%H:%M:%S
