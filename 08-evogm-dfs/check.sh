#!/bin/bash
# check.sh : status + last progress lines of cloudgoat/evogm-dfs (read-only); check.sh fetch : download the outputs to kout/
source /workspace/work/merge3/kenv.sh
if [ "$1" == "fetch" ]; then mkdir -p /workspace/work/evogm-dfs/kout; timeout 600 kaggle kernels output cloudgoat/evogm-dfs -p /workspace/work/evogm-dfs/kout 2>&1 | tail -3; ls /workspace/work/evogm-dfs/kout; exit; fi
timeout 60 kaggle kernels status cloudgoat/evogm-dfs 2>&1 | tail -1
cd /workspace/work/evogm && bash kw.sh evogm-dfs ${1:-25}
