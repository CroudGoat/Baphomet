#!/bin/bash
# check.sh [N] : status + last progress lines of cloudgoat/evogm2-main ; check.sh fetch : download outputs to kout/main
cd /workspace/work/evogm; source /workspace/work/merge3/kenv.sh
if [ "$1" == "fetch" ]; then mkdir -p /workspace/work/evogm2/kout/main; timeout 600 kaggle kernels output cloudgoat/evogm2-main -p /workspace/work/evogm2/kout/main 2>&1 | tail -3; ls /workspace/work/evogm2/kout/main; exit; fi
timeout 30 kaggle kernels status cloudgoat/evogm2-main 2>&1 | tail -1
bash kw.sh evogm2-main ${1:-25}
