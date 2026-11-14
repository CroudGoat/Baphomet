#!/bin/bash
# check_r2.sh : status + last progress lines of kernel cloudgoat/evogm-r2 ; check_r2.sh fetch : download outputs to kout/r2
cd /workspace/work/evogm; source /workspace/work/merge3/kenv.sh
if [ "$1" == "fetch" ]; then mkdir -p kout/r2; timeout 600 kaggle kernels output cloudgoat/evogm-r2 -p kout/r2 2>&1 | tail -3; ls kout/r2; exit; fi
timeout 30 kaggle kernels status cloudgoat/evogm-r2 2>&1 | tail -1
bash kw.sh evogm-r2 ${1:-25}
