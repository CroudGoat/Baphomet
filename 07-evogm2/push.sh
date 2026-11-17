#!/bin/bash
# NOT RUN YET (user approval needed). 1) create private dataset cloudgoat/evogm2-data from kds/, 2) push kernel cloudgoat/evogm2-main.
set -e
source /workspace/work/merge3/kenv.sh
cd /workspace/work/evogm2/kds && rm -rf __pycache__
kaggle datasets create -p . -r skip 2>&1 | tail -3          # private by default; later updates: kaggle datasets version -p . -m "msg"
echo "wait until the dataset is ready:"; for i in $(seq 1 30); do s=$(kaggle datasets status cloudgoat/evogm2-data 2>&1 | tail -1); echo "$s"; [[ "$s" == *ready* ]] && break; sleep 20; done
cd /workspace/work/evogm2/kernel_main && kaggle kernels push -p . 2>&1 | tail -3
