#!/bin/bash
# NOT RUN (needs the user's go). 1) create the private dataset cloudgoat/evogm-dfs-data from kds_dfs/ (v2_prev.json, 277 KB) and wait
# until it is ready, 2) push the kernel cloudgoat/evogm-dfs (private, T4 x2, internet, datasets evogm2-data + evogm-dfs-data).
# Resume after a crash: copy kernel/ to a new slug dir, set "kernel_sources": ["cloudgoat/evogm-dfs"], push (same code; it reads
# dfs_log / dfs_val_log / dfs_fit_outputs / dfs_val_outputs / dfs_state.pt / dfs_calib.json from /kaggle/input).
set -e
source /workspace/work/merge3/kenv.sh
cd /workspace/work/evogm-dfs/kds_dfs && rm -rf __pycache__
kaggle datasets create -p . -r skip 2>&1 | tail -3          # private by default
echo "wait until the dataset is ready:"; for i in $(seq 1 30); do s=$(kaggle datasets status cloudgoat/evogm-dfs-data 2>&1 | tail -1); echo "$s"; [[ "$s" == *ready* ]] && break; sleep 20; done
cd /workspace/work/evogm-dfs/kernel && rm -rf __pycache__ && kaggle kernels push -p . 2>&1 | tail -3
