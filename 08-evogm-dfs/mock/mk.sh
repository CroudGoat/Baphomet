#!/bin/bash
# EvoGM-DFS mock run: $1 = run dir, $2 = EVOGM_DFS_MOCK json extras (e.g. '"hard_h": 99'), $3 = optional dir whose kaggle/working becomes an input (resume)
# env passthrough: MOCK_TS MOCK_CRASH MOCK_RACEERR MOCK_BROKEN MOCK_BASEBFCL MOCK_PREV (path of a v2_prev.json to use instead of the mock one)
R=$1; pkill -9 -f "gpuworker_dfs.py \S+ $R/tmp/w" ; rm -rf $R;   # (a killed mock session leaves its mock workers running: kill them so a reused dir is not served twice)
 mkdir -p $R/kaggle/input/evogm2-data/bfcl_eval/bfcl_eval/constants $R/kaggle/input/evogm-dfs-data $R/kaggle/working $R/tmp
D=$R/kaggle/input/evogm2-data; K=/workspace/work/evogm2/kds; M=/workspace/work/evogm-dfs/mock; PY=/workspace/work/merge3/venv/bin/python
cp $K/{search_items.json,val_items.json,test_items.jsonl,subset_ids.json} $D/; touch $D/bfcl_eval/bfcl_eval/constants/enums.py
cp $M/{linmerge.py,fitscore.py} $D/
if [ -n "$MOCK_PREV" ]; then cp $MOCK_PREV $R/kaggle/input/evogm-dfs-data/v2_prev.json; else env -u MOCK_BROKEN $PY $M/make_prev.py $R/kaggle/input/evogm-dfs-data/v2_prev.json > $R/prev.txt; fi
if [ -n "$3" ]; then mkdir -p $R/kaggle/input/prev; cp -r $3/kaggle/working/. $R/kaggle/input/prev/; fi
export MOCK_TS=${MOCK_TS:-0.002}
export EVOGM_DFS_MOCK="{\"root\": \"$R\", \"ts\": $MOCK_TS, \"worker\": \"$M/gpuworker_mock.py\", \"dfslib\": \"/workspace/work/evogm-dfs/src/dfslib.py\", \"timeouts\": [30, 120, 60, 120, 30] ${2:+, $2}}"
cd $R && timeout 3000 $PY /workspace/work/evogm-dfs/kernel/evogm_dfs.py > $R/out.txt 2>&1; echo "exit $?" >> $R/out.txt
pkill -9 -f "gpuworker_dfs.py \S+ $R/tmp/w"
tail -3 $R/out.txt
