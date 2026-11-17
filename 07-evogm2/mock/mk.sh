#!/bin/bash
# EvoGM v2 mock run: $1 = run dir, $2 = EVOGM2_MOCK json extras (e.g. '"hard_h": 99'), $3 = optional dir whose kaggle/working becomes an input (resume)
# env passthrough: MOCK_TS MOCK_CRASH MOCK_RACEERR MOCK_BASEBFCL
R=$1; rm -rf $R; mkdir -p $R/kaggle/input/evogm2-data/bfcl_eval/bfcl_eval/constants $R/kaggle/working $R/tmp
D=$R/kaggle/input/evogm2-data; K=/workspace/work/evogm2/kds; M=/workspace/work/evogm2/mock
cp $K/{search_items.json,val_items.json,test_items.jsonl,subset_ids.json} $D/; touch $D/bfcl_eval/bfcl_eval/constants/enums.py
cp $M/{linmerge.py,fitscore.py} $D/
if [ -n "$3" ]; then mkdir -p $R/kaggle/input/prev; cp -r $3/kaggle/working/. $R/kaggle/input/prev/; fi
export MOCK_TS=${MOCK_TS:-0.002}
export EVOGM2_MOCK="{\"root\": \"$R\", \"ts\": $MOCK_TS, \"worker\": \"$M/gpuworker_mock.py\", \"timeouts\": [30, 120, 60, 120, 30] ${2:+, $2}}"
cd $R && timeout 3000 /workspace/work/merge3/venv/bin/python /workspace/work/evogm2/kernel_main/evogm2_main.py > $R/out.txt 2>&1; echo "exit $?" >> $R/out.txt
tail -3 $R/out.txt
