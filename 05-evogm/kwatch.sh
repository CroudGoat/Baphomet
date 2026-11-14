#!/bin/bash
# usage: kwatch.sh KERNEL [N]  -> status + last N progress lines from the live log
source /workspace/work/merge3/kenv.sh
timeout 25 kaggle kernels status cloudgoat/$1 2>&1 | grep -o '"KernelWorkerStatus.*"'
timeout 60 python /workspace/work/evogm/klogs.py $1 35 >/dev/null 2>&1
python3 - "$2" <<'P'
import json,sys,re
n=int(sys.argv[1] or 12); out=[]
for l in open('/tmp/klive.txt',encoding='utf-8',errors='replace'):
    if not l.startswith('data: '): continue
    try: d=json.loads(l[6:])
    except Exception: continue
    t=d.get('data','')
    for s in t.split('\n'):
        if re.match(r'\d\d:\d\d:\d\d \[', s): out.append(s[:400])
print('\n'.join(out[-n:]))
P
