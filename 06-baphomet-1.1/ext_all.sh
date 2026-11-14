for spec in agents_a1:0 dsv4pro:0 q38distill:0 qwopus_coder:0 qwopus_v3:0 arex_turbo:0 neohorse:1 tmax:0; do
  k=${spec%%:*}; r=${spec##*:}
  until grep -q "done.*" logs/dl_all.log && python - "$k" <<'P'
import sys,re
order=["agents_a1","dsv4pro","q38distill","qwopus_coder","qwopus_v3","arex_turbo","neohorse","tmax"]
n=open("logs/dl_all.log").read().count("\ndone ")+open("logs/dl_all.log").read().startswith("done ")
sys.exit(0 if n>order.index(sys.argv[1]) else 1)
P
  do sleep 15; done
  echo "extract $k $(date)"
  python extract.py m/base m/$k $r tv/$k > logs/ext_$k.log 2>&1; tail -n 1 logs/ext_$k.log
done
echo EXTALL
