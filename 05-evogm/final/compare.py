# Score raw_evogm.jsonl with bench scoring and compare paired vs raw_baphomet / raw_qwen35 on the 180-item test set.
import json, sys, os, shutil, subprocess
from math import comb
D = os.path.dirname(os.path.abspath(__file__)); B = "/workspace/work/bench"
for f in ["score.py", "bfcl_ast_checker.py", "items.jsonl", "subset_ids.json", "raw_baphomet.jsonl", "raw_qwen35.jsonl"]:
    if not os.path.exists(f"{D}/{f}"): shutil.copy(f"{B}/{f}", D)
if not os.path.exists(f"{D}/.pylib"): os.symlink(f"{B}/.pylib", f"{D}/.pylib")
subprocess.run([sys.executable, f"{D}/score.py"], check=True, cwd=D, stdout=subprocess.DEVNULL)
ids = json.load(open(f"{D}/subset_ids.json"))
def load(t): return {r["id"]: r for r in map(json.loads, open(f"{D}/scored_{t}.jsonl")) if r["id"] in set(ids)}
S = {t: load(t) for t in ["evogm", "baphomet", "qwen35"]}
common = set.intersection(*[set(v) for v in S.values()])
def p_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
out = dict(n_common=len(common), suites={}, pairs={})
for b in ["mmlupro", "gpqa", "bfcl", "all"]:
    ii = [i for i in common if b == "all" or S["evogm"][i]["bench"] == b]
    out["suites"][b] = {t: dict(n=len(ii), acc=round(sum(S[t][i]["correct"] for i in ii) / len(ii), 4),
                                trunc=round(sum(S[t][i]["truncated"] for i in ii) / len(ii), 4),
                                mean_tok=round(sum(S[t][i]["completion_tokens"] for i in ii) / len(ii), 1)) for t in S}
    for o in ["baphomet", "qwen35"]:
        e_only = sum(S["evogm"][i]["correct"] and not S[o][i]["correct"] for i in ii)
        o_only = sum(S[o][i]["correct"] and not S["evogm"][i]["correct"] for i in ii)
        out["pairs"][f"evogm_vs_{o}:{b}"] = dict(n=len(ii), evogm_only=e_only, other_only=o_only, exact_mcnemar_p=round(p_exact(e_only, o_only), 4))
json.dump(out, open(f"{D}/compare.json", "w"), indent=1)
print(json.dumps(out, indent=1))
