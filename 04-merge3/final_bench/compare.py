# Paired comparison on the fixed 180-item test: merge3 vs qwen35 (base) and vs baphomet. Exact McNemar per bench and overall,
# plus fitness-vs-final correlation across models with both a MERGE^3 fitness estimate and a 180-item score.
import json, sys
from math import comb
ids = set(json.load(open("subset_ids.json")))
def load(t): return {r["id"]: r for r in map(json.loads, open(f"scored_{t}.jsonl")) if r["id"] in ids}
S = {t: load(t) for t in ["qwen35", "baphomet", "merge3"]}
common = set.intersection(*[set(v) for v in S.values()])
def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)
out = dict(n_common=len(common), acc={}, pairs={})
for t in S:
    out["acc"][t] = {b: round(sum(S[t][i]["correct"] for i in common if S[t][i]["bench"] == b) / max(1, sum(1 for i in common if S[t][i]["bench"] == b)), 4)
                     for b in ["mmlupro", "gpqa", "bfcl"]}
    out["acc"][t]["all"] = round(sum(S[t][i]["correct"] for i in common) / len(common), 4)
    out["acc"][t]["trunc"] = round(sum(S[t][i]["truncated"] for i in common) / len(common), 4)
for other in ["qwen35", "baphomet"]:
    for b in ["mmlupro", "gpqa", "bfcl", "all"]:
        ii = [i for i in common if b == "all" or S["merge3"][i]["bench"] == b]
        m_only = sum(S["merge3"][i]["correct"] and not S[other][i]["correct"] for i in ii)
        o_only = sum(S[other][i]["correct"] and not S["merge3"][i]["correct"] for i in ii)
        out["pairs"][f"merge3_vs_{other}:{b}"] = dict(n=len(ii), merge3_only=m_only, other_only=o_only, exact_mcnemar_p=round(mcnemar(m_only, o_only), 4))
json.dump(out, open("compare.json", "w"), indent=1); print(json.dumps(out, indent=1))
