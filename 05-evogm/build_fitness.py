# Build the EvoGM fitness item set (generation format, bench prompts), disjoint from the 180-item test set and all bench pool items.
# MMLU-Pro / GPQA: IRT-informative selection (MERGE^3 IRT params, Fisher info p(1-p)||a||^2 at the mean endpoint theta).
# BFCL non-live: random stratified, 10 per category, excluding the bench's sampled ids.
import json, re, random, sys, numpy as np, pyarrow.parquet as pq
from collections import defaultdict, Counter
sys.path.insert(0, "/workspace/work/bench/.pylib")
N_MMLU, N_GPQA, N_BFCL_PER = 40, 30, 10
M3 = "/workspace/work/merge3"; BENCH = "/workspace/work/bench"
pool = json.load(open(f"{M3}/kdata/fitness_pool.json"))
for p in pool: p["pos"] = int(p["pos"])
IRT = np.load(f"{M3}/irt_out/irt_params.npz"); A, B = IRT["A"], IRT["B"]
E = np.load(f"{M3}/kout/run/irt_estimate.npz")
th = np.mean([E[k] for k in E if k.startswith("theta_")], axis=0)  # (1,15,1)
z = np.clip(A * th - B, -30, 30).sum(axis=1)[0]; P = 1 / (1 + np.exp(-z))
info = P * (1 - P) * (A[0] ** 2).sum(axis=0)
bench = [json.loads(l) for l in open(f"{BENCH}/items.jsonl")]
test_ids = set(json.load(open(f"{BENCH}/subset_ids.json")))
bench_ids = {b["id"] for b in bench}
assert test_ids <= bench_ids
items = []
# ---- MMLU-Pro
rows = {r["question_id"]: r for r in pq.read_table(f"{BENCH}/data/mmlupro_test.parquet").to_pylist()}
L = "ABCDEFGHIJ"
cand = [p for p in pool if p["task"] == "mmlu_pro" and f"mmlupro_{p['qid']}" not in bench_ids]
bycat = defaultdict(list)
for p in cand: bycat[rows[int(p["qid"])]["category"]].append(p)
tot = sum(len(v) for v in bycat.values())
quota = {c: N_MMLU * len(v) / tot for c, v in bycat.items()}
alloc = {c: int(q) for c, q in quota.items()}
for c in sorted(quota, key=lambda c: -(quota[c] - alloc[c]))[: N_MMLU - sum(alloc.values())]: alloc[c] += 1
for c in sorted(bycat):
    for p in sorted(bycat[c], key=lambda p: -info[p["pos"]])[: alloc[c]]:
        r = rows[int(p["qid"])]
        opts = "\n".join(f"{L[i]}. {o}" for i, o in enumerate(r["options"]))
        q = ("The following is a multiple choice question about " + c + ". Think step by step and then finish your answer with \"The answer is (X)\" where X is the correct letter choice.\n\n"
             f"Question: {r['question']}\nOptions:\n{opts}")
        items.append(dict(id=f"fit_mmlupro_{r['question_id']}", bench="mmlupro", category=c, messages=[{"role": "user", "content": q}],
                          answer=r["answer"], nopts=len(r["options"]), pool_key=p["key"], irt_info=float(info[p["pos"]]), irt_p=float(P[p["pos"]])))
# ---- GPQA (main/extended; every diamond question excluded = bench pool incl. the 50 test items)
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
dia = {norm(r["Question"]) for r in pq.read_table(f"{BENCH}/data/gpqa_diamond_full.parquet").to_pylist()}
best = {}
for p in pool:
    if p["task"] != "gpqa": continue
    m = re.match(r"What is the correct answer to this question:(.*)\nChoices:\n\(A\) (.*)\n\(B\) (.*)\n\(C\) (.*)\n\(D\) (.*)\nAnswer: ?$", p["ctx"], re.S)
    assert m, p["key"]
    qn = norm(m.group(1))
    if qn in dia: continue
    if qn not in best or info[p["pos"]] > info[best[qn][0]["pos"]]: best[qn] = (p, m)
print("gpqa eligible unique", len(best), Counter(v[0]["subtask"] for v in best.values()))
for p, m in sorted(best.values(), key=lambda v: -info[v[0]["pos"]])[:N_GPQA]:
    ch = [m.group(i).strip() for i in range(2, 6)]; ans = "ABCD"[int(p["gold"])]
    q = ("Answer the following multiple choice question. Think step by step and then finish your answer with \"The answer is (X)\" where X is one of A, B, C, D.\n\n"
         f"Question: {m.group(1).strip()}\n\nA) {ch[0]}\nB) {ch[1]}\nC) {ch[2]}\nD) {ch[3]}")
    items.append(dict(id=f"fit_{p['key'].replace(':', '_')}", bench="gpqa", category=p["subtask"], messages=[{"role": "user", "content": q}], answer=ans,
                      pool_key=p["key"], irt_info=float(info[p["pos"]]), irt_p=float(P[p["pos"]])))
# ---- BFCL non-live (bench used 50/cat sampled with Random(1234); exclude those)
from bfcl_eval.utils import load_dataset_entry
import bfcl_eval, os
pa_dir = os.path.join(os.path.dirname(bfcl_eval.__file__), "data/possible_answer")
rng = random.Random(4242)
for cat in ["simple_python", "multiple", "parallel", "parallel_multiple"]:
    ents = load_dataset_entry(cat, include_prereq=False, include_language_specific_hint=True)
    gt = {}
    for line in open(f"{pa_dir}/BFCL_v4_{cat}.json"):
        d = json.loads(line); gt[d["id"]] = d["ground_truth"]
    elig = [e for e in ents if e["id"] not in bench_ids]
    print("bfcl", cat, len(ents), "eligible", len(elig))
    for e in rng.sample(elig, N_BFCL_PER):
        items.append(dict(id=f"fit_{e['id']}", bfcl_id=e["id"], bench="bfcl", category=cat, messages=e["question"][0], tools=e["function"], answer=gt[e["id"]]))
assert not ({i["id"] for i in items} & bench_ids)
assert not ({i.get("bfcl_id") for i in items} & bench_ids)
json.dump(items, open("kds/fitness_items.json", "w"))
print(Counter(i["bench"] for i in items), Counter(i["category"] for i in items if i["bench"] != "mmlupro"))
for b in ["mmlupro", "gpqa"]:
    v = [i["irt_p"] for i in items if i["bench"] == b]; print(b, "irt p mean", np.mean(v), "info mean", np.mean([i["irt_info"] for i in items if i["bench"] == b]))
