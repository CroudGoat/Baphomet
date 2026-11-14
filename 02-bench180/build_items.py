import json, random, pyarrow.parquet as pq
from collections import defaultdict
SEED=1234
items=[]
# MMLU-Pro: 20/category
rows=pq.read_table('data/mmlupro_test.parquet').to_pylist()
bycat=defaultdict(list)
for r in rows: bycat[r['category']].append(r)
rng=random.Random(SEED)
L="ABCDEFGHIJ"
for cat in sorted(bycat):
    for r in rng.sample(bycat[cat],20):
        opts="\n".join(f"{L[i]}. {o}" for i,o in enumerate(r['options']))
        q=("The following is a multiple choice question about "+cat+". Think step by step and then finish your answer with \"The answer is (X)\" where X is the correct letter choice.\n\n"
           f"Question: {r['question']}\nOptions:\n{opts}")
        items.append(dict(id=f"mmlupro_{r['question_id']}",bench="mmlupro",category=cat,messages=[{"role":"user","content":q}],answer=r['answer'],nopts=len(r['options'])))
# GPQA diamond
g=pq.read_table('data/gpqa_diamond_full.parquet').to_pylist()
for i,r in enumerate(g):
    ch=[r['Correct Answer'].strip(),r['Incorrect Answer 1'].strip(),r['Incorrect Answer 2'].strip(),r['Incorrect Answer 3'].strip()]
    perm=list(range(4)); random.Random(SEED+i).shuffle(perm)
    shuffled=[ch[p] for p in perm]; ans="ABCD"[perm.index(0)]
    q=("Answer the following multiple choice question. Think step by step and then finish your answer with \"The answer is (X)\" where X is one of A, B, C, D.\n\n"
       f"Question: {r['Question'].strip()}\n\nA) {shuffled[0]}\nB) {shuffled[1]}\nC) {shuffled[2]}\nD) {shuffled[3]}")
    items.append(dict(id=f"gpqa_{i}",bench="gpqa",category=r['High-level domain'],messages=[{"role":"user","content":q}],answer=ans))
# BFCL v4 non-live AST
from bfcl_eval.utils import load_dataset_entry
import sys; sys.path.insert(0, ".pylib"); import bfcl_eval, os
pa_dir=os.path.join(os.path.dirname(bfcl_eval.__file__),'data/possible_answer')
for cat in ["simple_python","multiple","parallel","parallel_multiple"]:
    ents=load_dataset_entry(cat, include_prereq=False, include_language_specific_hint=True)
    gt={}
    for line in open(f"{pa_dir}/BFCL_v4_{cat}.json"):
        d=json.loads(line); gt[d['id']]=d['ground_truth']
    for e in random.Random(SEED).sample(ents,50):
        msgs=e['question'][0]
        items.append(dict(id=e['id'],bench="bfcl",category=cat,messages=msgs,tools=e['function'],answer=gt[e['id']]))
with open('items.jsonl','w') as f:
    for it in items: f.write(json.dumps(it)+"\n")
from collections import Counter
print(Counter(i['bench'] for i in items), len(items))
