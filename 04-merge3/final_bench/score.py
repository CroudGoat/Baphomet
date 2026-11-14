# Local scoring: python score.py  -> writes scored_*.jsonl, results.json, prints summary
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pylib"))
import json, re, glob, os, sys
from collections import defaultdict
from bfcl_ast_checker import ast_checker
from bfcl_eval.constants.enums import Language
import bfcl_ast_checker as AC

# our model names are not in BFCL's MODEL_CONFIG_MAPPING; Qwen-style models keep dots in names
AC.convert_func_name = lambda fn, model_name: fn

BASE = os.path.dirname(os.path.abspath(__file__))
items = {json.loads(l)["id"]: json.loads(l) for l in open(f"{BASE}/items.jsonl")}

def split_think(out):
    if "</think>" in out:
        return out.split("</think>", 1)[0], out.split("</think>", 1)[1], True
    return out, "", False

def mc_parse(ans_text, letters):
    L = "".join(letters)
    pats = [rf"answer is \(?\s*\**\s*([{L}])\s*\**\s*\)?", rf"[Aa]nswer\s*[:：]\s*\**\s*\(?([{L}])\)?", rf"\\boxed\{{\s*(?:\\text\{{)?\(?([{L}])\)?", ]
    for p in pats:
        m = re.findall(p, ans_text)
        if m: return m[-1]
    m = re.findall(rf"\(([{L}])\)", ans_text)
    if m: return m[-1]
    m = re.findall(rf"\b([{L}])\b", ans_text.strip()[-20:])
    if m: return m[-1]
    return None

def coerce(v):
    v = v.strip("\n")
    try: return json.loads(v)
    except Exception: pass
    try:
        import ast; return ast.literal_eval(v)
    except Exception: return v

def parse_calls(text, tools):
    calls = []
    tool_types = {t["name"]: t["parameters"].get("properties", {}) for t in tools}
    for blk in re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", text, re.S):
        blk = blk.strip()
        m = re.match(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", blk, re.S)
        if m:
            name = m.group(1).strip(); args = {}
            for pm in re.finditer(r"<parameter=([^>\n]+)>(.*?)</parameter>", m.group(2), re.S):
                k = pm.group(1).strip(); raw = pm.group(2).strip("\n")
                ptype = tool_types.get(name, {}).get(k, {}).get("type")
                if ptype == "string":
                    args[k] = raw
                else:
                    args[k] = coerce(raw)
            calls.append({name: args})
            continue
        try:
            d = json.loads(blk)
            calls.append({d["name"]: d.get("arguments", {})})
        except Exception:
            raise ValueError("unparseable tool_call block")
    if not calls:
        raise ValueError("no tool_call found")
    return calls

PERSONA = re.compile(r"Intern-?S1|Intern-?A1|InternLM|Intern\s?Agent|Agents-A1|DeepSeek|Claude|Qwopus|Anthropic|OpenAI|ChatGPT|GPT-4|Gemini|AREX|tmax|NeoHorse", re.I)

def score_file(path):
    recs = [json.loads(l) for l in open(path)]
    seen = {}
    for r in recs: seen[r["id"]] = r  # dedupe, last wins
    out = []
    for r in seen.values():
        it = items[r["id"]]
        think, ans, closed = split_think(r["output"])
        r["think_closed"] = closed
        r["truncated"] = r["finish_reason"] == "length"
        r["persona_hits"] = sorted(set(m.lower() for m in PERSONA.findall(r["output"])))
        if it["bench"] in ("mmlupro", "gpqa"):
            letters = "ABCDEFGHIJ"[: it.get("nopts", 4)] if it["bench"] == "mmlupro" else "ABCD"
            pred = mc_parse(ans, letters) if closed else None
            r["pred"] = pred; r["gold"] = it["answer"]; r["correct"] = pred == it["answer"]
            r["parse_fail"] = closed and pred is None and not r["truncated"]
        else:
            try:
                calls = parse_calls(ans if closed else "", it["tools"])
                res = ast_checker(it["tools"], calls, it["answer"], Language.PYTHON, it["category"], "local")
                r["pred"] = calls; r["correct"] = bool(res["valid"]); r["error"] = res.get("error"); r["error_type"] = res.get("error_type")
                r["parse_fail"] = False
            except Exception as e:
                r["pred"] = None; r["correct"] = False; r["error"] = [str(e)[:200]]; r["error_type"] = "parse_error" if closed else "no_answer_truncated"
                r["parse_fail"] = closed
        out.append(r)
    return out

def summarize(rs):
    n = len(rs)
    if not n: return {}
    return dict(n=n, acc=round(sum(r["correct"] for r in rs) / n, 4), correct=sum(r["correct"] for r in rs),
                trunc_rate=round(sum(r["truncated"] for r in rs) / n, 4), parse_fail=sum(r["parse_fail"] for r in rs),
                avg_out_tokens=round(sum(r["completion_tokens"] for r in rs) / n, 1))

results = {}
scored = {}
for path in sorted(glob.glob(f"{BASE}/raw_*.jsonl")):
    tag = os.path.basename(path)[4:-6]
    rs = score_file(path); scored[tag] = rs
    with open(f"{BASE}/scored_{tag}.jsonl", "w") as f:
        for r in rs: f.write(json.dumps(r) + "\n")
tags = list(scored)
# paired: only ids present for all models
common = set.intersection(*[set(r["id"] for r in scored[t]) for t in tags]) if tags else set()
for t in tags:
    R = {}
    rs = [r for r in scored[t] if r["id"] in common]
    for b in ["mmlupro", "gpqa", "bfcl"]:
        br = [r for r in rs if r["bench"] == b]
        R[b] = summarize(br)
        cats = defaultdict(list)
        for r in br: cats[r["category"]].append(r)
        R[b]["by_category"] = {c: summarize(v) for c, v in sorted(cats.items())}
        R[b]["persona_mentions"] = sum(1 for r in br if r["persona_hits"])
        R[b]["persona_terms"] = sorted(set(x for r in br for x in r["persona_hits"]))
        if b == "bfcl":
            et = defaultdict(int)
            for r in br:
                if not r["correct"]: et[r.get("error_type")] += 1
            R[b]["error_types"] = dict(et)
    R["all_ids_scored_unpaired"] = len(scored[t])
    results[t] = R
json.dump(dict(paired_items=len(common), results=results), open(f"{BASE}/results.json", "w"), indent=1)
for t in tags:
    print("==", t)
    for b in ["mmlupro", "gpqa", "bfcl"]:
        s = results[t][b]; print(b, {k: v for k, v in s.items() if k not in ("by_category",)})
