# Scoring identical to bench/score.py (MC answer parsing after </think>; BFCL official AST checker on parsed <tool_call>s).
import re, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfcl_ast_checker import ast_checker
from bfcl_eval.constants.enums import Language
import bfcl_ast_checker as AC
AC.convert_func_name = lambda fn, model_name: fn


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
                args[k] = raw if ptype == "string" else coerce(raw)
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


def score(it, output, finish_reason):
    """-> dict(correct, truncated, think_closed, pred)"""
    think, ans, closed = split_think(output)
    r = dict(truncated=finish_reason == "length", think_closed=closed)
    if it["bench"] in ("mmlupro", "gpqa"):
        letters = "ABCDEFGHIJ"[: it.get("nopts", 4)] if it["bench"] == "mmlupro" else "ABCD"
        pred = mc_parse(ans, letters) if closed else None
        r.update(pred=pred, correct=pred == it["answer"])
    else:
        try:
            calls = parse_calls(ans if closed else "", it["tools"])
            res = ast_checker(it["tools"], calls, it["answer"], Language.PYTHON, it["category"], "local")
            r.update(pred=calls, correct=bool(res["valid"]), error_type=res.get("error_type"))
        except Exception as e:
            r.update(pred=None, correct=False, error_type="parse_error" if closed else "no_answer_truncated")
    return r
