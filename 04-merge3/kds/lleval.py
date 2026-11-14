# Loglik multiple-choice scoring (lm-eval-harness semantics) on top of a vLLM engine.
# loglik(choice) = sum_j log p(cont_j | ctx + cont_<j). Each needed next-token distribution is obtained with one
# max_tokens=1 request returning the top-K raw logprobs (K=maxlp); identical prefixes are deduplicated, so MMLU-Pro
# (" A".." J", single token) needs 1 request/item and GPQA ("  (A)" -> [' ',' (','A',')']) needs 7.
# drop_shared=True (optional, off by default): continuation tokens forming the common prefix of ALL choices of an item
# add the same log-prob to every choice, so they are skipped. This changes every choice's score by the same constant
# (argmax / acc unchanged; acc_norm unchanged too because GPQA choices have equal byte length) and cuts GPQA to 5 req/item.
# A token outside the top-K gets the K-th logprob (upper bound) and is counted in nmiss.
import time


def encode_pair(tok, context, continuation):
    # lm_eval.api.model.TemplateLM._encode_pair; no special tokens (Qwen tokenizer adds no BOS)
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = tok(context + continuation, add_special_tokens=False).input_ids
    ctx = tok(context, add_special_tokens=False).input_ids
    return ctx, whole[len(ctx):]


def _lcp(seqs):
    n = 0
    while all(len(s) > n for s in seqs) and len(set(s[n] for s in seqs)) == 1:
        n += 1
    return n


def build_plan(tok, items, drop_shared=False):
    prompts, plist, plan = {}, [], []
    for it in items:
        encs = [encode_pair(tok, it["ctx"], c) for c in it["conts"]]
        start = _lcp([cont for _, cont in encs]) if drop_shared else 0
        if start and any(len(cont) == start for _, cont in encs):
            start -= 1  # never drop a choice's entire continuation
        ch = []
        for ctx, cont in encs:
            steps = []
            for j in range(start, len(cont)):
                p = tuple(ctx + cont[:j])
                if p not in prompts: prompts[p] = len(plist); plist.append(p)
                steps.append((prompts[p], cont[j]))
            ch.append(steps)
        plan.append((it, ch, len(encs[0][0]), start))
    return plist, plan


def run_requests(llm, plist, maxlp):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    order = sorted(range(len(plist)), key=lambda i: plist[i])  # shared prefixes adjacent -> prefix-cache hits
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=maxlp)
    outs = llm.generate([TokensPrompt(prompt_token_ids=list(plist[i])) for i in order], sp, use_tqdm=False)
    lps = [None] * len(plist)
    for i, o in zip(order, outs):
        lps[i] = {t: v.logprob for t, v in o.outputs[0].logprobs[0].items()}
    return lps


def score(plan, lps):
    res = {}
    for it, ch, nctx, start in plan:
        sc, nmiss = [], 0
        for steps in ch:
            s = 0.0
            for pi, t in steps:
                lp = lps[pi]
                if t in lp: s += lp[t]
                else: s += min(lp.values()); nmiss += 1
            sc.append(s)
        pred = max(range(len(sc)), key=lambda i: (sc[i], -i))
        res[it["key"]] = dict(pred=pred, correct=int(pred == it["gold"]), scores=sc, nmiss=nmiss, ntok=nctx, dropped=start)
    return res


def evaluate(llm, tok, items, maxlp=200, drop_shared=False):
    plist, plan = build_plan(tok, items, drop_shared)
    t1 = time.time()
    lps = run_requests(llm, plist, maxlp)
    t_gen = time.time() - t1
    res = score(plan, lps)
    return res, dict(t_gen=t_gen, n_requests=len(plist), prompt_tokens=sum(len(p) for p in plist), n_items=len(items))
