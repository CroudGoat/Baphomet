# Persistent per-GPU generation worker (gpuworker2 + racing). One in-process vLLM 0.33.0 engine, weights swapped in place
# (Worker.reload_weights with a streamed exact linear merge). File-based command protocol (see run script).
# op gen with "partial"+"stop": requests are fed through llm.llm_engine (add_request/step); finished outputs are written to
# `partial` every ~8 s; if the controller creates `stop`, the unfinished requests are aborted (finish_reason "aborted").
# Any failure of that path falls back to plain llm.generate for the same call (res["race_error"] says why).
# usage: CUDA_VISIBLE_DEVICES=g python gpuworker3.py BASE_DIR CMD_DIR GPUMEM EAGER(0/1) MAXSEQS
import os, sys, json, time, traceback
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAXLEN = 12288


def _rec(it, o):
    c = o.outputs[0]
    return dict(id=it["id"], prompt_tokens=len(o.prompt_token_ids or []), completion_tokens=len(c.token_ids),
                finish_reason=c.finish_reason, output=c.text)


def race_generate(llm, prompts, items, sp, partial, stop, tag):
    eng = llm.llm_engine
    try:
        from vllm.sampling_params import RequestOutputKind
        sp = sp.clone(); sp.output_kind = RequestOutputKind.FINAL_ONLY
    except Exception:
        pass
    ext = [f"{tag}_{k}" for k in range(len(prompts))]
    idmap = {}
    for k, (rid, p) in enumerate(zip(ext, prompts)):
        r = eng.add_request(rid, p, sp)
        idmap[rid] = k
        if isinstance(r, str): idmap[r] = k
    def kidx(rid):
        if rid in idmap: return idmap[rid]
        for e in ext:
            if rid.startswith(e + "-"): return idmap[e]
        return None
    done = {}; last = time.time(); aborted = []
    while eng.has_unfinished_requests():
        for o in eng.step():
            if getattr(o, "finished", False):
                k = kidx(o.request_id)
                if k is not None and k not in aborted: done[k] = _rec(items[k], o)
        if time.time() - last > 8 or not eng.has_unfinished_requests():
            last = time.time()
            json.dump([done[k] for k in sorted(done)], open(partial + ".tmp", "w")); os.replace(partial + ".tmp", partial)
            if os.path.exists(stop) and eng.has_unfinished_requests():
                aborted = [k for k in range(len(ext)) if k not in done]
                eng.abort_request([ext[k] for k in aborted])
                for _ in range(200):   # drain whatever the engine still reports
                    if not eng.has_unfinished_requests(): break
                    eng.step()
                break
    recs = []
    for k, it in enumerate(items):
        if k in done: recs.append(done[k])
        else: recs.append(dict(id=it["id"], prompt_tokens=0, completion_tokens=0, finish_reason="aborted", output=""))
    return recs, len(aborted)


def main():
    base, cdir, gpumem, eager, maxseqs = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4] == "1", int(sys.argv[5])
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import torch
    t0 = time.time()
    llm = LLM(base, dtype="float16", max_model_len=MAXLEN, gpu_memory_utilization=gpumem, enforce_eager=eager,
              enable_prefix_caching=False, seed=0, max_num_seqs=maxseqs, limit_mm_per_prompt={"image": 0, "video": 0})
    tok = AutoTokenizer.from_pretrained(base)
    t_init = time.time() - t0

    def ptrs(model):
        return {n: t.data_ptr() for n, t in list(model.named_parameters()) + list(model.named_buffers())}
    kv = None
    try:
        kv = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks
    except Exception:
        pass
    json.dump(dict(ok=True, t_init=t_init, eager=eager, gpumem=gpumem, num_gpu_blocks=kv,
                   free_gb=torch.cuda.mem_get_info()[0] / 2**30),
              open(f"{cdir}/ready.tmp", "w"))
    os.rename(f"{cdir}/ready.tmp", f"{cdir}/ready.json")
    n = 0
    while True:
        p = f"{cdir}/cmd_{n}.json"
        if not os.path.exists(p):
            time.sleep(0.3); continue
        cmd = json.load(open(p)); t1 = time.time(); res = dict(ok=True)
        try:
            op = cmd["op"]
            if op == "exit":
                res["bye"] = True
            elif op == "load_merge":
                before = llm.apply_model(ptrs)[0]
                def _fn(worker, base, tvs, lams):
                    import torch, linmerge
                    key = (base, tuple(tvs))
                    if getattr(worker, "_merger_key", None) != key:
                        worker._merger = linmerge.Merger(base, tvs); worker._merger_key = key
                    gen = worker._merger.iter_merged(lams, device=worker.device, keep_on_device=True)
                    worker.reload_weights(weights_iterator=gen, is_checkpoint_format=True)
                    torch.cuda.synchronize(); torch.cuda.empty_cache()
                    return True
                llm.collective_rpc(_fn, args=(base, cmd["tvs"], cmd["lams"]))
                after = llm.apply_model(ptrs)[0]
                res["ptr_changed"] = sum(1 for k in before if before[k] != after.get(k))
                res["t"] = time.time() - t1
                res["free_gb"] = torch.cuda.mem_get_info()[0] / 2**30
            elif op == "gen":
                items = json.load(open(cmd["items"]))
                prompts = []
                for it in items:
                    kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=True)
                    if it.get("tools"): kw["tools"] = it["tools"]
                    prompts.append(tok.apply_chat_template(it["messages"], **kw))
                sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, seed=0, max_tokens=int(cmd["max_tokens"]),
                                    skip_special_tokens=False)
                recs = None
                if cmd.get("partial"):
                    try:
                        recs, res["aborted"] = race_generate(llm, prompts, items, sp, cmd["partial"], cmd["stop"], f"c{n}")
                    except Exception as e:
                        res["race_error"] = repr(e)[:600] + " | " + traceback.format_exc()[-1500:]
                        try:
                            eng = llm.llm_engine
                            eng.abort_request([f"c{n}_{k}" for k in range(len(prompts))])
                            for _ in range(200):
                                if not eng.has_unfinished_requests(): break
                                eng.step()
                        except Exception:
                            pass
                        recs = None
                if recs is None:
                    outs = llm.generate(prompts, sp, use_tqdm=False)
                    recs = [_rec(it, o) for it, o in zip(items, outs)]
                json.dump(recs, open(cmd["out"], "w"))
                res["n"] = len(recs); res["gen_tokens"] = sum(r["completion_tokens"] for r in recs)
            else:
                raise ValueError(op)
        except Exception as e:
            res = dict(ok=False, error=repr(e), tb=traceback.format_exc()[-4000:])
        res["secs"] = time.time() - t1
        json.dump(res, open(f"{cdir}/res_{n}.tmp", "w")); os.rename(f"{cdir}/res_{n}.tmp", f"{cdir}/res_{n}.json")
        print(f"[gpuworker3] cmd {n} {cmd.get('op')} ok={res['ok']} {res['secs']:.1f}s aborted={res.get('aborted')} race_err={res.get('race_error','')[:200]}", flush=True)
        if cmd.get("op") == "exit":
            del llm
            return
        n += 1


if __name__ == "__main__":
    main()
