# Persistent per-GPU generation worker: one in-process vLLM 0.33.0 engine, weights swapped in place
# (Worker.reload_weights with a streamed exact linear merge). File-based command protocol (see run script).
# usage: CUDA_VISIBLE_DEVICES=g python gpuworker2.py BASE_DIR CMD_DIR GPUMEM EAGER(0/1) MAXSEQS
import os, sys, json, time, traceback
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAXLEN = 12288


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
    merger_args = {}
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
                outs = llm.generate(prompts, sp, use_tqdm=False)
                recs = []
                for it, o in zip(items, outs):
                    c = o.outputs[0]
                    recs.append(dict(id=it["id"], prompt_tokens=len(o.prompt_token_ids), completion_tokens=len(c.token_ids),
                                     finish_reason=c.finish_reason, output=c.text))
                json.dump(recs, open(cmd["out"], "w"))
                res["n"] = len(recs); res["gen_tokens"] = sum(r["completion_tokens"] for r in recs)
            else:
                raise ValueError(op)
        except Exception as e:
            res = dict(ok=False, error=repr(e), tb=traceback.format_exc()[-4000:])
        res["secs"] = time.time() - t1
        json.dump(res, open(f"{cdir}/res_{n}.tmp", "w")); os.rename(f"{cdir}/res_{n}.tmp", f"{cdir}/res_{n}.json")
        print(f"[gpuworker2] cmd {n} {cmd.get('op')} ok={res['ok']} {res['secs']:.1f}s", flush=True)
        if cmd.get("op") == "exit":
            del llm
            return
        n += 1


if __name__ == "__main__":
    main()
