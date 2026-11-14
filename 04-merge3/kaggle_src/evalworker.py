# Loglik multiple-choice evaluation (lm-eval-harness semantics) with vLLM, one model per process (see lleval.py).
# usage: CUDA_VISIBLE_DEVICES=i python evalworker.py MODEL_DIR ITEMS_JSON OUT_JSON [PREFIX_CACHE 0/1]
# ITEMS_JSON: list of {"key","ctx","conts":[...],"gold":int}.  Env: MAXLP (200), MAXLEN, GPUMEM, TOKENIZER, DROP_SHARED (0).
import os, sys, json, time
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lleval import encode_pair, evaluate  # noqa: F401  (encode_pair re-exported for older scripts)
MAXLP = int(os.environ.get("MAXLP", "200"))

def main():
    model_dir, items_path, out_path = sys.argv[1:4]
    prefix_cache = (len(sys.argv) <= 4 or sys.argv[4] == "1")
    from vllm import LLM
    t0 = time.time()
    items = json.load(open(items_path))
    llm = LLM(model_dir, dtype="float16", max_model_len=int(os.environ.get("MAXLEN", "16384")),
              gpu_memory_utilization=float(os.environ.get("GPUMEM", "0.88")), enforce_eager=True,
              enable_prefix_caching=prefix_cache, max_logprobs=MAXLP, seed=0,
              limit_mm_per_prompt={"image": 0, "video": 0}, tokenizer=os.environ.get("TOKENIZER", model_dir))
    tok = llm.get_tokenizer()
    t_load = time.time() - t0
    res, st = evaluate(llm, tok, items, MAXLP, drop_shared=os.environ.get("DROP_SHARED", "0") == "1")
    json.dump(dict(results=res, t_load=t_load, t_total=time.time() - t0, **st), open(out_path, "w"))
    print(f"evalworker done {len(res)} items, {st['n_requests']} requests, {st['prompt_tokens']} prompt tokens; load {t_load:.0f}s "
          f"gen {st['t_gen']:.0f}s ({st['prompt_tokens']/max(st['t_gen'],1e-9):.0f} tok/s incl. cache hits)", flush=True)

if __name__ == "__main__":
    main()
