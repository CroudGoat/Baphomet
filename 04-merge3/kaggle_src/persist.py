# Persistent vLLM engine per GPU with in-place weight reloading (vLLM 0.33.0).
# One process per GPU (CUDA_VISIBLE_DEVICES=i). The engine runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0), so
# LLM.collective_rpc(callable) runs directly on the worker and can stream merged tensors (GPU-resident) into
# Worker.reload_weights(weights_iterator=...) -> vllm.model_executor.model_loader.reload.reload_weights
#   (initialize_layerwise_reload -> model.load_weights(iter) -> finalize_layerwise_reload).
# Endpoints (whole checkpoints on disk) use Worker.reload_weights(weights_path=dir).
# After every swap the prefix cache (KV + hybrid/mamba state blocks) is reset, otherwise cached prefixes computed with
# the previous weights would be reused.
import os, time, hashlib
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import lleval


class Engine:
    def __init__(self, model_dir, gpumem=0.80, maxlen=16384, maxlp=200, prefix_cache=True, tokenizer=None):
        from vllm import LLM
        t0 = time.time()
        self.maxlp = maxlp
        self.llm = LLM(model_dir, dtype="float16", max_model_len=maxlen, gpu_memory_utilization=gpumem, enforce_eager=True,
                       enable_prefix_caching=prefix_cache, max_logprobs=maxlp, seed=0,
                       limit_mm_per_prompt={"image": 0, "video": 0}, tokenizer=tokenizer or model_dir)
        self.tok = self.llm.get_tokenizer()
        self.t_init = time.time() - t0
        self.current = model_dir

    def _after_swap(self):
        ok = self.llm.reset_prefix_cache()
        if not ok:
            raise RuntimeError("reset_prefix_cache failed")

    def load_dir(self, path):
        """In-place reload from a HF checkpoint dir (same architecture). Returns seconds."""
        t0 = time.time()
        self.llm.collective_rpc("reload_weights", kwargs=dict(weights_path=path, is_checkpoint_format=True))
        self._after_swap()
        self.current = path
        return time.time() - t0

    def load_merge(self, base_dir, tv_paths, weights, densities, seed, chunk_elems=None):
        """Compute the dare_ties merge on this GPU tensor-by-tensor and stream it straight into the model. Returns s."""
        def _fn(worker, base_dir, tv_paths, weights, densities, seed, chunk_elems):
            import torch, m3lib
            chunk_elems = chunk_elems or m3lib.CHUNK_ELEMS
            gen = m3lib.iter_merged(base_dir, tv_paths, weights, densities, seed=seed, device=worker.device,
                                    chunk_elems=chunk_elems, keep_on_device=True)
            worker.reload_weights(weights_iterator=gen, is_checkpoint_format=True)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return True
        t0 = time.time()
        self.llm.collective_rpc(_fn, args=(base_dir, tv_paths, weights, densities, seed, chunk_elems))
        self._after_swap()
        self.current = ("merge", tuple(weights), tuple(densities), seed)
        return time.time() - t0

    def checksum(self):
        """sha256 per parameter/buffer of the live model (bit-level), plus an overall digest."""
        def _f(model):
            import torch
            out = {}
            for n, t in list(model.named_parameters()) + list(model.named_buffers()):
                if t.device.type == "meta":
                    out[n] = "META"; continue
                b = t.detach().contiguous().cpu()
                if b.dtype in (torch.float16, torch.bfloat16): b = b.view(torch.int16)
                elif b.dtype == torch.float32: b = b.view(torch.int32)
                out[n] = hashlib.sha256(b.numpy().tobytes()).hexdigest()[:20]
            return out
        d = self.llm.apply_model(_f)[0]
        tot = hashlib.sha256("".join(f"{k}:{v};" for k, v in sorted(d.items())).encode()).hexdigest()[:20]
        return tot, d

    def evaluate(self, items, drop_shared=False):
        return lleval.evaluate(self.llm, self.tok, items, self.maxlp, drop_shared=drop_shared)
