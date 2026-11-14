# MERGE^3 on Kaggle — working notes

## Access
- Kaggle CLI works with legacy placeholder creds: `export PATH=$HOME/.local/bin:$PATH KAGGLE_USERNAME=cloudgoat KAGGLE_KEY=placeholder` (proxy injects real token). ~/.kaggle/access_token moved to access_token.bak.
- HF token connected as service "huggingface" (huggingface.co). Gated leaderboard details accepted per repo with `POST https://huggingface.co/datasets/<repo>/ask-access` (303) — then 200.
- Kaggle quota at start: 1.34h used / 30h (refresh Sat 2026-11-14 09:00 JST).

## Kaggle environment (debug kernel)
- T4 x2, Python 3.13, torch 2.11 preinstalled; `pip install vllm==0.33.0` -> torch 2.14, ~4-8 min. /tmp overlay ~950 GB free, /kaggle/working 20 GB, RAM 31 GB, 4 CPU.
- HF download of Qwen3.5-4B (8.8 GB) ~30-60 s.
- vLLM needs `if __name__=="__main__"` guard (spawn).
- prompt_logprobs path slow (~1.1k tok/s, no prefix cache) -> evalworker uses max_tokens=1 + top-K logprobs per needed prefix (dedup), prefix caching on.

## Design (user decisions + MERGE^3 faithful)
- Donors fixed (user override): Agents-A1-4B, DeepSeek-V4-Pro-Qwen3.5-4B, Qwen3.8-4B-Distill, Qwopus3.5-4B-v3, AREX-Turbo. No similarity step.
- Fitness = leaderboard v2 MMLU-Pro (5-shot) + GPQA (diamond+main+extended, 0-shot), loglik MC, no chat template (contexts copied from Qwen/Qwen2.5-7B details = no chat template). Test items excluded (80 MMLU-Pro by question_id, 50 GPQA questions by text in each subset -> 150 rows). Pool: MMLU-Pro 11,952, GPQA 1,042.
- IRT: Polo et al. modified py-irt (felipemaiapolo/py-irt) multidim_2pl hierarchical, D=15, lr .1, 2000 epochs, seed 42; fitted on leaderboard per-item responses of N randomly sampled models (seed 42). item_curve = sigmoid(sum_d(a_d*theta_d - b)).
- Extract: uniform random 20 items/task (seed 420) = paper default.
- Estimate: endpoint gammas via fit_theta on full pool (base + 5 donors).
- Evolve: NSGA-II pop 25, n_gen 7 (175 evals), seed 420, dare_ties genome [w1..w5,d1..d5] in [0,1], objectives -gmp-IRT (delta 0.5) per task. Seeds: equal (w=.2,d=1) and base (w=0). Selection: max mean gmp-IRT over all evaluated.
- TVs fp8 e4m3 per-row absmax; merge exact vs dequantized TVs (unit-tested).

## Files
- irt/collect.py (leaderboard responses -> irt/resp/*.json), build_pool.py -> kdata/, kaggle_src/*.py, kds/ (Kaggle dataset cloudgoat/merge3-data).

## Session 2 (2026-11-10 JST, subagent merge3-kaggle-2)
### Persistent engine + in-place reload (VALIDATED, kernel merge3-swap v2)
- kaggle_src/persist.py: one in-process vLLM 0.33.0 engine per GPU (VLLM_ENABLE_V1_MULTIPROCESSING=0, enforce_eager, fp16,
  gpumem 0.80, max_model_len 16384, prefix caching on, base tokenizer). Swaps use vLLM's own in-place API
  Worker.reload_weights (= reload.inplace: initialize_layerwise_reload -> model.load_weights(iter) -> finalize) via
  LLM.collective_rpc; merges are streamed tensor-by-tensor from m3lib.iter_merged (GPU) into it. Prefix cache reset after every swap.
- Exactness: merge M (w=[.3,.2,.2,.15,.15], d=[.6,.7,.5,.8,.9], seed 12345) loaded in place on GPU0 vs written to disk +
  fresh engine on GPU1: sha256 of all 363 params/buffers identical; loglik scores on 200 pool items (120 MMLU-Pro + 80 GPQA)
  bit-identical (200/200, max |diff| 0.0). Round trip base->M->base: params identical, scores identical. Repeat merge identical.
- Swap times (T4): merge+in-place load 93-97 s (5 donors active); full checkpoint dir reload 25-34 s; fresh engine init 420-505 s;
  merge-to-disk 123 s. Eval of the 40-item reduced set: 26 s. => ~2 min per evolution candidate per GPU.
- Fixes needed on the way (attempt 1 OOM'd): DARE chunk 32M->8M elements (m3lib.CHUNK_ELEMS = 1<<23; it is part of the DARE
  RNG seeding, so fixed for the whole run; nothing had been evaluated with the old value), tensors > 64M elements (embeddings)
  yielded on CPU during reload, gpumem .75->.80, expandable_segments. max_model_len 8192 -> 16384 because one GPQA-extended
  pool item (gpqa_extended:356) is 12,304 tokens (all others <= 2,761).
- lleval.py = shared loglik MC scorer (evalworker.py now a thin wrapper). Optional drop_shared (skip continuation tokens shared by
  all choices: GPQA 7 -> 5 requests/item). Test on all 1,042 GPQA items: preds agree 1039/1042, per-item offset not exactly
  constant (spread up to 0.093 nats; fp16 batch-composition noise) -> NOT identical -> NOT USED (user rule: only if identical).
### Tokenizers (step 3, sandbox, all 12,994 pool items, exact lm-eval encode_pair ctx + continuation ids)
- transformers 5.22.2 AutoTokenizer (what vLLM loads): all 5 donors identical to base on every item.
- Raw tokenizer.json: 3 variants (base; agents_a1/qwopus_v3/arex_turbo = same vocab/merges, merges in list format + 7 extra
  audio/tts special tokens; dsv4pro = OLD Qwen2 pre-tokenizer regex without \p{M}). dsv4pro raw differs on 2 MMLU-Pro items
  (mmlu_pro:11554, :11715); transformers rebuilds the regex from tokenizer_config 'pretokenize_regex', so no effect via AutoTokenizer.
- Decision: every endpoint/candidate is evaluated with the BASE tokenizer (persistent engine), which removes the issue entirely.
### Throughput (persistent engine, both GPUs busy, prefix cache on)
- MMLU-Pro 5-shot: 1,000 items 330 s (945k prompt tok, 2.9k tok/s). GPQA 0-shot, 1,042 items: 7 req/item 1,087 s; 5 req/item 787 s.
- Full pool per endpoint ~1.40 GPU-h (MMLU-Pro 11,952 ~1.10 h + GPQA 0.30 h) -> 7 endpoints ~9.8 GPU-device-h (~4.9 h wall on T4x2).
  1,000 MMLU-Pro + GPQA(5 req) ~0.31 GPU-h/endpoint. Anchors (200+100) ~3 min/endpoint.
### IRT fit (step 4)
- Collector finished: 967 response files (593 new ok, 32 missing tasks, 1 err). IRT fit kernel merge3-irt (CPU) pushed on the
  611-model snapshot (as instructed: >=400); response matrix 611 x 12,994 (HelpingAI/Priya-3B skipped: 27% missing).
  py-irt fork tested locally (pydantic 2 / torch 2.15 OK). A refit on all ~966 models is CPU-only if wanted.
- Caveat inherited from leaderboard details: GPQA doc_hash agreement across models mean 0.979, min 0.63 (some docs differ
  across harness versions).
### DEVIATION (user decision 2026-11-10 11:07 JST): endpoint abilities from tinyBenchmarks IRT anchors, not the full pool
- MERGE^3 fits endpoint theta (gamma_m) on the full fitness data. To save GPU time we instead evaluate each endpoint on
  tinyBenchmarks anchor points (anchor_points.ipynb, clustering='irt': KMeans on [A_1..A_15, B] per item, random_state 42,
  n_init auto, uniform balance weights, anchor = nearest item to each centre, weight = cluster mass): 200 MMLU-Pro + 100 GPQA,
  from the pool (180 test items excluded). gamma_m = fit_theta on those 300 responses (both tasks jointly). Theta estimation
  error therefore propagates into every candidate's gmp-IRT estimate. Reduced set (20/task, seed 420) and evolution defaults unchanged.
- Files: pick_anchors.py -> kdata/irt_anchors.json; kaggle_src/gpuworker.py + run.py (endpoints -> estimate -> evolve -> final).
- Final kernel no longer evaluates the winner on the full pool (cost ~1.4 GPU-h); it evaluates it on the 340 endpoint items.
### RESULT (2026-11-10 17:50 JST)
- merge3-run COMPLETE, 5.39 kernel-h (endpoints+estimate ~1h, evolution 175 evals 3.25h, final). Winner gen6: w=[0.0005,0.914,0.183,0.514,0.033] d=[0.438,0.268,0.989,0.569,0.769] (a1,dsv4pro,q38distill,qwopus,arex). fitness gmpIRT MMLU-Pro 0.559 / GPQA 0.468.
- Final 180 test: GPU0 vLLM failed to start (free 13.34 < 0.95*14.56 GiB, leftover memory) -> only odd half (90 items) evaluated. fp16 (T4), 0.95, 12288.
- 90 paired items: MMLU-Pro M 67.6 / B 75.7 / Q 64.9; GPQA 48.1/63.0/40.7; BFCL 69.2/100/100. All: M vs B 2/17 p=0.001; M vs Q 8/13 p=0.38. BFCL: 3 no tool_call, 3 wrong count parallel.
- Verdict: does NOT beat Baphomet. Fitness (loglik, no template, MMLU/GPQA only) did not predict generation / tool-use.
