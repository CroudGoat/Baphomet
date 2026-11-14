# EvoGM-style linear merge over the 8 Baphomet donors (Kaggle T4x2) — working notes

Approved by user 2026-11-11 09:30 JST. Subagent run started 2026-11-11 ~09:35 JST.

## Quota
- Kaggle GPU quota could NOT be verified: the browser's Google login (qrodpond@gmail.com) is not linked to the cloudgoat
  Kaggle account (Google SSO offered to create a new profile "PondQrod" — cancelled), and the CLI has no quota endpoint.
  No kernel ran since merge3-run (2026-11-10 03:08 UTC), so the last-known figure (~23 h left of 30 h, reset Sat 2026-11-14
  09:00 JST) should still hold. This job is capped at HARD_H=9.55 h kernel time (+ container start) by design.

## Design (files in kds/ = Kaggle dataset cloudgoat/evogm-data, private; kernel cloudgoat/evogm-run, private)
- Donor order (lambda order): agents_a1, dsv4pro, q38distill, qwopus_coder, qwopus_v3, arex_turbo, neohorse, tmax.
- Merge (linmerge.py): exact linear task arithmetic W = W_base + sum lam_i (W_i - W_base), fp32 accumulation, output in each
  base tensor's dtype (F32 A_log/norm stay F32), read straight from donor checkpoints (no TV quantization). NeoHorse keys
  model.* -> model.language_model.* (text only); keys a donor lacks contribute 0 (Agents-A1/AREX have no mtp; tmax/NeoHorse
  text only). Streamed into the persistent vLLM engine with Worker.reload_weights (in place, cudagraph refs preserved; the
  worker checks data_ptrs unchanged after every load).
- Baphomet effective lambda: goatmerge LinearKernel = base + sum w_i T_i; normalize/rescale are not used by the linear
  kernel (rescale only applies with sparsification, density=1) -> 0.1375 x 8 (sum 1.10). valmerge.py compares my merge with
  the published weights (every 4th tensor).
- Search (port of github.com/JiangTao97/evogm fusion_bench/method/evogm.py defaults): coefficients in [-1,1], init =
  8 one-hots + average (1/8) + Baphomet lambda + U(-1,1) randoms; winners = top 30% of the round history; dual MLP
  generators (128-256-256-256-128, ReLU, Tanh), loss = 100*(cycle MSE both ways) + 1*MSE(lose2win(losers), winner centroid),
  Adam lr 1e-4, 200 epochs per iteration, generator persists within a round; new = clip(lose2win(population)); population =
  top-P of population U new; max_iter 3; rounds with basis shift (top-8 of round history = next round experts; round-2
  one-hots reuse cached fitness instead of re-evaluating). seed 42.
- Scale (paper P = 2N = 16, repo 2 rounds x 3 iters, would be ~130 evals) chosen in-kernel from the measured cost of the first
  2 evals: P=12 & 2 rounds if >= 64 evals affordable, P=12 & 1 round if >= 36, P=10 & 1 round if >= 20, else stop.
  Iterations stop early when the remaining budget cannot hold them (round 1 reserves round 2's minimum).
- Fitness: generation, base chat template, enable_thinking, temp .6 top_p .95 top_k 20 seed 0, max_tokens 4096, fp16,
  vLLM 0.33.0 in-process (prefix caching off, max_num_seqs 48, max_model_len 12288). Plain accuracy; truncated = wrong.
  fitness = mean(acc MMLU-Pro, acc GPQA, acc BFCL). Early stop: BFCL generated first; if BFCL acc < 25% the MC items are not
  generated (counted wrong) — only affects obviously broken merges, and the generator uses only the winner/loser split.
- Items (build_fitness.py -> kds/fitness_items.json): 40 MMLU-Pro + 30 GPQA + 40 BFCL non-live (10 per category).
  Disjoint from all 678 bench pool items (incl. the 180 test items): MMLU-Pro by question_id, GPQA: every diamond question
  removed by normalized text (bench pool = all 198 diamond) -> 348 unique main/extended questions eligible; BFCL: bench's
  50/cat ids removed. MMLU-Pro/GPQA selected by IRT Fisher information p(1-p)||a||^2 at the mean endpoint theta of the
  MERGE^3 fit (merge3/irt_out/irt_params.npz, merge3/kout/run/irt_estimate.npz); MMLU-Pro allocated proportionally to the
  14 categories, top-info within each. BFCL random (Random(4242)). Mean loglik-IRT p: MMLU .52, GPQA .47.
- Scoring (fitscore.py) = bench/score.py logic (verified 180/180 identical on raw_baphomet).
- Final: same engines (no restart), winner merged in place, 180 test items split across GPUs, max_tokens 8192, output
  raw_evogm.jsonl in the bench client format -> score locally with bench/score.py + paired McNemar vs raw_baphomet/raw_qwen35.
  Difference vs bench: in-process LLM.generate instead of vllm serve (same sampling params & seed).

## Run log
- 2026-11-11 00:55 UTC: kernel evogm-run v1 pushed.
- v1 (cudagraphs+torch.compile engines, gpumem .86): died silently ~48 min in, while the 2 engines booted (no log, no worker
  logs; suspected host-RAM OOM from two concurrent torch.compile). Setup took ~39 min (pip vllm dominated). valmerge v1:
  my fp32 merge vs published Baphomet: max |diff| 0.00098 (= 1 bf16 ulp), 89/185 sampled tensors bit-identical,
  rel. err vs Baphomet's delta 0.32 (rounding-level differences relative to a small delta).
- v2 (eager) pushed 01:46 UTC, then v3 (HARD_H 9.0) pushed 01:47: unclear whether two sessions would run -> deleted kernel
  evogm-run (killed v2 after ~6 min) and pushed a fresh kernel cloudgoat/evogm-main (HARD_H 8.85) at 01:52 UTC.
- Quota via kagglesdk get_accelerator_quota_statistics (works with the CLI creds!): 2026-11-11 01:51 UTC timeUsed 35,839 s
  (9.96 h) this week, refresh Sat 2026-11-14 09:00 JST (the API also reports totalTimeAllowed 21,600 s — unclear meaning).
- Live logs: klogs.py / kwatch.sh (kagglesdk get_kernel_session_logs_stream).
- evogm-main v1 (01:52 UTC, eager): ROOT/TMP FILESYSTEM WENT READ-ONLY ~02:04 UTC at ~94 GB used (10 checkpoints 90 GB +
  pip) -> pip install vllm silently failed (20 min), workers could not create dirs (OSError 30), valmerge OOM'd (unchunked
  full tensors on GPU). Same pattern explains v1's silent death. Ended 02:14 (ERROR).
- Fix (evogm-main v2, pushed 02:27 UTC, HARD_H 8.55): fp8 TVs (MERGE^3 code), donors downloaded 2 at a time, extracted and
  deleted (peak disk ~60 GB), Baphomet check on 2 of 10 shards only, pip --no-cache-dir to pip.log with import check.
  Job GPU time before this attempt ~1.3 h (quota timeUsed 10.32 h at 02:24 UTC; ~9.05 h before the job started).

## Hardened attempt (v3 script, subagent 2026-11-11 15:05 JST)
- Quota at start (06:05 UTC): timeUsed 38,828 s = 10.79 h this week (refresh Sat 2026-11-14 09:00 JST). Attempt cap <= 10 kernel-h
  (smoke + full + final), keep >= 3 h unused.
- kernels/run/evogm_run.py rewritten (design unchanged): race-free extraction completion (atomic queue-get + in-flight set; error
  only for donors that never arrived), download retries x3, disk guard (our /tmp/m+/tmp/tv <= 70 GB), pip retry, call timeouts
  (load 1500 s, fitness gen 3600 s, final 4 h, boot 1800 s), per-candidate restart+retry once then fitness 0 (failed=True),
  workers dead between batches restarted, kill() waits for GPU memory release (<700 MiB) before reboot, final reuses the live
  engines (restart only if dead). Checkpoint: /kaggle/working/evo_log.jsonl (every candidate, fsync) + evo_state.pt (plan,
  basis, pop, history, generator+Adam state, numpy/torch RNG) at every step boundary; on start it merges every
  /kaggle/input/**/evo_log.jsonl into the cache and resumes from the most advanced /kaggle/input/**/evo_state.pt.
  -> To resume after a crash: push a NEW kernel slug with kernel_sources=[crashed kernel] (+ evogm-data), MODE full.
- Mock test harness mocktest/ (mk.sh builds a fake Kaggle tree; mock gpuworker/linmerge/fitscore with fault injection):
  verified extraction completion, download retry, worker crash/hang/error -> restart -> same results, kill+resume at (2,3)
  reproduces the uninterrupted run exactly (bug found+fixed: Dual() init consumed torch RNG after the restore).
- Smoke kernel cloudgoat/evogm-smoke v1 pushed 06:26 UTC (MODE smoke: extraction, valmerge 2 shards, baphomet+average evals,
  generator/state round trip, restart test on GPU1, mini final 6 items). Its evo_log carries into the full run via kernel_sources.
- evogm-smoke v1 RESULT (06:26-07:06 UTC, ~0.67 h): extraction fix OK (all 8 TVs at 0.305 h, peak disk 47 GB), pip 6 min,
  valmerge fp8 vs Baphomet 2 shards: max|diff| 0.00098 (1 bf16 ulp), rel_err 0.29, 12/63 tensors exact; both engines ready
  (t_init 393/522 s, gpumem .84, 179 KV blocks, 2.8 GB free); generator/state round trip identical; worker-1 restart OK (62 s,
  GPU mem freed to 3 MiB first); mini final 6 items @8192 OK (5/6 correct). BUG: the 2 smoke fitness evals were skipped by the
  time-budget check (HARD_H 1.6 too small for final_est) -> fixed (smoke ignores budget). Fitness eval path is therefore first
  exercised in the full run; added abort if the first 4 evaluations all fail.
- Full run: kernel cloudgoat/evogm-full v1 pushed 07:06 UTC (HARD_H 8.6, 3 download slots, staggered engine boots).
  Outputs to fetch: evo_log.jsonl, evo_state.pt, evo_result.json, history.json, plan.json, raw_evogm.jsonl, progress.log.

## evogm-full RESULT (COMPLETE, 2026-11-11 JST)
- GPU time 7.13 h kernel (search 6.83 h, final ~0.3 h), 40 candidates, 0 failed. Best fitness per iter: 0.686 -> 0.697 -> 0.722 -> 0.722 (iter3 no gain).
- Winner λ = [0.0066,0.1039,0.6271,0.0145,0.2335,0.0526,-0.0035,0.0105] (vs Baphomet 0.1375x8): mass on donor3 (0.63) and donor5 (0.23). fitness 0.722 (mmlu .80 gpqa .47 bfcl .90).
- 180q @8192: EvoGM 77.2% (MMLU-Pro 75.0, GPQA 62.0, BFCL 96.0), trunc 6.7%, 1498 tok; Baphomet 77.8% (77.5/58/98), trunc 15.6%, 3180 tok; Qwen3.5 70.0%.
- Paired: vs Baphomet all 8-9 p=1.0 (no diff); vs Qwen3.5 all 18-5 p=0.011, GPQA 12-2 p=0.013. Script: evogm/cmp.py; scored in final_bench/.

## Round 2+ (IRT) attempt (subagent 2026-11-13 JST)
- Approved 2026-11-13 10:11 JST: cap 8 kernel-h total for this attempt (incl. setup/final/any smoke), keep >= 3 h quota unused.
  Quota at start: timeUsed 66,884 s = 18.58 h (refresh Sat 2026-11-14 09:00 JST).
- Cost model (simt.py, fit on round-1 per-item tokens, 6% error): eager vLLM step time = 0.034 s + 0.0055 s x active seqs ->
  generation cost ~ proportional to generated tokens (5.5 ms per token per sequence), tail overhead ~0.034 s/step.
  Full 110-item eval ~1000 s gen + ~150 s swap; the 40-item subset ~440 s gen.
- IRT (irt.py / irt_final.py -> kds/irt_params.json, embedded base64 in the kernel): matrix irt_mat.npz (40 cand x 110 items,
  truncated = wrong). Per-task 2PL, joint MAP (theta~N(0,1), log a~N(0,.5^2), b~N(0,2^2)). Constant items: MMLU 7 all-right/6
  all-wrong, GPQA 1/8, BFCL 19/0. Subset = per task top Fisher info at the top-15 candidates' theta divided by sqrt(mean tokens):
  16 MMLU-Pro + 10 GPQA + 14 BFCL = 40 items (~66k tokens/cand vs ~165k). Fitness estimate = p-IRT: per task (observed subset correct
  + sum of 2PL p(theta_MAP) over the unobserved items)/n_task, mean of the 3 tasks -> same scale as round-1 fitness.
  LOO over the 40 round-1 candidates: MAE 0.013 vs full fitness (top-15 MAE 0.009, rank corr 0.93); naive subset accuracy MAE 0.066.
  Alternatives tried: 16/12/12 (MAE .013, 479 s), 18/8/14 (.016), alpha 0/1 (cost weighting) similar.
- Racing: worker op gen streams through llm.llm_engine add_request/step (vLLM 0.33 API checked), writes finished outputs every 8 s;
  controller polls every 10 s, after >= 50% of the subset finished computes the optimistic estimate (theta + 1.5 sd) and aborts the
  candidate if < the current P-th best population value. Offline simulation on round-1 iters 2-3: 4/20 pruned, ~9% time saved,
  0 wrongly pruned (2 sd: 1/20, 2%). Any race-path error -> plain llm.generate fallback in the same call + racing disabled.
- Completion (guard vs IRT error / winner's curse): after every iteration, if the best un-completed population member's estimate
  >= best full fitness - 0.02, the top-2 un-completed members are completed on the 70 missing items (one per GPU) -> exact
  110-item fitness (round-1 definition). Population/generator use the best available value (full if known, else IRT).
  Winner = max full-110 fitness over round 1 + completions; final 180-item test only if the winner changed (else round-1 result stands).
- Setup cache (plan item 3) NOT done: 8 fp8 TVs ~30 GB > Kaggle 20 GB output limit, and pip (7.6 min) runs in parallel with the
  donor downloads (critical path = downloads+extraction 0.31 h, engine boot ~0.15 h) -> caching would not reliably save time.
  cudagraph engines would cut the eager per-step overhead a lot, but were not tried (risk, untested on T4 hybrid model).
- Search: resume evo_state.pt (round 1, next (1,4)) -> basis shift (top-8 round-1 history = experts, values .722 .703 .697 .697 .692 .692
  .689 .686), round 2.. with P=10, up to 3 iterations/round, up to round 5; fresh generator per round (as reference); one-hots of the
  new basis reuse cached values. Rounds/iterations sized in-kernel from measured costs (prior 640 s/subset eval, 900 s/completion):
  a round starts if >= 1 iteration fits; the 3rd iteration is skipped when it would make an otherwise affordable next round (T=2) impossible.
  Budget always reserves max(final estimate, 1900 s) + 420 s. HARD_H 7.7.
- Code: kernels/r2/evogm_r2.py (= hardened v3 setup/engines + kernels/r2/part2.py + embedded kernels/r2/gpuworker3.py), mock tests
  mocktest/mk2.sh (r2a-e: normal, winner change + final, race_error fallback + worker crash/restart, kill + resume at (2,2) reproduces
  the uninterrupted run exactly).
- Kernel cloudgoat/evogm-r2 v1 pushed 2026-11-13 01:33 UTC (10:33 JST), private, T4x2, dataset evogm-data, kernel_sources evogm-full.
  Latest possible end ~09:20 UTC (18:20 JST). Monitor: ./check_r2.sh [N]; fetch: ./check_r2.sh fetch (-> kout/r2).
  Scoring: copy kout/r2/raw_evogm.jsonl to final_r2/raw_evogm.jsonl, cd final_r2 && python score.py, then python ../cmp_r2.py
  (final_r2 = copy of merge3/final_bench; cmp_r2.py verified to reproduce the round-1 numbers).
- monitor 2026-11-13 12:07 JST: RUNNING, round 2 iter 1 done (1.58h), 12 new evals (n=52), raced 0, completions r2i1: 0.7092->full 0.7111, 0.7411->full 0.7222 (ties r1); best_full 0.7222 (unchanged). No errors.
- monitor 13:38 JST: RUNNING, round 2 iter 3, n=70 evals, raced 0, best_full 0.7222, no errors
- monitor 15:07 JST: RUNNING (log tail unavailable this check, no error visible)
- monitor 16:40 JST: RUNNING, round3 iter3, n=104 evals, raced ~, best_full 0.7333, no errors

## evogm-r2 RESULT
- COMPLETE 2026-11-13 ~18:05 JST check. 6.53 h GPU (cap 8h). 64 new evals, 3 raced, 0 failed. Rounds 1-3 (round 4 skipped: budget).
- Winner λ=[0.0116,0.2039,0.5901,0.0013,0.2972,0.0323,0.0030,-0.0007] (winner_changed=true). IRT est 0.782 -> full110 0.764 (round-by-round best_full 0.722->0.733->0.764).
- 180q: EvoGM 79.4% (MMLU-Pro 81.2/GPQA 62/BFCL 94), trunc 10%, 1834 tok. Baphomet 77.8%, Qwen3.5 70.0%. McNemar vs Baphomet 10-7 p=0.63; vs Qwen3.5 20-3 p<0.001.
- Caveats: IRT fitness from 40q; winner's curse; temp 0.6 single sample; raced candidates stay estimates. quota.py timeUsed shows 4013s (quota window reset), so used hours taken from evo_result.

## Report update with rounds 2-3 (subagent 2026-11-13 ~19:50 JST)
- report/report_{en,ja}.tex rebuilt -> report/EvoGM_Baphomet_report_{en,ja}.pdf (en 9 pp pdflatex, ja 10 pp uplatex+dvipdfmx). Round-1 versions kept as report_{en,ja}_r1.tex.bak.
  Sources: report/analysis_r2.py -> data_r2.json (all numbers recomputed from scored files / evo_result / evo_log), charts_r2.py -> fig_*_{en,ja}.pdf (report, greyscale serif like figs.py) + charts/*_{en,ja}.png (Hark light theme, 2x).
  Section texts: abstract_*.tex, r2_sections_*.tex, tail_*.tex spliced in by /tmp build scripts (logic: replace abstract, insert before Discussion).
- Final model named Baphomet-1.1-4B (= r3 winner). Key: test 79.4% [Wilson 73.0-84.7] (MMLU-Pro 81.25, GPQA 62, BFCL 94), trunc 10.0%, 1834 tok; vs Bap-1.0 10-7 p=.63, vs r1 12-8 p=.50, vs Qwen 20-3 p=.00049.
  IRT vs exact on the 12 completions: MAE 0.0093, bias +0.008, Spearman 0.96. Fit-110: 1.1 .764 vs best donor q38distill .686, Bap-1.0 .603.
- Correction: best_full per ROUND is 0.722 (r1) -> 0.722 (r2, tie) -> 0.764 (r3); 0.733 is r3 iter 2, not round 2.
- Env note: the sandbox had lost matplotlib/uplatex/IPAex fonts; reinstalled (pip matplotlib scipy; apt texlive-lang-japanese fonts-ipaexfont texlive-latex-extra). \ttdefault=lmtt added (cm-super absent -> avoided bitmap fonts).
