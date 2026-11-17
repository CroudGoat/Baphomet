# EvoGM v2 (design agreed 2026-11-15)
- λ: per donor × {attention, MLP} = 16 dims (embeddings, norms and all other non-MLP tensors go in the attention group; user decided 2026-11-15).
- Search 200: MMLU-Pro 100 / GPQA 60 / BFCL non-live 40 (10/cat). Validation 150: 75 / 45 / 30 (8/8/7/7).
  build_sets.py, seed 20261115, random stratified (MMLU-Pro by category proportional; GPQA main/extended random).
  Disjoint from bench pool (incl. 180 test), v1 110 fitness set, and each other. Files: search_items.json, val_items.json (not for repo: GPQA text).
- Fitness = mean of per-task accuracy on search set. Round 1: full 200 → fit per-task 2PL; Round 2+: ~60-item Fisher subset (split by LOO) + racing early-cut + completion guard.
- Winner / stopping round chosen by validation score. 180 test untouched until the end.
- Open: BFCL ceiling (v1 R1: 19/40 items all-correct) → optional prescreen of BFCL search candidates.
- 2026-11-15 rebalance (user OK): BFCL prescreen (160 cands, 3 models x2) → only 20 mid-difficulty (0<p<1). Search now MMLU-Pro 110 / GPQA 70 / BFCL 20 (simple 5, multiple 8, parallel 4, parallel_multiple 3). rebalance_search.py, seed 20261116. Backups: search_items_random_bfcl.json (original), search_items_bfcl40_pre.json (40 with filler).
- BFCL guard: on validation (BFCL 30), a candidate whose BFCL accuracy is below Qwen3.5-4B base is not eligible as winner.

## v2 kernel BUILT (subagent 2026-11-15 ~20:00-21:00 JST) — NOT pushed, NOT uploaded
Files
- kds/ = dataset cloudgoat/evogm2-data (private, dataset-metadata.json ready): linmerge.py (v2, grouped lambda), fitscore.py,
  bfcl_ast_checker.py, bfcl_eval.tar.gz, search_items.json (200), val_items.json (150), test_items.jsonl + subset_ids.json (180 test).
- src/part_a.py (setup/engines = hardened v3/r2 code, path prefix + time scale hooks for the mock), src/part_b.py (items, IRT, evals,
  budget, generator), src/part_c.py (flow), src/gpuworker4.py (= gpuworker3, has __main__ guard), src/build.py -> kernel_main/evogm2_main.py
  (worker embedded base64) + kernel_main/kernel-metadata.json (cloudgoat/evogm2-main, private, GPU T4x2 "NvidiaTeslaT4", internet,
  dataset_sources [cloudgoat/evogm2-data]). push.sh (dataset create + kernel push; NOT run), check.sh (status/log/fetch -> kout/main).
- tests/test_linmerge_groups.py (+ .out), mock/ (mk.sh, gpuworker_mock.py, mock linmerge/fitscore, chain.sh/chain2.sh, compare.py).
Grouping (linmerge.group_of): MLP = key has a ".mlp." component -> 195 of 738 keys (96 text layers.*.mlp.{gate,up,down}_proj, 3 mtp mlp,
  96 vision blocks.*.mlp.linear_fc* — vision unused for text). Attention = the other 543 (330 text: self_attn 8 layers incl. q/k_norm,
  linear_attn 24 layers incl. A_log/conv1d/dt_bias/norm/in_proj_*/out_proj, input/post_attention layernorms, embed_tokens (tied lm_head;
  there is no lm_head key), final norm; mtp fc/norms/self_attn 12; vision non-MLP 201). Text params: attn 1.99 B (incl. 0.64 B embedding),
  MLP 2.34 B. lams16 = [attn x 8 donors, mlp x 8 donors] in linmerge.DONORS order; 8-dim lams still accepted.
  Unit test: equal halves == 8-dim merge bit-identical on all 738 tensors (4 lambdas incl. B11, random, 0.1375, 0); split halves = each
  tensor equals its own group's 8-dim merge; dtypes kept; length check. PASSED.
Algorithm as built
- Round 1 = INIT (10, priority order): B11 [λ11,λ11]; B10 [.1375]x16; crossovers [λ11|λ10], [λ10|λ11]; re-weights [1.25λ11|.75λ11],
  [.75λ11|1.25λ11]; q38distill one-hot (both groups); ablations [λ11|0], [0|λ11]; one local random λ11+N(0,.1). Reference "average"
  and U(-1,1) randoms dropped (expensive long-output full rows, v1 average .592). Each = full 200 items. Qwen3.5 base val (150) runs
  as the 3rd job of round 1 (λ=0). Budget may cut init members beyond MIN_CAL=8 (they need room for >= 1 iteration).
- IRT (in-kernel, ~3 s CPU for 10 rows): per-task 2PL joint MAP as irt.py; Fisher info at the top-half candidates' theta / tokens^0.5;
  splits tried (MMLU,GPQA,BFCL) = 30/20/10, 32/22/6, 28/24/8, 34/18/8, 26/26/8, 36/16/8, 30/26/4, 24/24/12, 28/20/12; LOO (refit
  without the candidate) MAE vs its full fitness; score = 0.5*MAE + 0.5*MAE(top half); chosen = min score -> irt_v2.json (LOO table).
- Generator iterations exactly as r2 (P=10, MAX_ITER 3, rounds up to 4 with basis shift = top-16 of the round history, dim 16):
  subset p-IRT fitness, racing (theta+1.5sd, >=50% observed, threshold = P-th best), completion of the top-2 un-completed within .02
  of best full. Round >=2 init = 16 experts (cached) + their average (1 subset eval); pop = top-P.
- Validation: candidates = top-4 by full search fitness + Baphomet-1.1 + best full of each round (cap 6). Eligible iff val BFCL >= base
  val BFCL. Winner = max val fitness among eligible (tie: full fitness). Final 180 @8192 only if winner != B11 (B11 = the 79.4% model;
  16-dim [λ11,λ11] is bit-identical to the 8-dim merge). No eligible -> final skipped, flagged. Outputs: evo_log.jsonl, val_log.jsonl,
  irt_v2.json, history.json, plan.json, val_result.json, evo_result.json, raw_evogm.jsonl + final_meta.json, fit_outputs/, val_outputs/.
- Budget: evo deadline = HARD_H - 420 s - max(final_est, 1900 s) - val reserve (base if pending + 5 val evals, ceil/lanes).
  Priors: full 200 eval 2400 s, val = .72 x full, subset/completion priors from round-1 measured times x token fraction.
  HARD_H set to 9.0 in the kernel.
- Resume: push a NEW kernel slug with kernel_sources=[crashed kernel] (+ evogm2-data): reads every evo_log/val_log/fit_outputs/
  val_outputs/evo_state.pt (v3) under /kaggle/input; state saved after IRT, after every iteration, after each basis shift, at evo end.
Mock tests (mock/, scaled time; real controller code)
- A/A2 normal (final path / winner=B11 skip path), C/C2 worker exit + CUDA-error + hang -> restart, same 73 candidates/winner/final as A
  (racing auto-disabled after an error, as designed), D/D2 race path error -> plain generate fallback, racing off, completes,
  E/E3 BFCL floor (base val BFCL forced .85-.9) -> lower-BFCL candidates marked not eligible, E2 floor .97 -> "no eligible winner" path,
  B/G chains: kill in round 1, at state (1,3), at (2,2), during validation, then finish -> REPRODUCED the uninterrupted run exactly
  (evo records, val records, winner, history, final). Racing made deterministic in the mock via a ".seen" ack file written by the
  controller's poll (the real worker ignores it).
- Final build (md5 4fde8ea7 kernel_main/evogm2_main.py, 76 KB) re-tested: A3 normal == A2 exactly; G-chain (kill r1:4 / state 1,3 /
  state 2,2 / val:3 -> finish) == A2 exactly; C2 crashes (exit, CUDA err, hang->timeout) same 71 candidates/winner, 0 failed; D2 race-error
  fallback OK; E3 floor .867 -> 4 of 5 val candidates excluded, B11 eligible (tie at floor counts as eligible).
- Budget fixes found by the realistic-budget mock (TS .03, HARD_H scaled): val reserve now uses the top-K candidates' measured cost
  (base val excluded from the val time model); subset/completion priors from the top-half round-1 rows (verbose ablations made them ~2x
  too high, so no iteration fit). Mock F5 (HARD_H 9 equiv., mock ~25% slower than real): setup+R1 (9 of 10 init, budget cut the last
  one) 4.7 h, 1 iteration 1.6 h, val 0.5 h, ended 7.3 h (+0.5 h final if the winner != B11). Never exceeded HARD_H in any budget run.
Cost projection (real, from v1/r2 per-item tokens via simt + r2 measured swap/subset/completion times)
- full 200 eval: 1.1-like ~1650 s gen (+50-180 s swap), Bap-1.0-like ~2700 s, q38 ~1050 s; round-1 10 init + base val ~ 24k lane-s
  = ~3.4 h wall on 2 GPUs (+ setup 0.45 h). Subset 60 ~700 s, completion 140 items ~1100 s -> one generator iteration ~1.3 h wall.
  Val 150 for a good candidate ~1300 s; final 180 @8192 ~0.45 h.
- HARD_H 9.0 (kernel default): R1 + 2 iterations (~20 generated + <=4 completions), val, final -> ~8.0-8.5 GPU-h.
  HARD_H 10.5: R1 + 3 iterations (one full generator round, ~30 generated) -> ~9.5-10 GPU-h.  (Kaggle session limit 12 h.)
- Quota 2026-11-15 ~20:00 JST (quota.py works): timeUsed 20,361 s = 5.66 h of 30 h, refresh Sat 2026-11-21 09:00 JST -> ~24.3 h left;
  either cap leaves >= 14 h.
Open decisions for the user: HARD_H (9 vs ~10.5); keep the 2 ablation members (diagnostic + IRT anchors, ~0.8 h) or drop them for
  ~+0.6 iteration; whether to refit the IRT with completion rows at round boundaries (not done; spec says round-1 data only).
To launch (after approval): ./push.sh (creates private dataset cloudgoat/evogm2-data from kds/, waits for ready, pushes
  kernel_main). Monitor ./check.sh [N]; fetch ./check.sh fetch -> kout/main. Resume after a crash: new kernel slug with
  kernel_sources=[cloudgoat/evogm2-main] + dataset evogm2-data, same code.

## 2026-11-16 finalize + launch (user approved 08:30 JST)
User decisions: HARD_H 10.5 (session cap; Kaggle limit 12 h); keep both ablation init members; NEW round-boundary IRT refit.
Changes (src/part_a.py, part_b.py, part_c.py; backup of the 11-15 sources in src_bak_20261116/):
- HARD_H default 10.5.
- irt_stage(prev, out, rnd): fits on ALL full-200 rows in the cache (round-1 rows + completion rows); a refit keeps the previous
  t_sub/t_comp priors (completion rows' t_gen is a subset-run time), records fit_round, prev_n_cal/prev_subset_ids, refits[] (round, n_cal,
  chosen split, subset overlap); asserts finite a/b/theta.
- refit_irt(rnd) at each round boundary (only if rnd < MAX_ROUND, after the ">= 16 evaluated members" check, before the basis shift):
  refit + LOO split choice + subset reselect -> irt_v2_refit_r{rnd}.json; any exception -> keep the previous fit, logged
  (refit_failures in IRT state / evo_result). Then rescore_subset_records(): every subset (IRT) record is re-estimated from its saved
  observed items with the fit in use (one scale for racing thresholds / pop ranking / basis shift); original kept as fitness_r1irt,
  irt_version tag; updated lines appended to evo_log (later line wins on resume). Deterministic, idempotent.
- Resume: new state next=(rnd, END_IT=MAX_ITER+1) saved when a round's iterations end (before the refit); resume there redoes the
  refit (identical) then shifts the basis and saves (rnd+1, 0) with the refit IRT. Mock kill point "refit:<rnd>" right after the refit.
- evo_result.irt now has fit_round / refits / refit_failures.
Build: kernel_main/evogm2_main.py md5 070f3e93, 80.2 KB.
Mock results (all PASS):
- A (max_round 4, hard_h 99): refits r1 10->12 rows (split 32/22/6 -> 28/20/12, overlap 43/60), r2 12->16, round 3/4 run, done.
- A2 (max_round 2) baseline; H-chain (mock/chain3.sh: kill r1:4 / state 1,3 / state 1,4 (iterations finished) / refit:1 (right after
  the refit) / state 2,2 / val:3 -> finish) == A2 exactly (71 evo records incl. rescored fitness, 6 val, winner, hist; refit json identical).
- A3 (MOCK_SHIFT 2, max_round 4): 3 refits (r3: 18 rows unchanged -> same split), final 180 path OK.
- C (exit + CUDA err + hang->timeout): 71 candidates, 0 failed, same winner/val as A2 (racing off after the error, by design).
- E (base BFCL .88): 3 candidates excluded by the floor, B11 winner. RF (forced refit failure): fallback to the round-1 fit, logged, completes.
- F realistic budget (TS .03, hard_h 0.315 = 10.5 h): R1 10/10 init by 5.0 h-equiv, 1 iteration, refit (no new rows), no time for round 2,
  val, ended 7.9 h-equiv (winner B11 -> final skipped; final reserve kept). Never exceeded HARD_H. Note: budget is conservative (iter
  estimate 2.7 h-equiv vs 1.9 actual; mock ~25% slower than real), so the real run likely gets 1-2 iterations and may not reach a
  round boundary -> the refit may not trigger in this session.
LAUNCHED 2026-11-16: quota before 20,361 s used (5.66 h of 30 h; API totalTimeAllowed 21,600 s is the known misleading field) ->
  ~24.3 h left. push.sh 08:56 JST: dataset cloudgoat/evogm2-data created + ready, kernel v1 pushed. Kernel log start 23:55:51 UTC
  (08:55:51 JST), 2x T4, pip OK (vllm 0.33.0, 401 s), downloads + TV extraction running normally at +12 min (status RUNNING).
  HARD_H ceiling -> ends by ~19:26 JST; projected ~8 h (~17:00 JST, +~0.45 h if the winner != B11 and the final 180 runs).
  https://www.kaggle.com/code/cloudgoat/evogm2-main  Monitor: ./check.sh (needs >120 s; run kw.sh in background), fetch: ./check.sh fetch.

## 2026-11-16 17:30 result: COMPLETE 8.27h, no eligible winner (all failed val BFCL floor 0.933); final skipped. Top val: generated r1i1 .684 (BFCL .867); B11 val .666 BFCL .90.

## 2026-11-16 17:40 user: loosen BFCL floor -> kernel cloudgoat/evogm2-final pushed (v1)
kernel_final/evogm2_final.py = main (md5 070f3e93) + BFCL_TOL = 2/30 (eligible if val BFCL >= base 0.933 - 0.067, i.e. <=2 of 30 below base).
kernel_sources=[evogm2-main]: evo_state next=(2,0) evo_done=True -> skips evolution; val all cached -> winner = generated r1i1 (lkey 0.027353...,
val .684) -> final 180 @8192. Expected ~1.2 h incl. setup. Check task 7f277cef (30 min from 18:45 JST).

## 2026-11-16 19:50 evogm2-final COMPLETE: winner generated r1i1 (val .684). Final 180 (strict score.py): MMLU-Pro 59/80 (.738), GPQA 28/50 (.56), BFCL 47/50 (.94) = 134/180 (.744).
Baphomet-1.1 same 180: 62/80, 29/50, 49/50 = 140/180 (.778). evogm2 loses by 6 (-3.3pt). (Earlier evogm v1: 143/180.) Next: ask user whether to HF-publish anyway / more search.

## 2026-11-17 00:44 DFS proposal (awaiting user go)
Donors r1i1 / B11 / B10. Qwen3.5-4B: 32 layers, full_attention_interval 4. Space: 8 blocks of 4 layers, per block source (3) + skip/x1/x2, W 3x3
(residual scale at donor transitions; fold as 1/w on all later o_proj/down_proj/linear-attn out proj — verify numerically in mock), max ~40 layers.
Init includes plain B11 and plain r1i1. Reuse IRT subset/racing/completion; BFCL floor base - 2/30. Quota ~15.3 h left (52,784 s used).
Final r1i1 180: 134/180 (MMLU-Pro 59/80, GPQA 28/50, BFCL 47/50) vs B11 140/180.
Revised 00:50 (user): binary per slot like Sakana — slots = r=2 x 3 donors x 8 blocks(4 layers) = 48 bits + W 3x3 = 57 dims; cap ~40 layers.
Evaluator = EvoGM system (200 search, IRT subset, racing, completion, val 150, BFCL floor base-2/30, final 180). Optimizer: EvoGM generator on continuous scores (bit = score>0). Awaiting go.

## 2026-11-17 EvoGM-DFS kernel BUILT + mock-tested (subagent, 01:00-02:05 JST) — NOT pushed, dataset NOT created
Dir /workspace/work/evogm-dfs/: src/{dfslib.py, gpuworker_dfs.py, part_a/b/c.py, build.py} -> kernel/evogm_dfs.py (md5 1ddf5a16, 106 KB; worker +
dfslib embedded b64) + kernel/kernel-metadata.json (cloudgoat/evogm-dfs, private, T4x2 "NvidiaTeslaT4", internet, datasets evogm2-data +
evogm-dfs-data). kds_dfs/ = NEW dataset cloudgoat/evogm-dfs-data (v2_prev.json 277 KB, built by build_prev.py from kout/main: donor lams,
base val record (BFCL .9333), v2 B11/r1i1 search+val+final180 records, all 16 v2 full rows (per-item correct/tokens), irt_v2_refit_r1.json).
push.sh (dataset create + kernel push; NOT run), check.sh (status/fetch -> kout/). tests/ (numeric), mock/ (mk.sh, gpuworker_mock.py,
make_prev.py, chain.sh, compare.py).
Space (dfslib.py): donors 0 r1i1, 1 b11, 2 b10 (exact v2 linmerge fp8-TV merges). 8 blocks x 4 layers (hybrid period). Genotype 57 = 48 slot
scores (bit = score>0) + 9 W coords. Slot order rep -> block -> donor (t = rep*24 + blk*3 + d); path = selected slots in t order. (Paper's
literal layout rep -> model -> layer can switch donors "backwards" only once per repetition with r=2; rbd makes all 3^8 same-depth per-block
donor mixes reachable in rep 0, rep 1 = second pass/duplication.) Path 6..10 blocks (repair: trim lowest / add highest scores; phenotype only).
W_ij = 2^x, log2 on a 1/8 grid in [-1,1] (|x|<1/16 -> 1; without the grid every near-1 W was a "new" model in the mock), applied at EVERY
block boundary (incl. same donor), cumulative C clamped [0.25, 4] (fp16 headroom). Folding: block i's o_proj / linear_attn.out_proj /
down_proj x 1/C_i. Non-layer tensors (embed/tied lm_head, final norm, mtp, visual) = B11 (user spec) -> plain B11 = exactly v2 B11 (v2
search/val records reused, no eval); plain r1i1 in DFS = r1i1 layers + B11 embedding = a NEW model (evaluated in init); the true v2 r1i1
(val .684) is kept as an external candidate in validation.
Engine: 40 layers (padding = B11 layer of the same position with the 3 output projections zeroed, at the end), boot dir = config with 40
layers (layer_types extended periodically) + symlinked B11 shards + pad file; gpumem [0.90, 0.86] (40 layers = +1.7 GiB weights; v2 0.84 had
2.89 GiB KV -> ~2.1 GiB at 0.90, ~47k tokens vs 81k). Candidate = in-place reload_weights from shards (op load_dfs; all tensors yielded,
fold-1 tensors passed through bit-identical). Setup: v2 download/TV pipeline -> 3 donors x 8 block shards + B11 non-layer (~24 GB, built on
both GPUs with linmerge.Merger.iter_merged; peak disk ~69 GB) -> TVs + base weights deleted -> boot.
Evaluator: v2 IRT fit reused (irt_v2_refit_r1, subset 26/26/8) as DFS fit_round 1; refit every ROUND_LEN=3 iterations if new DFS full rows
(v2 rows + DFS completions). NEW smoke screen per candidate: 16 subset items (8 BFCL + shortest others) @1024 tokens; reject if correct<2 or
finished<5 (v2 models: worst = attn-only 4 correct / 8 finished, B11 7/15) -> screened record (16-item IRT estimate); finished smoke
outputs are reused as subset observations. Racing (init thr = B11 full - .05, then P-th best), completion guard (.02, up to 2, resumable by
completion_tag), val 150 = DFS top-4 by full fitness + plain B11 (v2) + v2 r1i1 (v2), floor base - 2/30, final 180 @8192 unless the winner is
plain B11 or v2 r1i1. Session sanity (every session, ~3 min): plain B11 smoke (pads at end) vs v2 B11 on the same items, and pads moved to the
middle must give the same outputs -> abort (status aborted_sanity, quota saved) if the padded engine is broken. Optimizer: EvoGM dual
generator (dim 57, P=10, single coordinate system, no basis shift since 57 > evaluations), novelty repair (flip least-confident bits of
duplicates of previously evaluated phenotypes). Init 13: plain r1i1, plain b11 (cached), r1i1|b11 halves both ways, b11 at even / odd blocks,
r1i1 + b11 block 4 (9 blocks), r1i1 0-5 + 4-7 (10), r1i1 without block 6 (7), r1i1 + b10 tail, r1i1 with W=2^-1/8, 2 local randoms.
CMA-ES not used: ~50 evaluations in 57 dims is < 4 CMA generations (popsize 16); no clear advantage.
Numeric checks (tests/test_dfs_numeric.py, tiny random transformers Qwen3_5ForCausalLM: GDN x3 + gated attn, (1+w) norms, tied emb; 40-layer):
- identity padding: plain B11 padded 40 vs donor 32 max|dlogit| = 0.0 exactly; pads in the middle vs end = 0.0 exactly.
- W folding vs explicit residual scaling (forward pre-hooks on donor layer modules), 12 random genotypes incl. trims and clamp hits:
  float64 (HF internal fp32 casts redirected), eps=0: max rel err 1.8e-14 (exact). eps=1e-6: 4.8e-5 (emb std .02) / 6.8e-6 (emb std 1) =
  the eps term only. fp32: 1.7e-5. Folded out-projs rounded to fp16: 1.6e-3 (= fp16 weight rounding). Without folding: rel err 0.2-1.9.
  Nothing else non-scale-invariant found (GDN, gated norm, q/k norms, attention gate all read normed inputs).
- build_shards + iter_plan(plain, 32 layers) == linmerge full merge, bit-identical on 1275 tensors (3 donors); boot dir: 40 layers,
  layer_types periodic, pad outputs zero, tokenizer copied.
Mock tests (mock/, real controller code): A (max_iter 6) normal -> final path; H-chain (kill init eval:6 / state:1 / mid-iteration eval:30 /
refit:1 / state:6 / val:2 -> finish) == A exactly (60 records, 5 val, hist, winner, final) — found+fixed: novelty must ignore records of the
current iteration (else a resumed iteration generates a different batch); C (worker exit + CUDA err + hang->timeout): 0 failed, same winner
as the racing-off run; D (race error -> plain generate); E (base BFCL .97) -> plain B11 / v2 r1i1 / some DFS excluded, eligible DFS winner,
final runs; S (broken engine) -> calibration abort; G (real v2_prev.json) -> v2 IRT path, completes. Harness note: a mock-killed session
leaves its mock workers running; mk.sh now kills them (they served reused dirs twice in an earlier chain).
Budget (mock F: TS .03, HARD_H 10.5 h-equiv; mock per-eval ~1170 s/eval = ~1.5x a v2-calibrated estimate): setup+calibration 0.87 h, init
(12 evals + 2 completions) done 2.65 h, iteration 1 5.68 h, iteration 2 7.80 h, val 8.9 h, final done 9.23 h (< 10.5; final reserve kept).
31 candidate evals + 6 completions. Real estimate (v2 measured x 1.3-1.5): subset eval ~12-15 min/lane (swap ~45 s, smoke ~2 min, gen
~9-11 min), completion ~25-30 min, broken path ~3-4 min -> init ~1.7 h, ~1.4 h per 10-candidate iteration -> ~4 iterations, ~45-55
candidate evaluations + ~8-10 completions, ~9.5-10 h total.
Unverifiable here (no GPU): vLLM 0.33 behaviour of the 40-layer boot dir / layerwise reload with renamed layers (guarded by the session
sanity check, costs ~0.7 h of quota if it aborts), real swap/KV numbers at gpumem 0.90.
LAUNCHED 2026-11-17 08:28 JST: evogm-dfs-data + kernel cloudgoat/evogm-dfs v1 (md5 1ddf5a16, 40 layers). Monitor b7d46e28.

EvoGM-DFS RESULT (2026-11-17 17:0x JST): COMPLETE, 8.18h. Val(45 q/task): v2_r1i1 0.6844 (winner; not in DFS space) > plain_b11 .6659 > plain_r1i1 .6622 > generated b11[0]+r1i1[1]+b11[2]+r1i1[3]+b11[4-7] .6533 > even-mix .6511 (BFCL .833 < floor .867, ineligible) > odd-mix .6385; base .508. Final 180 skipped (winner = v2 r1i1, 134/180). No DFS candidate beat v2 r1i1.
