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
