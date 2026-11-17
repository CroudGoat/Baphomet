# Baphomet: context for agents

Read this first. It is the handoff for any agent (Hark, Claude Code, Codex, ...) working on this repo.
Owner: CloudGoat (GitHub CroudGoat, Hugging Face CloudGoat). Talk to the owner in casual Japanese.

## What Baphomet is
A family of 4B Japanese/English reasoning + tool-calling models made by **merging, not training**.
- Base: Qwen3.5-4B.
- 8 donor fine-tunes of the same base (lambda order used everywhere):
  `agents_a1, dsv4pro, q38distill, qwopus_coder, qwopus_v3, arex_turbo, neohorse, tmax`
  (exact HF repo ids and pinned shas: `05-evogm/donor_keys.json`, `06-baphomet-1.1/RECON_NOTES.md`).
- **Baphomet-1.0-4B**: linear task arithmetic, 0.1375 x 8 (sum 1.10), built with GoatMerge (owner's own merge engine, separate repo).
- **Baphomet-1.1-4B**: EvoGM (ICML 2026) style evolutionary search over the 8 lambdas, generation-accuracy fitness. Weights on HF `CloudGoat/Baphomet-1.1-4B` (public since 2026-11-15). Rebuild from lambda with `05-evogm/kds/linmerge.py` (CPU port in `06-baphomet-1.1/merge.py`).

## Results (keep these caveats whenever you report numbers)
| | 180-item test (MMLU-Pro / GPQA / BFCL, thinking on) |
|---|---|
| Qwen3.5-4B base | 70.0% |
| Baphomet-1.0-4B | 77.8% |
| EvoGM round-1 winner | 77.2% with ~half the thinking tokens |
| Baphomet-1.1-4B | 79.4% |

- 1.1 vs 1.0 is **not significant** (McNemar p=0.63). Don't call it a clear win.
- On the 110-item search set 1.1 scores 0.764 (MMLU-Pro 80.0, GPQA 56.7, BFCL 92.5) vs best donor q38distill 0.686. This is optimistic: 1.1 was selected on that set (winner's curse: estimated 0.782 -> measured 0.764). Single sample, SE ~0.04.
- The 8 donors were **never run on the 180-item test**. That is the open fair comparison.
- 1.1 weights were reconstructed on CPU; Kaggle merged on CUDA. Expected bit-identical, not verified on GPU.
- `generation_config.json` in the 1.1 upload was added by us (removable).
- After 1.1: EvoGM v2 and EvoGM-DFS (2026-11-16/17) did not beat 1.1. Main lesson: search/val were scored at a 4,096-token cap but the test at 8,192, which favours short reasoners; align the caps next time. See `08-evogm-dfs/report/findings_ja.pdf`.
- Earlier failure: MERGE3 (IRT-fitness evolutionary dare_ties) lost to 1.0 and broke tool calling; loglik MMLU-Pro/GPQA fitness did not predict generation results. See `04-merge3/NOTES.md`.

## Where things live
- `01-baphomet-1.0-merge/`  early merge logs / slerp script
- `02-bench180/`  the 180-item paired benchmark (items, scorer, BFCL AST checker, 1.0 vs base results)
- `03-bench90/`  90-item bench incl. MERGE3 run
- `04-merge3/`  MERGE3 attempt (IRT item selection, kernels, logs)
- `05-evogm/`  EvoGM run: NOTES.md (full design), kernels, kout logs, final / final_r2 results, report (ja/en PDF + TeX), upstream evogm repo copy in `repo/`
- `06-baphomet-1.1/`  CPU reconstruction, checks, model card, upload script
- `07-evogm2/`  EvoGM v2 (16-dim lambda, attention vs MLP per donor): winner r1i1 134/180 vs B11 140/180 (strict scorer); not released
- `08-evogm-dfs/`  EvoGM-DFS (layer-path search over r1i1 / B11 / B10): best in-space = plain B11. Findings: `08-evogm-dfs/report/findings_ja.pdf`
- Full map: `REPO_STRUCTURE.md`
- Also: private GitHub `CroudGoat/EvoGM-Qwen3.5-4B` (EvoGM code+report).

## Compute
- GPU: Kaggle user `cloudgoat`, T4 x2, 30 h/week, resets Saturday 09:00 JST. Keep >= 3 h spare.
- Owner's local GPU: 12 GB VRAM.

## Not in this repo (deliberately)
Model weights, task vectors, tokenizers, downloaded datasets (MMLU-Pro, GPQA, BFCL parquet/json), third-party repos (tinyBenchmarks, mergekit, MERGE3 repo, IRT response dumps), venvs, files > 20 MB, and **any API keys or tokens**. Never commit secrets here.
Note: item files contain GPQA questions; GPQA asks not to post them publicly, so keep this repo private or strip them before going public.
