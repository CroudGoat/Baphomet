# Baphomet-1.1-4B local reconstruction (2026-11-13 JST, CPU sandbox, aarch64 Neoverse-V2, torch 2.15.0+cpu)
- Output: Baphomet-1.1-4B/ (9.34 GB). Not uploaded.
- Recipe = evogm/kds/linmerge.py (extract_tv + Merger.iter_merged), ported to CPU in extract.py / merge.py (row chunks, CHUNK_ELEMS 2^23).
  lambda = kout/r2/evo_result.json best.lams (float32 values). Repos pinned to the HF shas current on 2026-11-13 (all last modified before the Kaggle run).
- TVs: tv/<donor>/ (fp8 e4m3 + per-row scale, grouped files + index.json). n/zero/missing identical to kout/r2/tv_stats.json for all 8; sqnorm matches to <=1e-8 rel.
- Checks: logs/checks.log (headers byte-identical to base, keys/dtypes/shapes equal, 0 nonfinite, 366/738 tensors changed; vision + MTP unchanged = no donor TV for them),
  logs/check_indep.log (independent whole-tensor recompute bit-identical, 6 tensors), logs/check_baph10.log + logs/check_exact.log (uniform 0.1375 merge vs published 1.0).
- Residual risk: Kaggle merged on CUDA; CPU here. Elementwise IEEE ops, fma in add_(alpha) on both (verified on CPU), RNE fp8 cast -> expected bit-identical, not verifiable without a GPU.
