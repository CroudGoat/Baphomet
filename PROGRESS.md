# Progress (as of 2026-11-14 JST)

## Done
- 2026-11-09  Baphomet-1.0-4B beats Qwen3.5-4B on the 180-item bench (mostly fewer truncations from shorter reasoning).
- 2026-11-10  MERGE3 attempt lost to 1.0 and broke tool calling.
- 2026-11-12  EvoGM round 1: ties 1.0 (77.2 vs 77.8) at ~half the thinking tokens.
- 2026-11-13  EvoGM round 2+ (6.5 GPU-h): new winner 79.4% = Baphomet-1.1-4B. Reconstructed on CPU, uploaded to HF (private). Report written (ja/en).
- 2026-11-13  1.1 vs the 8 donors on the 110-item search set: 1.1 above every donor (optimistic, see AGENTS.md).

## Open / waiting on owner
- [ ] Make `CloudGoat/Baphomet-1.1-4B` public?
- [ ] Submit to Open Japanese LLM Leaderboard (llm-jp)?
- [ ] Run the 8 donors on the 180-item test (Kaggle GPU) for a fair comparison.
- [ ] NLP2027 (言語処理学会) write-up. Deadline 2027-01-15 15:00 JST. Angle under consideration: efficiency, or handling lineage overlap among donors.
- [ ] Donor lineage check (e.g. modelDNA / Model Kinship) to see how much the 8 donors overlap.
- [ ] Fix the source mismatch for numbers in model card Section 6.
- [ ] MERGE3: 90 items left unevaluated (low priority).
