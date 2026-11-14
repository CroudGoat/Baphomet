# Qwen3.5-4B vs CloudGoat/Baphomet-1.0-4B (paired, 180 items, T4, thinking ON, max_tokens 8192)
| bench | n | Qwen acc | Baph acc | Qwen trunc | Baph trunc | Qwen avg tok | Baph avg tok | B-only/Q-only | exact McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| MMLU-Pro | 80 | 71.3% | 77.5% | 17.5% | 13.8% | 4385 | 3331 | 7/2 | 0.18 |
| GPQA-D | 50 | 42.0% | 58.0% | 54% | 34% | 7094 | 5915 | 9/1 | 0.021 |
| BFCL non-live AST | 50 | 96% | 98% | 0 | 0 | 229 | 205 | 1/0 | 1.0 |
Both-untruncated: MMLU-Pro n65 Q56 B58; GPQA n22 Q20 B20.
Persona leakage: 0 in both. Settings: vLLM 0.33.0 fp16, temp0.6 top_p0.95 top_k20 seed0, own chat templates.
