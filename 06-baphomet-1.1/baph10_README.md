---
language:
- en
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.5-4B
- InternScience/Agents-A1-4B
- Jackrong/Qwopus3.5-4B-Coder
- Jackrong/Qwopus3.5-4B-v3
- Jackrong/DeepSeek-V4-Pro-Qwen3.5-4B
- empero-ai/Qwen3.8-4B-Distill
- BAAI/AREX-Turbo
- TokenRhythm/NeoHorse-1-4B-GGUF
- allenai/tmax-4b
tags:
- merge
- goatmerge
- task-vector
- linear-merge
- qwen3_5
- vision-language
- gguf
---

<div align="center">

# Baphomet-1.0-4B

<p>Linear merge of 8 task-vectors based on Qwen3.5-4B</p>

</div>

<div align="center">
<table>
<tr>
<th>Base</th>
<th>Method</th>
<th>Task-Vectors</th>
<th>Output</th>
</tr>
<tr>
<td><code>Qwen/Qwen3.5-4B</code></td>
<td><code>linear</code> (<a href="https://github.com/CroudGoat/GoatMerge">goatmerge</a>)</td>
<td>8 vectors, 0.1375 each</td>
<td><code>safetensors ×10</code>, <code>Q8_0 GGUF</code></td>
</tr>
</table>
</div>

---

## 1. Overview

Baphomet-1.0-4B is a model produced by linearly combining 8 task-vectors (`T = W_donor − W_base`) into the base model `Qwen/Qwen3.5-4B`.
According to a [2025 paper (hitit et al., 2025)](https://arxiv.org/pdf/2603.09938) on model merging, pure Task Arithmetic is the only method that consistently improves performance across "in-the-wild" models. Based on that paper, I merged high-quality community fine-tuned versions of Qwen3.5-4B.

```text
W_merged = W_base + Σ αᵢ · Tᵢ   (αᵢ = 0.1375)
```

The configured weights sum to `0.1375 × 8 = 1.10`. According to `goatmerge.json`, `normalize: true, rescale: true` are enabled, so normalized coefficients are applied. Refer to the goatmerge implementation for the exact computation.

No claim is made about improved performance or inheritance of specific capabilities. No quantitative evaluation has been performed.

---

## 2. Merge configuration

### 2.1 Base model and settings

| Item | Value |
|:---|:---|
| Base | `Qwen/Qwen3.5-4B` |
| Base fingerprint | `62de7a218d003afe6dffca82532c20150488b68ec629e72f9b50a0615dd7c8c5` |
| Tool | [goatmerge](https://github.com/CroudGoat/GoatMerge) (format v1.1.0) |
| Method | `linear`, `density=1.0`, `normalize=true`, `rescale=true` |
| Chunk | `8388608` elements, `drop_caches=true` |

### 2.2 Task-vector list (α = 0.1375 each)

The table below transcribes the extraction source (`source_model`) recorded in each TV's `goatmerge.json`. No statement of effect is made.

| # | Task-vector | Recorded source_model |
|:---:|:---|:---|
| 1 | `Agents-A1-4B-TV` | `InternScience/Agents-A1-4B` |
| 2 | `DeepSeek-V4-Pro-Qwen3.5-4B-TV` | `Jackrong/DeepSeek-V4-Pro-Qwen3.5-4B` |
| 3 | `Qwen3.8-4B-Distill-TV` | `empero-ai/Qwen3.8-4B-Distill` |
| 4 | `Qwopus3.5-4B-Coder-TV` | `Jackrong/Qwopus3.5-4B-Coder` |
| 5 | `Qwopus3.5-4B-v3-TV` | `Jackrong/Qwopus3.5-4B-v3` |
| 6 | `AREX-Turbo-TV` | `BAAI/AREX-Turbo` |
| 7 | `NeoHorse-1-4B-TV` | `NeoHorse-1-4B-Aligned` (key-converted `TokenRhythm/NeoHorse-1-4B`) |
| 8 | `tmax-4B-TV` | `allenai/tmax-4b` |

<details>
<summary>recipe.yaml (for reproduction)</summary>

```yaml
base: Qwen/Qwen3.5-4B
out: ./Baphomet-1.0-4B-k8

merge_method: linear
chunk_elements: 8388608
drop_caches: true
skip_fingerprint_check: false

tv:
  - dir: ./Task-Vectors/Agents-A1-4B-TV
    weight: 0.1375
  - dir: ./Task-Vectors/DeepSeek-V4-Pro-Qwen3.5-4B-TV
    weight: 0.1375
  - dir: ./Task-Vectors/Qwen3.8-4B-Distill-TV
    weight: 0.1375
  - dir: ./Task-Vectors/Qwopus3.5-4B-Coder-TV
    weight: 0.1375
  - dir: ./Task-Vectors/Qwopus3.5-4B-v3-TV
    weight: 0.1375
  - dir: ./Task-Vectors/AREX-Turbo-TV
    weight: 0.1375
  - dir: ./Task-Vectors/NeoHorse-1-4B-TV
    weight: 0.1375
  - dir: ./Task-Vectors/tmax-4B-TV
    weight: 0.1375
```

</details>

Note: `NeoHorse-1-4B` uses a text-only key layout, so its keys were converted from `model.*` to `model.language_model.*` with `align_neohorse.py` before extraction. See Section 5 of the parent directory README for details.

---

## 3. Model specifications

Excerpt from `config.json`.

| Item | Value |
|:---|:---|
| Architecture | `Qwen3_5ForConditionalGeneration` |
| Text layers | 32 layers, hidden size `2560`, heads `16Q / 4KV (GQA)`, `head_dim 256` |
| Attention layer types | Repeating pattern of 3 `linear_attention` layers followed by 1 `full_attention` layer |
| FFN | `intermediate 9216`, `SiLU`, `tie_word_embeddings: true` |
| Vocabulary size | `248,320` (`Qwen2Tokenizer`, BPE) |
| Context length | `262,144` (RoPE θ=`10000000`, interleaved M-RoPE) |
| Vision | hidden size `1024`, 24 layers, `patch_size 16`, `spatial_merge_size 2`, `temporal_patch_size 2` |
| Special tokens | `<|im_start|>`, `<|im_end|>`, `<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`, `<|video_pad|>`, `<think>`, `<tool_call>`, and others |
| Dtype | `bfloat16` (safetensors) |

---

## 4. Included files

| File | Size | Description |
|:---|---:|:---|
| `model-00001-of-00010.safetensors` through `model-00010…` | Approx. 8.9GB total | Model weights (bf16, 10 shards) |
| `model.safetensors.index.json` | 66KB | Shard index |
| `Baphomet-1.0-4B-q8_0.gguf` | 4.2GB | Q8_0 quantization for llama.cpp |
| `config.json` | 3.1KB | Model configuration |
| `tokenizer.json` / `vocab.json` / `merges.txt` | 13MB / 6.5MB / 3.2MB | Tokenizer |
| `tokenizer_config.json` / `chat_template.jinja` | 17KB / 7.6KB | Chat template |
| `preprocessor_config.json` | 390B | `Qwen3VLProcessor` configuration |
| `goatmerge.json` | 1.6KB | Merge provenance (fingerprint, TV list) |

---

## 5. Usage

### 5.1 Transformers (safetensors)

```python
from transformers import AutoModelForVision2Seq, AutoProcessor

model_id = "Baphomet-1.0-4B"  # Replace with a local path or repository name

model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": "https://example.com/cat.png"},
        {"type": "text", "text": "Describe this image in English."},
    ]},
]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(out[0], skip_special_tokens=False))
```

### 5.2 llama.cpp (GGUF)

```bash
# Interactive chat
llama-cli -m Baphomet-1.0-4B-q8_0.gguf -cnv -c 8192

# Server mode
llama-server -m Baphomet-1.0-4B-q8_0.gguf -c 32768 --port 8080
```

### 5.3 vLLM

```bash
vllm serve ./Baphomet-1.0-4B-k8 --max-model-len 32768 --trust-remote-code
```

<details>
<summary>Chat format example</summary>

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Write a parallel downloader in Rust<|im_end|>
<|im_start|>assistant
<think>
Organizing the requirements…
</think>
```rust
// code here
```<|im_end|>
```

Tool calls may be emitted in `<tool_call><function=…>` form and reasoning in `<think>…</think>`. Refer to `chat_template.jinja` for the exact format.

</details>

---

## 6. Benchmark
![benchmark](https://pbs.twimg.com/media/HWrH2hOawAAovww?format=jpg&name=large)
| Suite | Condition | Base (`Qwen/Qwen3.5-4B`) | Baphomet-1.0-4B | Notes |
|:---|:---|:---:|:---:|:---|
| MMLU-Pro | Same prompts and settings as base | 71.3% | 77.5% | n=80 |
| GPQA-Diamond | Same prompts and settings as base | 42.0% | 58.0% | n=50 |
| BFCL non-live | Same prompts and settings as base | 96.0% | 98.0% | n=50 |

The base model (`Qwen3.5-4B`) and `Baphomet-1.0-4B` were evaluated under identical conditions for direct comparison.

---

## 7. Limitations

- The model is the result of a linear merge; there is no guarantee that characteristics of individual donor models are preserved.
- It may inherit the license terms and biases of each donor and the base model. Determine the scope of use after reviewing the license of each model.
- Benchmark evaluation is in progress (see Section 6). No results are reported yet, and no performance claims are made.
- Use for final decisions in high-risk domains such as medical or legal matters is not intended.

---

## 8. Credits

| Role | Model |
|:---|:---|
| Base | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Donors | InternScience/Agents-A1-4B, Jackrong/DeepSeek-V4-Pro-Qwen3.5-4B, empero-ai/Qwen3.8-4B-Distill, Jackrong/Qwopus3.5-4B-Coder, Jackrong/Qwopus3.5-4B-v3, BAAI/AREX-Turbo, TokenRhythm/NeoHorse-1-4B, allenai/tmax-4b |
| Merge tool | [GoatMerge](https://github.com/CroudGoat/GoatMerge) |
| References | [O. K. Hitit, C. Sankar, Y. K. Kim, A. Risteski, and A. Elgohary, "A systematic study of in-the-wild model merging for large language models," arXiv preprint arXiv:2511.21437, 2025.](https://arxiv.org/pdf/2511.21437)
