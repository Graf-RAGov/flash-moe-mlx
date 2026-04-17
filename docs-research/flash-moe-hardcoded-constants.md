# flash-moe Hardcoded Constants

All values are for the upstream target: **Qwen3.5-397B-A17B-4bit** (MLX 4-bit).
Every constant listed here must become a runtime lookup for the Qwen3.6 port.

**Bottom line:** PR #3 + nerds-odd-e fork do most of this already. Use them — don't redo.

---

## 1. Shape Constants in metal_infer/infer.m (lines 72-101)

| Constant | Value | Location | Qwen3.6 value |
|----------|-------|----------|---------------|
| `HIDDEN_DIM` | 4096 | infer.m:72 | **2048** |
| `NUM_LAYERS` | 60 | infer.m:73 | **40** |
| `NUM_ATTN_HEADS` | 32 | infer.m:74 | **16** |
| `NUM_KV_HEADS` | 2 | infer.m:75 | 2 (same) |
| `HEAD_DIM` | 256 | infer.m:76 | 256 (same) |
| `VOCAB_SIZE` | 248320 | infer.m:77 | 248320 (same) |
| `NUM_EXPERTS` | 512 | infer.m:78 | **256** |
| `NUM_EXPERTS_PER_TOK` | 10 | infer.m:79 | **8** (9 effective with shared) |
| `MOE_INTERMEDIATE` | 1024 | infer.m:80 | **512** |
| `SHARED_INTERMEDIATE` | 1024 | infer.m:81 | **512** |
| `GROUP_SIZE` | 64 | infer.m:84 | 64 (same — 8-bit affine group) |
| `LINEAR_NUM_V_HEADS` | 64 | infer.m:88 | **32** |
| `LINEAR_NUM_K_HEADS` | 16 | infer.m:89 | 16 (same) |
| `LINEAR_KEY_DIM` | 128 | infer.m:90 | 128 (same) |
| `LINEAR_VALUE_DIM` | 128 | infer.m:91 | 128 (same) |
| `LINEAR_TOTAL_KEY` | 2048 | infer.m:92 | derived: LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM |
| `LINEAR_TOTAL_VALUE` | 8192 | infer.m:93 | derived: LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM = **4096** |
| `LINEAR_CONV_DIM` | 12288 | infer.m:94 | derived from linear_attn in_proj shapes |
| `CONV_KERNEL_SIZE` | 4 | infer.m:95 | 4 (same) |
| `ROPE_THETA` | 10000000 | infer.m:96 | 10000000 (same) |
| `PARTIAL_ROTARY` | 0.25 | infer.m:97 | 0.25 (same) |
| `ROTARY_DIM` | 64 | infer.m:98 | 64 (same — 0.25 * head_dim=256) |

### Effective K at Runtime

Confusingly, `NUM_EXPERTS_PER_TOK=10` in the config but the code uses **K=4** at line 6533. This is the gorroai K=4 optimization (or an upstream approximation). For Qwen3.6, `num_experts_per_tok=8`; K=4 approximation still available if benchmarking justifies it.

---

## 2. Same Values Mirrored in extract_weights.py (lines 126-145)

`extract_weights.py` validates tensor shapes against hardcoded expected dimensions. All `HIDDEN_DIM`, `NUM_LAYERS`, `NUM_ATTN_HEADS`, etc. appear again here as shape checks. The nerds-odd-e fork's `+67/-32` rewrite of this file updates these to be config-driven.

Key constants duplicated in extract_weights.py:
- Expert tensor expected shapes derived from `NUM_EXPERTS * MOE_INTERMEDIATE * HIDDEN_DIM`
- Linear-attn tensor shapes derived from `LINEAR_TOTAL_KEY`, `LINEAR_TOTAL_VALUE`, `LINEAR_CONV_DIM`
- Layer count loop: `for layer_idx in range(NUM_LAYERS)`

---

## 3. Layer-Type Dispatch

### extract_weights.py:149-155

```python
if (i + 1) % 4 == 0:
    # full attention layer
else:
    # linear attention layer
```

This is the `full_attention_interval=4` pattern. Same for Qwen3.6 — no change needed.

### infer.m:4646

Comment notes the 4-interval pattern. Same constant embedded in the attention dispatch switch.

### NUM_LINEAR_LAYERS at infer.m:978

```c
#define NUM_LINEAR_LAYERS 45  // 60 - 15 full-attn
```

For Qwen3.6: `NUM_LINEAR_LAYERS = 30` (40 - 10 full-attn).

---

## 4. Expert Packing Format

### EXPERT_SIZE

```c
#define EXPERT_SIZE 7077888  // bytes per expert, infer.m:103
```

This is `Qwen3.5-397B`-specific. Derived from: `(gate_proj + up_proj + down_proj) weights + scales + biases` at 4-bit MLX affine quant with `HIDDEN_DIM=4096`, `MOE_INTERMEDIATE=1024`.

For Qwen3.6 (8-bit, `HIDDEN_DIM=2048`, `MOE_INTERMEDIATE=512`), EXPERT_SIZE will be different. Compute at config load:

```
EXPERT_SIZE = 9 * (weight_bytes + scales_bytes + biases_bytes)
```

Also mirrored at: `repack_experts.py:40`

### Sub-tensors per Expert (repack_experts.py:28-38)

9 sub-tensors at fixed byte offsets within each expert blob:

| Index | Name | Notes |
|-------|------|-------|
| 0 | gate_proj.weight | |
| 1 | gate_proj.scales | MLX affine per-group scale |
| 2 | gate_proj.biases | MLX affine per-group bias |
| 3 | up_proj.weight | |
| 4 | up_proj.scales | |
| 5 | up_proj.biases | |
| 6 | down_proj.weight | |
| 7 | down_proj.scales | |
| 8 | down_proj.biases | |

2-bit variant halved at `repack_experts_2bit.py:97-121`.

### File Layout

- One file per layer: `packed_experts/layer_XX.bin`
- Expert E at offset: `E * EXPERT_SIZE` within the layer file
- Non-expert weights: `model_weights.bin` (5.5 GB), 64-byte aligned, sanitized tensor names
- Manifest: `model_weights.json`

---

## 5. Tokenizer

### Custom BPET Format

- Magic bytes: `"BPET"` at offset 0 (`tokenizer.h:140`)
- vocab_size: dynamic from header (not hardcoded)

### Special Tokens — Hardcoded in infer.m:122-125

```c
#define EOS_TOKEN     248046
#define EOS_TOKEN2    248044
#define THINK_START   248068
#define THINK_END     248069
```

These are Qwen3.5-397B-specific token IDs. **Verify against Qwen3.6 tokenizer.json** — vocab is same size (248320) but special token assignments may differ. The nerds-odd-e fork's `model_config.h` likely makes these runtime-configurable; confirm before porting.

---

## 6. Non-Expert Weights File

| Item | Value |
|------|-------|
| Filename | `model_weights.bin` |
| Size (Qwen3.5-397B 4-bit) | ~5.5 GB |
| Alignment | 64-byte aligned |
| Tensor names | sanitized (`.` replaced with `_`, prefix stripped) |
| Manifest | `model_weights.json` (JSON array of {name, offset, shape, dtype}) |

For Qwen3.6 (8-bit, half the dimensions), expect ~2-3 GB for non-expert weights.

Tensor prefix to strip: `model.language_model.` (Qwen3.6) vs `model.` (Qwen3.5-397B). Update extract_weights.py accordingly.

---

## 7. Metal Shader Constraint

### shaders.metal:274, 372

```metal
threadgroup float x_shared[4096];
```

This shared memory array bakes in a maximum `hidden_dim` of 4096. Qwen3.6 `hidden_size=2048` is strictly less than 4096 — **no shader change needed**. The shader will use only the first 2048 slots.

If we ever port to a model with `hidden_size > 4096`, this constant would need updating (and verifying Metal threadgroup memory limits).

---

## 8. Quantization Details

| Item | Upstream (4-bit) | Qwen3.6 target (8-bit) |
|------|-----------------|------------------------|
| Format | MLX affine 4-bit | MLX affine 8-bit |
| group_size | 64 | 64 (likely — confirm from 8-bit config.json) |
| Packing | 8 nibbles per uint32 | 1 byte per element |
| Scale dtype | bf16 per group | bf16 per group |
| Bias dtype | bf16 per group | bf16 per group |
| GGUF support | None | None (do not use llama.cpp GGUF) |

The `dequant_matvec_8bit` Metal kernel from PR #14 handles the 8-bit case. Cherry-pick PR #14 as Step 2 of the port plan.
