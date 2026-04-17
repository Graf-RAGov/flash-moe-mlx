# Qwen3.6-35B-A3B Architecture Spec

For use in: `metal_infer/model_config.h`, `extract_weights.py`, `repack_experts.py`

## Sources

- https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json
- https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/model.safetensors.index.json
- https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-8bit
- https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF

---

## Architecture Parameters

| Param | Value | Notes |
|-------|-------|-------|
| architectures | Qwen3_5MoeForConditionalGeneration | Same class as Qwen3.5 |
| num_hidden_layers | 40 | vs 60 in Qwen3.5-397B |
| hidden_size | 2048 | vs 4096 in Qwen3.5-397B |
| num_attention_heads | 16 | full-attn layers only |
| num_key_value_heads | 2 | GQA 8:1 |
| head_dim | 256 | |
| num_experts | 256 | vs 512 in Qwen3.5-397B |
| num_experts_per_tok | 8 | plus 1 shared expert (effective K=9) |
| moe_intermediate_size | 512 | per-expert hidden; vs 1024 in Qwen3.5-397B |
| shared_expert_intermediate_size | 512 | |
| vocab_size | 248320 | same as Qwen3.5-397B |
| rope_theta | 10000000 | |
| partial_rotary_factor | 0.25 | 64 of 256 head dims rotated |
| max_position_embeddings | 262144 | 1.01M with YaRN factor 4.0 |
| rms_norm_eps | 1e-6 | |
| hidden_act | silu | SwiGLU experts |
| tie_word_embeddings | false | separate lm_head |
| layer_types | 3x linear_attention then 1x full_attention x 10 | full-attn at {3,7,11,15,19,23,27,31,35,39} |
| full_attention_interval | 4 | |
| linear_num_value_heads | 32 | GatedDeltaNet; vs 64 in Qwen3.5-397B |
| linear_num_key_heads | 16 | |
| linear_key_head_dim | 128 | |
| linear_value_head_dim | 128 | |
| linear_conv_kernel_dim | 4 | depthwise conv1d |
| attn_output_gate | true | NEW — sigmoid gate on attn output (not in Qwen3.5-397B) |
| attention_bias | false | |
| q_norm.weight / k_norm.weight | per full-attn layer | NEW — QK-norm (not in Qwen3.5-397B) |
| mtp_num_hidden_layers | 1 | extra MTP block — drop for text-only inference |
| has vision tower | yes (27-block ViT, 1152 hidden, 16 patch) | drop for text-only inference |

---

## Layer Type Map

40 total layers, repeating pattern of 3 linear_attention + 1 full_attention:

```
Layer  0: linear_attention
Layer  1: linear_attention
Layer  2: linear_attention
Layer  3: full_attention    <- MoE + full self-attn
Layer  4: linear_attention
Layer  5: linear_attention
Layer  6: linear_attention
Layer  7: full_attention
... (pattern repeats every 4 layers)
Layer 36: linear_attention
Layer 37: linear_attention
Layer 38: linear_attention
Layer 39: full_attention
```

Full-attention layers: {3, 7, 11, 15, 19, 23, 27, 31, 35, 39} — 10 layers total.
Linear-attention layers: all others — 30 layers total.
(Upstream Qwen3.5-397B: full-attn at every 4th of 60 = 15 full-attn, 45 linear-attn)

---

## Tensor Naming

All language model tensors live under `model.language_model.*` prefix.

### Top-level

```
model.language_model.embed_tokens.weight       # [vocab_size, hidden_size]
model.language_model.norm.weight               # [hidden_size]
lm_head.weight                                 # [vocab_size, hidden_size]
```

### Full-attention layers (N in {3,7,11,15,19,23,27,31,35,39})

```
model.language_model.layers.N.self_attn.q_proj.weight
model.language_model.layers.N.self_attn.k_proj.weight
model.language_model.layers.N.self_attn.v_proj.weight
model.language_model.layers.N.self_attn.o_proj.weight
model.language_model.layers.N.self_attn.q_norm.weight   # NEW — QK-norm
model.language_model.layers.N.self_attn.k_norm.weight   # NEW — QK-norm
model.language_model.layers.N.input_layernorm.weight
model.language_model.layers.N.post_attention_layernorm.weight
```

### Linear-attention layers (all other N)

```
model.language_model.layers.N.linear_attn.in_proj_qkv.weight
model.language_model.layers.N.linear_attn.in_proj_z.weight
model.language_model.layers.N.linear_attn.in_proj_a.weight
model.language_model.layers.N.linear_attn.in_proj_b.weight
model.language_model.layers.N.linear_attn.conv1d.weight
model.language_model.layers.N.linear_attn.out_proj.weight
model.language_model.layers.N.linear_attn.A_log
model.language_model.layers.N.linear_attn.dt_bias
model.language_model.layers.N.linear_attn.norm.weight
model.language_model.layers.N.input_layernorm.weight
model.language_model.layers.N.post_attention_layernorm.weight
```

### MoE (all layers — both full-attn and linear-attn have MoE FFN)

```
model.language_model.layers.N.mlp.gate.weight                    # routing gate
model.language_model.layers.N.mlp.experts.gate_up_proj           # 3D: [num_experts, 2*moe_intermediate, hidden]
model.language_model.layers.N.mlp.experts.down_proj              # 3D: [num_experts, hidden, moe_intermediate]
model.language_model.layers.N.mlp.shared_expert.gate_proj.weight
model.language_model.layers.N.mlp.shared_expert.up_proj.weight
model.language_model.layers.N.mlp.shared_expert.down_proj.weight
model.language_model.layers.N.mlp.shared_expert_gate.weight
```

### Skip (text-only inference)

```
mtp.*                # Multi-Token Prediction block
model.visual.*       # Vision tower (27-block ViT)
```

---

## Tokenizer

- **Type:** Qwen2Tokenizer (BPE)
- **Files:** `vocab.json` + `merges.txt`
- **Pretokenizer:** Standard GPT-2-family regex (byte-level BPE)
- **vocab_size:** 248320 — same as Qwen3.5-397B, reuse tokenizer.bin if already exported
- **Special tokens (same as Qwen3.5-397B):**
  - EOS: 151645 (in Qwen2 numbering — verify against actual tokenizer.json)
  - The hardcoded tokens in upstream infer.m (248046, 248044, 248068, 248069) are Qwen3.5-specific; re-check against Qwen3.6 tokenizer.json

---

## Quantization (mlx-community/Qwen3.6-35B-A3B-8bit)

- **Total size:** 37.75 GB across 8 safetensors shards
- **Format:** MLX affine 8-bit (`bits=8`)
- **Confirm at load time:** Check `config.json` in the 8-bit repo for `quantization.bits` and `quantization.group_size`
- This is the format flash-moe's existing dequant pipeline handles (with PR #14's `dequant_matvec_8bit` kernel)

---

## Delta vs Qwen3.5-397B (Porting Checklist)

| Item | Qwen3.5-397B upstream | Qwen3.6-35B-A3B | Action |
|------|-----------------------|-----------------|--------|
| hidden_size | 4096 | **2048** | Config update |
| num_hidden_layers | 60 | **40** | Config update |
| num_experts | 512 | **256** | Config update |
| num_experts_per_tok | 10 | **8** | Config update |
| moe_intermediate_size | 1024 | **512** | Config update |
| linear_num_value_heads | 64 | **32** | Config update |
| num_attention_heads | 32 | **16** | Config update |
| attn_output_gate | absent | **true** | New codepath needed |
| QK-norm | absent | **q_norm, k_norm** | New codepath needed |
| Tensor prefix | `model.*` | **`model.language_model.*`** | extract_weights.py fix |
| 3D expert tensors | per-expert files | **fused [N_exp, ...]** | repack slicing fix |
| Vision tower | absent | present (skip) | Filter in extract_weights |
| MTP block | absent | present (skip) | Filter in extract_weights |
