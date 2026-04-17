# MTP (Multi-Token Prediction) Self-Speculative Decoding — Research

Research notes for task #18: wire Qwen3.6-35B-A3B's bundled MTP head into flash-moe's Metal inference engine to run self-speculative decoding (lossless, target 1.6-1.9x speedup).

Author: Opus agent, worktree `agent-afdd7d37`, 2026-04-17.

---

## TL;DR

1. **MTP architecture** (per Qwen3-Next / DeepSeek-V3): one extra transformer block that takes the final hidden state of the main model + the embedding of the next-token candidate, normalises both, concatenates, projects back to `hidden_size` via `fc`, runs one MoE decoder layer, final RMSNorm, then the **shared `lm_head`** for logits.
2. **Published speedups**: DeepSeek-V3 reports ~1.8x at ~85% single-step acceptance. Qwen3-Next advertises 1.3-2.1x (vLLM / SGLang production reports).
3. **For Qwen3.6-35B-A3B specifically**: the MTP block exists in the original `Qwen/Qwen3.6-35B-A3B` release (shards 25 + 26 of 26, 19 tensors named `mtp.*`) but **the `mlx-community/Qwen3.6-35B-A3B-8bit` quantization dropped them entirely**. Our local `/Users/retry/qwen36-35b-a3b-8bit/model.safetensors.index.json` contains **zero `mtp.*` entries** (confirmed via `grep`).
4. **HF transformers does not yet ship a reference forward pass for the MTP head** for Qwen3-Next family — the `Qwen3NextPreTrainedModel` class explicitly sets `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`. The reference forward lives only in vLLM (`vllm/model_executor/models/qwen3_next_mtp.py`) and SGLang.
5. **Prior experiment on this codebase**: the results.tsv / CLAUDE.md explicitly lists `MTP speculative decoding | break-even | MoE I/O scales per-token (unlike dense)`. For a MoE streamed from SSD, each additional draft token pays the full K-expert SSD read cost, eliminating the classic dense-model speculative win.

---

## 1. MTP Architecture (Qwen3-Next / Qwen3.6 family)

### 1.1 Components (from vLLM reference and the HF weight index)

The full Qwen3-Next MTP block consists of the following named submodules (verified against the original Qwen3.6-35B-A3B safetensors index, see section 4):

| Submodule | Role | Notes |
|-----------|------|-------|
| `mtp.pre_fc_norm_embedding` | RMSNorm on the next-token embedding | weight `[hidden_size]` |
| `mtp.pre_fc_norm_hidden` | RMSNorm on the main model's hidden state | weight `[hidden_size]` |
| `mtp.fc` | Linear `2*hidden -> hidden`, concat(embed_norm, hidden_norm) -> h' | weight `[hidden_size, 2*hidden_size]`, no bias |
| `mtp.layers.0.*` | One full MoE transformer decoder layer (self-attn + MoE + norms) | Identical shape family to a `model.layers.N` full-attn block. Has 256 experts + 1 shared expert (same MoE shape as main layers). |
| `mtp.norm` | Final RMSNorm before lm_head | `[hidden_size]` |
| (tied) `lm_head` | Reused from the main model — **not duplicated** in the MTP block | The `mtp_use_dedicated_embeddings: false` in `config.json` signals that both `embed_tokens` and `lm_head` are shared. |

Sources:
- vLLM MTP model: [`vllm/model_executor/models/qwen3_next_mtp.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next_mtp.py) (web-fetched forward pass summary).
- HF transformers currently ignores these weights: `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` in [`modular_qwen3_next.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modular_qwen3_next.py).
- Local config `/Users/retry/qwen36-35b-a3b-8bit/config.json` lines 724-725: `"mtp_num_hidden_layers": 1`, `"mtp_use_dedicated_embeddings": false`.

### 1.2 Forward pass (single-draft step)

Given the main model's last hidden state `h_main` at position `t` and the token just emitted `tok_t`:

```
e = embed_tokens[tok_t]                        # lookup in shared embedding
e_n = rmsnorm(e,      mtp.pre_fc_norm_embedding.weight)
h_n = rmsnorm(h_main, mtp.pre_fc_norm_hidden.weight)
x   = concat([e_n, h_n], axis=-1)              # [2*hidden_size]
x   = mtp.fc @ x                               # -> [hidden_size]
x   = mtp.layers[0](x, position=t+1)           # full MoE decoder layer, self-attn uses its OWN per-step cache
x   = rmsnorm(x, mtp.norm.weight)
draft_logits = lm_head @ x                     # shared with main lm_head
draft_tok    = argmax(draft_logits)            # or sample
```

For N draft tokens you iterate this block N times, each time feeding back the previous `draft_tok` and the MTP layer's own hidden state. (Strictly one MTP block is trained for k=1 next-step prediction — iterating it produces progressively lower-accuracy drafts. DeepSeek-V3 trains k=2 and reports k=1 acceptance ~85%.)

### 1.3 Verification (self-speculative)

Exact lossless speculative decoding (Leviathan et al.):

1. Produce N draft tokens `d_1..d_N` from MTP.
2. Run the main model **once** on the sequence `[..., tok_t, d_1, d_2, ..., d_N]` with causal attention — yields `N+1` logits distributions (one per draft position + one for the token after d_N).
3. For i = 1..N: compare argmax of main-model logits at position i-1 with `d_i`. Accept if equal, otherwise take the main-model's token and stop.
4. If all N drafts accepted, also consume the bonus token (main logits at position N).

This matches DeepSeek-V3 and vLLM/SGLang reference implementations. When sampling at T>0, a probability-ratio acceptance test replaces argmax-equality.

---

## 2. Published acceptance rates and speedups

| Model / impl | Acceptance rate | Wall-clock speedup | Source |
|--------------|-----------------|---------------------|--------|
| DeepSeek-V3, MTP1 | 85-90% (next-token) | ~1.8x | [DeepSeek-V3 Technical Report, arXiv 2412.19437](https://arxiv.org/abs/2412.19437); [DeepWiki summary](https://deepwiki.com/deepseek-ai/DeepSeek-V3/4.4-multi-token-prediction-(mtp)) |
| Qwen3-Next (native MTP) | not published as %, > DeepSeek per Qwen blog | 1.3-2.1x (SGLang/vLLM) | [vLLM blog: Qwen3-Next support](https://blog.vllm.ai/2025/09/11/qwen3-next.html), [Qwen3-Next Alibaba Cloud blog](https://www.alibabacloud.com/blog/602580) |
| DeepSeek-V3 + SGLang on ROCm | 1.25-2.11x (Random), 1.36-1.80x (ShareGPT) | (same) | [ROCm tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/mtp.html) |

### 2.1 Expected acceptance math (Leviathan et al.)

For draft length γ and per-token acceptance rate α, the expected number of tokens *generated per verification step* is:

```
E[accepted] = (1 - α^{γ+1}) / (1 - α)   (Leviathan 2023, "Fast Inference from Transformers via Speculative Decoding")
```

At α=0.85, γ=4: E ≈ 3.66 tokens per main-model forward pass. Raw forward-pass speedup = 3.66x **in the limit of zero draft cost and zero per-token I/O overhead**. In practice:

- Draft cost (1 MoE layer per draft) ~= 1/40 of one main forward → ≈2.5% per draft.
- Per-token SSD I/O for the main model's verification forward costs the same as a single token step **only if the K=8 experts per layer needed at each verified position can be batched into the existing pread pipeline**. Otherwise cost grows linearly with γ. See §5 "Why MoE breaks the classical speculative win".

Sources: [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192), [DeepSeek Explained 4 — Shirley Li](https://medium.com/data-science-collective/deepseek-explained-4-multi-token-prediction-33f11fe2b868).

---

## 3. Reference implementations

| Project | File | Status |
|---------|------|--------|
| vLLM | `vllm/model_executor/models/qwen3_next_mtp.py` | Production. `Qwen3NextMTP` wraps `Qwen3NextMultiTokenPredictor` (`embed_tokens` / `pre_fc_norm_embedding` / `pre_fc_norm_hidden` / `fc` / `layers[step_idx]` / `norm`) + a shared `lm_head`. Confirmed by direct web fetch of the file. |
| SGLang | `python/sglang/srt/models/qwen3_next_mtp.py` | Production; referenced in the ROCm tutorial above. |
| HF transformers | `modeling_qwen3_next.py` | **Not implemented** — `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` drops the weights at load time. [GH source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modular_qwen3_next.py). |
| MLX-lm | — | No speculative branch for Qwen3-Next MTP found at time of writing. |
| llama.cpp | — | No Qwen3-Next MTP support; general speculative decoding works only with a separate draft model. |
| unsloth | — | GGUF of Qwen3.6 drops MTP (consistent with llama.cpp). |

Our practical implementation reference is vLLM's file. No open-source "flash-moe-style" engine has wired MTP for streamed-from-SSD MoE inference.

---

## 4. Qwen3.6-35B-A3B MTP tensors (from HF index)

Fetched from `https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/model.safetensors.index.json`:

| Tensor | Shard |
|--------|-------|
| `mtp.fc.weight` | 00026-of-00026 |
| `mtp.pre_fc_norm_embedding.weight` | 00026-of-00026 |
| `mtp.pre_fc_norm_hidden.weight` | 00026-of-00026 |
| `mtp.norm.weight` | 00026-of-00026 |
| `mtp.layers.0.input_layernorm.weight` | 00026-of-00026 |
| `mtp.layers.0.post_attention_layernorm.weight` | 00026-of-00026 |
| `mtp.layers.0.self_attn.q_proj.weight` | 00026-of-00026 |
| `mtp.layers.0.self_attn.k_proj.weight` | 00026-of-00026 |
| `mtp.layers.0.self_attn.v_proj.weight` | 00026-of-00026 |
| `mtp.layers.0.self_attn.o_proj.weight` | 00026-of-00026 |
| `mtp.layers.0.self_attn.q_norm.weight` | 00026-of-00026 |
| `mtp.layers.0.self_attn.k_norm.weight` | 00026-of-00026 |
| `mtp.layers.0.mlp.gate.weight` | 00026-of-00026 |
| `mtp.layers.0.mlp.experts.gate_up_proj` | **00025-of-00026** |
| `mtp.layers.0.mlp.experts.down_proj` | 00026-of-00026 |
| `mtp.layers.0.mlp.shared_expert.gate_proj.weight` | 00026-of-00026 |
| `mtp.layers.0.mlp.shared_expert.up_proj.weight` | 00026-of-00026 |
| `mtp.layers.0.mlp.shared_expert.down_proj.weight` | 00026-of-00026 |
| `mtp.layers.0.mlp.shared_expert_gate.weight` | 00026-of-00026 |

Total 19 tensors. Shapes are not published in the index — must be derived from a safetensors header parse after download.

### 4.1 Expected shapes (derived from config.json)

With `hidden_size=2048`, `num_experts=256`, `moe_intermediate_size=512`, `shared_expert_intermediate_size=512`, `num_key_value_heads=2`, `head_dim=256`, `num_attention_heads=16` (full-attention — the MTP decoder block is a full-attention layer, not linear-attention):

| Tensor | Shape | dtype (unquantized) |
|--------|-------|---------------------|
| `mtp.fc.weight` | `[2048, 4096]` | bf16 |
| `mtp.pre_fc_norm_embedding.weight` | `[2048]` | bf16 |
| `mtp.pre_fc_norm_hidden.weight` | `[2048]` | bf16 |
| `mtp.norm.weight` | `[2048]` | bf16 |
| `mtp.layers.0.input_layernorm.weight` | `[2048]` | bf16 |
| `mtp.layers.0.post_attention_layernorm.weight` | `[2048]` | bf16 |
| `mtp.layers.0.self_attn.q_proj.weight` | `[16*256, 2048]` = `[4096, 2048]` | bf16 |
| `mtp.layers.0.self_attn.k_proj.weight` | `[2*256, 2048]` = `[512, 2048]` | bf16 |
| `mtp.layers.0.self_attn.v_proj.weight` | `[512, 2048]` | bf16 |
| `mtp.layers.0.self_attn.o_proj.weight` | `[2048, 4096]` | bf16 |
| `mtp.layers.0.self_attn.q_norm.weight` | `[256]` | bf16 |
| `mtp.layers.0.self_attn.k_norm.weight` | `[256]` | bf16 |
| `mtp.layers.0.mlp.gate.weight` | `[256, 2048]` | bf16 |
| `mtp.layers.0.mlp.experts.gate_up_proj` | `[256, 1024, 2048]` | bf16 |
| `mtp.layers.0.mlp.experts.down_proj` | `[256, 2048, 512]` | bf16 |
| `mtp.layers.0.mlp.shared_expert.gate_proj.weight` | `[512, 2048]` | bf16 |
| `mtp.layers.0.mlp.shared_expert.up_proj.weight` | `[512, 2048]` | bf16 |
| `mtp.layers.0.mlp.shared_expert.down_proj.weight` | `[2048, 512]` | bf16 |
| `mtp.layers.0.mlp.shared_expert_gate.weight` | `[1, 2048]` | bf16 |

Rough unquantized sizes: fc ~16MB; experts tensors ~ (256 * 1024 * 2048 + 256 * 2048 * 512) * 2 B ≈ 1.5 GB. At 8-bit MLX affine quantisation (group_size=64) the experts would compress to ~400 MB, matching one additional `layer_40.bin` file of the same class as `layer_39.bin`.

Sources for shapes: [Qwen3.6-35B-A3B config.json](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json); cross-ref with per-expert shapes in this repo's `/Users/retry/Documents/code/flash-moe/repack_experts.py` `qwen3.6-35b-a3b` profile.

---

## 5. Why MoE-streamed-from-SSD breaks the classical speculative win

The per-layer timing breakdown in this repo's main `CLAUDE.md`:

```
CMD1 + CMD2: attn + routing + shared   ≈1.8ms GPU
I/O: pread K experts (~6.75 MB each)    ≈2.4ms SSD (60% of step)
CMD3: expert forward                    ≈0.04ms encode (deferred)
```

Speculative decoding amortises **one** main-model forward over (1 + E[accepted]) tokens. In a **dense** model that one forward's cost is fixed, so the win is direct. In flash-moe's MoE, verifying γ draft tokens means the main model must read **γ distinct K=8 expert sets per layer** (or do the worst-case union ≈ min(γK, 256) per layer). Empirically from the `Speculative early routing` row in `results.tsv`, reading even a small additional set of experts causes cache pollution and SSD DMA arbitration against GPU compute (unified memory, see `parallelism-exploration.md`).

The prior flash-moe MTP experiment (CLAUDE.md `Discarded` table: `MTP speculative decoding | break-even | MoE I/O scales per-token (unlike dense)`) likely had the same finding.

**There is still one regime where MTP can win here**: when draft tokens reuse mostly the **same** experts as the last-emitted token (temporal locality in expert routing is weak — 25% hit rate per `train_predictor.py` analysis) — so for γ=2 and very aggressive batching of main-model expert reads, some speedup may remain, but recovering 1.6-1.9x as DeepSeek-V3 dense-expert-matmul inference sees is unlikely.

Sources: [Flash-MoE CLAUDE.md](/Users/retry/Documents/code/flash-moe/CLAUDE.md) "Discarded" table; `docs-research/parallelism-exploration.md`; Leviathan et al. 2023.

---

## 6. Critical blockers for implementation in this worktree

### 6.1 Source safetensors are gone

`/Users/retry/qwen36-35b-a3b-8bit/` no longer holds `*.safetensors` (confirmed via `ls` — only `packed_experts/`, config files, and the index remain). More importantly the mlx-community 8bit distribution **never shipped MTP tensors** — `grep mtp model.safetensors.index.json` returns 0 matches. To get the MTP weights we would need to:

- **Option A — download shards 25 + 26 of the original Qwen/Qwen3.6-35B-A3B (bf16)**: sizes are 3.83 GB + 2.23 GB = **~6.06 GB total** (verified via `huggingface.co/api/models/Qwen/Qwen3.6-35B-A3B/tree/main`). These are bf16 not MLX 8-bit, so the infer.m pipeline would need a bf16 matvec path (the 8bit dequant kernel can't consume them as-is), OR we run a one-off MLX quant over just the MTP block (~minutes of CPU work once the pipeline exists).
- **Option B — download the mlx-community full (non-8bit) Qwen3.6 repo** if one exists — unverified.
- **Option C — skip weights, wire all the glue code** (CLI flag, speculative loop in infer.m, stub `mtp_draft_token` that returns a "decline to draft" sentinel) so the next agent with weights can drop in the forward pass. This is the cheapest path to meaningful progress.

### 6.2 The worktree's infer.m is still Qwen3.5-397B

Grep of the worktree's `infer.m` shows `HIDDEN_DIM 4096 / NUM_LAYERS 60 / NUM_EXPERTS 512 / NUM_EXPERTS_PER_TOK 10` and `MODEL_PATH_DEFAULT` pointing at the Qwen3.5 snapshot (lines 72-80, 127). No `MODEL_QWEN36_35B_A3B profile` or runtime profile switch exists in this branch. Implementing MTP first requires the Qwen3.6 port (tracked elsewhere — see `docs-research/qwen3.6-35b-a3b-arch.md` and `issue-20-other-qwen-models.md`).

### 6.3 MTP on this engine is already a known discarded experiment

From `/Users/retry/Documents/code/flash-moe/CLAUDE.md` line 105 (the "Discarded (58 experiments)" table): `MTP speculative decoding | break-even | MoE I/O scales per-token (unlike dense)`. The prior result (on Qwen3.5-397B, not 3.6) found that MoE expert streaming eliminates the classical speculative-decoding win.

---

## 7. Expected result if fully implemented

Revising the 1.6-1.9x target in light of §5 and §6.3:

- **Verification forward per step**: the main model reads K=8 experts per layer per position. For γ=2 drafts, worst-case expert reads scale ~2x per layer. With OS page cache hit rate ~71% and 30% of draft positions touching new experts, incremental I/O is ~0.3 x γ x 2.4 ms = 1.44 ms extra per step at γ=2.
- **Draft cost**: one MTP MoE layer ≈ 5-7ms (1 layer's worth of GPU + SSD I/O) per draft.
- **Net step time**: main_forward (~160 ms for 40 layers steady state target) + γ x draft_cost (~12 ms for γ=2) = ~172 ms for potentially 2 tokens → ~11.6 tok/s vs 9.7 tok/s baseline → **1.20x best-case**, not 1.8x.
- **Realistic** once SSD arbitration penalties are included: break-even to +15%, matching the prior experimental result.

The "published 1.8x" numbers are from dense models (DeepSeek-V3 served at FP8 weights in VRAM, no SSD streaming) and from MoE with full-weights-in-VRAM (vLLM / SGLang on 8xH100). Neither regime matches flash-moe.

---

## 8. Decision recommendation

Given (a) no MTP weights locally, (b) 6 GB download to obtain bf16 weights + requantisation work, (c) the Qwen3.6 port itself hasn't landed in this worktree, and (d) the prior break-even result on the same engine class, the correct call is:

1. **Do not re-run Phase 3 (code) in this 90-minute timebox**. The expected ceiling is +15%, well below the 1.6-1.9x task target.
2. Capture the research above plus the implementation plan (`opt-mtp-implementation.md`) so a future agent can pick up quickly if a dense or fully-in-RAM model class is added.
3. Track the blocker explicitly: "MTP requires source safetensors (~6 GB download) + Qwen3.6 port; expected +0-15% speedup on this streaming-MoE engine, not 1.8x".

---

## Appendix A — Commands used

```bash
# Confirm MTP tensors absent locally
grep -c mtp /Users/retry/qwen36-35b-a3b-8bit/model.safetensors.index.json
# => 0

# Confirm safetensors deleted
ls /Users/retry/qwen36-35b-a3b-8bit/*.safetensors
# => no matches

# Confirm original HF repo has MTP shards
# via https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/model.safetensors.index.json
# (see §4 for tensor list, §6.1 for shard sizes)
```

## Appendix B — Sources

- DeepSeek-V3 paper: https://arxiv.org/abs/2412.19437
- DeepSeek MTP deep dive: https://deepwiki.com/deepseek-ai/DeepSeek-V3/4.4-multi-token-prediction-(mtp)
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023): https://arxiv.org/abs/2211.17192
- Qwen3-Next vLLM support blog: https://blog.vllm.ai/2025/09/11/qwen3-next.html
- Qwen3-Next Alibaba Cloud overview: https://www.alibabacloud.com/blog/602580
- SGLang + DeepSeek MTP on ROCm: https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/mtp.html
- HF transformers modular_qwen3_next.py (no MTP class, weights ignored): https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modular_qwen3_next.py
- vLLM qwen3_next_mtp.py (the reference forward pass): https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next_mtp.py
- Qwen3.6-35B-A3B HF index: https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/model.safetensors.index.json
- Qwen3.6-35B-A3B config: https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json
- Flash-MoE CLAUDE.md "Discarded" table (MTP break-even line): /Users/retry/Documents/code/flash-moe/CLAUDE.md
- vLLM MTP bug reports (context on Qwen3NextMTP as an architecture): https://github.com/vllm-project/vllm/issues/29945, https://github.com/vllm-project/vllm/issues/36331
