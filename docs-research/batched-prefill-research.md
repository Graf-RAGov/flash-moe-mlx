# Batched Prefill — Research Survey

Task #19, worktree `agent-a2c6c991`. Target: replace flash-moe's serial per-token prefill (~109 ms/token on Qwen3.6-35B-A3B 8-bit, M1 Pro) with a batched prefill that processes N prompt tokens per attention pass. Goal 5-10× TTFT speedup. Steady-state decode path untouched.

Baseline confirmation: `docs-research/benchmark-ram-tokspeed.md` rows 80-88 show TTFT scales linearly at ~109 ms/prompt-token, no batching present.

## 1. Prior art — how production engines batch prefill

### 1.1 llama.cpp

- `llama_decode` accepts a `llama_batch` with `n_tokens` token slots, each with its own `pos` and `seq_id`. Prefill is just "a decode call where `n_tokens > 1`" — the same kernel path serves both phases. [llama.cpp/discussions/4130](https://github.com/ggml-org/llama.cpp/discussions/4130) [deepwiki llama.cpp batch processing](https://deepwiki.com/ggml-org/llama.cpp/3.5-batch-processing-pipeline)
- Two-tier sizing:
  - `n_batch` (`--batch-size`, default 2048) = logical batch size. Limits size of one `llama_decode` call's logits/embeddings buffer.
  - `n_ubatch` (`--ubatch-size`, default 512) = physical batch size sent to the compute kernels. Big batches are split into ubatches and each ubatch runs as one forward pass. [llama.cpp/discussions/6328](https://github.com/ggml-org/llama.cpp/discussions/6328)
- Prefill is compute-bound (GEMM over N × hidden tokens), decode is memory-bound (GEMV one token at a time). A single llama_decode can mix prefill-chunk tokens with decode tokens across sequences — this is continuous batching. [Profile llama.cpp with Arm Streamline](https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama_cpp_streamline/2_llama.cpp_intro/)
- Each token in the batch gets its own `KQ_mask` row so cross-sequence isolation is preserved; causal mask is just the ubatch-local one. [deepwiki batch processing pipeline](https://deepwiki.com/ggml-org/llama.cpp/3.5-batch-processing-pipeline)

Takeaway for flash-moe: the "llama.cpp model" is exactly what we need — one unified forward pass, `n_ubatch` parameter controlling physical batch size.

### 1.2 vLLM — chunked prefill (original reference implementation)

- Long prompts are split into chunks (default `max_num_batched_tokens=512` on A100) and each chunk forms one scheduler iteration. Decode tokens from other requests share the same batch slot. [vLLM optimization docs](https://docs.vllm.ai/en/stable/configuration/optimization/) [Don Moon blog](https://donmoon.medium.com/llm-inference-optimizations-2-chunked-prefill-764407b3a67a)
- Rationale: prefill is compute-bound, decode is memory-bound. Co-scheduling in the same batch keeps both units busy, reported +50 % total token throughput. [vLLM RFC #3130](https://github.com/vllm-project/vllm/issues/3130)
- On Qwen3-Next (hybrid linear + full attention, same family as Qwen3.6), vLLM uses **Prefill Context Parallel (PCP)** — zigzag ring attention for full-attn layers, batch-dim partitioning for linear-attn layers. [vLLM Qwen3-Next issue #37995](https://github.com/vllm-project/vllm/issues/37995) [vLLM blog — Qwen3-Next](https://blog.vllm.ai/2025/09/11/qwen3-next.html)

Takeaway: **chunk size ~512 is the empirical sweet spot** on datacenter GPUs. On M1 Pro with 16 GB of usable GPU RAM the upper bound will be lower, likely 64-128.

### 1.3 mlx-lm / SwiftLM — Apple Silicon specifics

- mlx-lm exposes `--prefill-size` (default 512), equivalent to llama.cpp's batch size. SwiftLM reimplements it explicitly to avoid O(N²) unified-memory blowup on huge prompts. [SwiftLM README](https://github.com/SharpAI/SwiftLM)
- MLX itself does **not** implement "true" chunked prefill in the vLLM sense yet — long (100k+) inputs still have poor TTFT. [MLX Apple Silicon comparative study](https://yage.ai/share/mlx-apple-silicon-en-20260331.html)
- LM Studio ships a `lmstudio-mlx-patch` that specifically speeds up Apple Silicon prompt processing ~2× by fixing MLX's prefill batching. [thornad/lmstudio-mlx-patch](https://github.com/thornad/lmstudio-mlx-patch)
- Practical chunk size on Apple Silicon: **default 512, 4096 for throughput, 8192+ causes memory pressure.** Qualitative; no hard numbers for M1 Pro specifically. [MLX comparative study](https://yage.ai/share/mlx-apple-silicon-en-20260331.html)

Takeaway: on a 16 GB-GPU M1 Pro with 40-layer × ~2.6 GB hidden working set, the safe ceiling is well below 512. **We pick 32-64 as the MVP target.**

### 1.4 FlashAttention-2 — multi-query tile-based kernel

- FA-2 splits the attention computation across (head, Q-tile) pairs. One threadgroup produces one Q-tile × full-KV output block. Online softmax tracks per-row max and denominator to avoid materializing the full `N × N` matrix. [FA-2 paper (arxiv 2307.08691)](https://arxiv.org/pdf/2307.08691) [Princeton NLP FA-2 blog](https://princeton-nlp.github.io/flash-atttention-2/)
- On Apple Silicon, the usable tile is `Br=Bc=32`, giving `32×32=1024` threads/threadgroup — exactly the Metal `maxTotalThreadsPerThreadgroup` limit. [tiny-llm Metal FA walkthrough](https://skyzh.github.io/tiny-llm/week2-04-flash-attention.html)
- Existing Metal FA-2 reference impls: [harvestingmoon/flash_attn_metal_cpp](https://github.com/harvestingmoon/flash_attn_metal_cpp). GQA/MQA integration requires separate handling because KV heads are fewer than Q heads.

Shared-memory budget per tile for Qwen3.6-35B-A3B full-attn layer:
- Heads: 16 Q × 2 KV × head_dim 256. GQA ratio 8.
- Per threadgroup working set for batch B: Q tile (B × 256 × 4 B) + K tile (Bc × 256 × 4 B) + V tile (Bc × 256 × 4 B) + softmax scratch (B × Bc × 4 B).
- B=1 (current decode): 1×256×4 + 32×256×4 × 2 + 32×4 = 1 KB + 64 KB + 128 B — already exceeds 32 KB at tile 32.
- Practical plan: **keep the per-query kernel (current `attn_scores_batched` + `attn_softmax_batched` + scores@V) and run it B times with shared K/V cache reads.** The "flash" part is forfeited; we gain from avoiding B separate command buffers, not from on-chip softmax. For B=32 this still collapses 32 × 3 command buffers into 1 × 3 encoders.

### 1.5 Metal threadgroup memory hard limits on M1 Pro

- macOS enforces a **32 KB threadgroup memory limit** for compute kernels. M1 family max threads/threadgroup = 1024. [Apple dev forum 674385](https://developer.apple.com/forums/thread/674385) [mlc-llm issue #3293](https://github.com/mlc-ai/mlc-llm/issues/3293) [Metal feature set tables PDF](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- `[MTLComputePipelineState maxTotalThreadsPerThreadgroup]` is the runtime query. [Apple docs setThreadgroupMemoryLength](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443142-setthreadgroupmemorylength)

**Implication for batched `attn_scores`:** The current `attn_scores_batched` uses a threadgroup per (pos, head) and only 32 × 4 B = 128 B of shared memory (the SIMD reduction scratch). For a multi-query version with B query tokens, we have two shapes:
- **Extend outer dim:** grid = (seq_len × num_heads × B), same kernel body. No new shared memory. **Trivially safe.**
- **Batch Q in shared:** put the B × head_dim = B × 1 KB Q slice in tgmem. Cap: B ≤ 24 (24 KB, leaves 8 KB for scratch). Gain: one load of K/V per threadgroup × B queries.

MVP picks extend-outer, later iteration picks batch-Q-in-shared.

## 2. GatedDeltaNet prefill — the hard part

### 2.1 Architecture recap

flash-moe's Qwen3.6 (and 3.5) port uses GatedDeltaNet for 3 out of 4 layers (`FULL_ATTN_INTERVAL=4`). The layer's GPU pipeline inside `fused_layer_forward` (`infer.m` lines 4062-4190) currently:

1. `conv1d_step` — 1D causal conv, kernel size 4, maintains a 4-tap ring buffer (`buf_conv_state`).
2. `rms_norm_qk` — per-head RMS norm on q,k.
3. `compute_decay_beta` — compute gate `g`, `β` from `α` and `A_log`.
4. **`delta_net_step`** — the recurrence `S_t = g_t * S_{t-1} + β_t * (v_t - S_{t-1} k_t) k_t^T`, on a 64 × 128 × 128 persistent state per layer (`buf_delta_state`).
5. `gated_rms_norm` — output norm with z gate.

Steps 1 and 4 are **sequential by construction** — ring buffer history + recurrent state update each advance one position at a time.

### 2.2 What the literature says about batching GatedDeltaNet

- The ICLR 2025 GatedDeltaNet paper [pdf](https://jankautz.com/publications/GatedDeltaNet_ICLR25.pdf) and companion Songlin Yang blog post [DeltaNet Explained Part II](https://sustcsonglin.github.io/blog/2024/deltanet-2/) present a **chunk-parallel** form: split the sequence into chunks of size **C=64** (default), then within each chunk use a WY-representation (Bischoff & Van Loan, 1985) to re-parameterize the Householder products. This turns the sequential recurrence into a batched GEMM of shape `(C × d_k) × (d_k × d_v)` plus diagonal decay multiplies.
- Complexity `O(n · C · d²)` — linear in seq length, fully SIMD-parallel within the chunk. [NeurIPS 2024 — Parallelizing Linear Transformers with the Delta Rule](https://openreview.net/pdf?id=y8Rm4VNRPH)
- Wall-clock: 45 Kt/s on H100 for 1.3B at C=64. [Gated DeltaNet — emergentmind review](https://www.emergentmind.com/topics/gated-deltanet)
- vLLM's Qwen3-Next support implements this as the "linear attention batch-dim partition" path mentioned in §1.2.

Implementation complexity: high. The recurrence is a **rank-1 outer-product update** per token; the chunk-parallel reformulation re-derives the state after C steps as a closed-form product involving `cumprod(g)` and a structured matrix. Writing this correctly in Metal needs new kernels (a chunked `compute_cumprod_gate`, a `chunked_delta_net` doing a batched GEMM + triangular mask, and fused decay multiplies).

### 2.3 Pragmatic MVP recommendation

Per task constraints ("If stuck on GatedDeltaNet batching: implement batched only for full-attn layers, leave linear_attn as per-token"):

- **Phase-1 MVP:** batch only the 15 full-attention layers across N prompt tokens. Run the 45 linear layers per-token as today. Expected per-token prefill cost dropdown: full-attn dominates ~30 % of layer time, MoE I/O ~50 %, linear-attn ~20 %. If full-attn goes to zero-marginal-cost within a batch of 32, per-token cost drops from 109 ms to ~85 ms — only **1.3× speedup**.
- The **real TTFT win** comes from hoisting expert dequant + I/O across the batch (§3 below), not from full-attn batching itself.

## 3. MoE batching — where the speedup actually comes from

### 3.1 The key insight from the MoE literature

- "After the gate function in the MoE layer calculates the expert selection for all tokens, the LLM serving system groups the tokens by their assigned experts, ensuring that each expert performs a batch-processing operation for all associated tokens, **so each expert weight only needs to be fetched once**." [HF MoE blog](https://huggingface.co/blog/moe) [Mixture of Experts explained]
- During prefill, routes aggregated across tokens become effectively dense. Mixtral can activate **all** experts in a layer when the batch is large enough. [HF MoE blog](https://huggingface.co/blog/moe)
- Union-of-experts size vs batch: for K=4 active, 512 total experts, with B tokens, expected distinct experts per layer is `512 × (1 - (1 - K/512)^B)` ≈ `min(K·B, 512·(1 - exp(-K·B/512)))`. At B=32 that's ~ 123 / 512 (24 %); at B=64 ~ 221 (43 %). So experts get reused ~2× at B=64.

**What this means for flash-moe:** in the current serial prefill each prompt token does K=4 × 40 layers = 160 expert preads (~528 MB). With B=32 batched tokens, distinct experts per layer ≈ 123, so we'd do 123 × 40 = 4920 expert preads total for 32 tokens = **154 preads/token vs 160 today** — barely any savings from I/O dedup at K=4. The savings come from **eliminating the per-token GPU kernel launch overhead** and from **batching the dequant compute** across the tokens each expert serves.

### 3.2 Per-layer breakdown from `docs-research/optimization-10x-ideas.md` (line 4)

Per-layer time on M1 Pro (warm, K=4, seq=1):
- `cmd1_wait 1.09 ms` — attention QKV projection wait
- `cmd2_wait 0.47 ms` — o_proj + norm + routing wait
- `expert_io 0.71 ms` — SSD pread, parallel K=4
- `cmd3 encode 0.07 ms` + `submit 0.04 ms`
- `cpu_attn 0.018 ms` — negligible

Total ~2.38 ms/layer × 40 layers ≈ 95 ms/token. Observed 109 ms/token prefill matches within measurement noise.

### 3.3 Where batching wins per-layer

For batch B, best-case per-layer cost:
- CMD1 attn projections: B queries against the same weights — **GEMV → GEMM**, ~1.5× speedup at B=8 (memory-bandwidth shared), ~5× at B=32 (compute-bound regime).
- CMD2 o_proj + norm: same, **1.5-3× amortization**.
- Expert I/O: 123 experts at B=32 instead of 128 (4×32) — **1.04× at I/O boundary**, near zero if page cache is warm.
- Expert dequant + matmul: B tokens share each expert's dequantized weights — **≈B× amortization** because current kernel is memory-bound.
- **Net per-layer: ~B/2 to ~B speedup** once B > 4 and experts overlap.

At B=32 on M1 Pro, 5-10× total prefill speedup is **plausible** if all four pipeline stages batch cleanly. Main risks:
- Metal memory pressure: B=32 × hidden 4096 × 40 layers × 4 B = 20 MB of extra live activation per layer-boundary. Safe.
- KV cache append: must write B positions per layer — cheap (memcpy).
- GatedDeltaNet: if kept per-token in MVP, it becomes **the** bottleneck. Its `delta_net_step` kernel is 0.5-0.8 ms on M1 Pro × 45 layers × B tokens = 22-36 ms/token of serial work that batch can't amortize. Ceiling: 2-3× speedup.

## 4. Recommended batch size for M1 Pro (8-core GPU)

Inputs:
- Threadgroup memory budget: 32 KB hard limit. [Metal feature set tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- Qwen3.6 full-attn: 16 Q heads × 2 KV heads × head_dim 256. GQA ratio 8.
- Existing `attn_scores_batched` uses 128 B of tgmem (SIMD-reduction scratch). Plenty of headroom.
- Current `buf_attn_scores` pre-sized for `GPU_KV_SEQ=8192` per head × num_heads. Needs ×B extension for batched-Q output.

Hard constraints:
- Peak working-set activations for B: `B × HIDDEN_DIM × 4 B + B × num_heads × GPU_KV_SEQ × 4 B`. At B=64 that's 1 MB + 2 MB per layer — fine.
- Per-layer K,V projection output scales with B. B=32: 32 × 256 heads × 4 B = 32 KB — fine.
- B=128 is the practical upper bound before activation memory collides with non-expert mmap residency on 16 GB GPU address space.

**Recommended:** default `--prefill-batch 32`, with empirical sweep 1 / 8 / 32 / 64 / 128. Sweet spot on M1 Pro expected at 32 based on MLX/SwiftLM precedents (§1.3) and the memory budget above.

## 5. Which parts of flash-moe's pipeline are batch-friendly

| Stage | Per-token cost | Batch-friendly? | Notes |
|---|---|---|---|
| Embedding lookup | negligible | yes (vectorize) | Already done by `pt->count > 1` path (infer.m:6881) |
| Input RMS norm | 20 μs | yes | Per-row norm, trivially parallel |
| **Full-attn QKV projection** | 0.3-0.6 ms | **yes** | GEMV → GEMM, ideal batch target |
| **Full-attn scores+softmax+V** | 0.15-0.3 ms | **yes** | Need new kernel or B× outer loop |
| KV cache append | < 0.01 ms | yes | Memcpy B rows |
| **GatedDeltaNet conv1d+recurrence** | 0.6-0.9 ms | **NO (sequential)** | Needs chunk-parallel rewrite (§2.2-§2.3) |
| O-proj + routing + shared expert | 0.4-0.5 ms | yes | GEMV → GEMM |
| Top-K routing | < 0.01 ms | yes | Per-row argmax |
| **Expert pread I/O** | 0.5-0.7 ms | **partial** | B=32 gives ~4 % I/O dedup at K=4 |
| **Expert forward (gate/up/SwiGLU/down)** | 0.3-0.4 ms | **yes, big** | Each expert serves many batch-tokens; GEMV → GEMM |
| Combine + residual + norm | 0.1 ms | yes | Per-row |
| Final norm + lm_head | once per prefill | — | Last token only; same as today |

Summary: **~80 % of per-token cost is batch-friendly. GatedDeltaNet is the sole sequential tax** — and it's 20-30 % of layer time.

## 6. Causal mask and RoPE correctness across tiles

- Full-attn with batch B queries attending to `KV_len = past + B` keys: mask is a `B × KV_len` lower-triangular shifted by `past`. Row `i` (0..B-1) in the batch attends to all positions `p < past + i`. [llama.cpp KQ_mask construction](https://deepwiki.com/ggml-org/llama.cpp/3.5-batch-processing-pipeline)
- RoPE for prefill batch: apply `rotate_embeddings(q_b, pos=past+b)` per batch slot `b`. Same as today per-token, just batched. Partial rotary (`PARTIAL_ROTARY=0.25` → 64 dims) applies to the same subset.
- Softmax numerical stability: online softmax across tiles with running max is mandatory for FA-style kernels, not needed for the outer-loop approach (each Q row computes its own max over the single K scan).

## 7. Final recommendation for Phase 2 implementation

1. **Add `--prefill-batch N` CLI flag (default 1).** N=1 reproduces today's bit-exact serial behavior. Implemented as a new code path in `main()`'s prefill loop.
2. **MVP: batch MoE + full-attn only.** Keep GatedDeltaNet sequential. Expected speedup: 2-3×.
3. **Stretch: chunk-parallel GatedDeltaNet** with C=64. Adds ~500 LOC of Metal. Gets to 5-10×.
4. **Expert-aware MoE grouping** (batch B tokens by expert): saves one dequant pass per reused expert. Largest single optimization once batching is in place.

Citations for every claim anchored inline above.

## Sources

- [llama.cpp — discussions #4130 — Parallelization/Batching Explanation](https://github.com/ggml-org/llama.cpp/discussions/4130)
- [llama.cpp — discussions #6328 — batch-size vs ubatch-size](https://github.com/ggml-org/llama.cpp/discussions/6328)
- [llama.cpp — deepwiki batch processing pipeline](https://deepwiki.com/ggml-org/llama.cpp/3.5-batch-processing-pipeline)
- [Arm — Profile llama.cpp with Streamline](https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama_cpp_streamline/2_llama.cpp_intro/)
- [vLLM docs — chunked prefill configuration](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [vLLM RFC #3130 — Upstream Chunked Prefill](https://github.com/vllm-project/vllm/issues/3130)
- [vLLM blog — Qwen3-Next](https://blog.vllm.ai/2025/09/11/qwen3-next.html)
- [vLLM RFC #37995 — Prefill Context Parallel for Qwen3.5 Hybrid Attention](https://github.com/vllm-project/vllm/issues/37995)
- [Don Moon — LLM Inference Optimizations: Chunked Prefills and Decode Maximal Batching](https://donmoon.medium.com/llm-inference-optimizations-2-chunked-prefill-764407b3a67a)
- [SwiftLM README — --prefill-size design](https://github.com/SharpAI/SwiftLM)
- [thornad — lmstudio-mlx-patch (Apple Silicon 2x prompt processing)](https://github.com/thornad/lmstudio-mlx-patch)
- [MLX Apple Silicon comparative study](https://yage.ai/share/mlx-apple-silicon-en-20260331.html)
- [FlashAttention-2 paper arxiv 2307.08691](https://arxiv.org/pdf/2307.08691)
- [Princeton NLP — FlashAttention-2 blog](https://princeton-nlp.github.io/flash-atttention-2/)
- [tiny-llm — Flash Attention on Metal walkthrough](https://skyzh.github.io/tiny-llm/week2-04-flash-attention.html)
- [harvestingmoon — flash_attn_metal_cpp](https://github.com/harvestingmoon/flash_attn_metal_cpp)
- [Apple Developer Forum — M1 max threadgroup size](https://developer.apple.com/forums/thread/674385)
- [Apple — Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- [mlc-llm issue #3293 — threadgroup memory limit exceeded](https://github.com/mlc-ai/mlc-llm/issues/3293)
- [Apple docs — setThreadgroupMemoryLength](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443142-setthreadgroupmemorylength)
- [Gated DeltaNet — ICLR 2025 paper](https://jankautz.com/publications/GatedDeltaNet_ICLR25.pdf)
- [Songlin Yang — DeltaNet Explained Part II](https://sustcsonglin.github.io/blog/2024/deltanet-2/)
- [NeurIPS 2024 — Parallelizing Linear Transformers with the Delta Rule](https://openreview.net/pdf?id=y8Rm4VNRPH)
- [Gated DeltaNet — emergentmind review](https://www.emergentmind.com/topics/gated-deltanet)
- [HuggingFace — Mixture of Experts Explained](https://huggingface.co/blog/moe)
- [arxiv 2509.07379 — DuoServe-MoE](https://arxiv.org/html/2509.07379v2)
- [arxiv 2503.04398 — Speculative MoE](https://arxiv.org/html/2503.04398v3)
- [arxiv 2410.17954 — ExpertFlow](https://arxiv.org/html/2410.17954v1)
- [arxiv 2503.09716 — MoE-Gen module batching](https://arxiv.org/html/2503.09716v1)
- [flash-moe — docs-research/benchmark-ram-tokspeed.md (local)](benchmark-ram-tokspeed.md)
- [flash-moe — docs-research/optimization-10x-ideas.md, section B (local)](optimization-10x-ideas.md)
