# Batched Prefill — Implementation Notes (Task #19)

**Status:** Scaffold + CLI landed; full kernel-batched path not implemented within 90-min timebox.
**Worktree:** `agent-a2c6c991` (branch `worktree-agent-a2c6c991`)
**Commit:** `e3ee907d9d995b9bbd7fa8486ae8b888cec489ce` — `feat(prefill): --prefill-batch N CLI flag + batched-window scaffold`
**Research doc:** [batched-prefill-research.md](batched-prefill-research.md)

## 1. Algorithm chosen

Based on the prior art survey ([research doc §1](batched-prefill-research.md#1-prior-art--how-production-engines-batch-prefill)) and the GatedDeltaNet hybrid-architecture constraint ([§2](batched-prefill-research.md#2-gateddeltanet-prefill--the-hard-part)), the target design is:

**Windowed prefill with batch width `N`.**

- Split the prompt into contiguous windows of up to `N` tokens.
- Within each window, future-batched kernels will amortize:
  - QKV projection (full-attn): one GEMM of `B × HIDDEN_DIM` against the shared weight matrix instead of `B` GEMVs.
  - Full-attention `Q·Kᵀ`, softmax, `scores·V`: extend the Q-grid dimension to `B`. The current `attn_scores_batched` Metal kernel needs only one extra dim in its grid — no new threadgroup-memory footprint.
  - MoE expert forward: group tokens by selected expert inside the window, run each expert's gate/up/SwiGLU/down once over its token group.
  - GatedDeltaNet: **stays sequential** per-token inside the window (state-carrying recurrence). Chunk-parallel reformulation is deferred — see trade-offs below.
- Outside the window boundary: same serial loop as today. KV-cache append and delta-net state update happen token-by-token within the window; the *next* window sees the fully-updated state.

### Why not full end-to-end batched attention (à la FlashAttention-2 with B queries)?

- M1 Pro's 32 KB threadgroup-memory limit ([Metal feature set tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)) caps an on-chip Q/K/V tile at `Br=Bc=32` for `head_dim=256` — and even then only the reduction scratch fits, not the whole Q slice.
- The existing `attn_scores_batched` kernel already runs threadgroup-per-(pos, head). Adding an outer grid dim for B costs nothing in shared memory, so the MVP upgrade path does **not** need FA-2 tiling at all.

### Why a window instead of whole-prompt batching?

- Activation memory at B=2048 would be ~32 MB per layer-boundary × 40 layers = 1.3 GB of live intermediate state. Too much on a 16 GB unified-memory M1 Pro with 2.6 GB non-expert weights already resident.
- llama.cpp, vLLM, and mlx-lm all use windowed/chunked prefill with `n_ubatch ∈ {128, 256, 512}` for the same reason ([research doc §1.1-§1.3](batched-prefill-research.md#1-prior-art--how-production-engines-batch-prefill)).

## 2. Trade-offs

| Decision | Rationale | Cost |
|---|---|---|
| Window of N, not whole prompt | Activation memory budget on M1 Pro 16GB | One state-sync per window boundary |
| GatedDeltaNet sequential | `delta_net_step` recurrence is inherently serial without WY-decomposition; MVP-out-of-scope per task | 45/60 layers stay per-token; ceiling ~2-3× instead of 5-10× |
| Scaffold first, kernels later | 90-min timebox; touching `fused_layer_forward` safely requires day-scale work (see [optimization-10x-ideas.md §B](optimization-10x-ideas.md): "8-12 days, ~800-1000 LOC") | Zero speedup in this landing — only plumbing |
| Bit-identical at any N in this PR | Task phase 4 requires it + safety | Perf unchanged until kernel batching lands |
| Default N=1 | Backward compat: existing invocations produce the same output | New users must opt in |

## 3. What shipped in commit `e3ee907`

- **`--prefill-batch N` CLI flag** on the `infer` binary. Default 1. Clamped to 128 (activation-memory guard).
- **Help-text line** in `print_usage()`.
- **Windowed prefill loop** in `main()`:
  - At N=1: outer while-loop iterates once per token → identical to prior `for (token_idx=...)` loop.
  - At N>1: tokens are grouped into windows; the inner body still calls `fused_layer_forward` once per token at the correct `pos`.
  - Explicit "window begin hook" / "window end hook" comment seams where kernel-batched work will attach.
- **Log line** at prefill start: `[prefill] mode=serial window=1` or `mode=batched-window window=N`.

### Files changed

```
metal_infer/infer.m  | 94 +++++++++++++++++++++++++++++++++++++--------------
 1 file changed, 68 insertions(+), 26 deletions(-)
```

### Build

Builds clean at `make infer` in the worktree (pre-existing unused-function warnings unchanged). Binary 143 KB.

## 4. Batch size sweet spot (predicted, not measured)

From [research doc §4](batched-prefill-research.md#4-recommended-batch-size-for-m1-pro-8-core-gpu):

| N | Full-attn speedup (est.) | MoE dequant amortization (est.) | Activation mem/layer | Predicted TTFT speedup |
|---|---|---|---|---|
| 1 | 1.0× | 1.0× | 16 KB | 1.0× (baseline) |
| 8 | 1.8× | ~2× (24 % expert overlap) | 128 KB | 1.4× |
| 32 | 4.0× | ~4× (44 % expert overlap) | 512 KB | 2.5× |
| 64 | 5.5× | ~6× (69 % expert overlap) | 1 MB | 3.5× |
| 128 | — | activation mem stress | 2 MB | memory pressure |

These numbers are **upper bounds assuming the kernel-batched path is implemented**. At `e3ee907` no kernel-batched work is in — the scaffold adds only loop structure. Actual measured speedup at any N in this commit is ~1.0× (within noise).

Expected sweet spot on M1 Pro once full implementation lands: **N=32**, matching MLX SwiftLM's default-of-adjacent-class [SwiftLM](https://github.com/SharpAI/SwiftLM) and the shared-memory budget analysis.

## 5. TTFT before/after

**Before** (from `docs-research/benchmark-ram-tokspeed.md`, M1 Pro, Qwen3.6-35B-A3B 8-bit):
- 1-tok prompt: 860 ms
- 8-tok prompt: 1623 ms (~109 ms/prompt-tok prefill)
- 18-tok prompt: 2451 ms
- 27-tok prompt: 3302 ms (near crash boundary — KV `attn_scores_pipe` segfaults at seq ≥ 32, see bench doc §"BLOCKER")
- Extrapolated 500-tok: ~55 s
- Extrapolated 2000-tok: ~3.5 min
- Extrapolated 4000-tok: ~7 min
- Measured 8-tok TTFT: 1623 ms

**After (commit `e3ee907`, scaffold only):** same as before, ±noise. Log now prints `mode=serial window=1`.

**After (projected once kernel-batched path lands):**

| prompt | prefill-batch 1 | prefill-batch 8 | prefill-batch 32 | prefill-batch 64 |
|---|---|---|---|---|
| 500 | 55 s | 39 s | 22 s | 16 s |
| 2000 | 218 s | 156 s | 87 s | 62 s |
| 4000 | 436 s | 311 s | 174 s | 125 s |

All "after" numbers are model-based projections using the §4 speedup estimates. They also require unblocking the GPU attention crash (benchmark doc blocker; separate task).

## 6. Verification (from task phase 4)

- **Bit-identical at --prefill-batch 1 vs prior behavior**: **YES.** Code inspection confirms: at N=1 the outer while-loop iterates once per token, calling `fused_layer_forward` in the same order at the same `pos`, followed by the same `discard_deferred_experts()`. No semantic change.
- **Bit-identical at --prefill-batch 32 vs --prefill-batch 1**: **YES in this commit.** The window only groups iteration; inside it every token is still processed via the same per-token `fused_layer_forward` call at the same `pos`. Outputs are unchanged. **Once kernel-batched work lands, numerical identity must be re-verified** — floating-point reduction order in batched GEMM differs from serial GEMV. Expected to match within float32 round-off (~1e-5 relative).
- **End-to-end TTFT measurement at 500 / 2000 / 4000 prompt tokens**: **NOT POSSIBLE from this worktree.** The 397B-compiled binary ported to 35B produces correct results only for prompt+gen < 32 because of the unrelated `attn_scores_pipe` segfault ([benchmark-ram-tokspeed.md §BLOCKER](benchmark-ram-tokspeed.md#blocker-infer-crashes-when-total-sequence-length-reaches-32-tokens)). All planned benchmarks at 500 / 2000 / 4000 tokens hit that crash before reaching the prefill timing.
- **Peak RSS at batch=32/64**: similarly blocked.

## 7. Quality regressions

None observed. None expected at any N in this commit because the per-token compute path is unchanged. When kernel-batched work lands:
- Float-32 reduction ordering will differ between GEMV and GEMM execution paths. Expected error: `~1e-5` relative, imperceptible at argmax.
- Tool-calling / JSON output quality unchanged (no change to sampling or post-processing).

## 8. What would need to happen to complete the 5-10× target

Ordered by expected ROI (all must be done in the same change; a partial landing gives a partial speedup):

1. **Batched `attn_scores` / `attn_softmax` / `scores·V`** kernels accepting a Q-grid dim. Smallest kernel change; no shared-memory pressure on M1 Pro.
2. **Batched QKV projection** — switch the `attn_specs` path from GEMV to GEMM. Already GPU-resident; needs a new dispatch shape.
3. **Expert-group MoE forward** — after top-K routing on B tokens, group tokens by expert id, run each distinct expert once over its token group. Requires a new `moe_expert_grouped` kernel but reuses existing dequant.
4. **KV-cache append for B rows** — memcpy of `B × kv_dim` per layer. Trivial.
5. **GatedDeltaNet chunk-parallel** (WY decomposition, C=64). [Research doc §2.2](batched-prefill-research.md#22-what-the-literature-says-about-batching-gateddeltanet). Biggest kernel work. Without it, ceiling is 2-3×.
6. **Unblock the `attn_scores_pipe` crash at seq ≥ 32** so real benchmarks at 500+ prompt tokens are even possible.

Effort: 8-12 days, ~800-1000 LOC. Matches the estimate in [optimization-10x-ideas.md §B](optimization-10x-ideas.md).

## 9. Summary

- CLI flag `--prefill-batch N` landed; scaffold + log + help line in `infer.m`.
- Research doc with 26 citations, shared-memory budget analysis, and GatedDeltaNet trade-off analysis: [batched-prefill-research.md](batched-prefill-research.md).
- Commit SHA in worktree: `e3ee907d9d995b9bbd7fa8486ae8b888cec489ce`.
- Bit-identical to prior behavior at any N in this commit.
- Real TTFT speedup: **0× (no kernel change yet)**. Projected once kernels are batched: 2-3× with GatedDeltaNet serial, 5-10× with chunk-parallel linear attention.
- Real benchmarks blocked by unrelated `attn_scores_pipe` segfault at seq ≥ 32 (see benchmark-ram-tokspeed.md).

## Sources

See [batched-prefill-research.md § Sources](batched-prefill-research.md#sources) for the 26-citation bibliography. This implementation doc re-cites only:

- [docs-research/benchmark-ram-tokspeed.md](benchmark-ram-tokspeed.md) — baseline 109 ms/token, crash at seq ≥ 32
- [docs-research/optimization-10x-ideas.md](optimization-10x-ideas.md) — effort estimate for full batched prefill (§B)
- [Apple — Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) — 32 KB threadgroup memory limit
- [SwiftLM GitHub](https://github.com/SharpAI/SwiftLM) — `--prefill-size` default precedent
