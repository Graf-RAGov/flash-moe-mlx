# Research: fused GPU-resident token forward pass (Task #20)

## Baseline measurement (warm, 1-tok prompt, 10 gen, K=4, M1 Pro 32GB)

Per-layer phase breakdown, averaged over 360 layers from a single `--timing` run:

| phase | ms |
|---|---|
| deferred_wait | 0.000 |
| deferred_cpu  | 0.000 |
| input_norm    | 0.001 |
| cmd1_submit   | 0.020 |
| cmd1_wait     | 0.693 |
| cpu_attn      | 0.009 |
| cmd2_encode   | 0.019 |
| cmd2_wait     | 0.487 |
| routing_cpu   | 0.002 |
| expert_io     | 0.000 |
| cmd3_encode   | 0.240 (deferred — runs async with next layer) |
| **total/layer** | **1.472** |
| cmd_buffers/layer | 3 (CMD1 + CMD2 + CMD3) |
| sync_waits/layer  | 2 (CMD1, CMD2; CMD3 deferred) |

End-to-end: **15.41 tok/s** warm (baseline `c17fb48` at ctx 500 was 8.08 tok/s cold; warm-burst up to 15 tok/s for 1-tok prompt).

`cmd1_wait (0.693) + cmd2_wait (0.487) = 1.18 ms/layer = 47 ms/tok` of GPU-bound waits.

## The hypothesis in the task description

> "Hidden state migrates GPU↔CPU round-trip PER LAYER for routing decision."

On Qwen3.6-35B-A3B the main-branch state is NOT that bad:

- `routing_cpu` is only **2 µs** (120 ns/expert × 256 experts softmax + 4 × 256 topk scan).
- The CPU "intervention" between CMD2 and CMD3 is:
  `gpu_flush_batch_results` (small readbacks of gate_scores 256 floats, shared_gate 512, shared_up 512, shared_gate_score 1 — all managed buffers with direct pointer) + `cpu_softmax(gate_scores, 256)` + `cpu_topk(gate_scores, 256, 4, ...)` + `cpu_normalize_weights`.
- Fusing softmax+topK into Metal saves the 2 µs/layer = **80 µs/tok total**. Negligible.

## The real bottleneck

`cmd1_wait + cmd2_wait` = **1.18 ms/layer**. Breaking that down:
- GPU encoder launch + dispatch overhead: each command buffer commit has ~50-150 µs of driver/OS cost independent of kernel work.
- For linear-attn layers (`gpu_linear_attn=1`, which is on for 30/40 layers), ALL of CMD1's work is already on GPU (5 encoders: conv1d, rms_norm_qk, compute_decay_beta, delta_net_step, gated_rms_norm) feeding `batch_out[6]`.
- CPU work between CMD1_wait and CMD2_commit on linear-attn path: trivial (sets some pointers, copies residual from CPU to buf_residual).

Therefore: **fuse CMD1+CMD2 into one command buffer** on linear-attn layers. This eliminates one commit + one waitUntilCompleted per layer × 30 linear-attn layers = 30 fewer syscalls per token.

On full-attn layers (10/40): CPU does q_norm, k_norm, RoPE, and KV cache updates between CMD1 and CMD2, so a simple merge is trickier. We leave those alone in Phase 1.

## What can realistically be fused

1. **CMD1 + CMD2 for linear-attn layers** — directly mergeable. Just encode CMD2's dispatches (o_proj, residual_add, rms_norm, routing, shared gate/up) into the same buffer as CMD1's (proj + linear attn). **Estimated saving**: 1 wait = ~0.15 ms/layer × 30 layers = ~4.5 ms/tok → ~7-10% tok/s.

2. **Eliminate residual memcpy** — prev layer's `buf_moe_hidden` already holds the hidden state. CMD2's residual_add currently memcpys `residual` (= CPU-side hidden) to `buf_residual`. Replace `buf_residual` binding with `buf_moe_hidden` (with correct ownership). **Estimated saving**: one 16KB memcpy/layer × 40 = ~0.5 ms/tok (negligible but clean).

3. **Fused softmax+topK kernel** — negligible (2 µs/layer). Skip.

4. **CMD3 already merged with CMD1(N+1)** — the `gpu_combine` path in CMD3 already produces `buf_input` for next layer, and the FAST PATH submits CMD1 without a CPU readback. This is done.

## PR #11 applicability

The diff is written for the 397B model (hardcoded `NUM_EXPERTS=512`, `MOE_INTERMEDIATE=1024`). Interesting parts:

- **partial_softmax GPU kernel** (claimed 128× routing reduction): not a direct win here, routing_cpu is already 2 µs. Skip.
- **matvec_v3_small** (4KB shared-mem kernel for down_proj when in_dim ≤ 1024): could help the 397B path but not 35B — MOE_INTERMEDIATE=512 uses matvec_v3 with 2KB shared which already fits. Skip.
- **cache-aware routing** (`--cache-aware` flag, LRU substitution of experts): interesting but a quality/semantics change, not pure pipeline reorg. Explicitly out of scope (task says "Do NOT break steady-state semantics — this is purely a pipeline reorg").

Verdict: PR #11 does not cleanly apply to the 35B model on current main. We skip it and focus on the CMD1+CMD2 fusion.

## Metal API constraints

- A command buffer is a sequence of encoder passes. Multiple compute encoders in the same buffer serialize in order; no encoder break is required as long as kernels use the same queue.
- Writing a buffer (e.g., `buf_conv_output`) from one kernel and reading it in the next is safe across encoders in the same command buffer.
- The only reason to split into separate commits is: need a CPU decision between them. In the linear-attn path, we don't need any CPU decision between CMD1 and CMD2.
- One-kernel-per-forward (persistent kernel style from FlashAttention-3): Metal doesn't expose CUDA graphs directly. `MTLIndirectCommandBuffer` exists but is overkill and adds its own overhead. Keep it simple: one dispatch per logical op, encoded into fewer command buffers.

## Plan (Phase 2)

1. Merge CMD1 + CMD2 into one cmd buffer for linear-attn layers where `gpu_linear_attn=1`. Full-attn layers keep current 3-buffer flow.
2. Do NOT rewrite routing kernel (not the bottleneck).
3. Bit-identical verification: same first-10-token sequence vs main 129a25e.
4. Bench at ctx 500 / 2000 (matching K=4 warm cells in results.tsv).

## Measurement boundary

A pre-existing timing field split `cmd1_wait`, `cmd2_wait`. After merge these become one `cmd12_wait` for linear-attn layers. Adjust `print_timing` to report combined phase when fused.
