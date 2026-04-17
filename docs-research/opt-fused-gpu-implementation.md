# Opt: Fused CMD1+CMD2 command buffer on linear-attn layers (Task #20)

Base commit: **129a25e** (worktree `agent-af56a28a`, branch `worktree-agent-af56a28a`).

## What was fused

Only one fusion was applied — the only one that is a pure pipeline reorg with no numerical change on Qwen3.6-35B-A3B:

**Merge CMD1 and CMD2 into a single MTLCommandBuffer on linear-attention layers
that are taking the FAST PATH (gpu_linear_attn=1, prev_gpu_combined=1).**

Gate condition: `g_fuse_cmd12 && gpu_linear_attn` (enabled by default; override
via `FLASHMOE_FUSE_CMD12=0`).

Saves one `commit` + one `waitUntilCompleted` pair per qualifying layer. On
Qwen3.6-35B-A3B with FULL_ATTN_INTERVAL=4, 30 of 40 layers are linear-attn;
layer 0 of each token takes the SLOW PATH (prev_gpu_combined=0), so 29 out of
40 layers per generation token qualify for fusion.

### Data-flow changes

1. CMD1's commit/wait is skipped when `cmd12_fused=1`. CMD1 stays open so the
   CMD2 encoders (o_proj + residual_add + rms_norm + routing-gate + shared
   expert gate/up) are appended into the same command buffer. Single commit
   and single wait at the end of the CMD2 block.
2. The `residual_add` encoder binds the previous layer's `buf_moe_hidden`
   directly as its residual source instead of copying the CPU-side `residual`
   back to `buf_residual`. This is safe by serial-queue ordering: CMD3(N-1)
   writes `buf_moe_hidden`, then CMD12(N)'s `residual_add` reads it, then
   CMD3(N) overwrites it. Eliminates one 8 KB GPU→CPU→GPU round-trip per layer.
3. `finalize_deferred_experts()` runs AFTER the fused commit+wait instead of
   between CMD1_wait and CMD2_commit. Semantically equivalent because the wait
   still implies CMD3(N-1) completion (serial queue).
4. Predicted-expert async pread starts AFTER the fused wait. No effect on
   default runs (prediction is off by default).

### What was NOT fused

- **Softmax + top-K routing kernel (from PR #11)**: on the 35B model with
  NUM_EXPERTS=256 and K=4, CPU `cpu_softmax`+`cpu_topk` is **2 µs/layer**,
  measured on a warm run. Fusing it into Metal saves 80 µs/token, negligible
  compared to the ~10 ms/token we save on CMD1/CMD2 overhead. Skipped.
- **Full-attention layers (10/40)**: CMD1→CMD2 requires CPU-side Q/K RMSNorm,
  RoPE, and KV-cache update. Cannot fuse without additional kernels. Skipped
  (keeps 3-buffer flow on these 10 layers).
- **PR #11 partial_softmax** and `matvec_v3_small`: diff is written against the
  397B model (NUM_EXPERTS=512, MOE_INTERMEDIATE=1024, hardcoded). Neither
  kernel applies unchanged to the 35B shape (NUM_EXPERTS=256,
  MOE_INTERMEDIATE=512). Porting them would be a separate task and the 35B
  model's routing cost isn't the bottleneck.
- **Cache-aware routing from PR #11**: changes routing semantics (quality/perf
  trade-off), explicitly out of task scope ("purely a pipeline reorg").

## Numerical identity — PASS

Prompt `"Explain quantum"`, `--tokens 20 --k 4`.

Baseline (`FLASHMOE_FUSE_CMD12=0`) and fused (`FLASHMOE_FUSE_CMD12=1`) produce
**bit-identical** generated token ID sequences:

```
303 4145 3665 271 248068 198 8160 579 264 7047
1817 25 271 16 13 220 2972 15771 2598 2570
```

`diff /tmp/f0_ids.txt /tmp/f1_ids.txt` → empty. Verified for this prompt and
for `"Hello"` with 10 tokens generated. No numerical drift since the fusion
is a pure reordering of GPU encoders (same kernels, same inputs, same output
buffers). The only semantic change is that `residual_add` reads
`buf_moe_hidden` instead of `buf_residual` — both hold the same data on the
FAST PATH (the CPU `residual` upload is just a round-tripped copy of the
prev-layer GPU hidden state).

## Per-layer phase timing

Measured on M1 Pro 32 GB, Qwen3.6-35B-A3B 8-bit, `--tokens 30 --k 4`,
`--prompt "Explain quantum"` (3 tokens), `--timing`, warm run (fused binary
warmed before the F0 measurement so page cache is stable).

| phase | baseline (F0) | fused (F1) | Δ |
|---|---|---|---|
| cmd1_submit | 0.023 | 0.023 | — |
| cmd1_wait | 1.691 | 0.349 | **−79 %** |
| cmd2_encode | 0.024 | 0.023 | — |
| cmd2_wait | 0.680 | 1.810 | +166 % (now includes CMD1's GPU kernels) |
| cmd1+cmd2 waits | **2.371** | **2.159** | **−9.0 %** |
| expert_io | 0.706 | 0.704 | — |
| cmd3_encode | 0.056 | 0.054 | — |
| **total_layer** | **3.210** | **2.982** | **−7.1 %** |
| cmd_buffers/token | 120 (3 × 40) | 81 (2 × 29 + 3 × 11 + 29 layer-0 etc.) | |
| sync_waits/token | 80 (2 × 40) | 51 | −36 % |

`cmd12_fused_count = 841` out of `1160` total layer invocations in that run
(29 fused layers × 29 tokens (27 gen + 2 prefill) = 841).

## End-to-end tok/s

| run | F0 tok/s | F1 tok/s | delta |
|---|---|---|---|
| warm burst (gen after 2 prefill toks), ctx~30 | 7.15 | 7.61 | +6.4 % |
| 10-run A/B avg (Hello, 30 tok) — outliers trimmed | 8.60 | 9.08 | +5.6 % |
| first-settled (Explain quantum, 30 tok) | 9.18 | 10.58 | +15.2 % |

Variance is high across runs because of SSD page-cache state. The **−7 %
per-layer-time** number from the timing breakdown is the most reliable signal:
~0.23 ms saved per layer × 40 layers = ~9 ms saved per token. At the 8-10 tok/s
operating point that's +7 % tok/s. The single-run +15 % was a warm-cache burst.

## Why not the 1.5-2× the task predicted

The task description hypothesised "~400 host memcpys per token" from CPU
routing being the #1 GPU-utilisation killer. On Qwen3.6-35B-A3B as currently
built (on commit 129a25e, which already has the GPU-side combine in CMD3,
FAST PATH, and the vDSP+BLAS CPU port from `c17fb48`):

- `routing_cpu` is 2 µs/layer.
- `deferred_wait` and `deferred_cpu` are each <0.003 ms.
- The GPU-side combine already keeps the hidden state resident in
  `buf_moe_hidden` / `buf_input` between CMD3(N-1) and CMD1(N).

So the "CPU routing stall" that motivated the 1.5-2× target is already gone
on main. What's left is pure GPU-dispatch overhead (commit + wait +
intermediate encoder-break cost). Fusing 29 of 40 layers' CMD1+CMD2 shaves
the command-buffer count by ~25 % and the sync-wait count by ~36 %, which
maps to the ~7 % tok/s gain — consistent with the measured 0.15-0.2 ms
per-layer commit+wait overhead predicted in the research doc.

## Rollback / disable

`FLASHMOE_FUSE_CMD12=0 ./infer ...` falls back to the original 3-buffer flow.
The fusion is per-layer — on full-attention layers and on layer 0 of each
token (SLOW PATH) it's automatically bypassed.

## Worktree & commit

Branch: `worktree-agent-af56a28a`
Commit: **e64bcd9** (Task #20: fuse CMD1+CMD2 on linear-attn layers)
Worktree: `/Users/retry/Documents/code/flash-moe/.claude/worktrees/agent-af56a28a`

Based on `129a25e` on main (pipeline guard audit, vDSP+BLAS opt already applied).
Not committed to main. Main's 35B `model_weights.bin` and `packed_experts/`
are symlinked, not copied (as instructed).
