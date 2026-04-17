# Parallelism exploration — Qwen3.6-35B-A3B 8-bit on M1 Pro

9.7 tok/s ≈ 103 ms/tok. Per-layer avg (benchmark-ram-tokspeed.md:138, 1160 layers):
`cmd1_wait 1.085 + cmd2_wait 0.472 + expert_io 0.713 + cpu_attn 0.018 +
cmd3_encode 0.047 + submits 0.042 = 2.38 ms` × 40 ≈ 95 ms.

## 1. CPU bottleneck map (main thread, single-threaded)

| Work | File:line |
|---|---|
| `cpu_rms_norm` (input 4407, final 7880, Q/K per-head 4694–4711) | infer.m:718 |
| `cpu_softmax` over 256 experts | infer.m:746 (call 5327) |
| `cpu_topk` K=4 from 256 | infer.m:763 (call 5330) |
| `cpu_argmax` VOCAB=248320 (~6 ms/tok) | infer.m:823 (call 7978) |
| `apply_rotary_emb` | infer.m:2074 (call 4714) |
| `finalize_deferred_experts` HIDDEN memcpy | infer.m:3970 |
| Host↔GPU HIDDEN memcpys, ≥10/layer | 4413, 4744, 5060, 5065, 5267–5269, 5574 |
| CPU full-attn fallback (kv_len<32 forced — §4.2) | infer.m:4751–4775 |

I/O uses 4 pthreads (`NUM_IO_THREADS=4`, infer.m:3060; pool 3094–3177). BLAS
`cblas_sscal/sgemv/sger` (infer.m:4915–4940) only fires on `--cpu-linear`; 35B
default is fused GPU delta-net (infer.m:4238, 4267). On the GPU path, total CPU
math per layer < 0.1 ms (`cpu_attn 0.018`); **the pegged P-core is ~entirely
`[cmd waitUntilCompleted]` at 4365 and 5262 plus host-memcpys.**

## 2. GPU utilization hypotheses (~50%)

**H1 — CMD3 finishes before next CMD1+CMD2.** Serial queue (infer.m:4253,
4363). CMD3 ~0.7 ms; during `cmd2_wait` readback (5267) GPU is idle. Verify:
`sudo powermetrics --samplers gpu_power -i 100` — expect alternating ~1 ms
busy/idle.

**H2 — Low GPU occupancy on narrow 35B tensors.** Routing gate matvec:
out_dim=256 → 256/8 = **32 threadgroups** (shaders.metal:519, dispatch
infer.m:1764). 14-core M1 Pro holds ~56 concurrent TGs → ~60 % fill, µs
runtime. Per-encoder launch overhead (~25 µs) is a large fraction.
Verify: Instruments GPU Trace, `setLabel:` each encoder (4274, 5085, 5681).

**H3 — Synchronous host memcpys serialize the pipeline.** ≥10 HIDDEN-sized
copies/layer × 40 = ~400 main-thread memcpys/token; none overlap GPU.
Verify: `os_signpost` around memcpys in Instruments.

## 3. Theoretical ceiling

- **Memory bandwidth warm:** 4×40×3.34 MB = 534 MB / 200 GB/s = **2.7 ms →
  370 tok/s**. Non-expert weights cached in `wf_buf` (infer.m:889).
- **Compute:** 6 GFLOP / 5.2 TFLOPS ≈ **1.2 ms → 830 tok/s**.
- **Cold SSD:** 534 MB / 6.5 GB/s ≈ **82 ms → 12 tok/s**.
- **Encoder+commit overhead:** ~22 × 40 × 25 µs + 120 × 50 µs ≈ **28 ms/tok**
  irreducible on current shape.

9.7 tok/s = ~2.6 % of warm ceiling, ~80 % of cold-SSD ceiling. 34 GB experts on
32 GB RAM → page-cache residency is the binding constraint (paper §6 "71 %
hit rate"), not compute.

## 4. Top 3 optimizations

1. **Parallelise main-thread CPU work via Accelerate vDSP + GCD.** `cpu_argmax`
   (823), `cpu_softmax` (746), `cpu_rms_norm` (718), and HIDDEN memcpys → port
   to `vDSP_*` / `dispatch_apply`. Accelerate already linked (Makefile:19);
   `VECLIB_MAXIMUM_THREADS` is unset (Grep-verified). **Effort**: ~150 LOC.
   **Expected**: +3–8 % tok/s by letting CMD1 submit earlier, shrinking H1
   gap. **Risk**: low; numerical drift bounded. **Rollback**: one file.

2. **Fix GPU-attn seq_len ≥ 32 segfault** (infer.m:4740; diagnosed in
   benchmark-ram-tokspeed.md:22–46). `buf_attn_scores` / `NUM_FULL_ATTN_LAYERS`
   sized for 397B don't fit 35B head layout. At kv_len=8 K × 15 full-attn,
   CPU softmax+dot adds ~22 ms/tok → 15–25 % cliff. **Effort**: ~30 LOC.
   **Expected**: no short-ctx change, prevents long-ctx cliff. **Risk**:
   medium. **Rollback**: keep the `kv->len<32` CPU gate.

3. **Batched prefill** (TODO:215). Prefill is 1 tok/call today (infer.m:7809),
   TTFT ~109 ms/tok. Pack N tokens per CMD1; experts load once/layer amortize.
   **Effort**: ~600 LOC. **Expected**: 5–10× TTFT; no steady-state change.
   **Risk**: high (KV cache, RoPE, linear-attn state per-token). **Rollback**:
   feature flag.

## 5. What NOT to do

- **`F_RDADVISE` prefetch** — results.tsv:29 net 0 % (−31 % I/O / +73 %
  cmd2_wait from unified-memory contention). CLAUDE.md:67: serial pipeline
  "hardware-optimal". 35B shares the same controller; contention proportional.
- **LZ4 expert compression** — results.tsv:28 (−13 %). Same decompress cost at
  3.34 MB as at 7 MB.
- **Custom expert cache (Metal LRU / malloc)** — results.tsv:5,6,22. OS page
  cache wins every A/B.
- **File clustering / `dispatch_io`** — results.tsv:33 (0 %), 34 (−70 %).
- **Speculative early routing** — results.tsv:7 (−38 %); already disabled at
  infer.m:4548 (`spec_routing_enabled=0`).
- **Spin-poll GPU wait** — results.tsv:30 (−23 %); CPU thermal fights GPU.
- **MTP speculative decoding** — CLAUDE.md break-even; each speculated token
  still needs its own K experts → no MoE I/O reuse.

## 6. Sanity: Accelerate already multi-threaded?

No. `VECLIB_MAXIMUM_THREADS` / `OMP_NUM_THREADS` / `setenv` absent from
`metal_infer/` (Grep). Active BLAS calls (infer.m:4915, 4921, 4934) operate on
128×128 — below Accelerate's sgemv multi-thread threshold (~512×512) — so
single-threaded even when the path runs. On default GPU-delta-net path they
don't run at all. Single-core appearance = main thread in `waitUntilCompleted`
+ HIDDEN memcpys; I/O pool parks on `pthread_cond_wait` (infer.m:3113) between
0.7 ms bursts. Activity Monitor's 1 Hz sample shows one core → accurate but
misleading.
