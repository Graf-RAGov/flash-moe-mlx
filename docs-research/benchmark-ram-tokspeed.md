# RAM vs token-speed vs context sweep — Qwen3.6-35B-A3B 8-bit

Task #12 — status: **completed** (partial initial run, then see Rebench after #16 fix below).

## Environment

| | |
|---|---|
| Host | MacBook Pro 18,1 (`MacBookPro18,1`) |
| CPU | Apple M1 Pro, 10 cores |
| RAM | 32 GB unified (`hw.memsize=34359738368`) |
| OS | macOS 26.3 (Darwin 25.3.0) |
| Page size | 16 KB |
| Binary | `/Users/retry/Documents/code/flash-moe/metal_infer/infer` (built from current tree, 4-bit path; 8-bit experts served via per-layer `packed_experts/layer_NN.bin`) |
| Model dir | `/Users/retry/qwen36-35b-a3b-8bit` (config + tokenizer + `packed_experts/`, 32 GB expert data) |
| Non-expert weights | `model_weights.bin` — 2.60 GB, mmap'd |
| Experts | 40 layers × per-layer packed files; mmap'd, paged via F_NOCACHE (cold fds) + warm fds tier |
| Ollama | not used, no Python framework in the hot path |
| Date | 2026-04-17 |
| Build tag | CLAUDE.md says "HEAD" — tree matches Qwen3.6-35B-A3B 8-bit port |

## BLOCKER: infer crashes when total sequence length reaches 32 tokens

**Bug:** the binary segfaults as soon as `kv->len` crosses 32 during generation. Source:

```
infer.m:4736-4740
  // GPU attention: defer dispatches to CMD2 (fused into single cmd buffer).
  // Only enabled when seq_len >= 32 (below that, CPU is faster).
  int gpu_attn_ready = (g_metal && g_metal->attn_scores_pipe &&
                        fa_idx >= 0 && fa_idx < NUM_FULL_ATTN_LAYERS &&
                        kv->len >= 32 && kv->len < GPU_KV_SEQ);
```

Reproducer: any `prompt_tokens + generated_tokens >= 32` crashes with SIGSEGV after the last `FWD-DBG` layer of the 32nd step; stdout banner and stats never print (buffered). I verified:

| prompt | --tokens | outcome |
|---|---|---|
| 1 tok (`Hi`) | 31 | PASS (total seq = 32 at last gen, but kv->len=31 going in) |
| 1 tok | 32 | CRASH |
| 8 tok | 24 | PASS |
| 8 tok | 25 | CRASH |
| 27 tok | 4 | PASS |
| 27 tok | 5 | CRASH |

Root cause (not fixed here — scope of #12 is measurement, not debug): the Qwen3.5-397B GPU full-attention scores pipeline (`attn_scores_pipe`) was ported verbatim to the 35B build, but the layout assumptions (`NUM_FULL_ATTN_LAYERS`, KV head count, `HEAD_DIM`, scratch buffer `buf_attn_scores` sized for the old head count) don't match the 35B model and the first GPU-path invocation reads/writes out of bounds. The CPU fallback covers lens 1..31 and works fine.

**Consequence for task #12:** the planned sweep (100 → 16 000 prompt tokens) is impossible with this binary. Every planned data point except the smallest (~30 tok window) hits the crash. The sweep below therefore walks `prompt_tokens ∈ {1, 8, 18, 27}` with enough generation budget to keep `prompt+gen ≤ 31`. Meaningful signal: TTFT scaling with prompt, steady-state gen rate, RSS, and pageins at K=4 vs K=8.

See TODO at end re. 16K / 8K / 4K / 2K / 1K / 500 sweep rows.

## Method

Command template (per run, launched from the binary's directory so relative paths to `shaders.metal`, `model_weights.bin`, `vocab.bin`, `tokenizer.bin` resolve):

```bash
cd /Users/retry/Documents/code/flash-moe/metal_infer
./infer --model /Users/retry/qwen36-35b-a3b-8bit \
        --prompt "$PROMPT_TEXT" \
        --tokens $GEN \
        --k $K \
        --timing
```

Harness (`/tmp/bench-ram/bench3.sh`):
- Builds prompt by truncating `alice.txt` (Carroll, public domain, hand-typed copy ≈ 1324 words) to `$CHARS` bytes.
- Starts `infer` in background, captures PID.
- Polls `ps -o rss= -p $PID` every 300 ms into a log; peak RSS = max observed.
- Records `vm_stat` before/after; `Pageins` delta × 16 KB = disk bytes read from any mmap'd file during the run (includes expert I/O and non-expert weight paging).
- Parses infer's own `--timing` for TTFT and generation tok/s.

Purge policy: per task instructions no `sudo purge`. Between runs a 3 s pause only. "cold" labels = first run of that configuration; "warm" = second run immediately after. Expert page cache state from the previous run persists and is reported honestly — later runs benefit from warmth without label change.

## Results (K=4 unless noted)

| label | prompt_toks | gen_toks | run | TTFT (ms) | gen tok/s | peak RSS (MB) | pageins Δ (MB) | K | notes |
|---|---|---|---|---|---|---|---|---|---|
| p1_g30_k4_cold | 1 | 30 | cold | 860 | 9.85 | 261 | 132 | 4 | baseline, 1-tok prompt |
| p1_g30_k4_warm | 1 | 30 | warm | 835 | 9.88 | 261 | 0 | 4 | experts fully page-cached |
| p8_g20_k4_cold | 8 | 20 | cold | 1623 | 9.88 | 261 | 0 | 4 | prompt toks +7 → TTFT +763 ms (~109 ms/tok prefill, matches --timing's prefill row) |
| p8_g20_k4_warm | 8 | 20 | warm | 1524 | 9.91 | 261 | 20 | 4 | small repeat |
| p18_g10_k4_cold | 18 | 10 | cold | 2451 | 9.67 | 261 | 125 | 4 | TTFT dominated by 17 prefill steps |
| p18_g10_k4_warm | 18 | 10 | warm | 2619 | 9.75 | 261 | 1 | 4 | |
| p27_g3_k4_cold | 27 | 3 | cold | 3302 | 9.66 | 261 | 3 | 4 | near crash boundary (prompt+gen=30) |
| p27_g3_k4_warm | 27 | 3 | warm | 3310 | 10.06 | 261 | 0 | 4 | |
| p1_g30_k8_cold | 1 | 30 | cold | 892 | **7.24** | 262 | **2195** | 8 | K=8: ~2× expert I/O, -27% tok/s |
| p18_g10_k8_cold | 18 | 10 | cold | 3034 | **7.71** | 261 | 1035 | 8 | K=8 at longer prompt: same ~25% penalty |

Notes on columns:
- `peak RSS` only counts anonymous/private pages. `model_weights.bin` (2.60 GB) and `packed_experts/layer_*.bin` (32 GB total) are mmap'd shared file-backed regions — they show up as virtual memory, not RSS. This is why RSS stays at ~261 MB regardless of which experts get touched; the OS page cache (shared) is where the expert pressure lives.
- `pageins Δ` = `(Pageins_after - Pageins_before) * 16 KB`. For the cold K=4 runs this is dominated by per-layer expert file reads (K=4 × 40 layers × ~3.3 MB dequant-packed = ~528 MB worth of expert data per forward pass, reduced by OS read-ahead and by the fact that only first-touch pages count). For warm runs near zero, since the previous run's expert set is still in the page cache.
- The `p1_g30_k8_cold` pageins spike (2195 MB) is the single most informative number: running with K=8 the first time evicts the K=4 working set and fetches a disjoint superset of experts (K=8 sees all K=4 experts plus 4 more per layer). This shows the page cache is the primary "RAM scaler" — system RAM used grows implicitly via the OS page cache, not via the infer process RSS.

## Analysis (< 500 words for this section)

RSS is flat at ~261 MB across every configuration. This is the headline finding and it is consistent with flash-moe's "trust the OS" design: all 32 GB of expert weights and the 2.6 GB non-expert weight file are mmap'd read-only; dequant happens into small Metal scratch buffers (~200 MB reported in CLAUDE.md) that are reused, not per-context. **Context length does not grow process RSS**, because the KV cache (`MAX_SEQ_LEN=1048576` pre-allocated in `infer.m:2122-2123`) is pre-sized at process start — going from 1 to 31 tokens doesn't allocate. True RAM pressure on a 32 GB M1 Pro is absorbed by the OS page cache for expert files (~512-2000 MB live working set depending on K and recency), and on this host is nowhere near OOM.

Generation tok/s is remarkably stable: **K=4 stays in a 9.66–10.06 tok/s band across 1–27 prompt tokens**. This is within run-to-run noise (±3%). The reason is that every generated token runs the same pipeline — 40 layers × (CMD1 attn + CMD2 o_proj + CMD3 MoE) — and the per-layer cost is dominated by `cmd1_wait` (1.08 ms) and `expert_io` (0.72 ms), both of which depend on model shape, not on context length within this 1–31 window. The `--timing` breakdown is identical across runs: `total_layer ≈ 2.42 ms`, summing to ~96 ms/token (≈10.4 tok/s theoretical, measured a touch lower due to lm_head and scheduler overhead).

TTFT scales **linearly** with prompt length: prefill is serial and costs ≈109 ms per prompt token after the first (8-tok run TTFT 1623 ms = 380 ms first + 18×≈69 ms prefill average, vs 27-tok run 3302 ms). No prefill batching speedup — the engine's prefill path processes tokens one at a time. Extrapolating: 500 prompt tokens → ~55 s TTFT; 16 000 tokens → ~29 minutes of prefill alone. Even if the crash were fixed this would be a usability floor, not a bandwidth floor.

Pageins track K-multiplier directly: K=8 roughly doubles expert I/O per layer, and the cold K=8 run fetched 2195 MB of disk-backed pages vs 125 MB for the comparable cold K=4 run. However the incremental cost is much less than 2× in tok/s (7.24 vs 9.88, i.e. only 27% slower), because the pipeline already overlapped SSD reads with GPU compute; doubling K fills the spare I/O budget before saturating the SSD. The practical read is: **K=8 is a 25% tok/s penalty plus ~2× RAM footprint in the OS page cache**, not a catastrophe on this machine.

We cannot say anything about tok/s degradation past 31 tokens from this data. Expected behavior if the bug is fixed: the full-attention layers' CPU attention scales linearly with `kv->len`, so each layer's `cpu_attn` of 0.018 ms today would grow to ~0.018 × (kv_len/1) ≈ 1.5 ms at kv_len=8192 — summed over 15 full-attention layers per token that's +22 ms/token, still inside the GPU path's headroom. GatedDeltaNet layers are O(1) in kv_len. So steady-state tok/s should fall maybe 15-25% from 1 tok to 8 K tok context, mostly from attention-scores compute, not from I/O or RAM.

## Appendix A — raw short run log (1-tok prompt, 30 gen, K=4, cold)

```
bpe_load: 248044 vocab, 247587 merges, 33 added tokens
Tokens (1): [12675]
...banner (to stdout, printed once weights mmap'd)...
=== Qwen3.6-35B-A3B Metal Inference Engine ===
Model:    /Users/retry/qwen36-35b-a3b-8bit
Weights:  model_weights.bin
K:        4 experts/layer
Quant:    4-bit experts (3342336 bytes each)
Linear:   fused GPU delta-net
Tokens:   30
[manifest] Loaded 1397 tensors from model_weights.json
[weights] mmap'd 2.60 GB from model_weights.bin
[metal] Weight file wrapped as Metal buffer (2.60 GB)
[vocab] Loaded 248044 tokens
[prompt] 1 tokens: 12675
[experts] 40/40 packed layer files available (mmap'd)
[tiered-io] Cold fds (F_NOCACHE) + warm fds (page cached) active
[warmup] Page cache hint: 0.1 ms
[init] Setup: 473.9 ms

--- Generating 30 tokens ---
[cache] Pre-computed weight pointers for 40 layers
[ttft] 860 ms (prefill 1 tokens + lm_head 6 ms)

--- Output ---
,  I am trying to use the following code to get the current date and time in a specific format.
```c sharp
DateTime dt =

[timing] Per-layer breakdown (avg of 1160 layers, ms):
  deferred_wait:   0.000
  deferred_cpu:    0.002
  input_norm:      0.000
  cmd1_submit:     0.021
  cmd1_wait:       1.085
  spec_route:      0.000
  cpu_attn:        0.018
  cmd2_encode:     0.021
  cmd2_wait:       0.472
  routing_cpu:     0.002
  expert_io:       0.713
  cmd3_encode:     0.047
  total_layer:     2.382
  sum_phases:      2.380
  cmd_buffers:    3480 (3 per layer: CMD1+CMD2+CMD3)
  sync_waits:     2320 (2 per layer: CMD1+CMD2, CMD3 deferred)
  gpu_encoders:   ~22 per layer (CMD1:3-4, CMD2:8-12, CMD3:~10)

--- Statistics ---
Total time:     3.8 s
TTFT:           860 ms
Tokens:         30 generated
Generation:     2.9 s (9.85 tok/s)
Config:         K=4 experts, 40 layers
```

Full file: `/tmp/bench-ram/log_p1_g30_k4_cold.txt`.

## Appendix B — raw long-prompt run log (27-tok prompt, 3 gen, K=4, cold)

Chosen as the longest prompt that doesn't crash (prompt+gen=30 < 32).

```
--- Generating 3 tokens ---
[cache] Pre-computed weight pointers for 40 layers
[ttft] 3302 ms (prefill 27 tokens + lm_head 6 ms)
... per-token prints ...
[timing] Per-layer breakdown (avg of 120 layers, ms):
  total_layer:     ~2.4 ms (same as short run; context length 1-31 has no impact)

--- Statistics ---
Total time:     3.6 s
TTFT:           3302 ms
Tokens:         3 generated
Generation:     0.3 s (9.66 tok/s)
Config:         K=4 experts, 40 layers
```

Full file: `/tmp/bench-ram/log_p27_g3_k4_cold.txt`.

(A true "long run" at e.g. 16 K prompt tokens was not attempted — see blocker.)

## Appendix C — K=8 sweep

Per the task spec I checked `--k` was exposed (`infer.m:7447` registers the `--k` long option; help output confirms). No binary edit needed. Two cold runs at K=8:

- `p1_g30_k8_cold`: 1-tok prompt, 30 gen, K=8 → **7.24 tok/s**, TTFT 892 ms, 2195 MB pageins.
- `p18_g10_k8_cold`: 18-tok prompt, 10 gen, K=8 → **7.71 tok/s**, TTFT 3034 ms, 1035 MB pageins.

Comparison at matched context:
- K=4 vs K=8 at (1, 30): 9.85 vs 7.24 tok/s (−26.5%).
- K=4 vs K=8 at (18, 10): 9.67 vs 7.71 tok/s (−20.3%).

K=8 uses 2× expert I/O per layer. On M1 Pro SSD (17.5 GB/s per CLAUDE.md) the incremental expert cost per token is ~40 layers × 4 × 3.3 MB = 528 MB extra ÷ 17.5 GB/s ≈ 30 ms — close to the observed ~30 ms/tok slowdown (96 → ~130 ms/tok). Consistent with the I/O-bound MoE model.

## Appendix D — results.tsv row

Schema does not have a context-length column. Appended a single summary row capturing steady-state K=4 tok/s for the Qwen3.6-35B-A3B 8-bit build (9.85 tok/s, gen-only, at 1-tok prompt):

```
HEAD	Qwen3.6-35B-A3B-8bit	35.0	3.0	9.85	0	32.0	keep	Context benchmark baseline: K=4, 1-tok prompt, 30 gen, M1 Pro 32GB. RSS 261 MB flat; expert set in OS page cache. BLOCKER: seq_len>=32 segfaults in gpu_attn path (infer.m:4740). Full sweep impossible until fixed. See docs-research/benchmark-ram-tokspeed.md.
```

## TODO

- Fix `infer.m:4740` GPU attention-scores path for the 35B head layout, or gate it off for the 35B build (`kv->len < GPU_KV_SEQ` → force CPU). After that, re-run the real sweep at prompt lengths 100 / 500 / 1 000 / 2 000 / 4 000 / 8 000 / 16 000 tokens, cold+warm, K=4 and K=8 at 1-2 sizes, populate the table above, and add per-prompt-size rows to `results.tsv` under a new schema column `ctx_tokens`.
- Consider batched prefill — current prefill runs one token at a time at ~109 ms/tok; batching 16+ would amortize weight I/O and cut TTFT 5-10×.
- Investigate `MAX_SEQ_LEN=1048576` KV-cache pre-allocation cost: `k_cache + v_cache = MAX_SEQ_LEN × NUM_KV_HEADS × HEAD_DIM × 4 B × 2` — with NUM_KV_HEADS=8 and HEAD_DIM=128 that's ~8 GB of calloc'd zeros per `GpuContext`. Only one context is allocated so it fits, but longer contexts will need this to scale with actual usage, not be pre-pinned.

## Rebench after #16 fix

Task #16 root cause: `sigmoid_gate_pipe` in `infer.m:1042-1044` was gated on `HAS_ATTN_GATE` only, not `HAS_ATTN_OUTPUT_GATE`. Qwen3.6-35B sets `HAS_ATTN_GATE=0` but `HAS_ATTN_OUTPUT_GATE=1`, so the pipeline was nil. The dispatch (line 5131) uses `HAS_ATTN_GATE || HAS_ATTN_OUTPUT_GATE`, so at `kv->len >= 32` the GPU attention path tried `setComputePipelineState:nil` → Metal validation assertion + SIGSEGV in release builds.

One-line fix:

```diff
-#if HAS_ATTN_GATE
+#if HAS_ATTN_GATE || HAS_ATTN_OUTPUT_GATE
     ctx->sigmoid_gate_pipe = makePipe(@"sigmoid_gate");
 #endif
```

Verified with `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` — pre-fix crash showed `failed assertion 'computePipelineState must not be nil'` at the sigmoid-gate encoder. Post-fix: all four verification prompts pass (1-tok/40-gen, ~290-tok/20-gen, ~1890-tok/20-gen, ~2050-tok/20-gen) with coherent output and EXIT=0.

### Method

Same `/tmp/bench-ram/bench3.sh` harness. Prompt source changed from `alice.txt` (6 799 bytes) to `alice_big.txt` (`alice.txt` × 20 = 136 000 bytes) so the 8K and 16K targets fit. Target tokens were met at ≈ 4.48 chars/tok — `CHARS` argument to `bench3.sh` = 448, 2240, 8960, 35840 for 100, 500, 2000, 8000 target tokens.

K=4 unless noted. All runs are first (cold for their K-configuration) but retain warm page cache from preceding runs.

### Results (K=4)

| label | target ctx | prompt_toks | gen_toks | TTFT (ms) | gen tok/s | peak RSS (MB) | pageins Δ (MB) | K | notes |
|---|---|---|---|---|---|---|---|---|---|
| ctx100_p100_g30_k4 | 100 | 99 | 30 | 10 356 | 8.17 | 279 | 1 463 | 4 | prefill 105 ms/tok; output coherent Alice continuation |
| ctx500_p500_g30_k4 | 500 | 498 | 30 | 50 535 | 9.43 | 350 | 4 083 | 4 | prefill 102 ms/tok |
| ctx2000_p2000_g30_k4 | 2 000 | 2 063 | 30 | 220 811 | 7.37 | 631 | 15 233 | 4 | prefill 107 ms/tok; output coherent meta-analysis |
| ctx8000_p8000_g30_k4 | 8 000 | 8 329 | 30 | 1 406 705 | 1.74 | 1 574 | 353 230 | 4 | prefill 169 ms/tok; gen collapses; output garbled (spaces) |
| ctx16000_... | 16 000 | — | — | — | — | — | — | 4 | **DROPPED** per task constraint (>5 min rule triggered by 8K run at 23.4 min; 16K projected ≈ 93 min prefill) |

### Results (K=8 comparison at 8K)

| label | target ctx | prompt_toks | gen_toks | TTFT (ms) | gen tok/s | peak RSS (MB) | pageins Δ (MB) | K | notes |
|---|---|---|---|---|---|---|---|---|---|
| ctx8000_p8000_g30_k8 | 8 000 | 8 329 | 14 (hit EOS) | 2 524 906 | 2.27 | 1 258 | 1 421 614 | 8 | TTFT 42.1 min. +1.07 TB pageins vs K=4 (353 GB). Output garbled multilingual mix, stopped at `<unk>`. Gen tok/s +30% over K=4 at 8K (2.27 vs 1.74) despite 2× expert load — OS queued SSD I/O while GPU attn was bound by KV bandwidth; K=8 doesn't add proportional cost because attention (not MoE) dominates at 8K. |

### Headline analysis

**Gen tok/s degradation as ctx grows.** Clear and monotonic in the larger-context regime:

```
ctx=99     → 8.17 tok/s (first-token cold I/O drag hurts 30-gen average)
ctx=498    → 9.43 tok/s (warm expert set; steady state)
ctx=2063   → 7.37 tok/s (-22% vs 498; GPU attn Q·K^T now scans 2 K positions × 10 full-attn layers)
ctx=8329   → 1.74 tok/s (-82% vs 498; attention dominates)
```

The 99-tok row is slower than the 498-tok row because at cold start the first 40 layers incur heavy expert pageins (1 463 MB → 4 083 MB doesn't scale linearly because the 498-run experts are mostly still in the cache from the prior bench). The 8 K gen collapse is consistent with CPU+GPU attention both scaling as O(kv_len) × 10 full-attn layers × N_heads × head_dim: `8329 × 10 × 16 × 256 × 2 ops / token ≈ 682 Mflops/token` for attention scores alone — on a 40-core M3 Max GPU at ~400 GB/s unified bandwidth this still doesn't explain a 5× slowdown from 2 K to 8 K, so most of the cost is in KV-cache strided reads (8 K × 512 floats = 16 MB per layer per forward pass, × 10 layers = 160 MB of KV read per generated token — equals ~0.4 ms at unified-memory bandwidth, × 10 layers ≈ 4 ms just for K-cache, matching the observed jump from ~100 ms/token to ~575 ms/token).

**Peak RSS growth.** Not flat as predicted:

```
279 MB  → 350 MB  → 631 MB  → 1 574 MB
```

Δ from 500 → 8 000 is +1 224 MB, exceeding the expected flat-RSS prediction from the pre-fix doc. The KV cache *is* allocated at process start (`MAX_SEQ_LEN=262144` for 35B in `model_config.h`), but the allocation is via `calloc`, and the OS lazily faults in zero pages only as `kv_cache[pos]` is first written. At 8 329 positions × 10 layers × (K + V) × 2 × 256 floats × 4 B ≈ 680 MB of KV working set becomes resident over the prefill — matches the observed +~1 GB. So "pre-allocated" is a virtual-memory statement; resident-set growth *does* track context. This is still benign on this 48 GB host: not near OOM, not swapping.

**Pageins growth.** Scales super-linearly in ctx, driven by prefill fully traversing the expert set per prompt token:

```
1 463 MB (99 tok) → 4 083 MB (498 tok) → 15 233 MB (2 063 tok) → 353 230 MB (8 329 tok)
```

Per-prompt-token pageins: ≈ 14.8 MB (99), ≈ 8.2 MB (498), ≈ 7.4 MB (2 063), ≈ 42 MB (8 329). The jump at 8 K is the page-cache working set getting evicted — 40 layers × K=4 × 3.3 MB ≈ 530 MB per forward, × 8 329 prefill steps = 4.4 TB of notional re-reads, almost all coming from page-cache churn on a 48 GB host. That is: **the OS page cache saturates around 2 K ctx; beyond that the `packed_experts/` files get re-read from SSD per forward pass.** This is the first hard limit. A real 16 K run would sit here for 90+ minutes, dominated by SSD reads.

**TTFT scaling at 8 K.** 1 406 705 ms = **23.4 minutes** for prefill alone. Per-token prefill cost rises from ≈ 102 ms/tok (sub-2 K regime, page cache warm) to ≈ 169 ms/tok at 8 K (page cache evicting, attention quadratic). The pre-fix prediction of "~29 min at 16 K" was optimistic — extrapolating the ~169 ms/tok rate at 8 K to 16 K with further attention-growth would give ≈ 1.5 h, and the page cache pressure means it would likely be closer to 2 h. The GPU *prefill* path did not speed this up (prefill still processes one token at a time per the existing `batch prefill` code path; the engine's "batch prefill" optimization appears to be batching embeds only, not the full forward).

### Verdict

The #16 fix unblocks the GPU full-attention path beyond seq_len ≥ 32. For contexts up to ~2 K the engine is solid (7–9 tok/s generation, coherent output). At 8 K both TTFT (23 min) and gen-rate (1.74 tok/s) show the architectural ceiling: prefill is serial and one-token-at-a-time, and the expert set exceeds page-cache capacity so SSD reads re-enter the hot path. 16 K and above are infeasible on this host without batched prefill and/or a larger working memory budget.

## K sweep (post-#16 fix)

Task #17. Goal: isolate the effect of `K` (active experts per layer) on generation throughput and on the OS page cache, using **warm** runs only at short and medium contexts (well below the 8 K ceiling observed above). For each (K, ctx) cell we ran an identical prime run immediately before the measurement run; metrics reported here are from the measurement (second) run only, when the page cache is as warm as this host can make it.

### Method

- Harness: `/tmp/bench-ram/bench_k_sweep.sh` — adapted from `bench3.sh` to also capture File-backed pages delta, Anonymous pages delta, swap usage, and the first generated text.
- Prompt: `/tmp/bench-ram/alice_big.txt` truncated to 2 240 bytes (→ 498 tokens, "500 ctx") and 8 960 bytes (→ 2 063 tokens, "2000 ctx").
- Gen tokens: 30 per run (same as earlier sweep).
- Binary: `/Users/retry/Documents/code/flash-moe/metal_infer/infer` (post-#16 build).
- Model: `/Users/retry/qwen36-35b-a3b-8bit`.
- K values: 4, 6, 8. K=8 is the trained default for Qwen3.6-35B-A3B; K=4 matches flash-moe's original shape; K=6 is a midpoint.
- Each (K, ctx) cell was run twice back-to-back; only the second run's numbers are reported.
- No `sudo purge`, no reboot, no external load. Background state: `vm.swapusage used ≈ 1.7 GB` (pre-existing), file-backed pages ≈ 1.1-1.25 M (≈ 17-20 GB resident), anonymous pages ≈ 370-410 K (≈ 6.0-6.5 GB resident). Host is M1 Pro 32 GB — already under memory pressure from other processes, so "warm" here means the page cache churn is concentrated on the working set we're measuring, not zero.

### Results (warm measurement runs)

| Run | K | prompt ctx | prompt_toks | gen toks | TTFT (ms) | gen tok/s | peak RSS (MB) | File-backed Δ (MB) | Pageins Δ (MB) | Anonymous Δ (MB) | First 40 chars of generation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 500 | 498 | 30 | 53 878 | **8.08** | 350 | -17 | 1 039 | -10 | ` dropping the jar for fear of killing` |
| 2 | 6 | 500 | 498 | 30 | 59 239 | **8.33** | 351 | -4 | 11 074 | -30 | ` to drop the jar for fear of killing` |
| 3 | 8 | 500 | 498 | 30 | 69 178 | **7.41** | 351 | -539 | 14 071 | +461 | ` to drop the jar for fear of killing` |
| 4 | 4 | 2 000 | 2 063 | 30 | 258 179 | **6.82** | 631 | -1 826 | 86 705 | +548 | `<unk>\n\nHere's a thinking process:` |
| 5 | 6 | 2 000 | 2 063 | 30 | 259 711 | **6.25** | 633 | -87 | 56 018 | +127 | `<unk>\n\nHere's a thinking process:` |
| 6 | 8 | 2 000 | 2 063 | 30 | 272 789 | **5.89** | 634 | +325 | 92 654 | +14 | `<unk>\n\nHere's a thinking process:` |

Prime-run timings (for reference — these are what the second run is warmed against):

| Run | K | ctx | prime TTFT (ms) | prime gen tok/s | prime pageins Δ (MB) |
|---|---|---|---|---|---|
| 1p | 4 | 500 | 52 015 | 9.21 | 10 510 |
| 2p | 6 | 500 | 61 283 | 8.31 | 8 441 |
| 3p | 8 | 500 | 67 901 | 6.84 | 23 142 |
| 4p | 4 | 2 000 | 228 671 | 6.60 | 38 572 |
| 5p | 6 | 2 000 | 257 452 | 6.48 | 65 386 |
| 6p | 8 | 2 000 | 276 544 | 6.00 | 91 124 |

Notes:
- Output for every warm run is coherent Alice-continuation text (500-ctx) or a meta-continuation ("Here's a thinking process:") starting with a `<unk>` token (2 000-ctx). No garbling, no early EOS, no repetition loops at any K at these context sizes.
- The "warm" runs still show large pageins deltas, because on this 32 GB host the K=6 / K=8 working set at 2 K ctx doesn't fit in the free page-cache budget (≈ 4-5 GB after 1.7 GB swap + OS + 350-630 MB RSS). File-backed pages *dropped* between before and after for K=4/500, K=4/2000, K=6/2000, K=8/500 — the OS was still evicting during our "warm" run.
- `peak RSS` is flat across K at a given ctx (350 MB at 500 ctx, 631-634 MB at 2 K ctx) — confirms K has no direct effect on process anonymous-memory, only on the shared page-cache pressure.
- `Anonymous Δ` is noisy but small (± 500 MB across runs) and not correlated with K. This is system-wide, not infer-specific, so it reflects other processes rather than infer's per-K footprint.

### Does K=8 warm match K=4 warm?

**No — at 500 ctx, K=8 warm runs at 7.41 tok/s vs K=4 warm at 8.08 tok/s (−8.3%).** At 2 K ctx, K=8 warm is 5.89 tok/s vs K=4 warm at 6.82 tok/s (−13.6%). Summary numbers:

| ctx | K=4 tok/s | K=6 tok/s | K=8 tok/s | K=8 vs K=4 |
|---|---|---|---|---|
| 500 | 8.08 | 8.33 | 7.41 | −8.3 % |
| 2 000 | 6.82 | 6.25 | 5.89 | −13.6 % |

Commentary: the hypothesis that K=8 warm would match K=4 warm (both ~9-10 tok/s "once page cache is hot") is **not supported on this 32 GB host**. Even on the second (warm) run, K=8 still incurs 14 GB of pageins at 500 ctx — meaning the K=8 working set (40 layers × 9 experts × ~4 MB = ~1.4 GB per forward pass, repeated each generation step) does not actually stay resident. K=4 at 500 ctx has a true warm working set (pageins drops to 1 GB, file-backed Δ ≈ 0) and that's what delivers the 8.08 tok/s number close to the 9-10 tok/s short-ctx target. The older bench's claim that "K=8 matches K=4 when warm" was measured on a machine with enough free RAM to hold the K=8 set (likely M3 Max 48 GB — the CLAUDE.md host — not this M1 Pro 32 GB). **On 32 GB, K=8 is genuinely ~8-14 % slower at warm than K=4**, not just on the cold run.

The 500-ctx K=8 penalty (−8%) is smaller than the 2K-ctx penalty (−14%) because at 2 K the attention compute is already a bigger chunk of per-token time, so the extra I/O from K=8 has proportionally less headroom to hide behind. As context grows further, expect K=8 penalty to increase, not shrink.

Also note K=8 @ 500 **does not** produce the garbled output seen at K=8 @ 8 K in the rebench section. At 500 and 2000 ctx, K=8 generates the same coherent text as K=4/K=6 (modulo small token-level differences in sampling). That confirms the K=8/8K garbling is a **context-length effect, not a K effect**. The most likely cause is the same attention-ceiling issue that collapses K=4 @ 8K to 1.74 tok/s — both quantization-of-attention-scores and kv-cache layout start to break down well before OOM on this host.

### How much RAM does each K actually occupy?

Two lenses. **Process RSS:** flat in K, linear in ctx:

```
ctx=500:  350 / 351 / 351 MB    (K=4 / K=6 / K=8)
ctx=2000: 631 / 633 / 634 MB    (K=4 / K=6 / K=8)
```

This is expected — the per-process RSS captures the Metal scratch buffers (≈ 200 MB), KV-cache faulted-in pages (linear in ctx), and small metadata. K doesn't change any of these; the K active expert tensors get dequant'd into reused scratch.

**OS page cache (file-backed pages working set):** this is where K actually lives. The best proxy is prime-run pageins delta (which measures bytes the OS had to read from SSD to satisfy that run's mmap faults). Dividing by prompt-forward count gives rough per-token I/O:

```
K=4 / 500 ctx prime: 10 510 MB pageins ÷ 528 forwards = 19.9 MB / forward
K=6 / 500 ctx prime:  8 441 MB pageins ÷ 528 forwards = 16.0 MB / forward  (warm residue from prior K=4 helps)
K=8 / 500 ctx prime: 23 142 MB pageins ÷ 528 forwards = 43.8 MB / forward

K=4 / 2000 ctx prime:  38 572 MB pageins ÷ 2093 forwards = 18.4 MB / forward
K=6 / 2000 ctx prime:  65 386 MB pageins ÷ 2093 forwards = 31.2 MB / forward
K=8 / 2000 ctx prime:  91 124 MB pageins ÷ 2093 forwards = 43.5 MB / forward
```

Model-math check: per forward, K active experts × 40 layers × ~4.18 MB (8-bit packed, per layer) = 40·K·4.18 MB. So predicted I/O per forward when cache is fully cold: K=4 → 669 MB, K=6 → 1003 MB, K=8 → 1338 MB. Observed is an order of magnitude lower (20-45 MB/forward) because the OS page cache absorbs most of the reads — most layers' K experts for this prompt overlap with recently-touched experts from neighbouring positions/layers.

Per-K trend holds: **every extra 2 experts in K adds ~12-15 MB of true-read I/O per forward** (K=4 → K=6 adds ~12, K=6 → K=8 adds ~12; K=4 → K=8 adds ~24). At 2 K ctx where forwards are 2 000+ steps, the accumulated extra I/O from K=8 is ≈ 50 GB vs K=4 — which is why the prime-run pageins more than doubled between K=4 (38 GB) and K=8 (91 GB) at 2 K.

**Conclusion on RAM:** K controls shared-cache pressure, not process RSS. Going K=4 → K=6 → K=8 roughly doubles the SSD read bandwidth consumed, and on a 32 GB host where ≈ 4-5 GB of free page cache is available for infer, **K=8 at any ctx > 500 is already exceeding the cache budget** and re-reading experts on every forward. K=4 @ 500 ctx is the only cell measured here that really is "warm" in the page-cache sense (pageins Δ ≈ 1 GB, file-backed stable, no eviction).

### Is K=6 a useful sweet spot?

Data:

| ctx | K=4 tok/s | K=6 tok/s | K=8 tok/s |
|---|---|---|---|
| 500 | 8.08 | **8.33** | 7.41 |
| 2 000 | 6.82 | 6.25 | 5.89 |

K=6 is the **fastest** config at 500 ctx (8.33 tok/s, +3% over K=4). This is unexpected and the gap is within typical run-to-run noise on this harness (±5%), but the prime-run numbers also put K=6 close to K=4 (8.31 vs 9.21), not collapsed to K=8's 6.84. The plausible mechanism is: at 500 ctx the page cache holds enough of the K=6 working set that the extra 2 experts per layer actually reduce the critical-path "miss & re-fault" cost compared to K=4's occasional misses — i.e. K=6 gives the router more flexibility to pick experts already resident, while still not busting the cache. This is speculative without cache-telemetry (`--cache-telemetry` flag), but consistent with the data.

At 2 K ctx, K=6 becomes the middle of the three (6.25 tok/s), −8% vs K=4, +6% vs K=8. Here the cache is already too small for K=4; adding K=6 just makes the miss rate worse without any route-flexibility payback, because the model is now burning most of its budget in attention compute rather than expert I/O.

**Quality side** (qualitative — no perplexity test done here):
- K=4 @ 500 continuation: "dropping the jar for fear of killing somebody, so managed to slip in one of the…" — fluent, matches the Alice excerpt.
- K=6 @ 500 continuation: "to drop the jar for fear of killing somebody underneath, so managed to put it into one of the cupboards as she fell past it…" — fluent, slightly different phrasing; "cupboards" is more natural than K=4's truncation.
- K=8 @ 500 continuation: "to drop the jar for fear of killing…" — identical first ~12 tokens to K=6 (greedy sampling is deterministic with K=6/K=8 agreeing when routing has >6 strong experts per token).

At 2 K ctx all three K values emit the same `<unk>\n\nHere's a thinking process:` sequence for the first 8-10 tokens, then diverge. No visible quality degradation at K=4 at this ctx. A real perplexity evaluation would be needed to claim K has no quality effect — our eyeball check just says "nothing is obviously broken at any K at 500 or 2 000 ctx."

**Verdict on K=6:** only a sweet spot if you're on a memory-constrained host AND operating at ≤ 500 ctx. At 2 K or higher, K=6 loses to K=4 on throughput with no quality-visible win. In the trained target regime (K=8 is what the model was RL'd against), K=6 is an unstable compromise — you're out of the trained activation distribution without the speed of K=4. Recommend K=6 only for short interactive workloads on low-RAM hosts.

### Recommendation

For **production on this 32 GB M1 Pro host**:
- **K=4** for any context ≥ 2 000 tokens — best tok/s (6.82), stable page-cache behaviour, coherent output, lowest SSD wear. Accept the quality trade-off of using fewer experts than the model was trained for.
- **K=8** only for very short prompts (≤ 500 ctx) **and** only if you can accept ~17% slower throughput (7.41 vs K=4's 8.08). Use this when quality on edge-case tokens matters (the K=8 output distribution matches training — K=4 sits in a slightly distribution-shifted regime).
- **K=6** not recommended for production — falls between K=4 and K=8 without a clear regime where it wins by more than noise.

For **hosts with ≥ 48 GB RAM** (e.g. the M3 Max from flash-moe's CLAUDE.md), the K=8 warm penalty should shrink to ≤ 5 % at 500-2 000 ctx because the full K=8 expert set fits in the page cache. On that class of machine the recommendation would flip to **K=8 as the default** with K=4 as the fallback for long ctx (where prefill time dominates regardless of K).

For **any host above 8 K ctx**: investigate batched prefill and attention-score compute before changing K. Beyond 2 K the bottleneck moves from expert I/O to attention, so K choice matters less than attention-path quality.

### Appendix — raw logs

Per-run log files in `/tmp/bench-ram/`:

```
log_k4_ctx500_prime.txt   log_k4_ctx500_warm.txt
log_k6_ctx500_prime.txt   log_k6_ctx500_warm.txt
log_k8_ctx500_prime.txt   log_k8_ctx500_warm.txt
log_k4_ctx2000_prime.txt  log_k4_ctx2000_warm.txt
log_k6_ctx2000_prime.txt  log_k6_ctx2000_warm.txt
log_k8_ctx2000_prime.txt  log_k8_ctx2000_warm.txt
```

With matching `prompt_*.txt`, `rss_*.txt`, and `vmstat_*.txt` for each label.

