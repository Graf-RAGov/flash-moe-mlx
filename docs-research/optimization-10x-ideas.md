# 10× paths to 100 tok/s — Qwen3.6-35B-A3B 8-bit on M1 Pro 32 GB

**Baseline:** 9.7 tok/s warm, K=4, 1-tok prompt. Per-layer 2.38 ms × 40 ≈ 95 ms/tok. GPU ~50 %.
Time map (ms/layer): `cmd1_wait 1.09 + cmd2_wait 0.47 + expert_io 0.71 + cpu_attn 0.018 + encode 0.07 + submit 0.04`. Non-expert warm weights fit; expert working set is paged but mostly resident after first pass.

**Revised empirical ceiling on this box:** not the textbook 370 tok/s (bandwidth math at 200 GB/s). Realistic ceilings after removing specific bottlenecks:

| Scenario | Ceiling |
|---|---|
| Fix pipeline gaps (H1/H3), remove CPU memcpys, keep current kernels | **13–16 tok/s** (per-layer floor ~1.7 ms: cmd1 wait is real GPU work, not idle) |
| + fuse routing into GPU (eliminate CPU round-trips) | **18–22 tok/s** (encoder overhead 25 µs × 22 × 40 = 22 ms still real) |
| + ICB/argument-buffers collapse 3 submits → 1 per layer | **25–35 tok/s** |
| + speculative decoding 2.5× accept | **60–90 tok/s** (multiplicative over the above) |
| + batched prefill (TTFT only) | **~1000 tok/s prefill**, no gen change |
| Pure warm steady-state wall (200 GB/s ÷ 3.34 MB × 40 layers × K=4) | ~370 tok/s — requires eliminating ALL kernel-launch overhead (unreachable with current Metal API) |

**100 tok/s (10×) requires stacking ~3 of the below.** No single path gets there.

---

## A. MTP-head self-speculative (Qwen3.6 native)
**Mechanism:** Qwen3.6 ships one dedicated MTP transformer block (`mtp_num_hidden_layers: 1`, shared embeddings). Use it as the draft head over the main model's hidden state to predict token `t+1`; verify with one K=4 forward pass of the main model. Acceptance then amortizes the expert I/O over 2 tokens.
**Gain:** ~1.6–1.9× (single MTP head; DeepSeek-V3 reports ~1.8× MTP acceptance, ~85 % on code). The CLAUDE.md 397B finding that MTP "breaks even" is because at 17B active × K=4=128 experts/layer per verified token, expert I/O scaled linearly with speculated tokens. **At 35B K=4 active, the expert superset overlap is much higher** (K=4 draft + K=4 verify from the same hidden; overlap ~30–50 %), so MoE I/O does reuse. Not an automatic carry-over of the 397B null result.
**Effort:** 5–8 days; ~800 LOC. Must extract MTP weights (currently skipped in `extract_weights.py`), add draft pass, and a verify-accept loop. Paper refs: DeepSeek-V3 (2024, §4.3), GLM-4.5 MTP.
**Risk:** low (verification preserves exact sampling). Quality: **zero loss by construction.**
**Fits:** yes. MTP block is one transformer layer (~200 MB 8-bit).

## B. Batched prefill (TTFT only)
**Mechanism:** Replace per-token prefill loop (infer.m:7809) with one forward pass of N prompt tokens through the same MoE stack, batching the attention Q·K^T and reusing expert reads across the N-token slice.
**Gain:** TTFT 5–20× at 500–8 K prompts (from `benchmark-ram-tokspeed.md`: 220 s → 10–40 s at 2 K; 23 min → 2–4 min at 8 K). **Zero steady-state tok/s gain.**
**Effort:** 8–12 days, ~800–1000 LOC. Non-trivial because (i) GatedDeltaNet recurrence is serial per token — must process sequentially but batch the MoE I/O across ~64 tokens; (ii) RoPE and KV writes need batch variants; (iii) full-attn layers need proper causal mask.
**Risk:** medium (KV-cache layout, mrope-interleaved RoPE correctness). Quality: none if done correctly.
**Fits:** yes. This is the critical usability win for anything longer than one-shot chat. Already on TODO (line 215 of benchmark doc).

## C. Metal ICBs + argument buffers (pipeline collapse)
**Mechanism:** Replace 3 command buffers × 40 layers + ~80 `waitUntilCompleted` calls with one Indirect Command Buffer recorded once and re-dispatched per token. Residency sets (macOS 14) pre-pin expert buffers; argument buffers eliminate per-encoder binding.
**Gain:** submit/wait overhead today ≈ 120 × 50 µs + 80 × 100 µs = 14 ms/tok ≈ 14 % of total. Collapsing to O(1) submits reclaims ~12 ms → **1.13×**. Combined with removing the CMD3→CMD1 serialization gap (H1 in parallelism doc), **1.3–1.5×** is realistic.
**Effort:** 10–15 days, ~600 LOC. ICBs are a big API switch; argument buffers Tier-2 require rewriting every encoder's `setBuffer:` chain. Apple's own FA examples are a good template.
**Risk:** medium (expert pointers must be pre-bound via argument buffer tables; routing is dynamic — use `useResources:` with a superset).
**Fits:** yes, M1 Pro supports Tier 2 argument buffers and ICBs.

## D. Persistent GPU-resident token pass (fused layers)
**Mechanism:** Keep hidden state on GPU across all 40 layers. Fuse routing softmax + topK into Metal (extends PR #11's partial softmax). Eliminate the ~400 host-memcpy/token and the CMD2-wait. One command buffer per token.
**Gain:** Removes `cpu_softmax + cpu_topk + HIDDEN memcpys`. Today this is ~0.02 ms CPU math + blocking CPU→GPU bounce. Real win is not the math, it's unblocking the pipeline: cmd1_wait is currently 1.09 ms of which ~0.3–0.5 ms is scheduling bubble from the next-layer routing dependency. **Expected 1.5–2×**.
**Effort:** 8–12 days, ~500 LOC Metal + infer.m glue. Partial-softmax direction already merged in PR #11.
**Risk:** low. Quality: identical.
**Fits:** yes.

## E. mlock'd hot-expert arena (cold-path only)
**Mechanism:** Pre-fault + `mlock` the top-N most frequently visited experts (say 12 GB of them), page the tail via current mmap. CLAUDE.md's "trust the OS" null was at 397B where working set ≫ RAM. Here experts are 17–26 GB vs 32 GB — arguably the OS already wins in steady-state (warm runs show 0 pageins).
**Gain:** **~0 % warm steady-state** (OS page cache already near-optimal at 1-tok gen; bench shows warm K=4 has 0 MB pageins). Cold start / long-ctx prefill win: 1.3× maybe. Not the path.
**Effort:** 5 days, 300 LOC. Need expert-use histogram → LRU pin list.
**Risk:** low; has been tried as Metal LRU (results.tsv rows 5, 6, 22) → all slower. `mlock` differs from that but the structural issue is the same.
**Fits:** yes, but **not a 10× candidate.**

## F. int8 matmul (skip dequant)
**Mechanism:** Direct `simdgroup_matrix<char, ...>` int8 GEMV over packed experts; no fp16 dequant intermediate.
**Gain:** Dequant is ~30 % of expert matvec time per CLAUDE.md FMA entry. Skipping saves ~1.3×. But M1 Pro does **not** have hardware int8 matmul — only `simdgroup_matrix<half, float>` (no int8 type until M3+). So this path is actually slower on M1 Pro.
**Effort:** N/A — architecturally blocked on M1 Pro.
**Risk:** —
**Fits:** **No, skip.**

## G. Per-layer K-expert kernel batching
**Mechanism:** Stack K=4 expert activations, run one matmul per gate/up/down across the 4 experts (4×512 output instead of 4 separate 512 matvecs). Metal threadgroup occupancy problem (H2 in parallelism doc) goes away.
**Gain:** Expert matvec today runs at ~60 % occupancy per H2. Bringing it to ~95 % on the 3 expert projections (gate/up/down) with 4× wider output gives ~1.25–1.4× on `cmd2_wait + cmd3` specifically. **Overall 1.1–1.15×**.
**Effort:** 5 days, 300 LOC Metal + infer.m expert-loop restructuring.
**Risk:** low.
**Fits:** yes.

## H. Smaller quant (MXFP4/IQ4)
**Mechanism:** 8-bit → MXFP4 or IQ4_XS experts (halve expert file size).
**Gain:** Expert file 26 GB → 13 GB; fully page-resident warm. Steady-state win negligible here (already warm). Cold/long-ctx: 1.3–1.5× on TTFT.
**Quality caveat:** OpenAI's MXFP4 on gpt-oss reports near-BF16 perplexity at 4.25 bits — so the Q4 quality objection in CLAUDE.md (2-bit broke JSON) does not apply at 4.25 bits with proper block-scale format. For Qwen3.6 there's no reference requant yet; would need to port MLX's quantizer.
**Effort:** 4 days to requant + adapt kernel (has prior art: current 4-bit/2-bit kernels).
**Risk:** medium — requires a validation pass on tool calling.
**Fits:** yes.

## I. Draft-model speculative (Qwen3-1.7B)
**Mechanism:** Resident 1.7B dense draft, generate 4–8 tokens, verify with MoE in one forward. Well-documented (vLLM, llama.cpp).
**Gain:** Published EAGLE/Medusa accept ratios 60–75 %; with 4 draft tokens → 2.5–3.5× speedup on dense models. For MoE it depends on I/O reuse — at K=4 with 8 draft tokens, verify-pass needs one MoE forward per draft (if tree-verified), which costs 8× the expert I/O — killing the win. **Unless** verify can batch 8 tokens through one MoE pass (shares expert fetches if the K-sets overlap).
**Honest estimate:** 1.8–2.5× if batched verify is built, ~1.0× otherwise.
**Effort:** 10–14 days, ~1200 LOC. Needs Qwen3-1.7B port (different architecture, no GatedDeltaNet), persistent draft cache, tree-verification logic.
**Risk:** medium. Quality: lossless.
**Fits:** 1.7B Q8 is ~1.8 GB resident — fits.

## J. ANE utilization
**Mechanism:** Run attention projections on the 16-core ANE via Core ML while GPU does MoE.
**Gain:** ANE peak ~15 TOPS int8 but latency-oriented, not throughput. Core ML round-trip is ~1 ms just for the submit. Realistic: no gain or regression. No published LLM has shown ANE wins over Metal for batch-1 decode.
**Effort:** 4+ weeks. Very high.
**Risk:** high. **Skip.**

## K. Misc
- **Streaming prefill during typing** — UX only.
- **KV-cache quantization to int8** — at current ctx (<2 K on this box) KV is ~150 MB; not the bottleneck. At 8 K it's ~600 MB and *is* a bottleneck per bench (KV-cache strided read dominates at 8 K gen). 1.3× at long ctx, 0× at short.
- **N-gram/Lookahead decoding** — lossless, ~1.3–1.5× on code (Fu et al., 2024), zero I/O. Cheap win worth stacking.

---

## Ranking (gain × feasibility)

| # | Idea | Gain | Effort (days) | Verdict |
|---|---|---|---|---|
| 1 | **A. MTP self-speculative** | 1.6–1.9× | 5–8 | Highest ROI. Native head already in model. |
| 2 | **D. Fused GPU-resident pass** | 1.5–2× | 8–12 | Compounds with everything. Partial prior art. |
| 3 | **C. ICB + argument buffers** | 1.3–1.5× | 10–15 | Structural. Unblocks further kernel fusion. |
| 4 | **B. Batched prefill** | 5–20× TTFT | 8–12 | Different axis; required for any long-ctx use. |
| 5 | **K-lookahead / G. expert batching** | 1.15–1.4× each | 4–5 each | Cheap, compose cleanly. |

## Roadmap

**1-week budget:** MTP head (A) alone. ~6 days implementation + 1 day validation. Expected **1.7× → ~16 tok/s**. Lossless, cleanest ROI.

**1-month budget:** Stack A + D + C + G + lookahead. Multiplicative ceiling: 1.7 × 1.7 × 1.35 × 1.2 × 1.3 = **~6.1×** → **~58 tok/s**. This is the realistic 1-month ceiling. **Not 100 tok/s.** 10× requires either (i) also landing MXFP4 + int8 path on M2/M3 hardware, or (ii) adding draft-model speculative (I) in a batched-verify variant for another 1.8× on top. Then stacking yields ~10×, but effort rises to ~6 weeks.

**Honest note:** the warm 370 tok/s number is unreachable on current hardware at batch=1. Each Metal encoder launch is ~25 µs, each command buffer submit/wait ~50–100 µs, and with 22 encoders × 40 layers you can't go below ~22 ms/tok for launch overhead alone — a hard **45 tok/s ceiling** without ICBs. ICBs drop this to ~2 ms/tok launch budget, opening ~200 tok/s in theory; kernel work itself floors at ~3 ms/tok. So the practical M1 Pro steady-state ceiling is **80–120 tok/s**, i.e. 10× is at the edge of physically possible, not 2–3× headroom.

## Interaction between ideas (stacking vs conflict)

**Stacks cleanly:** MTP speculative (A) × GPU-fused pass (D) × ICB (C) × expert batching (G) × lookahead are orthogonal — each attacks a different bottleneck (token economy, CPU round-trip, launch overhead, occupancy, draft strategy) and multiplies. **Batched prefill (B) is independent** (different code path, TTFT only).
**Conflicts:** MTP (A) + draft model (I) don't stack — both exhaust the speculation budget; pick one. **mlock arena (E) conflicts with ICB residency sets (C)** — ICBs want Metal-managed residency, mlock wants OS-managed; mixing causes duplicate resident copies. **int8 matmul (F) is moot on M1 Pro**, so no conflict, just absent. **MoE-aware MXFP4 (H)** risks reopening the tool-calling quality regression observed at 2-bit; validate before stacking. **G (expert batching)** changes the Metal encoder shape ICBs record — do G first, then record ICBs around the new shape.

---

## Appendix — Sources

- **MTP / DeepSeek-V3** — Liu et al. 2024, *DeepSeek-V3 Technical Report*, §4.3 "Multi-Token Prediction". Reports ~1.8× speedup with MTP-depth=1, 85–90 % acceptance on code.
- **EAGLE** — Li et al. 2024, *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*. 2.7–3.5× on Vicuna/LLaMA-2-Chat.
- **EAGLE-2** — Li et al. 2024, *EAGLE-2: Faster Inference of LLMs with Dynamic Draft Trees*. 3.0–4.3×.
- **Medusa** — Cai et al. 2024, *Medusa: Simple LLM Inference Acceleration*. 2.2–3.6× on LLaMA-2.
- **Lookahead decoding** — Fu et al. 2024, *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding*. 1.5–2.3× lossless, N-gram cache.
- **Speculative decoding** — Leviathan et al. 2023, *Fast Inference from Transformers via Speculative Decoding*. 2–3× on T5/LaMDA.
- **MXFP4 / gpt-oss** — OpenAI, *gpt-oss Model Card* 2025, 4.25-bit MXFP4 claim of near-BF16 quality.
- **Apple Metal ICB** — Apple Developer, "Using Indirect Command Buffers to Reduce CPU Overhead"; Residency Sets in macOS 14+.
- **LLM in a Flash** — Alizadeh et al. 2023 (Apple). Inspires flash-moe.
- **flash-moe local** — `results.tsv`, CLAUDE.md discarded list, `parallelism-exploration.md` per-layer breakdown, `benchmark-ram-tokspeed.md` context sweep including 8K-ctx degradation.

## Appendix — 397B vs 35B MTP asymmetry (why this repo's prior null shouldn't stop us)

The CLAUDE.md entry "MTP speculative decoding — break-even; each speculated token still needs its own K experts → no MoE I/O reuse" is correct for 397B at K=4 **with I/O-bound expert streaming from SSD at ~17.5 GB/s for 6.75 MB experts**. At that model scale, expert file fetch is the dominant cost and per-speculated-token expert reads kill the gain.

At 35B 8-bit:
- Expert size per layer: ~3.3 MB (vs 6.75 MB).
- Experts at K=4 warm are **fully page-resident** after first pass (bench: 0 MB pageins on warm K=4).
- Verify pass at speculate-N re-touches the already-resident 40×4×3.3 MB = 528 MB working set — free in the page cache.
- MTP head is *one layer*, ~200 MB fp16 / 100 MB 8-bit, negligible.

So the 397B argument (expert I/O per speculated token) does not hold: on 35B warm the speculation cost is pure compute, and compute is not the bottleneck. This is the key inversion that makes (A) worth doing despite the prior null.
