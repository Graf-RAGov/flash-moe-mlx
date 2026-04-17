# docs-research/

Research cache + session artifacts for porting flash-moe to Qwen3.6-35B-A3B on 32GB Mac. Not upstream docs.

Session result: working port at 9.7 tok/s warm steady-state. Commits `b82ac06..129a25e` on branch `qwen36-35b-a3b`.

---

## Session plan

| File | What it is |
|------|-----------|
| [session-plan.md](session-plan.md) | Full 5-phase plan that drove the port. Context, architecture deltas, disk budget, verification strategy. |

## Upstream diffs (reference copies)

| File | What it is |
|------|-----------|
| [pr-3-runtime-config.diff](pr-3-runtime-config.diff) | PR #3 — ModelConfig struct replaces ~54 hardcoded #defines; adds --model flag and model_manager.py (4,838-line diff, open unmerged) |
| [pr-11-pure-wins.diff](pr-11-pure-wins.diff) | PR #11 — pure perf wins: partial softmax routing, fused kernels, no architectural changes |
| [pr-13-qwen3-coder-next.diff](pr-13-qwen3-coder-next.diff) | PR #13 — Qwen3-Coder-Next port + dequant_matvec_8bit Metal kernel (closed unmerged; kernel subset lives in PR #14) |
| [pr-14-8bit-dequant.diff](pr-14-8bit-dequant.diff) | PR #14 — dequant_matvec_8bit kernel alone; cherry-picked at commit 21a3b1a for 8-bit weight support |

## Upstream fork / issue context

| File | What it is |
|------|-----------|
| [fork-nerds-odd-e.md](fork-nerds-odd-e.md) | Summary of nerds-odd-e/flash-moe fork — most active fork, de-hardcodes model ID, already runs Qwen3-Coder-480B. Used as base branch. |
| [fork-gorroai.md](fork-gorroai.md) | Summary of gorroai/flash-moe fork — benchmarking focus, K=4 routing fix (+5.6%), M2R2 AoT experiment |
| [issue-15-setup-gotchas.md](issue-15-setup-gotchas.md) | GitHub issue #15 verbatim — full setup guide with disk budget math and every known gotcha |
| [issue-17-expert-index.md](issue-17-expert-index.md) | GitHub issue #17 verbatim — incomplete expert_index.json (layer 0 only); must regenerate all layers |
| [issue-20-other-qwen-models.md](issue-20-other-qwen-models.md) | GitHub issue #20 verbatim + our answer — yes, same architecture family works for Qwen3.6 and Qwen3-Coder variants |

## Target architecture

| File | What it is |
|------|-----------|
| [qwen3.6-35b-a3b-arch.md](qwen3.6-35b-a3b-arch.md) | Full architecture spec for Qwen3.6-35B-A3B — config params, tensor naming, layer types, tokenizer, quant details |
| [flash-moe-hardcoded-constants.md](flash-moe-hardcoded-constants.md) | Map of every hardcoded constant in upstream flash-moe with file:line citations — the porting checklist |
| [port-plan.md](port-plan.md) | Condensed 5-step port plan with disk budget and expected performance figures |

## Benchmarks + audits (produced during this session)

| File | What it is |
|------|-----------|
| [benchmark-ram-tokspeed.md](benchmark-ram-tokspeed.md) | Context-length sweep (100/500/2K/8K/16K), K sweep (K=4/6/8), RAM + pageins + tok/s. Practical ceiling ~2K ctx. |
| [parallelism-exploration.md](parallelism-exploration.md) | Why GPU is 50% utilized, why CPU looks single-core. Theoretical ceiling math. Top 3 wins ranked. |
| [pipeline-guard-audit.md](pipeline-guard-audit.md) | Audit of Metal pipeline creation vs dispatch macro guards — sigmoid_gate_pipe-class bugs. None new found. |
| [optimization-10x-ideas.md](optimization-10x-ideas.md) | 11 optimization paths researched (MTP, batched prefill, fused GPU, ICBs, int8 matmul, draft models, ANE, etc.). Honest ceiling ~58 tok/s (6×), not 10×. |
| [fused-gpu-pass-performance.md](fused-gpu-pass-performance.md) | Rebench log for CMD1+CMD2 merged linear-attn path (3 passes, TTFT/tok-s + per-phase timing + merged-layer counters). |

## Optimizations tried

| File | What it is |
|------|-----------|
| [opt-vdsp-gcd.md](opt-vdsp-gcd.md) | +17% tok/s — Accelerate vDSP + BLAS on main-thread scalar loops. Bit-identical. Cherry-picked to main at c17fb48. |
| [mtp-research.md](mtp-research.md) | Research on MTP self-speculative (DeepSeek-V3 pattern, vLLM reference). |
| [opt-mtp-implementation.md](opt-mtp-implementation.md) | MTP plan + blocker analysis. Not implemented — MLX 8-bit repo ships NO MTP tensors + SSD-streamed MoE inverts the speedup math (I/O scales per draft token). |
| [batched-prefill-research.md](batched-prefill-research.md) | (in-flight / check agent status) |
| [fused-gpu-pass-research.md](fused-gpu-pass-research.md) | (in-flight / check agent status) |
