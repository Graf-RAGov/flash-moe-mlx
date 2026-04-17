# fork-gorroai/flash-moe

## Key Facts

- **URL:** https://github.com/gorroai/flash-moe
- **Focus:** Research and benchmarking, not a model port
- **This fork does not run a different model** — it improves performance on Qwen3.5-397B

## Key Work

### K=4 Routing Fix (+5.6%, 21.48 tok/s on M3 Max)

Upstream uses `num_experts_per_tok=10` in routing but the original flash-moe paper describes K=4 as the sweet spot for this routing architecture. gorroai's campaign `6ca3340` pins K=4 and measures +5.6% throughput:

- Baseline: ~20.36 tok/s
- With K=4: **21.48 tok/s** (M3 Max reference)

Note: Qwen3.5-397B has K=10 in config. K=4 is a deliberate approximation — uses the top 4 experts only. Quality impact not fully characterized; gorroai's testing shows coherent output.

### M2R2 Ahead-of-Time (AoT) Experiment

Commit `4df3af8` documents an M2R2 AoT compilation experiment:

- Cold baseline binary: **15.2 tok/s**
- AoT pre-compiles Metal shaders to eliminate first-token compilation stall
- Still slower than JIT-warmed runtime (21.48 tok/s) but useful for measuring cold-start vs warm-cache contribution

### Autoresearch Infrastructure

Commit `7a119d5` adds an autoresearch validator script — automated benchmarking loop that records tok/s at each config change. Useful reference if we want to do automated perf regression tracking during the Qwen3.6 port.

## Notable Commits

| Hash | Message |
|------|---------|
| `6ca3340` | K=4 campaign 21.48 tok/s |
| `4df3af8` | docs: M2R2 AoT experiment results |
| `7a119d5` | autoresearch validator |

## Claimed Quant Kernels (Unverified)

Web search results mention gorroai adding IQ3_XXS / IQ4_XS / Q5_K quantization kernels. These do **not** appear in the first 15 visible commits. Verify before using — the commits may be on a branch or the web index may be wrong.

## Relevance to Qwen3.6-35B-A3B Port

Qwen3.6-35B-A3B uses `num_experts_per_tok=8` (plus 1 shared expert, effective K=9), and has 256 experts total (vs 512 in Qwen3.5-397B). Partial softmax over 256 experts is still a meaningful win over full softmax.

The K=4 concept applies: if quality allows, routing with K=4 out of 256 would further improve tok/s. The gorroai fork's benchmarking methodology is the reference for measuring this trade-off once the basic port is running.

**Do not base the port on this fork** — it has no multi-model infrastructure. Use nerds-odd-e as the base; apply gorroai's K-routing and softmax concepts as optimization passes after correctness is verified.
