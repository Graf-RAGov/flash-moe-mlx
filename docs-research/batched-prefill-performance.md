# Batched Prefill Performance (Feature Scaffold)

Date: 2026-04-17

## Scope

This measures the current `--prefill-batch` implementation (windowed prefill loop scaffold) against the immediate pre-change baseline.

- Baseline: detached clean commit `81861b2` (no `--prefill-batch` option)
- Feature: current worktree with `--prefill-batch`

Note: this environment runs in CPU fallback (`ERROR: No Metal device`), so numbers are only for functional comparison, not GPU throughput claims.

## Test setup

- Binary model config: `MODEL=qwen3.6-35b-a3b`
- Model path: `/Users/retry/qwen36-35b-a3b-8bit`
- Weights/manifest/vocab: `/Users/retry/Documents/code/flash-moe/metal_infer/*`
- Prompt: 25 tokens (`"hello "` repeated 24 times)
- Generation: `--tokens 1`
- Experts: `--k 4`
- Timing: `--timing`

## Results

| Variant | Prefill Mode | Prefill (ms) | TTFT (ms) | Total (s) | Delta vs baseline |
|---|---|---:|---:|---:|---|
| Baseline (`81861b2`) | serial | 61520 | 64986 | 65.0 | — |
| Feature (`--prefill-batch 1`) | serial (`window=1`) | 60698 | 64173 | 64.2 | prefill -1.34%, TTFT -1.25%, total -1.23% |
| Feature (`--prefill-batch 32`) | batched-window (`window=24`) | 59988 | 63397 | 63.4 | prefill -2.49%, TTFT -2.45%, total -2.46% |

Additional batch sweep on feature build (same setup):

| `--prefill-batch` | Mode | Effective Window | Prefill (ms) | TTFT (ms) | Total (s) |
|---:|---|---:|---:|---:|---:|
| 1 | serial | 1 | 61147 | 64740 | 64.7 |
| 8 | batched-window | 8 | 60552 | 64163 | 64.2 |
| 32 | batched-window | 24 | 60413 | 64085 | 64.1 |
| 64 | batched-window | 24 | 60327 | 63967 | 64.0 |

`window=24` at `32/64` is expected because intermediate prefill tokens are `25-1=24`.

## Conclusion

Against the immediate baseline, this feature does **not** show a material performance improvement yet. The observed 1-2.5% reductions are within typical run-to-run noise for this CPU-fallback setup.

This matches the implementation intent: current code adds batched-window control flow, but does not yet batch heavy kernels (attention/MoE compute) inside each window.
