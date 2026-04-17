# Fused GPU Pass Performance — Rebench (April 17, 2026)

## Scope

This note records measured runtime behavior of the `CMD1+CMD2` merge for the linear-attention GPU path in:

- `/Users/retry/.codex/worktrees/483b/flash-moe/metal_infer/infer.m`

Feature-related code paths touched:

- merged wait accounting: `cmd12_wait`, `cmd12_layers`
- linear path merge gating: `can_merge_cmd12_linear`, `cmd12_merged`
- merged residual source binding (`buf_moe_hidden` fast path)
- merged command buffer submit/wait path (`cmd_fused = cmd12 ? ...`)

## Benchmark Setup

- Date: **April 17, 2026**
- Host: **Apple M1 Pro** (Metal)
- Build:
  - `cd /Users/retry/.codex/worktrees/483b/flash-moe/metal_infer`
  - `make MODEL=qwen3.6-35b-a3b infer`
- Model/expert path:
  - `--model /Users/retry/qwen36-35b-a3b-8bit`
- Non-expert weights/manifest/vocab:
  - `--weights /Users/retry/Documents/code/flash-moe/metal_infer/model_weights.bin`
  - `--manifest /Users/retry/Documents/code/flash-moe/metal_infer/model_weights.json`
  - `--vocab /Users/retry/Documents/code/flash-moe/metal_infer/vocab.bin`
- Runtime args:
  - `--prompt "Hello, what is" --tokens 20 --k 4 --timing`
- Logs:
  - `/tmp/rebench_clean1.log`
  - `/tmp/rebench_clean2.log`
  - `/tmp/rebench_clean3.log`

## Results (3 Clean Passes)

| Run | TTFT (ms) | Gen tok/s | cmd1_wait (ms/layer) | cmd12_wait (ms/layer) | cmd2_wait (ms/layer) | expert_io (ms/layer) | total_layer (ms/layer) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2906 | 1.83 | 1.583 | 7.665 | 0.796 | 2.344 | 12.640 |
| 2 | 2974 | 1.71 | 1.870 | 8.564 | 0.924 | 1.702 | 13.283 |
| 3 | 2710 | 1.75 | 1.487 | 8.153 | 0.876 | 2.369 | 13.129 |
| **Average** | **2863.3** | **1.76** | **1.647** | **8.127** | **0.865** | **2.138** | **13.017** |

## Evidence That Merge Path Is Active

All three runs report:

- `cmd_buffers: 1710 (base 3/layer, merged cmd12 layers=570)`
- `sync_waits: 950 (base 2/layer, merged cmd12 layers=570)`

Given 760 timed layers total, 570 merged layers matches the expected linear-layer share and confirms the fused path is executing.

## Notes

- This benchmark reflects **current tree state**, including active debug prints in `infer.m`.
- Treat these numbers as a rebench snapshot and merge-path verification, not a final production ceiling.
- A clean A/B speedup attribution still needs a toggle (merged on/off) in the same binary with identical logging settings.
