# Port flash-moe to Qwen3.6-35B-A3B (MLX 8-bit) on 32GB Mac

## Context

User wants flash-moe adapted to run a ~35B MoE model with q8 weights on a 32GB Mac — no ollama, direct fork of `danveloper/flash-moe`. Initial framing was `qwen3.6:35b-a3b-q8_0` (the ollama tag), but the cleanest path uses `mlx-community/Qwen3.6-35B-A3B-8bit` (37.75 GB, MLX affine 8-bit format) because flash-moe's entire pipeline (repack, dequant kernels, expert streaming) is built around MLX affine quant — not llama.cpp GGUF block_q8_0. Switching to MLX 8-bit weights means reusing existing infrastructure instead of writing a GGUF dequant backend from scratch.

Flash-moe with mmap-backed SSD streaming is the mechanism that lets a model bigger than RAM run — so fitting 37.75GB of weights on 32GB RAM is the design case, not a fight.

## Research leverage (don't start from main)

Three pieces of upstream/fork work do most of the de-hardcoding already. Use them — don't redo.

| Source | What it gives | State |
|---|---|---|
| **PR #3** (Alexintosh) | `ModelConfig` struct replaces ~54 `#define`s, reads HF `config.json` at startup. Adds `--model <path>`. `model_manager.py`. | Open, not merged. 4,838-line diff. |
| **PR #13 / PR #14** (userFRM) | `dequant_matvec_8bit` Metal kernel for MLX 8-bit tensors. Fixes MoE gate routing when gates are 8-bit and experts 4-bit. Core infra for any 8-bit MLX model. | #13 closed unmerged; #14 open (subset, the dequant kernel alone). |
| **nerds-odd-e/flash-moe fork** | `model_config.h` (262 lines, runtime model id), `INSTALL.md`, `install.sh`, big `infer.m` refactor (+899/-251), `extract_weights.py` rewrite, `export_vocab.py`. Got Qwen3-Coder-480B running. Active Apr 2–11 2026. | 7 commits ahead of main, clean tree. |
| **Issue #15 comment** | Full setup guide with disk-budget math and gotchas (git-lfs double-copy trap, `huggingface-cli` fragility). | Open, unresolved. |

All artifacts already downloaded to `/tmp/fm-research/` (PR diffs 3/11/13/14, nerds-compare.json). Step 1 moves them into the repo.

## Step 1 — Save research into repo docs folder [status: completed]

Target: `/Users/retry/Documents/code/flash-moe/docs-research/` (in the existing clone, alongside existing `docs/`).

Files to write:
- `docs-research/pr-3-runtime-config.diff` — copy of `/tmp/fm-research/pr-3.diff`
- `docs-research/pr-13-qwen3-coder-next.diff` — `/tmp/fm-research/pr-13.diff`
- `docs-research/pr-14-8bit-dequant.diff` — `/tmp/fm-research/pr-14.diff`
- `docs-research/pr-11-pure-wins.diff` — `/tmp/fm-research/pr-11.diff`
- `docs-research/fork-nerds-odd-e.md` — compare summary, commit list, list of changed files, pull command
- `docs-research/fork-gorroai.md` — summary + K=4 fix note (no model port, but perf)
- `docs-research/issue-15-setup-gotchas.md` — verbatim body from issue #15 (disk-budget guide)
- `docs-research/issue-17-expert-index.md` — verbatim body (warns about `expert_index.json` scope bug)
- `docs-research/issue-20-other-qwen-models.md` — one-liner with our findings appended
- `docs-research/qwen3.6-35b-a3b-arch.md` — architecture spec for the port target (numbers below)
- `docs-research/flash-moe-hardcoded-constants.md` — map from Phase-1 Explore agent (file:line citations)
- `docs-research/port-plan.md` — condensed version of Steps 2–5 of this plan, in-repo
- `docs-research/README.md` — index pointing at the above, explaining this is research cache, not upstream docs

The JSON blob `/tmp/fm-research/nerds-compare.json` stays out — it's noisy and reproducible via `gh api`.

## Step 2 — Foundation: fork nerds-odd-e, cherry-pick PR #14

Base the work on `nerds-odd-e/flash-moe` (has `model_config.h` and a working "different Qwen" adaptation) rather than upstream main.

Sequence:
1. Add remote: `git remote add nerds https://github.com/nerds-odd-e/flash-moe && git fetch nerds`.
2. Create working branch from `nerds/main`: `git checkout -b qwen36-35b-a3b nerds/main`.
3. Cherry-pick or patch in PR #14 (`dequant_matvec_8bit` + CPU fallback). Apply manually if cherry-pick conflicts with nerds-odd-e's infer.m changes.
4. Optional: cherry-pick PR #11's bit-identical perf wins (partial softmax for routing is meaningful since Qwen3.6 has 256 experts, K=8).
5. Do NOT pull PR #3 wholesale — nerds-odd-e's `model_config.h` overlaps. Treat PR #3 as a reference for tensor-name patterns only.

## Step 3 — Architecture deltas (Qwen3.6-35B-A3B vs flash-moe's Qwen3.5-397B) [status: completed]

From `Qwen/Qwen3.6-35B-A3B/config.json` + `model.safetensors.index.json`:

| Param | flash-moe target | Qwen3.6-35B-A3B | Action |
|---|---|---|---|
| num_hidden_layers | 60 | **40** | Update config |
| hidden_size | 4096 | **2048** | Update config; check shader `x_shared[4096]` (still OK, 2048 ≤ 4096) |
| num_attn_heads / num_kv_heads | 32 / 2 | **16 / 2** | Update config |
| head_dim | 256 | **256** | Same |
| num_experts | 512 | **256** | Update config + repack offsets |
| num_experts_per_tok (K) | 4 (code) / 10 (spec) | **8** routed + 1 shared | Update K; partial-softmax routing from PR #11 applies |
| moe_intermediate_size | 1024 | **512** | Update config + per-expert byte size |
| shared_expert_intermediate_size | 1024 | **512** | Update config |
| vocab_size | 248320 | **248320** | Same |
| rope_theta / partial_rotary | 1e7 / 0.25 | **1e7 / 0.25** | Same |
| layer_types dispatch | `(i+1) % 4 == 0` (15 full / 45 linear) | same rule, **10 full / 30 linear** | Auto-works; 10 full-attn layers at {3,7,11,15,19,23,27,31,35,39} |
| linear_num_value_heads | 64 | **32** | Update config |
| linear_num_key_heads | 16 | **16** | Same |
| linear_key/value_head_dim | 128 | **128** | Same |
| rms_norm_eps | 1e-6 | **1e-6** | Same |
| tie_word_embeddings | false | **false** | Same |

**New codepaths required** (these are NOT in flash-moe upstream):
1. **QK-norm** on full-attention layers: `self_attn.q_norm.weight`, `self_attn.k_norm.weight` — an extra RMSNorm on Q and K after RoPE. Small shader addition.
2. **Attention output gate**: `attn_output_gate: true` in config → sigmoid gate on attention output before `o_proj`. New kernel or fused into existing output path.
3. **Fused expert tensor layout**: safetensors pack experts as 3D tensors `mlp.experts.gate_up_proj` and `mlp.experts.down_proj` (expert axis included) instead of per-expert 2D tensors. `repack_experts.py` must slice along axis 0 per expert.

**MTP head** (`mtp.*`, 1 extra transformer block) — drop, not needed for plain next-token.
**Vision tower** (`model.visual.*`, 27-block ViT) — drop, text-only.

## Step 4 — Repack + run

1. Download weights to `~/qwen36-35b-a3b-8bit/` (37.75 GB, 8 safetensors shards). Check disk first — need ~75 GB free peak (37.75 download + ~35 GB repacked experts + temp). User has 69 GB free → delete source safetensors immediately after repack (follow issue #15's staged-cleanup guide).
2. Run updated `extract_weights.py` → writes `model_weights.bin` (non-expert weights: embed, norms, full-attn proj, DeltaNet, shared expert, lm_head) + `model_weights.json` manifest.
3. Run updated `repack_experts.py` → writes `packed_experts/layer_00.bin` ... `layer_39.bin`. Per-expert byte size is quarter of Qwen3.5-397B's because MLX 8-bit doubles bytes-per-element but moe_intermediate halves and experts halve: roughly 256 experts × ~3.4 MB/expert × 8-bit ≈ ~870 MB per layer × 40 ≈ **35 GB packed**.
4. Update `expert_index.json` generation to cover all 40 layers (issue #17 shows upstream ships a layer-0-only index).
5. Build: `cd metal_infer && make infer chat`.
6. Smoke test: `./infer --prompt "Reply with exactly: OK" --tokens 5 --timing`.

## Step 5 — Verification

- `./infer` with `--timing` prints per-layer breakdown. Expect first-token stall while SSD warms page cache; steady-state 2–4 tok/s on 32 GB Mac with 37.75 GB weights > RAM (flash-moe paper + issue #21 "M4 Pro 24GB: 3.50 tok/s at 4-bit" benchmark transfers).
- Numerical verify: compare first-layer outputs against MLX python reference (`mlx_lm.generate` with `mlx-community/Qwen3.6-35B-A3B-8bit`) for a fixed prompt. PR #14's verification protocol applies.
- Monitor: `vm_stat 1` (pageins), `sysctl vm.swapusage`, `sudo memory_pressure`. Raise Metal wired cap if load fails: `sudo sysctl iogpu.wired_limit_mb=26000`.
- If OOM or unbearable tok/s: fallback to `mlx-community/Qwen3.6-35B-A3B-6bit` or `-5bit`; flash-moe has no 6/5-bit kernels, so this becomes a separate port. Alternative fallback: keep MLX 4-bit (mlx-community/Qwen3.6-35B-A3B-4bit) and reuse existing 4-bit kernels — only 8-bit gate dequant needs adding (PR #14).

## Critical files

| File | Change |
|---|---|
| `metal_infer/model_config.h` (nerds-odd-e) | Add Qwen3.6-35B-A3B profile: 40 layers, 256 experts, K=8, hidden 2048, etc. |
| `metal_infer/infer.m` | Replace remaining hardcoded constants with config lookups; add QK-norm path; add attn output gate; update layer-type dispatch count |
| `metal_infer/shaders.metal` | Add `dequant_matvec_8bit` from PR #14; add QK-norm kernel if not present; add attn output gate kernel |
| `metal_infer/extract_weights.py` | Handle fused `mlp.experts.gate_up_proj` + `down_proj` 3D tensors; skip `model.visual.*` and `mtp.*`; emit Qwen2Tokenizer vocab export |
| `repack_experts.py` | Slice 3D expert tensors into per-expert blobs; recompute per-layer byte layout for new dims + 8-bit |
| `metal_infer/export_vocab.py` (nerds-odd-e) | Confirm it reads Qwen2Tokenizer `vocab.json + merges.txt` correctly |
| `expert_index.json` generator | Fix to cover all 40 layers (issue #17) |
| `docs-research/*` | New — see Step 1 |

## Unknowns / verify live before coding

1. Whether `mlx-community/Qwen3.6-35B-A3B-8bit` uses MLX affine 8-bit (matches PR #14 kernel) or a newer MLX `quantization_config` variant. Inspect `config.json` → `quantization.bits`.
2. Whether gates (`mlp.gate.weight`, `mlp.shared_expert_gate.weight`) are 8-bit or fp16 in this specific repo (PR #13 found them 8-bit in the 4-bit model; 8-bit model might have them fp16 or 8-bit — changes dequant wiring).
3. Exact fused-expert tensor shape: `[num_experts, 2*moe_inter, hidden]` vs `[num_experts, moe_inter, 2*hidden]` — slice direction matters.
4. Whether nerds-odd-e's `model_config.h` already handles fused expert tensors (Qwen3-Coder-Next uses the same pattern) or only flat ones.
5. Peak disk during repack — confirm incremental delete of source shards post-repack is safe.

---

## Post-port decisions (added in-session)

### MTP speculative decoding — NOT worth implementing on this setup

Researched (`docs-research/mtp-research.md`, `docs-research/opt-mtp-implementation.md`) and rejected. Reasons:

1. **Weights not in our quant.** `mlx-community/Qwen3.6-35B-A3B-8bit` ships ZERO MTP tensors (`grep mtp model.safetensors.index.json` = 0 matches) — the quantizer dropped them. Getting them requires re-downloading bf16 shards 25+26 of the original repo (~6 GB more).

2. **Prior art says break-even.** `CLAUDE.md` "Discarded" table: `MTP speculative decoding | break-even | MoE I/O scales per-token (unlike dense)`. Flash-moe's upstream team already tried it on 397B.

3. **SSD-streamed MoE inverts the speedup math.** DeepSeek-V3's published 1.8× assumes weights-resident in GPU. On our setup, verifying γ draft tokens requires γ × K × 40 expert reads from SSD — I/O cost is 56 % of per-layer time, so extra reads eat the parallel-verify win.

4. **Realistic gain here: +0-15 %, not 1.6-1.9×.** The target tok/s range (16-18 from 9.7) is not reachable via MTP while weights stream from SSD.

**Revisit when:** the model fits fully in RAM (Q5 variant on this 32 GB box, or Q8 on a 48 GB+ host). Dense-case math applies there and the ~800-1000 LOC implementation becomes justified.
