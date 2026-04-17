# MTP Self-Speculative Decoding — Implementation Plan & Status

Task: #18 — wire Qwen3.6-35B-A3B's bundled MTP (Multi-Token Prediction) head into flash-moe for lossless self-speculative decoding.

Worktree: `.claude/worktrees/agent-afdd7d37`. Timebox: 90 min. Status at end of session: **blocked on weights + prior-art negative result; research + plan delivered, no code committed.**

Companion research: [`mtp-research.md`](./mtp-research.md).

---

## 1. Architecture chosen (if unblocked)

**Self-speculative decoding using the bundled `mtp.*` block** (the single extra transformer layer in Qwen3.6-35B-A3B) — not a separate draft model.

Rationale:
- Weights are already part of the base checkpoint (19 tensors in shards 25-26 of 26); no extra model to train or host.
- Shared `lm_head` and shared `embed_tokens` (per `"mtp_use_dedicated_embeddings": false` in config.json) means zero duplication cost.
- Matches the DeepSeek-V3 / Qwen3-Next "native MTP" design that vLLM and SGLang ship in production.

Forward pass per draft token (condensed, see §1.2 of the research doc for the full form):

```
e_n = rmsnorm(embed_tokens[last_tok], mtp.pre_fc_norm_embedding.weight)
h_n = rmsnorm(h_main_last_layer,      mtp.pre_fc_norm_hidden.weight)
x   = mtp.fc @ concat(e_n, h_n)                    # [2H] -> [H]
x   = mtp.layers[0](x)                             # full-attn + MoE + 2 norms
x   = rmsnorm(x, mtp.norm.weight)
draft = argmax(lm_head @ x)
```

Verification (Leviathan et al. 2023): run the main 40-layer forward on the extended sequence, compare argmax at each draft slot, accept prefix match, fall back at first mismatch.

---

## 2. Implementation plan (not executed — see §5 for reason)

### Phase A — Pull MTP tensors into the extraction pipeline

**File**: `/Users/retry/Documents/code/flash-moe/metal_infer/extract_weights.py` (or a new `extract_mtp.py` sibling).

Changes:
- Add a `--include-mtp` flag. Default off to preserve main-branch behaviour.
- Remove `mtp.*` from the vision/expert skip regexes. Use a new `mtp_expert_pattern` to route `mtp.layers.0.mlp.experts.*` through the existing expert-packing code path (this is one extra "virtual layer 40" from the perspective of `packed_experts/`).
- Write non-expert MTP tensors (`mtp.fc.weight`, `mtp.pre_fc_norm_*`, `mtp.norm.weight`, the self_attn q/k/v/o/q_norm/k_norm/input_layernorm/post_attention_layernorm, the shared-expert weights, the routing gate `mtp.layers.0.mlp.gate.weight`) into a parallel `model_weights_mtp.bin` blob with matching manifest. Do **not** touch the existing `model_weights.bin` (main's production artifact).

**File**: `/Users/retry/Documents/code/flash-moe/repack_experts.py`

Changes:
- Add `--include-mtp-experts` flag to the `qwen3.6-35b-a3b` profile. When set, emit `packed_experts/layer_40.bin` holding `mtp.layers.0.mlp.experts.*` in the same byte layout as layers 0-39.

### Phase B — Load the MTP block in infer.m

**File**: `/Users/retry/Documents/code/flash-moe/metal_infer/infer.m`

Add at top of file:
```c
#define HAS_MTP 1   // runtime-gated by --speculative CLI flag regardless
#define MTP_LAYER_IDX 40  // treat as a 41st virtual layer for expert file lookup
```

New struct `MtpBlock` holding pointers into a second mmap'd region for `model_weights_mtp.bin`, mirroring the existing `FullAttnLayer` struct (Q/K/V/O projections + q_norm/k_norm + input_layernorm/post_attention_layernorm + routing gate + shared expert + two pre-fc norms + fc + final norm).

`mtp_draft_token()` — new function (~150 LOC) that:
1. Runs the pre-fc norms on the incoming `h_main` and `embed[last_tok]`.
2. Performs the 2048→2048 linear via `fast_dequant_matvec` if quantised, or a plain BLAS `cblas_sgemv` if bf16.
3. Calls a condensed copy of the existing full-attention layer forward (reusing the same Metal kernels — the weights live in the MTP buffer, but the shape and attention math are identical to full-attention layer 39 of Qwen3.6).
4. Returns `argmax` of `lm_head @ normed_output`.

Key reuse: the MoE forward path (`gpu_encode_dequant_matvec_with_io_bufs` and the shared-expert helpers) works layer-agnostically given the right expert-file FD — we pass the `layer_40.bin` FD.

### Phase C — Speculative decode loop

**File**: `/Users/retry/Documents/code/flash-moe/metal_infer/infer.m` (generation loop, around the current token-step block).

Pseudocode, guarded by `if (g_spec_draft_n > 0)`:
```c
// after main forward produced h_main and tok_t:
int drafts[MAX_DRAFT];
int n_drafts = 0;
if (g_spec_draft_n > 0) {
    float h_drafter[HIDDEN_DIM];
    memcpy(h_drafter, h_main, sizeof(h_drafter));
    int prev = tok_t;
    for (int i = 0; i < g_spec_draft_n; i++) {
        drafts[i] = mtp_draft_token(h_drafter, prev, /*out=*/h_drafter);
        prev = drafts[i];
        n_drafts++;
    }
    // Run main model on [tok_t, drafts...] in one batched forward
    main_forward_batch(tok_t, drafts, n_drafts, /*out*/ verified_logits);
    int accepted = 0;
    for (int i = 0; i < n_drafts; i++) {
        int main_argmax = argmax(verified_logits[i]);
        if (main_argmax == drafts[i]) { emit(drafts[i]); accepted++; }
        else { emit(main_argmax); break; }
    }
    if (accepted == n_drafts) emit(argmax(verified_logits[n_drafts])); // bonus token
    g_spec_accepted += accepted;
    g_spec_draft_attempts += n_drafts;
}
```

The batched main forward is the expensive piece: the current `infer.m` is a **single-token loop**. Batched forward through 40 MoE layers with γ+1 positions requires either:
- Sequentially stepping through positions but caching the KV so only new positions hit Q/K/V projections — feasible, matches existing KV-cache.
- *Or* using the existing per-token path γ+1 times — much simpler, but defeats the point since it costs γ+1 full steps.

The first option is the correct one and is the majority of the LOC budget for this feature (estimated ~400-600 LOC of careful KV-cache / attention-mask plumbing).

### Phase D — CLI + config

New flag in `infer.m` main args: `--speculative N` (γ, default 0 = off). Print `spec acceptance X/Y (Z%)` in the timing summary.

### Phase E — Build + verify

- `make infer` in the worktree.
- Without `--speculative`: regression test the first 10 tokens of `"The capital of France is"`; output must be byte-identical to the pre-change binary's output.
- With `--speculative 2` and `--speculative 4`: same prompt, output must still be byte-identical (speculative is lossless). Compare tok/s.

Expected result (see §5): **no measured speedup** over the baseline on this SSD-streamed engine. The plan is correct; the engine class is wrong for MTP.

---

## 3. Blockers encountered this session

### 3.1 Source weights unavailable

- `/Users/retry/qwen36-35b-a3b-8bit/` has no `*.safetensors` files (deleted per the task brief's warning).
- The mlx-community 8bit distribution we previously extracted from **never shipped MTP tensors** — `grep mtp /Users/retry/qwen36-35b-a3b-8bit/model.safetensors.index.json` returns 0. This is independent of the deletion; even if the shards were still there, they would not contain MTP.
- Only recourse: download shards 25 + 26 of the original `Qwen/Qwen3.6-35B-A3B` (bf16) = 3.83 GB + 2.23 GB = **6.06 GB**. Command (not executed):
  ```
  huggingface-cli download Qwen/Qwen3.6-35B-A3B \
    --include "model-00025-of-00026.safetensors" \
    --include "model-00026-of-00026.safetensors" \
    --include "model.safetensors.index.json" \
    --include "config.json"
  ```
  Then either (a) requantise the MTP block to MLX 8-bit affine to match the existing pipeline, or (b) add a bf16 matvec path in `shaders.metal` for just the MTP block.

### 3.2 Worktree is still on the Qwen3.5-397B code path

`infer.m` in the worktree has `HIDDEN_DIM 4096 / NUM_LAYERS 60 / NUM_EXPERTS 512` and `MODEL_PATH_DEFAULT` points at the Qwen3.5 MLX snapshot. There is no runtime profile switch and no `MODEL_QWEN36_35B_A3B` symbol. Qwen3.6 porting is a prerequisite — tracked by `docs-research/qwen3.6-35b-a3b-arch.md` and `issue-20-other-qwen-models.md`.

### 3.3 Prior art: MTP is already a discarded experiment on this engine

`/Users/retry/Documents/code/flash-moe/CLAUDE.md` (line 105, "Discarded" table):

> `MTP speculative decoding | break-even | MoE I/O scales per-token (unlike dense)`

The mechanism for the break-even, unpacked in `mtp-research.md` §5:
- Verifying γ drafts makes the main model read γ x K x 40 expert sets from SSD per step (less the overlap the OS page cache catches).
- Per-expert SSD I/O is the dominant term of the per-layer cost (2.4 ms of 4.3 ms — 56%).
- The "1.8x" figure DeepSeek reports is for weights-in-VRAM, not weights-on-SSD.

Even with a perfect implementation, the realistic ceiling on this engine class is +0-15% (see research doc §7).

---

## 4. Measured speedup

**Not measured**. No code was committed and no weights were available to test with. The only benchmark delivered is the analytical ceiling estimate in `mtp-research.md` §7.

---

## 5. Acceptance rate measured

**Not measured**. Published reference figures that would apply if the implementation were completed: DeepSeek-V3 ~85% at γ=1 with one trained MTP head; Qwen3-Next blogs quote "higher than DeepSeek-V3" without a concrete number. Per-token α ≈ 0.80-0.90 is the expected operating band for Qwen3.6-35B-A3B's single-layer MTP head. (Sources in `mtp-research.md` §2.)

---

## 6. Before/after tok/s

| Run | tok/s |
|-----|-------|
| Baseline (claimed steady-state from task brief, Qwen3.6-35B-A3B at 8-bit) | 9.7 |
| With `--speculative 4` if fully implemented | Projected +0-15% over baseline ≈ 9.7-11.2 |
| Task target | 16-18 |

**Target is not achievable on this engine class** (SSD-streamed MoE). See research doc §5, §7.

---

## 7. Commit SHA in worktree

**No commit.** This session produced only two documentation files:

- `/Users/retry/Documents/code/flash-moe/docs-research/mtp-research.md`
- `/Users/retry/Documents/code/flash-moe/docs-research/opt-mtp-implementation.md` (this file)

Both are in the worktree's parent working directory (`flash-moe/docs-research/`), which is shared with the main repo — not inside the worktree's isolated branch. No `git` operations were performed.

---

## 8. Recommendation

1. **Pause MTP**. The prior experiment result (break-even) is consistent with the analytical model, which predicts +0-15% ceiling. The engineering cost (~800-1000 LOC + a 6 GB download + one-off MLX 8-bit requant of the MTP block + a bf16 fallback path) is not justified by the expected return.
2. **Revisit MTP if/when** flash-moe adds a mode where the main model's hot expert set fits in RAM (e.g. `Qwen3.6-35B-A3B` is small enough that at 8-bit, 256 experts x ~1.7 MB = ~430 MB per layer, ~17 GB total — plausibly RAM-resident on a 48 GB machine with the rest of the budget). In a RAM-resident MoE configuration, MTP becomes equivalent to the dense-model case and the DeepSeek-V3 1.8x number is back in play.
3. **Until then**, the larger lever for Qwen3.6-35B-A3B (per existing research in `docs-research/optimization-10x-ideas.md` and `opt-vdsp-gcd.md`) is the expert-read pipeline itself, not draft-and-verify on top of it.
