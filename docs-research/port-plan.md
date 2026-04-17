# Port Plan: flash-moe to Qwen3.6-35B-A3B

Condensed working plan. Full context in parent plan: `/Users/retry/.claude/plans/groovy-wiggling-waterfall.md`

---

## Step 1 — Base on nerds-odd-e fork

The nerds-odd-e fork is 7 commits ahead of upstream main, has `model_config.h` (262 lines), and has already run a second model (Qwen3-Coder-480B). Start here, not from upstream main.

```bash
cd ~/Documents/code/flash-moe
git remote add nerds https://github.com/nerds-odd-e/flash-moe
git fetch nerds
git checkout -b qwen36-35b-a3b nerds/main
```

Why not PR #3 as base: nerds-odd-e's `model_config.h` overlaps with PR #3's `ModelConfig` struct and was written after PR #3 filed. Treat PR #3 (`pr-3-runtime-config.diff`) as a reference for tensor-name patterns and any missed constants — do not wholesale-apply.

---

## Step 2 — Cherry-pick PR #14 for MLX 8-bit dequant kernel

PR #14 adds `dequant_matvec_8bit` in `shaders.metal` plus a CPU fallback path. This is the minimum required to use `mlx-community/Qwen3.6-35B-A3B-8bit` weights.

```bash
# Apply from the saved diff
git apply docs-research/pr-14-8bit-dequant.diff

# Or cherry-pick the PR commits if the remote is available:
# git fetch origin pull/14/head:pr14
# git cherry-pick <relevant commits>
```

Conflict risk: nerds-odd-e modified `infer.m` extensively (+899/-251). The dequant kernel additions in PR #14 target `shaders.metal`, not `infer.m` — conflict is unlikely but review the diff before applying. If conflicts occur, apply the `dequant_matvec_8bit` kernel manually to `shaders.metal`.

Optional: also apply PR #11 (`pr-11-pure-wins.diff`) for the partial-softmax routing optimization. Qwen3.6 has 256 experts (vs 512), so partial softmax over 256 is still a win. Apply after Step 4 and measure.

---

## Step 3 — Add Qwen3.6 profile to model_config.h

Edit `metal_infer/model_config.h` to add a Qwen3.6-35B-A3B profile. Exact numbers from [qwen3.6-35b-a3b-arch.md](qwen3.6-35b-a3b-arch.md):

```c
// In model_config.h — add alongside existing profiles
static const ModelConfig QWEN36_35B_A3B_CONFIG = {
    .model_id         = "Qwen3.6-35B-A3B",
    .hidden_dim       = 2048,
    .num_layers       = 40,
    .num_attn_heads   = 16,
    .num_kv_heads     = 2,
    .head_dim         = 256,
    .vocab_size       = 248320,
    .num_experts      = 256,
    .num_experts_per_tok = 8,
    .moe_intermediate = 512,
    .shared_intermediate = 512,
    .linear_num_v_heads = 32,
    .linear_num_k_heads = 16,
    .linear_key_dim   = 128,
    .linear_value_dim = 128,
    .conv_kernel_size = 4,
    .rope_theta       = 10000000.0f,
    .partial_rotary   = 0.25f,
    .rotary_dim       = 64,
    .group_size       = 64,
    .full_attn_interval = 4,
    .num_linear_layers = 30,  // 40 - 10 full-attn
    .has_qk_norm      = true,   // NEW
    .attn_output_gate = true,   // NEW
    .tensor_prefix    = "model.language_model.",  // different from Qwen3.5
};
```

Then add new codepaths in `infer.m`:

1. **QK-norm** (`q_norm`/`k_norm`): after Q and K projections in full-attn layers, apply RMSNorm using loaded `q_norm.weight`/`k_norm.weight`. Upstream does not have this — write new path gated on `config.has_qk_norm`.
2. **Attention output gate** (`attn_output_gate`): after `o_proj`, multiply by `sigmoid(gate)`. Write new path gated on `config.attn_output_gate`.
3. **Tensor prefix**: update `extract_weights.py` to strip `model.language_model.` prefix instead of `model.` for Qwen3.6.
4. **3D expert tensors**: Qwen3.6's `mlp.experts.gate_up_proj` and `mlp.experts.down_proj` are fused 3D tensors `[num_experts, ...]`. The nerds-odd-e repack already handles per-expert slicing — verify it reads these fused tensors correctly and slices along axis 0.

---

## Step 4 — Add new codepaths: QK-norm, attn output gate, fused 3D expert slicing

This is the substantive new C code. Checklist:

- [ ] Load `q_norm.weight` and `k_norm.weight` for all 10 full-attn layers into `model_weights.bin` extraction
- [ ] Add RMSNorm kernel call after Q/K projection in `infer.m` full-attn path (can reuse existing `rms_norm` kernel, just with different weights)
- [ ] Load `shared_expert_gate.weight` — a scalar gate on the shared expert output (sigmoid activation)
- [ ] In `infer.m` attention output path: `output = output * sigmoid(gate)` for Qwen3.6 profile
- [ ] In `repack_experts.py`: slice fused `mlp.experts.gate_up_proj[E]` and `mlp.experts.down_proj[E]` per expert E, verify shape `[moe_intermediate, hidden]` matches EXPERT_SIZE calculation
- [ ] Verify `mlp.shared_expert.{gate,up,down}_proj` are included in `model_weights.bin` extraction
- [ ] Skip `mtp.*` and `model.visual.*` tensors in extract_weights.py

---

## Step 5 — Download, repack, build, smoke test, numerical verify

### Disk Budget

| Phase | Disk usage | Action |
|-------|-----------|--------|
| Download `mlx-community/Qwen3.6-35B-A3B-8bit` | +37.75 GB | 8 shards |
| Git LFS cache (hidden `.git/lfs/`) | +37.75 GB | Delete immediately after `git lfs checkout` |
| After LFS cache deletion | 37.75 GB | `rm -rf .git/lfs/` |
| `extract_weights.py` | +~3 GB | `model_weights.bin` + `model_weights.json` |
| `repack_experts.py` | +~35 GB | `packed_experts/` — 40 layer files x 256 experts |
| Peak (source + packed) | **~76 GB** | Delete source shards 1-by-1 as you go if tight |
| After deleting source shards | **~38 GB** | Final footprint |

**User has 69 GB free.** 76 GB peak exceeds available space by ~7 GB. Use staged deletion:

```bash
# After each shard is fully consumed by repack_experts.py, delete it:
rm model-0000X-of-00008.safetensors
```

The nerds-odd-e `repack_experts.py` processes shards sequentially — delete each source shard after the script moves past it. See issue #15 comments for the staged-cleanup pattern (the same technique described there for the 210 GB Qwen3.5 model).

### Download

```bash
git clone https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-8bit qwen36-35b-a3b-8bit
cd qwen36-35b-a3b-8bit
git lfs fetch --all
git lfs checkout
rm -rf .git/lfs/  # Free LFS cache immediately
```

### Build and Run

```bash
cd ~/Documents/code/flash-moe/metal_infer
python3 extract_weights.py --model ~/qwen36-35b-a3b-8bit
python3 ../repack_experts.py --model ~/qwen36-35b-a3b-8bit
make
./infer --model ~/qwen36-35b-a3b-8bit --prompt "Hello" --tokens 20 --timing
```

### Smoke Test Checklist

- `[experts] 40/40 packed layer files available`
- `hidden rms after final_norm` is a real number (not `nan`)
- Output is coherent text (not `!!!!!` repeated)
- `--timing` shows `expert_io` time (SSD streaming is working)

### Numerical Verification

Compare output logits against `mlx_lm` reference on the same prompt:

```bash
pip install mlx-lm
python3 -c "
from mlx_lm import load, generate
model, tokenizer = load('mlx-community/Qwen3.6-35B-A3B-8bit')
print(generate(model, tokenizer, prompt='Hello', max_tokens=20))
"
```

Tokens should match (or near-match accounting for fp32 vs bf16 accumulation differences).

---

## Expected Performance on M-series 32 GB

| Reference | Hardware | tok/s | Notes |
|-----------|----------|-------|-------|
| flash-moe paper | 48 GB / ~400 GB/s | 4.4 | Qwen3.5-397B 4-bit |
| Issue #21 report | M4 Pro 24 GB | 3.50 | Qwen3.5-397B |
| Issue #15 rafaelkupper | M4 Pro 48 GB | 3.1 | Qwen3.5-397B |
| **Qwen3.6 estimate** | **32 GB M-series** | **2-4** | Smaller model, 8-bit, fits better in page cache |

First token latency: **30-60 seconds** on cold SSD page cache. Subsequent tokens faster as page cache warms. All expert weights (35 GB packed) exceed the 32 GB RAM — streaming from SSD is expected and normal.
