> Copied from https://github.com/danveloper/flash-moe/issues/20 on 2026-04-17. License: governed by upstream issue terms.

---

# Issue #20: Other Qwen models

**Opened by:** scepeda78 | **Created:** 2026-04-01 | **State:** open

---

## Issue Body

Can it be done the same to other Qwen models?

---

## Comments

### Comment by hangtoo (2026-04-03)

Is Qwen/Qwen3.5-122B-A10B available for use?

---

### Comment by Ma-Dan (2026-04-12)

Yes, please check this branch https://github.com/Ma-Dan/flash-moe/tree/Qwen3.5-122B
Original weight is https://modelscope.cn/models/mlx-community/Qwen3.5-122B-A10B-4bit
usage.txt shows how to genarate necessary files.
Currently it can run at 5.30 tok/s on 4bit, 8.92 tok/s on 2bit.

---

## Our Answer (Added 2026-04-17)

**Yes.** The architecture class `Qwen3_5MoeForConditionalGeneration` (the HuggingFace architecture string in `config.json`) is shared across the entire Qwen3.x MoE family. The C-code skeleton in flash-moe — expert packing, SSD streaming, Metal MoE kernels — works for all of them once dimensions are runtime-configurable rather than hardcoded.

Confirmed working or in-progress instances:

| Model | Status | Source |
|-------|--------|--------|
| Qwen3.5-397B-A17B | Working (upstream) | danveloper/flash-moe main |
| Qwen3.5-122B-A10B | Working (fork) | Ma-Dan/flash-moe Qwen3.5-122B branch, 5.30 tok/s 4-bit |
| Qwen3-Coder-Next | Working (PR) | PR #13 (closed unmerged) |
| Qwen3-Coder-480B | Working (fork) | nerds-odd-e/flash-moe, commit e5952fd |
| **Qwen3.6-35B-A3B** | **This port** | nerds-odd-e base + PR #14 cherry-pick |

**Architecture family note:** All of these report `architectures: ["Qwen3_5MoeForConditionalGeneration"]` in their `config.json`. Key structural differences to handle per-model:

- `num_hidden_layers` (40 for Qwen3.6, 60 for Qwen3.5-397B, varies for others)
- `num_experts` (256 for Qwen3.6, 512 for Qwen3.5-397B)
- `num_experts_per_tok` (8 for Qwen3.6, 10 for Qwen3.5-397B)
- `hidden_size` (2048 for Qwen3.6, 4096 for Qwen3.5-397B)
- `moe_intermediate_size` (512 for Qwen3.6, 1024 for Qwen3.5-397B)
- `linear_num_value_heads` (32 for Qwen3.6, 64 for Qwen3.5-397B)

Qwen3.6-35B-A3B introduces two **new** features not in the 397B: QK-norm (`q_norm`/`k_norm` per full-attn layer) and an attention output gate (`attn_output_gate: true`). These require new codepaths; they are not present in any existing fork. Everything else is a dimension change, handled by `model_config.h` in the nerds-odd-e fork.

See [qwen3.6-35b-a3b-arch.md](qwen3.6-35b-a3b-arch.md) for the full spec and [port-plan.md](port-plan.md) for the porting sequence.
