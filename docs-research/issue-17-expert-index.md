> Copied from https://github.com/danveloper/flash-moe/issues/17 on 2026-04-17. License: governed by upstream issue terms.

---

# Issue #17: Missing complete expert_index.json for Qwen3.5-397B-A17B-4bit model

**Opened by:** ysyx2008 | **Created:** 2026-03-30 | **State:** open

---

## Issue Body

## Problem Description

When trying to run the flash-moe project with the Qwen3.5-397B-A17B-4bit model, I encountered an issue where the provided `expert_index.json` file only contains expert information for **layer 0**, but the project requires expert information for **all 60 layers**.

## Steps to Reproduce

1. Clone the repository and set up the environment
2. Download the Qwen3.5-397B-A17B-4bit model from ModelScope (46 safetensors files, ~224GB total)
3. Run `extract_weights.py` to create `model_weights.bin` and `model_weights.json` (successful)
4. Run `repack_experts.py` to create packed expert files
5. Try to run `./metal_infer --full` or `./infer` with the model

## Error Message

```
ERROR: Cannot open /path/to/model/packed_experts/layer_01.bin: No such file or directory
```

## Root Cause Analysis

1. **Current `expert_index.json` structure**: Only contains expert information for layer 0

   ```json
   {
     "model_path": "...",
     "expert_reads": {
       "0": { ... }  // Only layer 0!
     }
   }
   ```

2. **Expected structure**: Should contain expert information for all 60 layers (0-59)

   ```json
   {
     "model_path": "...",
     "expert_reads": {
       "0": { ... },
       "1": { ... },
       // ... layers 2-58 ...
       "59": { ... }
     }
   }
   ```

3. **Impact**: The `repack_experts.py` script can only process layer 0, leaving layers 1-59 without packed expert files.

## Workarounds Attempted

1. **Modified `expert_index.json`**: Tried to manually add layer information, but need accurate offsets from all 46 safetensors files
2. **Partial testing**: Can only test layer 0 performance, cannot run full 60-layer inference
3. **Code inspection**: Found that `repack_experts.py` expects a complete index but the provided one is incomplete

## Questions for the Author

1. **Is there a script to generate the complete `expert_index.json`** for all 60 layers from the 46 safetensors files?
2. **Can you provide the complete `expert_index.json`** file for the Qwen3.5-397B-A17B-4bit model?
3. **What's the intended workflow** for users who download the model from ModelScope/Hugging Face?

## Environment

- **Project**: flash-moe (commit 3601d41)
- **Model**: mlx-community/Qwen3.5-397B-A17B-4bit from ModelScope
- **Hardware**: MacBook Pro M4 Max, 128GB RAM
- **OS**: macOS (Darwin Kernel Version 24.6.0)

## Additional Context

The project is amazing and the performance claims are impressive! I was able to successfully:

- Download the 224GB model
- Extract non-expert weights (5.5GB `model_weights.bin`)
- Run single-layer MoE benchmarks (showing ~160 tok/s theoretical throughput)
- Build all binaries successfully

The only blocker is the missing expert information for layers 1-59.

## Suggested Solution

1. **Option A**: Provide a script that analyzes the 46 safetensors files and generates the complete `expert_index.json`
2. **Option B**: Share the complete `expert_index.json` file in the repository
3. **Option C**: Document the exact process for users to generate this index themselves

Thank you for creating this incredible project! Looking forward to running the full 397B model on my MacBook Pro.

---

## Comments

### Comment by kekekekekeshi (2026-04-02)

貌似你需要处理成mlx的

*(Translation: "It seems you need to process it as MLX format.")*

---

## Port Note

For the Qwen3.6-35B-A3B port we must generate a **40-layer** index (not 60). See [port-plan.md](port-plan.md) Step 7 (repack phase). The nerds-odd-e fork's updated `repack_experts.py` (+379/-37 vs upstream) handles multi-model repacking and likely resolves this issue — it was written after this bug was filed. Verify that it auto-generates the full index by scanning all safetensors shards, rather than reading from a pre-baked JSON.

The `expert_index.json` path hardcode mentioned in issue #15 comment by Shivox also applies here — update the `model_path` field after generating, or patch to read from `--model` arg.
