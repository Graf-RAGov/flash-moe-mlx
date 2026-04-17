# fork-nerds-odd-e/flash-moe

## Key Facts

- **URL:** https://github.com/nerds-odd-e/flash-moe
- **Status:** 7 commits ahead of danveloper/flash-moe main, 0 behind. Clean tree.
- **Default branch:** main
- **Most active fork** as of Apr 2026

## Added Files

| File | Notes |
|------|-------|
| `INSTALL.md` | End-user installation guide |
| `install.sh` | Automated setup script |
| `requirements.txt` | Python dependency list |
| `metal_infer/model_config.h` | 262 lines — runtime model configuration header |
| `metal_infer/export_vocab.py` | Vocab export script |

## Heavily Modified Files

| File | Change | Notes |
|------|--------|-------|
| `metal_infer/infer.m` | +899 / -251 | Major refactor — de-hardcodes model identity, dynamic dispatch |
| `repack_experts.py` | +379 / -37 | Repack logic updated for multi-model support |
| `metal_infer/extract_weights.py` | +67 / -32 | Weight extraction updated |
| `metal_infer/Makefile` | +20 / -14 | Build system updates |

## Commit History (newest first)

| Hash | Date | Message |
|------|------|---------|
| `840aff8` | 2026-04-11 | :sparkles: fix install error with qwen3.5 |
| `b127bbc` | 2026-04-04 | :sparkles: support tool call for qwen3 coder |
| `3723847` | 2026-04-03 | :sparkles: model id is not hard coded now |
| `e5952fd` | 2026-04-03 | :sparkles: make qwen3 coder 480B running |
| `557c229` | 2026-04-03 | :wrench: update python venv folder for install.sh |
| `3c91c56` | 2026-04-02 | create install.sh and requirements.txt from INSTALL.md |
| `2990980` | 2026-04-02 | add install.md following issue #15 + code refine |

## Why This Matters for the Qwen3.6-35B-A3B Port

1. **De-hardcoded model ID** (`model id is not hard coded now`) — the single most important structural change. Upstream bakes model dimensions as `#define`s; this fork reads them at runtime.
2. **Proven second-model path** — Qwen3-Coder-480B is a different model from Qwen3.5-397B. Getting it running proves the refactor actually works, not just compiles.
3. **model_config.h** (262 lines) is the header we extend to add the Qwen3.6-35B-A3B profile. Already has the right shape for a config-driven dispatch.
4. **export_vocab.py** — the Qwen3.6 tokenizer is Qwen2BPE with the same byte-level decode pattern; reusing this script saves work.
5. **install.sh / requirements.txt** — reduces setup friction for anyone verifying the port.

**Use this as the base branch**, not upstream main.

## Clone Command

```bash
git remote add nerds https://github.com/nerds-odd-e/flash-moe
git fetch nerds
git checkout -b qwen36-35b-a3b nerds/main
```
