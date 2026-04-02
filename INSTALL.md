# Flash-MoE Setup Guide — The Real One

## What This Is

A step-by-step guide to running Qwen3.5-397B-A17B (397 billion parameter MoE model) locally on an Apple Silicon Mac using [danveloper/flash-moe](https://github.com/danveloper/flash-moe). Written from an actual setup on an M4 Max 64GB MacBook Pro — including every gotcha we hit.

**End result:** ~5 tok/s interactive chat + OpenAI-compatible API server. Zero cloud dependency.

---

## Hardware Requirements

- Apple Silicon Mac (M3 Max, M4 Pro, M4 Max, or better)
- **Minimum 48GB unified memory** (64GB+ recommended for better page cache hit rates)
- **~450GB free disk space during setup** (drops to ~215GB after cleanup)
- 1TB+ SSD (all Apple Silicon Macs qualify)
- macOS 26.2+ (Darwin 25.2.0+)

### Disk Space Budget — Read This First

This is the #1 thing that will bite you. The setup has three phases of disk usage:

| Phase | Cumulative Disk Used | Notes |
|-------|---------------------|-------|
| Download MLX 4-bit model | ~210 GB | Source safetensors files |
| Git LFS cache (hidden) | ~420 GB | `.git/lfs/` holds a second copy |
| After `git lfs fetch --all` cleanup | ~210 GB | Delete `.git/lfs/` to reclaim |
| After `repack_experts.py` | ~420 GB | 210GB source + 209GB packed experts |
| After deleting source model | **~215 GB** | Final footprint |

**You need ~450GB free to start.** Plan your cleanup steps. On a 1TB drive, this means you need most of your disk empty.

**Critical cleanup commands** (safe to run at each stage):

```bash
# After git lfs checkout completes — reclaim the LFS cache copy
rm -rf ~/qwen35-397b-4bit/.git/lfs/

# After repack_experts.py completes and you've verified inference works
# Move packed_experts out, then delete the source model
# (see "Final Directory Layout" below)
```

---

## Phase 0: Prerequisites

```bash
# Check available disk space
df -h /

# Check macOS version
sw_vers

# Xcode command line tools (needed for make/clang)
xcode-select --install

# Python 3
python3 --version

# Git LFS (required for downloading the model)
git lfs --version
# If not installed:
brew install git-lfs
git lfs install
```

### Python Dependencies

```bash
pip3 install --user safetensors numpy
```

### A Note on huggingface-cli

The official way to download HuggingFace models is via `huggingface-cli`. However, the pip install may fail to create the CLI binary depending on your Python installation (common with macOS system Python or `/usr/local/bin/python3`). The symptoms:

- `pip3 install --user huggingface_hub` succeeds
- `huggingface-cli` returns "command not found"
- The package installs to `~/Library/Python/3.x/lib/` but no binary appears in `~/Library/Python/3.x/bin/`

**Don't waste time debugging this.** Use `git lfs` instead — it works perfectly and is already installed via Homebrew.

---

## Phase 1: Clone Flash-MoE

```bash
cd ~
git clone https://github.com/danveloper/flash-moe.git
cd flash-moe
```

---

## Phase 2: Build the Inference Engine

```bash
cd ~/flash-moe/metal_infer
make
make chat
```

Compiler warnings are normal. Verify both binaries exist:

```bash
ls -la infer chat
```

`make` only builds `infer`. The `chat` binary is a separate target — you must run `make chat` explicitly.

---

## Phase 3: Download the Model (~210GB)

```bash
cd ~
git clone https://huggingface.co/mlx-community/Qwen3.5-397B-A17B-4bit qwen35-397b-4bit
```

This clones the repo metadata but may not pull all LFS files. The actual weight download happens via:

```bash
cd ~/qwen35-397b-4bit
git lfs fetch --all
git lfs checkout
```

**Download speed:** ~50-60 MB/s typical. Expect 60-90 minutes for the full download.

### Common Download Issues

**`git lfs pull` exits silently without downloading:**
This happens when LFS thinks the files are already present (they're actually pointer stubs). Use `git lfs fetch --all` followed by `git lfs checkout` instead.

**How to check for incomplete downloads (LFS pointer stubs):**

```bash
# This checks the first bytes of each file — LFS stubs start with "version"
for f in *.safetensors; do
  head -c 7 "$f" | grep -q 'version' && echo "STUB: $f"
done
```

If any files show as STUB, re-run `git lfs fetch --all && git lfs checkout`.

**How to verify file integrity:**

```bash
# All safetensors should be ~4.6-4.9 GB except the last one (~1.8 GB)
ls -lh *.safetensors | awk '{print $5, $9}' | sort -n | head -5
```

If any file (except model-00046-of-00046) is significantly smaller than ~4.6 GB, it's truncated. Delete it and re-fetch:

```bash
rm model-000XX-of-00046.safetensors
curl -L -o model-000XX-of-00046.safetensors \
  "https://huggingface.co/mlx-community/Qwen3.5-397B-A17B-4bit/resolve/main/model-000XX-of-00046.safetensors"
```

**After download completes — free up the LFS cache immediately:**

```bash
rm -rf ~/qwen35-397b-4bit/.git/lfs/
```

This reclaims ~210GB. The checked-out safetensors files are separate and unaffected.

---

## Phase 4: Export Vocab and Tokenizer

The repo does not ship `vocab.bin` or `tokenizer.bin`. You must generate them from the model's `tokenizer.json`. The repo includes `export_vocab.py` and `export_tokenizer.py` but they may not exist in all versions. If missing, see the "Manual Vocab Export" section below.

### Export tokenizer.bin

```bash
cd ~/flash-moe/metal_infer
python3 export_tokenizer.py ~/qwen35-397b-4bit/tokenizer.json
```

Expected output: `tokenizer.bin` (~7.8 MB, 248044 vocab, 247587 merges)

### Export vocab.bin (with Byte-Level BPE Decoding)

**Important:** The naive vocab export produces garbled output — `Ġ` instead of spaces, `Ċ` instead of newlines. This is because Qwen3.5 uses GPT-style byte-level BPE encoding where printable Unicode characters map to raw bytes.

**Use this script instead** (creates a properly decoded vocab.bin):

```bash
cd ~/flash-moe/metal_infer
python3 export_vocab.py --model ~/qwen35-397b-4bit
```

This reads `~/qwen35-397b-4bit/tokenizer.json` and writes `vocab.bin` in the current directory.

### How to Tell If Your vocab.bin Is Wrong

If you see output like this in chat:

```
HereĠisĠaĠsimpleĠPythonĠscript:ĊĊ
```

Instead of:

```
Here is a simple Python script:
```

Your vocab.bin was built without byte-level BPE decoding. Re-run the script above.

---

## Phase 5: Extract Non-Expert Weights

```bash
cd ~/flash-moe/metal_infer
python3 extract_weights.py --model ~/qwen35-397b-4bit
```

This creates:
- `model_weights.bin` (~5.5 GB) — all non-expert weights, mmap'd at runtime
- `model_weights.json` — tensor manifest

Takes ~4 seconds.

**If this fails with `MemoryError` or `FileNotFoundError`:** Your safetensors files are incomplete. Re-check for LFS stubs (see Phase 3).

---

## Phase 6: Repack Expert Weights (~209GB)

**Check disk space first:**

```bash
df -h /
# You need ~209 GB free
```

```bash
cd ~/flash-moe/metal_infer
python3 ../repack_experts.py --model ~/qwen35-397b-4bit
```

This reads all 46 safetensors files and creates `packed_experts/` containing per-layer binary files optimized for the SSD streaming engine.

- **Time:** ~3-4 minutes at ~1 GB/s write speed
- **Output:** `~/qwen35-397b-4bit/packed_experts/` (~209 GB)
- Each layer is verified automatically (watch for "verification PASSED")

**If repack fails with `OSError: Short read`:** One of your safetensors files is truncated. Check file sizes (see Phase 3) and re-download the bad file.

After repack completes, symlink the packed experts into the metal_infer directory:

```bash
cd ~/flash-moe/metal_infer
ln -s ~/qwen35-397b-4bit/packed_experts packed_experts
```

---

## Phase 7: Test Inference

### Quick Smoke Test

```bash
cd ~/flash-moe/metal_infer
./infer --model ~/qwen35-397b-4bit --prompt "Hello" --tokens 20 --timing
```

**What to look for:**
- `[experts] 60/60 packed layer files available` — all experts loaded
- `hidden rms after final_norm` should be a real number (e.g., 1.72), NOT `nan`
- Output should be coherent English, not repeated `!` characters

**If you see `nan` and garbage output:** The weights weren't extracted correctly. Delete and re-run `extract_weights.py`.

**If you see `0/60 packed layer files`:** The `packed_experts/` directory isn't being found. Check your symlink or `--model` path.

### Interactive Chat

Start the server in one terminal:

```bash
cd ~/flash-moe/metal_infer
./infer --model ~/qwen35-397b-4bit --serve 8000
```

Start the chat client in another:

```bash
cd ~/flash-moe/metal_infer
./chat --show-think
```

### OpenAI-Compatible API

With the server running on port 8000, you can hit it from any tool that speaks OpenAI API:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

This is how you'd configure Continue.dev, VS Code extensions, or other tools to use the local model as a fallback.

---

## Phase 8: Cleanup

Once inference is verified, delete the source model to reclaim ~210GB:

```bash
# Keep only the packed_experts directory from the source model
# (already symlinked into metal_infer/)
cd ~/qwen35-397b-4bit
ls -d packed_experts  # Verify it exists

# Delete everything except packed_experts
find ~/qwen35-397b-4bit -maxdepth 1 ! -name packed_experts ! -name . -exec rm -rf {} +
```

Or if you want a cleaner layout, move packed_experts somewhere permanent:

```bash
mv ~/qwen35-397b-4bit/packed_experts ~/flash-moe/metal_infer/packed_experts_data
rm -rf ~/qwen35-397b-4bit
cd ~/flash-moe/metal_infer
rm packed_experts  # Remove old symlink
ln -s packed_experts_data packed_experts
```

### Final Disk Usage

| Component | Location | Size |
|-----------|----------|------|
| packed_experts | ~/flash-moe/metal_infer/packed_experts | ~209 GB |
| model_weights.bin | ~/flash-moe/metal_infer/ | ~5.5 GB |
| model_weights.json | ~/flash-moe/metal_infer/ | < 1 MB |
| vocab.bin | ~/flash-moe/metal_infer/ | < 1 MB |
| tokenizer.bin | ~/flash-moe/metal_infer/ | ~7.8 MB |
| infer + chat binaries | ~/flash-moe/metal_infer/ | < 1 MB |
| **Total** | | **~215 GB** |

---

## Performance Notes

### Expected Performance by Hardware

| Machine | RAM | Bandwidth | Expected tok/s |
|---------|-----|-----------|---------------|
| M3 Max (reference) | 48 GB | ~400 GB/s | 4.4 |
| M4 Max | 64 GB | ~546 GB/s | 5.0-5.5+ |

### Performance Tips

- **Close memory-heavy apps during inference.** Every GB of free RAM improves the OS page cache hit rate for expert weights, directly improving tok/s.
- **Subsequent prompts are faster.** The page cache warms up as you use the model. First prompt is always slowest.
- **Use `--timing` to diagnose bottlenecks.** The `expert_io` line shows how much time is spent reading from SSD vs cache.
- **Don't use `--2bit`** unless you don't need tool calling or structured JSON output. The 2-bit mode breaks quote characters.

---

## Troubleshooting Quick Reference

| Symptom | Cause | Fix |
|---------|-------|-----|
| All output is `!!!!` | 0/60 experts loaded OR NaN weights | Check packed_experts symlink; re-run extract_weights |
| `hidden rms = nan` | Corrupted model_weights.bin | Delete and re-run extract_weights.py |
| `MemoryError` in extract_weights | Safetensors files are LFS stubs | Run `git lfs fetch --all && git lfs checkout` |
| `OSError: Short read` in repack | Truncated safetensors file | Check file sizes, re-download the small one |
| `Ġ` and `Ċ` in output | vocab.bin missing byte-level BPE decode | Rebuild vocab.bin with the BPE decode script |
| `<unk>` at start of responses | Thinking tokens not in vocab | Cosmetic only; model output is correct |
| `huggingface-cli: command not found` | pip entry point not created | Skip it, use `git lfs` instead |
| `git lfs pull` exits silently | LFS thinks files are present | Use `git lfs fetch --all` then `git lfs checkout` |
| Chat says "Server not running" | infer --serve not started | Start `./infer --model PATH --serve 8000` first |
| `./chat --model` unrecognized | chat doesn't take --model | Chat connects to the server; configure model on infer |
