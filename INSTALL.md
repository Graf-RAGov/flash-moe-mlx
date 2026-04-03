# Flash-MoE Setup Guide — The Real One

## What This Is

A step-by-step guide to running Qwen3.5-397B-A17B (397 billion parameter MoE model) locally on an Apple Silicon Mac using [danveloper/flash-moe](https://github.com/danveloper/flash-moe). Written from an actual setup on an M4 Max 64GB MacBook Pro — including every gotcha we hit.

**End result:** ~5 tok/s interactive chat + OpenAI-compatible API server. Zero cloud dependency.

If you'd rather automate the manual steps below, the repo also ships `./install.sh` with the same flow and an optional guarded cleanup step.

**Important:** `./install.sh` only supports `mlx-community/Qwen3.5-397B-A17B-4bit`. `--model-dir` changes the location, not the model family.

---

## Hardware Requirements

- Apple Silicon Mac (M3 Max, M4 Pro, M4 Max, or better)
- **Minimum 48GB unified memory** (64GB+ recommended for better page cache hit rates)
- **~430GB free disk space during setup** (drops to ~215GB after cleanup)
- 1TB+ SSD (all Apple Silicon Macs qualify)
- macOS 26.2+ (Darwin 25.2.0+)

### Disk Space Budget — Read This First

This is the #1 thing that will bite you. With `hf download --local-dir`, there is no giant Git LFS shadow copy, but you still need enough space for both the source safetensors and the repacked experts.

| Phase | Cumulative Disk Used | Notes |
|-------|---------------------|-------|
| Download MLX 4-bit model | ~210 GB | Source safetensors files in `~/qwen35-397b-4bit` |
| After `extract_weights.py` | ~216 GB | Adds `model_weights.bin` (~5.5 GB) |
| After `repack_experts.py` | ~425 GB | 210GB source + 209GB packed experts + extracted weights |
| After deleting source model | **~215 GB** | Final footprint |

**You need ~430GB free to complete the setup comfortably.** Plan your cleanup steps. On a 1TB drive, this still means most of your disk needs to be empty.

**Important:** `hf download --local-dir` writes files directly into your target directory. The `.cache/huggingface/` folder created there is only resume/update metadata, not a second 210GB copy of the model.

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
```

Python dependencies are installed from the repo's `requirements.txt` after cloning in Phase 1.

---

## Phase 1: Clone Flash-MoE

```bash
cd ~
git clone https://github.com/danveloper/flash-moe.git
cd flash-moe
```

### Create the Python Environment

All Python dependencies live in `requirements.txt`, including the `hf` CLI used for model download.

```bash
cd ~/flash-moe
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Recommended in mainland China: use the Hugging Face mirror
export HF_ENDPOINT=https://hf-mirror.com

# Verify the CLI is available
hf --version
```

If you open a new terminal later, run `source ~/flash-moe/.venv/bin/activate` again before using `hf` or the Python helper scripts.

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
source ~/flash-moe/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
hf download mlx-community/Qwen3.5-397B-A17B-4bit --local-dir ~/qwen35-397b-4bit
```

`hf download --local-dir` writes the model directly into `~/qwen35-397b-4bit` and keeps only a small `.cache/huggingface/` metadata directory there for resume/update tracking. It does **not** create a second Git LFS checkout.

**Download speed:** Depends heavily on your network and mirror. The good news is that `hf download` is resumable — if the connection drops, just run the same command again.

### Common Download Issues

**`hf: command not found`:**
You are probably in a new shell. Re-activate the repo venv:

```bash
source ~/flash-moe/.venv/bin/activate
hf --version
```

**Download interrupted or partially completed:**
Re-run the exact same command. `hf download --local-dir` will only fetch missing or stale files.

```bash
cd ~
source ~/flash-moe/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
hf download mlx-community/Qwen3.5-397B-A17B-4bit --local-dir ~/qwen35-397b-4bit
```

**How to check that all weight shards are present:**

```bash
cd ~/qwen35-397b-4bit
ls model-*.safetensors | wc -l
# Should print 46
```

**How to verify file integrity:**

```bash
# All safetensors should be ~4.6-4.9 GB except the last one (~1.8 GB)
ls -lh *.safetensors | awk '{print $5, $9}' | sort -n | head -5
```

If any file (except model-00046-of-00046) is significantly smaller than ~4.6 GB, it's truncated. Delete it and re-fetch:

```bash
rm model-000XX-of-00046.safetensors
source ~/flash-moe/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
hf download mlx-community/Qwen3.5-397B-A17B-4bit model-000XX-of-00046.safetensors \
  --local-dir ~/qwen35-397b-4bit
```

---

## Phase 4: Export Vocab and Tokenizer

The repo does not ship `vocab.bin` or `tokenizer.bin`. You must generate them from the model's `tokenizer.json`. The repo includes `export_vocab.py` and `export_tokenizer.py` but they may not exist in all versions. If missing, see the "Manual Vocab Export" section below.

### Export tokenizer.bin

```bash
cd ~/flash-moe/metal_infer
python export_tokenizer.py ~/qwen35-397b-4bit/tokenizer.json
```

Expected output: `tokenizer.bin` (~7.8 MB, 248044 vocab, 247587 merges)

### Export vocab.bin (with Byte-Level BPE Decoding)

**Important:** The naive vocab export produces garbled output — `Ġ` instead of spaces, `Ċ` instead of newlines. This is because Qwen3.5 uses GPT-style byte-level BPE encoding where printable Unicode characters map to raw bytes.

**Use this script instead** (creates a properly decoded vocab.bin):

```bash
cd ~/flash-moe/metal_infer
python export_vocab.py --model ~/qwen35-397b-4bit
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
python extract_weights.py --model ~/qwen35-397b-4bit
```

This creates:
- `model_weights.bin` (~5.5 GB) — all non-expert weights, mmap'd at runtime
- `model_weights.json` — tensor manifest

Takes ~4 seconds.

**If this fails with `MemoryError` or `FileNotFoundError`:** Your safetensors files are incomplete or missing. Re-check the shard count and file sizes from Phase 3, then re-run `hf download`.

---

## Phase 6: Repack Expert Weights (~209GB)

**Check disk space first:**

```bash
df -h /
# You need ~209 GB free
```

```bash
cd ~/flash-moe/metal_infer
python ../repack_experts.py --model ~/qwen35-397b-4bit
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
| `MemoryError` in extract_weights | Safetensors files are incomplete or missing | Re-run `hf download`, then re-check shard count and file sizes |
| `OSError: Short read` in repack | Truncated safetensors file | Check file sizes, delete the bad shard, and re-run `hf download` for that file |
| `Ġ` and `Ċ` in output | vocab.bin missing byte-level BPE decode | Rebuild vocab.bin with the BPE decode script |
| `<unk>` at start of responses | Thinking tokens not in vocab | Cosmetic only; model output is correct |
| `hf: command not found` | `.venv` not activated or requirements not installed | `source ~/flash-moe/.venv/bin/activate` then `python -m pip install -r requirements.txt` |
| `hf download` stops midway | Network or mirror interruption | Re-run the same `hf download --local-dir ...` command; it resumes |
| Chat says "Server not running" | infer --serve not started | Start `./infer --model PATH --serve 8000` first |
| `./chat --model` unrecognized | chat doesn't take --model | Chat connects to the server; configure model on infer |
