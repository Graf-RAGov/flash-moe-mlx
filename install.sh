#!/usr/bin/env bash
set -euo pipefail

BLUE=$'\033[1;34m'
GREEN=$'\033[1;32m'
YELLOW=$'\033[1;33m'
RED=$'\033[1;31m'
RESET=$'\033[0m'

log_info() {
  printf '%s[INFO]%s %s\n' "$BLUE" "$RESET" "$*"
}

log_ok() {
  printf '%s[OK]%s %s\n' "$GREEN" "$RESET" "$*"
}

log_warn() {
  printf '%s[WARN]%s %s\n' "$YELLOW" "$RESET" "$*"
}

log_error() {
  printf '%s[ERROR]%s %s\n' "$RED" "$RESET" "$*" >&2
}

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

Automates the INSTALL.md flow:
  1. Create/update .venv and install requirements.txt
  2. Build metal_infer/infer and metal_infer/chat
  3. Download the model with hf download
  4. Export tokenizer/vocab, extract weights, repack experts
  5. Create the packed_experts symlink
  6. Run a short smoke test

Options:
  --repo-dir PATH        flash-moe repo directory (default: script directory)
  --model-dir PATH       model directory (default: ~/qwen35-397b-4bit)
  --skip-build           skip make / make chat
  --skip-download        skip hf download and reuse an existing model dir
  --skip-smoke-test      skip the final ./infer smoke test
  --cleanup              after success, prompt before deleting source safetensors
  -h, --help             show this help

Environment:
  HF_ENDPOINT            Hugging Face endpoint (default: https://hf-mirror.com)
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log_error "Missing required command: $1"
    exit 1
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    log_error "Required file not found: $1"
    exit 1
  fi
}

ensure_symlink() {
  local link_path="$1"
  local target_path="$2"

  if [[ -d "$link_path" && ! -L "$link_path" ]]; then
    log_error "$link_path already exists as a directory; move it aside before rerunning."
    exit 1
  fi

  rm -f "$link_path"
  ln -s "$target_path" "$link_path"
}

confirm_cleanup() {
  local model_dir="$1"

  if [[ ! -t 0 ]]; then
    log_error "--cleanup requires an interactive terminal."
    exit 1
  fi

  log_warn "Cleanup will delete all source files under $model_dir except packed_experts."
  read -r -p "Continue with cleanup? [y/N] " reply
  if [[ ! "$reply" =~ ^[Yy]$ ]]; then
    log_info "Cleanup skipped."
    return
  fi

  find "$model_dir" -maxdepth 1 ! -name packed_experts ! -name . -exec rm -rf {} +
  log_ok "Source model files removed from $model_dir"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$SCRIPT_DIR"
MODEL_DIR="$HOME/qwen35-397b-4bit"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SKIP_BUILD=0
SKIP_DOWNLOAD=0
RUN_SMOKE_TEST=1
DO_CLEANUP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      if [[ $# -lt 2 ]]; then
        log_error "--repo-dir requires a path argument"
        exit 1
      fi
      REPO_DIR="$2"
      shift 2
      ;;
    --model-dir)
      if [[ $# -lt 2 ]]; then
        log_error "--model-dir requires a path argument"
        exit 1
      fi
      MODEL_DIR="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    --skip-smoke-test)
      RUN_SMOKE_TEST=0
      shift
      ;;
    --cleanup)
      DO_CLEANUP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      log_error "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

REPO_DIR="$(cd "$REPO_DIR" && pwd -P)"
MODEL_DIR="${MODEL_DIR/#\~/$HOME}"
VENV_DIR="$REPO_DIR/flash-moe-env"
METAL_DIR="$REPO_DIR/metal_infer"
MODEL_REPO="mlx-community/Qwen3.5-397B-A17B-4bit"

require_cmd python3
require_cmd make
require_cmd xcode-select

if [[ "$(uname -s)" != "Darwin" ]]; then
  log_error "This installer is intended for macOS."
  exit 1
fi

if ! xcode-select -p >/dev/null 2>&1; then
  log_error "Xcode command line tools not found. Run: xcode-select --install"
  exit 1
fi

require_file "$REPO_DIR/requirements.txt"
require_file "$METAL_DIR/Makefile"
require_file "$METAL_DIR/export_tokenizer.py"
require_file "$METAL_DIR/export_vocab.py"
require_file "$METAL_DIR/extract_weights.py"
require_file "$REPO_DIR/repack_experts.py"

free_kb="$(df -Pk "$HOME" | awk 'NR==2 {print $4}')"
recommended_kb=$((430 * 1024 * 1024))
if (( free_kb < recommended_kb )); then
  log_warn "Less than ~430GB is free on $HOME. The install may fail during repack."
fi

log_info "Using repo dir: $REPO_DIR"
log_info "Using model dir: $MODEL_DIR"
log_info "Using HF endpoint: $HF_ENDPOINT"

log_info "Creating/updating virtual environment"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

log_info "Installing Python dependencies from requirements.txt"
python -m pip install --upgrade pip
python -m pip install -r "$REPO_DIR/requirements.txt"
export HF_ENDPOINT
require_cmd hf
log_ok "hf CLI ready: $(hf --version | head -n 1)"

if (( SKIP_BUILD == 0 )); then
  log_info "Building metal_infer binaries"
  make -C "$METAL_DIR"
  make -C "$METAL_DIR" chat
  log_ok "Build completed"
else
  log_warn "Skipping build"
fi

if (( SKIP_DOWNLOAD == 0 )); then
  mkdir -p "$MODEL_DIR"
  log_info "Downloading model with hf download"
  hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
  log_ok "Model download finished"
else
  log_warn "Skipping download"
fi

require_file "$MODEL_DIR/tokenizer.json"
require_file "$MODEL_DIR/model.safetensors.index.json"

shard_count="$(find "$MODEL_DIR" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
if [[ "$shard_count" != "46" ]]; then
  log_warn "Expected 46 safetensors shards, found $shard_count"
fi

log_info "Exporting tokenizer.bin and vocab.bin"
(
  cd "$METAL_DIR"
  python export_tokenizer.py "$MODEL_DIR/tokenizer.json"
  python export_vocab.py --model "$MODEL_DIR"
)

log_info "Extracting non-expert weights"
(
  cd "$METAL_DIR"
  python extract_weights.py --model "$MODEL_DIR"
)

log_info "Repacking expert weights"
(
  cd "$METAL_DIR"
  python ../repack_experts.py --model "$MODEL_DIR"
)

ensure_symlink "$METAL_DIR/packed_experts" "$MODEL_DIR/packed_experts"
log_ok "packed_experts symlink updated"

if (( RUN_SMOKE_TEST == 1 )); then
  log_info "Running smoke test"
  (
    cd "$METAL_DIR"
    ./infer --model "$MODEL_DIR" --prompt "Hello" --tokens 20 --timing
  )
  log_ok "Smoke test completed"
else
  log_warn "Skipping smoke test"
fi

if (( DO_CLEANUP == 1 )); then
  confirm_cleanup "$MODEL_DIR"
else
  log_info "Cleanup not requested. See INSTALL.md Phase 8 if you want to reclaim space later."
fi

log_ok "Flash-MoE setup flow completed"
