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
  --profile PROFILE      model profile: qwen35-397b or qwen3-coder
                         (default: auto-detect from model-dir, fallback qwen35-397b)
  --skip-build           skip make / make chat
  --skip-download        skip hf download and reuse an existing model dir
  --skip-smoke-test      skip the final ./infer smoke test
  --cleanup              after success, prompt before deleting source safetensors
  -h, --help             show this help

Supported models:
  qwen35-397b            Qwen3.5-397B-A17B-4bit (default)
  qwen3-coder            Qwen3-Coder-480B-A35B-Instruct-4bit

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

validate_model_dir() {
  local model_dir="$1"

  require_file "$model_dir/tokenizer.json"
  require_file "$model_dir/model.safetensors.index.json"

  log_info "Validating model metadata"
  python3 "$REPO_DIR/repack_experts.py" --model "$model_dir" --profile "$PROFILE" --validate-model >/dev/null
  log_ok "Model metadata verified (profile: $PROFILE)"
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
MODEL_DIR=""
PROFILE=""
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
    --profile)
      if [[ $# -lt 2 ]]; then
        log_error "--profile requires a value: qwen35-397b or qwen3-coder"
        exit 1
      fi
      PROFILE="$2"
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

# Auto-detect profile from model directory name if not explicitly set
auto_detect_profile() {
  local dir_name
  dir_name="$(basename "$1")"
  case "$dir_name" in
    *[Cc]oder*480[Bb]*)
      echo "qwen3-coder"
      ;;
    *)
      # Try config.json if available
      local config_file="$1/config.json"
      if [[ -f "$config_file" ]]; then
        local num_experts
        num_experts="$(python3 -c "import json; print(json.load(open('$config_file')).get('num_experts', 0))" 2>/dev/null || echo "0")"
        if [[ "$num_experts" == "160" ]]; then
          echo "qwen3-coder"
          return
        fi
      fi
      echo "qwen35-397b"
      ;;
  esac
}

if [[ -z "$PROFILE" && -n "$MODEL_DIR" ]]; then
  expanded_model_dir="${MODEL_DIR/#\~/$HOME}"
  PROFILE="$(auto_detect_profile "$expanded_model_dir")"
  log_info "Auto-detected profile: $PROFILE"
elif [[ -z "$PROFILE" ]]; then
  PROFILE="qwen35-397b"
fi

# Validate profile
case "$PROFILE" in
  qwen35-397b|qwen3-coder) ;;
  *)
    log_error "Unknown profile: $PROFILE (expected qwen35-397b or qwen3-coder)"
    exit 1
    ;;
esac

# Profile-derived settings
case "$PROFILE" in
  qwen35-397b)
    MODEL_REPO="mlx-community/Qwen3.5-397B-A17B-4bit"
    MAKE_MODEL_ARG=""
    [[ -z "$MODEL_DIR" ]] && MODEL_DIR="$HOME/qwen35-397b-4bit"
    EXPECTED_SHARD_COUNT=46
    ;;
  qwen3-coder)
    MODEL_REPO="mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit"
    MAKE_MODEL_ARG="MODEL=qwen3-coder"
    [[ -z "$MODEL_DIR" ]] && MODEL_DIR="$HOME/qwen3-coder-480b-4bit"
    EXPECTED_SHARD_COUNT=""  # not validated for this model
    ;;
esac

REPO_DIR="$(cd "$REPO_DIR" && pwd -P)"
MODEL_DIR="${MODEL_DIR/#\~/$HOME}"
VENV_DIR="$REPO_DIR/flash-moe-env"
METAL_DIR="$REPO_DIR/metal_infer"

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

if (( SKIP_DOWNLOAD == 1 )); then
  validate_model_dir "$MODEL_DIR"
fi

free_kb="$(df -Pk "$HOME" | awk 'NR==2 {print $4}')"
recommended_kb=$((430 * 1024 * 1024))
if (( free_kb < recommended_kb )); then
  log_warn "Less than ~430GB is free on $HOME. The install may fail during repack."
fi

log_info "Using repo dir: $REPO_DIR"
log_info "Using model dir: $MODEL_DIR"
log_info "Using profile: $PROFILE"
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
  log_info "Building metal_infer binaries (profile: $PROFILE)"
  make -C "$METAL_DIR" $MAKE_MODEL_ARG
  make -C "$METAL_DIR" $MAKE_MODEL_ARG chat
  log_ok "Build completed"
else
  log_warn "Skipping build"
fi

if (( SKIP_DOWNLOAD == 0 )); then
  mkdir -p "$MODEL_DIR"
  log_info "Downloading model with hf download"
  hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
  log_ok "Model download finished"
  validate_model_dir "$MODEL_DIR"
else
  log_warn "Skipping download"
fi

shard_count="$(find "$MODEL_DIR" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
if [[ -n "$EXPECTED_SHARD_COUNT" && "$shard_count" != "$EXPECTED_SHARD_COUNT" ]]; then
  log_warn "Expected $EXPECTED_SHARD_COUNT safetensors shards, found $shard_count"
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
  python extract_weights.py --model "$MODEL_DIR" --profile "$PROFILE"
)

log_info "Repacking expert weights"
(
  cd "$METAL_DIR"
  python ../repack_experts.py --model "$MODEL_DIR" --profile "$PROFILE"
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
