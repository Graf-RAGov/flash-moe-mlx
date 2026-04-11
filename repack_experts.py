#!/usr/bin/env python3
"""Repack expert weights from scattered safetensors into contiguous per-layer binary files.

Creates one binary file per layer: packed_experts/layer_XX.bin
Each file = N_EXPERTS x EXPERT_SIZE bytes

Within each expert block, 9 components packed in fixed order:
  gate_proj.weight, gate_proj.scales, gate_proj.biases,
  up_proj.weight,   up_proj.scales,   up_proj.biases,
  down_proj.weight,  down_proj.scales,  down_proj.biases

Usage:
    python repack_experts.py --model ~/qwen35-397b-4bit
    python repack_experts.py --model ~/qwen3-coder-480b-4bit --profile qwen3-coder
    python repack_experts.py --model ~/qwen35-397b-4bit --layers 0-4
    python repack_experts.py --model ~/qwen35-397b-4bit --dry-run
    python repack_experts.py --model ~/qwen35-397b-4bit --verify-only 0
    python repack_experts.py --model ~/qwen35-397b-4bit --validate-model
"""

import argparse
import json
import os
import re
import struct
import time
import sys

# ============================================================================
# Model profiles
# ============================================================================

MODEL_PROFILES = {
    "qwen35-397b": {
        "components": [
            {"name": "gate_proj.weight",  "offset": 0,       "size": 2097152, "dtype": "U32", "shape": [1024, 512]},
            {"name": "gate_proj.scales",  "offset": 2097152,  "size": 131072,  "dtype": "BF16", "shape": [1024, 64]},
            {"name": "gate_proj.biases",  "offset": 2228224,  "size": 131072,  "dtype": "BF16", "shape": [1024, 64]},
            {"name": "up_proj.weight",    "offset": 2359296,  "size": 2097152, "dtype": "U32", "shape": [1024, 512]},
            {"name": "up_proj.scales",    "offset": 4456448,  "size": 131072,  "dtype": "BF16", "shape": [1024, 64]},
            {"name": "up_proj.biases",    "offset": 4587520,  "size": 131072,  "dtype": "BF16", "shape": [1024, 64]},
            {"name": "down_proj.weight",  "offset": 4718592,  "size": 2097152, "dtype": "U32", "shape": [4096, 128]},
            {"name": "down_proj.scales",  "offset": 6815744,  "size": 131072,  "dtype": "BF16", "shape": [4096, 16]},
            {"name": "down_proj.biases",  "offset": 6946816,  "size": 131072,  "dtype": "BF16", "shape": [4096, 16]},
        ],
        "expert_size": 7077888,
        "num_experts": 512,
        "num_layers": 60,
        "weight_prefix": "language_model.",
        "supported_config": {
            "hidden_size": 4096,
            "num_hidden_layers": 60,
            "num_experts": 512,
            "num_experts_per_tok": 10,
        },
        "supported_shard_count": 46,
        "supported_bits": 4,
        # regex: group(1)=layer, group(2)=proj, group(3)=part (weight/scales/biases)
        "expert_key_pattern": r"language_model\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)\.(weight|scales|biases)",
        "expert_key_type": "individual",  # one tensor per expert
    },
    "qwen3-coder": {
        "components": [
            {"name": "gate_proj.weight",  "offset": 0,        "size": 7864320,  "dtype": "U32", "shape": [2560, 768]},
            {"name": "gate_proj.scales",  "offset": 7864320,   "size": 491520,   "dtype": "BF16", "shape": [2560, 96]},
            {"name": "gate_proj.biases",  "offset": 8355840,   "size": 491520,   "dtype": "BF16", "shape": [2560, 96]},
            {"name": "up_proj.weight",    "offset": 8847360,   "size": 7864320,  "dtype": "U32", "shape": [2560, 768]},
            {"name": "up_proj.scales",    "offset": 16711680,  "size": 491520,   "dtype": "BF16", "shape": [2560, 96]},
            {"name": "up_proj.biases",    "offset": 17203200,  "size": 491520,   "dtype": "BF16", "shape": [2560, 96]},
            {"name": "down_proj.weight",  "offset": 17694720,  "size": 7864320,  "dtype": "U32", "shape": [6144, 320]},
            {"name": "down_proj.scales",  "offset": 25559040,  "size": 491520,   "dtype": "BF16", "shape": [6144, 40]},
            {"name": "down_proj.biases",  "offset": 26050560,  "size": 491520,   "dtype": "BF16", "shape": [6144, 40]},
        ],
        "expert_size": 26542080,
        "num_experts": 160,
        "num_layers": 62,
        "weight_prefix": "",
        "supported_config": {
            "hidden_size": 6144,
            "num_hidden_layers": 62,
            "num_experts": 160,
            "num_experts_per_tok": 8,
        },
        "supported_shard_count": None,  # not validated
        "supported_bits": 4,
        # regex: group(1)=layer, group(2)=proj, group(3)=part (weight/scales/biases)
        "expert_key_pattern": r"model\.layers\.(\d+)\.mlp\.switch_mlp\.(\w+)\.(weight|scales|biases)",
        "expert_key_type": "packed",  # all experts in one tensor [N, ...]
    },
}

# Default profile
ACTIVE_PROFILE = None
COMPONENTS = None
EXPERT_SIZE = None
NUM_EXPERTS = None
NUM_LAYERS = None
LAYER_SIZE = None
WEIGHT_PREFIX = None
SUPPORTED_MODEL_CONFIG = None
SUPPORTED_SHARD_COUNT = None
SUPPORTED_BITS = None
EXPERT_KEY_PATTERN = None
EXPERT_KEY_TYPE = None


def set_profile(name):
    """Set the active model profile."""
    global ACTIVE_PROFILE, COMPONENTS, EXPERT_SIZE, NUM_EXPERTS, NUM_LAYERS
    global LAYER_SIZE, WEIGHT_PREFIX, SUPPORTED_MODEL_CONFIG
    global SUPPORTED_SHARD_COUNT, SUPPORTED_BITS
    global EXPERT_KEY_PATTERN, EXPERT_KEY_TYPE

    if name not in MODEL_PROFILES:
        print(f"ERROR: unknown model profile '{name}'. Available: {', '.join(MODEL_PROFILES.keys())}", file=sys.stderr)
        sys.exit(1)

    p = MODEL_PROFILES[name]
    ACTIVE_PROFILE = name
    COMPONENTS = p["components"]
    EXPERT_SIZE = p["expert_size"]
    NUM_EXPERTS = p["num_experts"]
    NUM_LAYERS = p["num_layers"]
    LAYER_SIZE = NUM_EXPERTS * EXPERT_SIZE
    WEIGHT_PREFIX = p["weight_prefix"]
    SUPPORTED_MODEL_CONFIG = p["supported_config"]
    SUPPORTED_SHARD_COUNT = p["supported_shard_count"]
    SUPPORTED_BITS = p["supported_bits"]
    EXPERT_KEY_PATTERN = p["expert_key_pattern"]
    EXPERT_KEY_TYPE = p["expert_key_type"]


def parse_layers(spec):
    """Parse layer specification like '0-4' or '0,5,10' or 'all'."""
    if spec is None or spec == 'all':
        return list(range(NUM_LAYERS))
    layers = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def load_index(index_path):
    """Load expert_index.json and return expert_reads dict."""
    with open(index_path) as f:
        idx = json.load(f)
    return idx['expert_reads']


def generate_expert_index(model_path):
    """Generate expert_reads from safetensors headers for packed-expert models.

    For models like Qwen3-Coder where all experts are stored in a single tensor
    per component per layer (shape [N_experts, ...]), we scan safetensors headers
    to build the same index structure as pre-built expert_index.json.
    """
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_index = json.load(f)
    weight_map = weight_index["weight_map"]

    pattern = re.compile(EXPERT_KEY_PATTERN)
    file_headers = {}
    expert_reads = {}

    if EXPERT_KEY_TYPE == "packed":
        # Packed format: model.layers.X.mlp.switch_mlp.{proj}.{part}
        # Each tensor has shape [N_experts, ...], all experts contiguous
        for key in sorted(weight_map.keys()):
            m = pattern.match(key)
            if not m:
                continue
            layer_idx, proj_name, part_name = m.groups()
            comp_name = f"{proj_name}.{part_name}"
            shard_file = weight_map[key]

            if layer_idx not in expert_reads:
                expert_reads[layer_idx] = {}

            if shard_file not in file_headers:
                shard_path = os.path.join(model_path, shard_file)
                with open(shard_path, "rb") as f:
                    header_size = struct.unpack("<Q", f.read(8))[0]
                    header = json.loads(f.read(header_size))
                file_headers[shard_file] = (header, header_size)

            header, h_size = file_headers[shard_file]
            tensor_info = header[key]
            shape = tensor_info["shape"]
            data_start = tensor_info["data_offsets"][0] + 8 + h_size
            data_end = tensor_info["data_offsets"][1] + 8 + h_size
            total_size = data_end - data_start
            num_experts = shape[0]
            expert_size = total_size // num_experts

            expert_reads[layer_idx][comp_name] = {
                "file": shard_file,
                "abs_offset": data_start,
                "expert_stride": expert_size,
                "expert_size": expert_size,
                "total_size": total_size,
                "shape": shape,
            }
    else:
        # Individual format: each expert is a separate tensor
        # Pattern captures layer, expert_idx, proj, part
        for key in sorted(weight_map.keys()):
            m = pattern.match(key)
            if not m:
                continue
            layer_idx, expert_idx_str, proj_name, part_name = m.groups()
            expert_idx = int(expert_idx_str)
            comp_name = f"{proj_name}.{part_name}"
            shard_file = weight_map[key]

            if expert_idx != 0:
                continue  # only need expert 0 to determine strides

            if layer_idx not in expert_reads:
                expert_reads[layer_idx] = {}

            if shard_file not in file_headers:
                shard_path = os.path.join(model_path, shard_file)
                with open(shard_path, "rb") as f:
                    header_size = struct.unpack("<Q", f.read(8))[0]
                    header = json.loads(f.read(header_size))
                file_headers[shard_file] = (header, header_size)

            header, h_size = file_headers[shard_file]
            tensor_info = header[key]
            shape = tensor_info["shape"]
            data_start = tensor_info["data_offsets"][0] + 8 + h_size
            data_end = tensor_info["data_offsets"][1] + 8 + h_size
            expert_size = data_end - data_start

            expert_reads[layer_idx][comp_name] = {
                "file": shard_file,
                "abs_offset": data_start,
                "expert_stride": expert_size,
                "expert_size": expert_size,
                "total_size": expert_size * NUM_EXPERTS,
                "shape": [NUM_EXPERTS] + shape,
            }

    print(f"Generated expert index: {len(expert_reads)} layers, "
          f"{len(next(iter(expert_reads.values())))} components/layer")
    return expert_reads


def load_model_metadata(model_path):
    """Load config/index metadata for user-facing compatibility checks."""
    config_path = os.path.join(model_path, "config.json")
    index_path = os.path.join(model_path, "model.safetensors.index.json")

    metadata = {
        "config_path": config_path,
        "index_path": index_path,
        "config": {},
        "weight_index": {},
        "bits": None,
        "group_size": None,
        "hidden_size": None,
        "num_hidden_layers": None,
        "num_experts": None,
        "num_experts_per_tok": None,
        "shard_count": None,
        "total_parameters": None,
    }

    if os.path.isfile(config_path):
        with open(config_path) as f:
            metadata["config"] = json.load(f)
        quant = metadata["config"].get("quantization_config") or metadata["config"].get("quantization") or {}
        metadata["bits"] = quant.get("bits")
        metadata["group_size"] = quant.get("group_size")
        # Multimodal models (e.g. Qwen3.5) nest text params under text_config
        text_cfg = metadata["config"].get("text_config", {})
        for key in SUPPORTED_MODEL_CONFIG:
            metadata[key] = metadata["config"].get(key) or text_cfg.get(key)

    if os.path.isfile(index_path):
        with open(index_path) as f:
            metadata["weight_index"] = json.load(f)
        shard_files = sorted(set(metadata["weight_index"].get("weight_map", {}).values()))
        metadata["shard_count"] = len(shard_files)
        metadata["total_parameters"] = metadata["weight_index"].get("metadata", {}).get("total_parameters")

    return metadata


def validate_supported_model(model_path, expert_reads):
    """Fail fast with a clear error when the model directory is incompatible."""
    metadata = load_model_metadata(model_path)
    problems = []

    for key, expected in SUPPORTED_MODEL_CONFIG.items():
        actual = metadata.get(key)
        if actual != expected:
            problems.append(f"{key}={actual!r} (expected {expected})")

    if metadata["bits"] != SUPPORTED_BITS:
        problems.append(f"quantization bits={metadata['bits']!r} (expected {SUPPORTED_BITS})")

    if SUPPORTED_SHARD_COUNT is not None and metadata["shard_count"] != SUPPORTED_SHARD_COUNT:
        problems.append(f"shard_count={metadata['shard_count']!r} (expected {SUPPORTED_SHARD_COUNT})")

    missing_files = []
    if expert_reads is not None:
        expected_files = sorted({
            info["file"]
            for layer_info in expert_reads.values()
            for info in layer_info.values()
        })
        missing_files = [
            fname for fname in expected_files
            if not os.path.isfile(os.path.join(model_path, fname))
        ]
        if missing_files:
            problems.append(
                f"missing {len(missing_files)} shard(s) referenced by expert_index.json"
            )

    if problems:
        detected = [
            f"hidden_size={metadata['hidden_size']!r}",
            f"num_hidden_layers={metadata['num_hidden_layers']!r}",
            f"num_experts={metadata['num_experts']!r}",
            f"num_experts_per_tok={metadata['num_experts_per_tok']!r}",
            f"bits={metadata['bits']!r}",
            f"shards={metadata['shard_count']!r}",
        ]
        if metadata["total_parameters"] is not None:
            detected.append(f"total_parameters={metadata['total_parameters']}")

        print(f"ERROR: unsupported model directory: {model_path}", file=sys.stderr)
        print(f"Active profile: {ACTIVE_PROFILE}", file=sys.stderr)
        print(f"Detected metadata: {', '.join(detected)}", file=sys.stderr)
        print("Compatibility check failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)

        if SUPPORTED_SHARD_COUNT is not None and metadata["shard_count"] not in (None, SUPPORTED_SHARD_COUNT):
            print(
                f"Hint: --model/--model-dir points at a different MLX model. "
                f"Check that --profile matches your model.",
                file=sys.stderr,
            )
        elif missing_files:
            preview = ", ".join(missing_files[:3])
            if len(missing_files) > 3:
                preview += ", ..."
            print(f"Missing files include: {preview}", file=sys.stderr)
            print(
                "Hint: the model download is incomplete. Re-run hf download to fetch the missing shards.",
                file=sys.stderr,
            )
        return False

    print(f"Model metadata verified (profile: {ACTIVE_PROFILE})")
    return True


def verify_component_sizes(expert_reads):
    """Verify that component sizes in the index match expected sizes."""
    expected = {c['name']: c['size'] for c in COMPONENTS}
    for layer_key, comps in expert_reads.items():
        for comp_name, info in comps.items():
            if comp_name not in expected:
                print(f"WARNING: unknown component {comp_name} in layer {layer_key}")
                continue
            if info['expert_size'] != expected[comp_name]:
                print(f"MISMATCH: layer {layer_key}, {comp_name}: "
                      f"index says {info['expert_size']}, expected {expected[comp_name]}")
                return False
    print("Component sizes verified: all match expected layout")
    return True


def open_source_files(expert_reads, model_path, layers):
    """Open all needed safetensors files, return {filename: fd}."""
    needed_files = set()
    for layer_idx in layers:
        layer_key = str(layer_idx)
        if layer_key not in expert_reads:
            print(f"WARNING: layer {layer_idx} not found in expert_reads")
            continue
        for info in expert_reads[layer_key].values():
            needed_files.add(info['file'])

    fds = {}
    for fname in sorted(needed_files):
        path = os.path.join(model_path, fname)
        fds[fname] = os.open(path, os.O_RDONLY)
    print(f"Opened {len(fds)} source safetensors files")
    return fds


def repack_layer(layer_idx, expert_reads, fds, output_dir, dry_run=False):
    """Repack all 512 experts for one layer into a contiguous binary file.

    Returns (bytes_written, elapsed_seconds).
    """
    layer_key = str(layer_idx)
    if layer_key not in expert_reads:
        print(f"  Layer {layer_idx}: NOT FOUND in index, skipping")
        return 0, 0.0

    layer_info = expert_reads[layer_key]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    if dry_run:
        # Just verify we can compute all offsets
        for expert_idx in range(NUM_EXPERTS):
            for comp in COMPONENTS:
                info = layer_info[comp['name']]
                src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
                dst_offset = expert_idx * EXPERT_SIZE + comp['offset']
        print(f"  Layer {layer_idx:2d}: DRY RUN OK — would write {LAYER_SIZE:,} bytes to {out_path}")
        return LAYER_SIZE, 0.0

    t0 = time.monotonic()

    # Pre-allocate output file with zeros
    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, LAYER_SIZE)

    bytes_written = 0

    # Build read plan: group reads by source file for better locality
    # Each entry: (src_fd, src_offset, dst_offset, size)
    read_plan = []
    for expert_idx in range(NUM_EXPERTS):
        for comp in COMPONENTS:
            info = layer_info[comp['name']]
            src_fd = fds[info['file']]
            src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
            dst_offset = expert_idx * EXPERT_SIZE + comp['offset']
            read_plan.append((src_fd, src_offset, dst_offset, comp['size']))

    # Sort by (src_fd, src_offset) for sequential read locality
    read_plan.sort(key=lambda x: (x[0], x[1]))

    # Execute reads and writes
    for src_fd, src_offset, dst_offset, size in read_plan:
        data = os.pread(src_fd, size, src_offset)
        if len(data) != size:
            raise IOError(f"Short read: expected {size}, got {len(data)} "
                          f"at offset {src_offset}")
        os.pwrite(fd_out, data, dst_offset)
        bytes_written += size

    os.close(fd_out)
    elapsed = time.monotonic() - t0

    return bytes_written, elapsed


def verify_layer(layer_idx, expert_reads, fds, output_dir):
    """Read back several experts from packed file and compare to originals."""
    layer_key = str(layer_idx)
    layer_info = expert_reads[layer_key]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    if not os.path.exists(out_path):
        print(f"  Layer {layer_idx}: packed file not found")
        return False

    fd_packed = os.open(out_path, os.O_RDONLY)

    # Spot-check: first, second, middle, last experts
    check_indices = sorted(set([0, 1, NUM_EXPERTS // 2, NUM_EXPERTS - 1]))

    mismatches = 0
    for expert_idx in check_indices:
        for comp in COMPONENTS:
            info = layer_info[comp['name']]
            src_fd = fds[info['file']]
            src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
            dst_offset = expert_idx * EXPERT_SIZE + comp['offset']

            original = os.pread(src_fd, comp['size'], src_offset)
            packed = os.pread(fd_packed, comp['size'], dst_offset)

            if original != packed:
                print(f"  MISMATCH: layer {layer_idx}, expert {expert_idx}, {comp['name']}")
                mismatches += 1

    os.close(fd_packed)

    indices_str = ", ".join(str(i) for i in check_indices)
    if mismatches == 0:
        print(f"  Layer {layer_idx}: verification PASSED (experts {indices_str})")
    else:
        print(f"  Layer {layer_idx}: verification FAILED ({mismatches} mismatches)")

    return mismatches == 0


def write_layout(output_dir):
    """Write layout.json describing the packed format."""
    layout = {
        "expert_size": EXPERT_SIZE,
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "components": COMPONENTS,
    }
    path = os.path.join(output_dir, "layout.json")
    with open(path, 'w') as f:
        json.dump(layout, f, indent=2)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Repack expert weights into contiguous per-layer binary files")
    default_index = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'expert_index.json')
    parser.add_argument('--model', required=True,
                        help='Path to model directory containing safetensors shards')
    parser.add_argument('--profile', default='qwen35-397b',
                        choices=list(MODEL_PROFILES.keys()),
                        help='Model profile (default: qwen35-397b)')
    parser.add_argument('--index', default=default_index,
                        help='Path to expert_index.json')
    parser.add_argument('--layers', default=None,
                        help='Layer spec: "all", "0-4", "0,5,10" (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Verify offsets without writing')
    parser.add_argument('--verify-only', type=int, default=None, metavar='LAYER',
                        help='Verify a specific layer against originals')
    parser.add_argument('--validate-model', action='store_true',
                        help='Check that --model matches the supported layout and exit')
    args = parser.parse_args()

    set_profile(args.profile)
    print(f"Model profile: {args.profile}")
    print(f"Expert size: {EXPERT_SIZE:,} bytes, Experts/layer: {NUM_EXPERTS}, Layers: {NUM_LAYERS}")

    model_path = os.path.abspath(os.path.expanduser(args.model))
    index_path = os.path.abspath(os.path.expanduser(args.index))

    if not os.path.isdir(model_path):
        print(f"ERROR: model directory not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # --validate-model: only check config.json metadata, no index needed
    if args.validate_model:
        if not validate_supported_model(model_path, expert_reads=None):
            sys.exit(1)
        return

    # Load or generate expert index
    use_auto_index = EXPERT_KEY_TYPE == "packed"
    if use_auto_index:
        print("Generating expert index from safetensors headers...")
        expert_reads = generate_expert_index(model_path)
    else:
        if not os.path.isfile(index_path):
            print(f"ERROR: expert index not found: {index_path}", file=sys.stderr)
            sys.exit(1)
        print("Loading expert index...")
        expert_reads = load_index(index_path)
        print(f"Index path: {index_path}")
    print(f"Model path: {model_path}")
    print(f"Layers in index: {len(expert_reads)}")

    # Verify component sizes
    if not verify_component_sizes(expert_reads):
        print("ABORTING: component size mismatch")
        sys.exit(1)

    if not validate_supported_model(model_path, expert_reads):
        sys.exit(1)

    output_dir = os.path.join(model_path, "packed_experts")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Determine which layers to process
    if args.verify_only is not None:
        layers = [args.verify_only]
    else:
        layers = parse_layers(args.layers)

    print(f"Layers to process: {layers[0]}-{layers[-1]} ({len(layers)} layers)")

    if not args.dry_run and args.verify_only is None:
        total_bytes = len(layers) * LAYER_SIZE
        print(f"Total data to write: {total_bytes / (1024**3):.1f} GB")

        # Check free disk space
        stat = os.statvfs(output_dir)
        free_bytes = stat.f_bavail * stat.f_frsize
        free_gb = free_bytes / (1024**3)
        needed_gb = total_bytes / (1024**3)
        print(f"Free disk space: {free_gb:.1f} GB, needed: {needed_gb:.1f} GB")
        if free_bytes < total_bytes:
            print(f"WARNING: Not enough free space! Need {needed_gb:.1f} GB but only {free_gb:.1f} GB free.")
            print(f"Hint: use --layers to process a subset, e.g. --layers 0-{int(free_gb / 3.63) - 1}")
            sys.exit(1)

    # Open source files
    fds = open_source_files(expert_reads, model_path, layers)

    if args.verify_only is not None:
        verify_layer(args.verify_only, expert_reads, fds, output_dir)
        for fd in fds.values():
            os.close(fd)
        return

    # Write layout.json
    write_layout(output_dir)

    # Repack each layer
    t_start = time.monotonic()
    total_written = 0

    for i, layer_idx in enumerate(layers):
        t_layer = time.monotonic()
        bytes_written, elapsed = repack_layer(
            layer_idx, expert_reads, fds, output_dir, dry_run=args.dry_run
        )
        total_written += bytes_written

        if not args.dry_run and bytes_written > 0:
            throughput = bytes_written / elapsed / (1024**3) if elapsed > 0 else float('inf')
            overall_elapsed = time.monotonic() - t_start
            overall_throughput = total_written / overall_elapsed / (1024**3) if overall_elapsed > 0 else 0
            eta = (len(layers) - i - 1) * (overall_elapsed / (i + 1))
            print(f"  Layer {layer_idx:2d}: {bytes_written/1024**3:.2f} GB in {elapsed:.1f}s "
                  f"({throughput:.1f} GB/s) | "
                  f"Total: {total_written/1024**3:.1f}/{len(layers)*LAYER_SIZE/1024**3:.1f} GB "
                  f"({overall_throughput:.1f} GB/s avg) | "
                  f"ETA: {eta:.0f}s")

            # Verify this layer immediately
            if not verify_layer(layer_idx, expert_reads, fds, output_dir):
                print(f"ABORTING: verification failed for layer {layer_idx}")
                sys.exit(1)

    # Close source files
    for fd in fds.values():
        os.close(fd)

    # Final summary
    total_elapsed = time.monotonic() - t_start
    if not args.dry_run and total_written > 0:
        print(f"\n{'='*60}")
        print(f"DONE: {total_written:,} bytes ({total_written/1024**3:.1f} GB) written")
        print(f"Time: {total_elapsed:.1f}s")
        print(f"Throughput: {total_written/total_elapsed/1024**3:.1f} GB/s")
        print(f"Output: {output_dir}")
    elif args.dry_run:
        print(f"\nDRY RUN complete: {len(layers)} layers validated")


if __name__ == '__main__':
    main()
