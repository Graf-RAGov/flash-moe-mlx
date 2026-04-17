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
    python repack_experts.py --model ~/qwen36-35b-a3b-8bit --profile qwen3.6-35b-a3b
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
    # ---------------------------------------------------------------------------
    # Qwen3.6-35B-A3B — MLX 8-bit affine, fused gate_up_proj + down_proj tensors
    # ---------------------------------------------------------------------------
    # Tensor layout on disk (mlx-community/Qwen3.6-35B-A3B-8bit):
    #
    #   model.language_model.layers.N.mlp.experts.gate_up_proj
    #       .weight  [256, 1024, 256]  uint32  (8-bit: 4 vals/uint32, last dim = 2048/8)
    #       .scales  [256, 1024, 32]   bf16    (group_size=64, last dim = 2048/64)
    #       .biases  [256, 1024, 32]   bf16
    #   model.language_model.layers.N.mlp.experts.down_proj
    #       .weight  [256, 2048, 64]   uint32  (8-bit: 512/8=64)
    #       .scales  [256, 2048, 8]    bf16    (512/64=8)
    #       .biases  [256, 2048, 8]    bf16
    #
    # gate_up_proj convention: first 512 rows = gate, last 512 rows = up
    # (i.e. fused along axis 1 of the per-expert slice, gate then up)
    #
    # Per-expert EXPERT_SIZE (shape-derived, do not hardcode globally):
    #   gate.weight   [512, 256] uint32  = 512*256*4 = 524288
    #   gate.scales   [512,  32] bf16    = 512* 32*2 =  32768
    #   gate.biases   [512,  32] bf16    =             32768
    #   up.weight     [512, 256] uint32  =             524288
    #   up.scales     [512,  32] bf16    =              32768
    #   up.biases     [512,  32] bf16    =              32768
    #   down.weight  [2048,  64] uint32  = 2048*64*4 = 524288
    #   down.scales  [2048,   8] bf16    = 2048* 8*2 =  32768
    #   down.biases  [2048,   8] bf16    =              32768
    #   TOTAL = 1769472 bytes/expert
    #
    # EXPERT_SIZE is computed at runtime from actual shapes to avoid hardcoding.
    # The values above are reference checkpoints verified against the spec.
    "qwen3.6-35b-a3b": {
        # Qwen3.6-35B-A3B (mlx-community/Qwen3.6-35B-A3B-8bit):
        # Packed 3D layout: each tensor is [num_experts, rows, cols] — one tensor
        # for all experts, same as Qwen3-Coder.  Keys use switch_mlp (not experts).
        #
        # Actual safetensors shapes (from shard headers):
        #   switch_mlp.gate_proj.weight [256, 512, 512] uint32  (8-bit: 4 vals/u32, hidden=2048/4=512)
        #   switch_mlp.gate_proj.scales [256, 512,  32] bf16
        #   switch_mlp.gate_proj.biases [256, 512,  32] bf16
        #   switch_mlp.up_proj.weight   [256, 512, 512] uint32
        #   switch_mlp.up_proj.scales   [256, 512,  32] bf16
        #   switch_mlp.up_proj.biases   [256, 512,  32] bf16
        #   switch_mlp.down_proj.weight [256, 2048, 128] uint32 (moe_inter=512/4=128)
        #   switch_mlp.down_proj.scales [256, 2048,   8] bf16
        #   switch_mlp.down_proj.biases [256, 2048,   8] bf16
        #
        # Per-expert sizes (total / 256):
        #   gate/up weight: 512*512*4 = 1,048,576   scales/biases: 512*32*2 = 32,768
        #   down weight:    2048*128*4 = 1,048,576  scales/biases: 2048*8*2 = 32,768
        #   EXPERT_SIZE = 3*1,048,576 + 6*32,768 = 3,342,336 bytes
        "components": [
            {"name": "gate_proj.weight", "offset": 0,         "size": 1048576, "dtype": "U32",  "shape": [256, 512, 512]},
            {"name": "gate_proj.scales", "offset": 1048576,   "size": 32768,   "dtype": "BF16", "shape": [256, 512, 32]},
            {"name": "gate_proj.biases", "offset": 1081344,   "size": 32768,   "dtype": "BF16", "shape": [256, 512, 32]},
            {"name": "up_proj.weight",   "offset": 1114112,   "size": 1048576, "dtype": "U32",  "shape": [256, 512, 512]},
            {"name": "up_proj.scales",   "offset": 2162688,   "size": 32768,   "dtype": "BF16", "shape": [256, 512, 32]},
            {"name": "up_proj.biases",   "offset": 2195456,   "size": 32768,   "dtype": "BF16", "shape": [256, 512, 32]},
            {"name": "down_proj.weight", "offset": 2228224,   "size": 1048576, "dtype": "U32",  "shape": [256, 2048, 128]},
            {"name": "down_proj.scales", "offset": 3276800,   "size": 32768,   "dtype": "BF16", "shape": [256, 2048, 8]},
            {"name": "down_proj.biases", "offset": 3309568,   "size": 32768,   "dtype": "BF16", "shape": [256, 2048, 8]},
        ],
        "expert_size": 3342336,
        "num_experts": 256,
        "num_layers": 40,
        "weight_prefix": "language_model.",
        "supported_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_experts": 256,
            "num_experts_per_tok": 8,
        },
        "supported_shard_count": None,   # 8 shards; set None to skip count check
        "supported_bits": 8,
        # Packed 3D layout: language_model.model.layers.N.mlp.switch_mlp.{proj}.{part}
        "expert_key_pattern": (
            r"language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\."
            r"(gate_proj|up_proj|down_proj)\.(weight|scales|biases)"
        ),
        "expert_key_type": "packed",
    },
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
    EXPERT_SIZE = p["expert_size"]       # None for "fused" profiles — set later
    NUM_EXPERTS = p["num_experts"]
    NUM_LAYERS = p["num_layers"]
    # LAYER_SIZE deferred for "fused" profiles where EXPERT_SIZE is shape-driven
    LAYER_SIZE = (NUM_EXPERTS * EXPERT_SIZE) if EXPERT_SIZE is not None else None
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


# ---------------------------------------------------------------------------
# MLX 8-bit size helpers
# ---------------------------------------------------------------------------
# MLX 8-bit affine quantization packs 4 uint8 values into one uint32.
# Safetensors stores uint32 data (dtype "I32" in MLX headers).
# So the byte size of a weight tensor with logical shape [rows, cols_uint8]
# stored as uint32 is: rows * ceil(cols_uint8 / 4) * 4.
# For shapes already expressed as [rows, cols/8] uint32 (as in the spec),
# each uint32 element = 4 bytes, so total = prod(shape) * 4.
#
# Scales and biases are bf16 (2 bytes each).

def _u32_bytes(shape):
    """Byte size of a tensor whose safetensors shape is already in uint32 units."""
    result = 4
    for s in shape:
        result *= s
    return result


def _bf16_bytes(shape):
    """Byte size of a bf16 tensor."""
    result = 2
    for s in shape:
        result *= s
    return result


def _derive_fused_expert_layout(gate_up_w_shape, gate_up_s_shape,
                                down_w_shape, down_s_shape):
    """Derive per-expert component layout from fused 3D tensor shapes.

    Parameters
    ----------
    gate_up_w_shape : list[int]
        Shape of gate_up_proj.weight in the safetensors header, e.g. [256, 1024, 256].
        Axes: [num_experts, 2*moe_inter_rows, hidden/8_as_u32].
    gate_up_s_shape : list[int]
        Shape of gate_up_proj.scales, e.g. [256, 1024, 32].
        Axes: [num_experts, 2*moe_inter_rows, hidden/group_size].
    down_w_shape : list[int]
        Shape of down_proj.weight, e.g. [256, 2048, 64].
        Axes: [num_experts, hidden_rows, moe_inter/8_as_u32].
    down_s_shape : list[int]
        Shape of down_proj.scales, e.g. [256, 2048, 8].
        Axes: [num_experts, hidden_rows, moe_inter/group_size].

    Returns
    -------
    components : list[dict]
        Ordered list of component dicts with name/offset/size/dtype/shape.
    expert_size : int
        Total bytes per expert.
    """
    assert len(gate_up_w_shape) == 3, (
        f"Expected gate_up_proj.weight to be 3D, got shape {gate_up_w_shape}"
    )
    assert len(down_w_shape) == 3, (
        f"Expected down_proj.weight to be 3D, got shape {down_w_shape}"
    )

    n_exp = gate_up_w_shape[0]
    fused_rows = gate_up_w_shape[1]   # 2 * moe_inter
    w_cols_u32 = gate_up_w_shape[2]   # hidden_size / 8 (uint32 units)

    assert fused_rows % 2 == 0, (
        f"gate_up_proj fused axis (dim 1) must be even (gate+up), got {fused_rows}"
    )
    half_rows = fused_rows // 2        # moe_inter

    s_cols = gate_up_s_shape[2]        # hidden_size / group_size

    down_rows = down_w_shape[1]        # hidden_size
    down_w_cols_u32 = down_w_shape[2]  # moe_inter / 8 (uint32 units)
    down_s_cols = down_s_shape[2]      # moe_inter / group_size

    # Per-expert sub-shapes
    gate_w_shape  = [half_rows, w_cols_u32]
    gate_s_shape  = [half_rows, s_cols]
    up_w_shape    = [half_rows, w_cols_u32]
    up_s_shape    = [half_rows, s_cols]
    down_w_shape_ = [down_rows, down_w_cols_u32]
    down_s_shape_ = [down_rows, down_s_cols]

    # Byte sizes per expert
    gate_w_sz  = _u32_bytes(gate_w_shape)
    gate_s_sz  = _bf16_bytes(gate_s_shape)
    gate_b_sz  = gate_s_sz               # biases same shape as scales
    up_w_sz    = _u32_bytes(up_w_shape)
    up_s_sz    = _bf16_bytes(up_s_shape)
    up_b_sz    = up_s_sz
    down_w_sz  = _u32_bytes(down_w_shape_)
    down_s_sz  = _bf16_bytes(down_s_shape_)
    down_b_sz  = down_s_sz

    # Reference check: validate against known spec values for Qwen3.6-35B-A3B
    # (hidden=2048, moe_inter=512, group_size=64).
    # These asserts fire only if actual shapes deviate from the documented spec.
    # Comment them out when porting to a different model.
    if n_exp == 256 and half_rows == 512 and down_rows == 2048:
        assert gate_w_sz == 524288, (
            f"gate.weight size mismatch: expected 524288 for Qwen3.6-35B-A3B, got {gate_w_sz}"
        )
        assert gate_s_sz == 32768, (
            f"gate.scales size mismatch: expected 32768 for Qwen3.6-35B-A3B, got {gate_s_sz}"
        )
        assert down_w_sz == 524288, (
            f"down.weight size mismatch: expected 524288 for Qwen3.6-35B-A3B, got {down_w_sz}"
        )
        assert down_s_sz == 32768, (
            f"down.scales size mismatch: expected 32768 for Qwen3.6-35B-A3B, got {down_s_sz}"
        )

    components = []
    offset = 0
    for name, sz, dtype, shape in [
        ("gate_proj.weight", gate_w_sz,  "U32",  [n_exp] + gate_w_shape),
        ("gate_proj.scales", gate_s_sz,  "BF16", [n_exp] + gate_s_shape),
        ("gate_proj.biases", gate_b_sz,  "BF16", [n_exp] + gate_s_shape),
        ("up_proj.weight",   up_w_sz,    "U32",  [n_exp] + up_w_shape),
        ("up_proj.scales",   up_s_sz,    "BF16", [n_exp] + up_s_shape),
        ("up_proj.biases",   up_b_sz,    "BF16", [n_exp] + up_s_shape),
        ("down_proj.weight", down_w_sz,  "U32",  [n_exp] + down_w_shape_),
        ("down_proj.scales", down_s_sz,  "BF16", [n_exp] + down_s_shape_),
        ("down_proj.biases", down_b_sz,  "BF16", [n_exp] + down_s_shape_),
    ]:
        components.append({
            "name": name,
            "offset": offset,
            "size": sz,
            "dtype": dtype,
            "shape": shape,
        })
        offset += sz

    expert_size = offset
    return components, expert_size


def generate_expert_index_fused(model_path):
    """Generate expert_reads for models with fused 3D expert tensors (8-bit MLX).

    Handles the Qwen3.6-35B-A3B layout where:
      - model.language_model.layers.N.mlp.experts.gate_up_proj.{weight,scales,biases}
        are 3D tensors [num_experts, 2*moe_inter, hidden/8_u32]
      - model.language_model.layers.N.mlp.experts.down_proj.{weight,scales,biases}
        are 3D tensors [num_experts, hidden, moe_inter/8_u32]

    The fused gate_up_proj is split into gate (rows 0:moe_inter) and up (rows moe_inter:)
    for kernel compatibility. All shapes and byte sizes are derived from the safetensors
    header — nothing is hardcoded.

    Returns
    -------
    expert_reads : dict[str, dict]
        Layer-keyed dict, each value is a component dict matching the schema used
        by the 4-bit path (file, abs_offset, expert_stride, expert_size, total_size,
        shape). Also sets the global EXPERT_SIZE, LAYER_SIZE, and COMPONENTS.
    """
    global EXPERT_SIZE, LAYER_SIZE, COMPONENTS

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    assert os.path.isfile(index_path), (
        f"model.safetensors.index.json not found at {index_path}"
    )
    with open(index_path) as f:
        weight_index = json.load(f)
    weight_map = weight_index["weight_map"]

    pattern = re.compile(EXPERT_KEY_PATTERN)
    file_headers = {}
    # raw_index[layer_str][proj_name][part_name] = {file, shape, data_offsets}
    raw_index = {}

    for key in sorted(weight_map.keys()):
        m = pattern.match(key)
        if not m:
            continue
        layer_idx, proj_name, part_name = m.groups()
        shard_file = weight_map[key]

        if layer_idx not in raw_index:
            raw_index[layer_idx] = {}
        if proj_name not in raw_index[layer_idx]:
            raw_index[layer_idx][proj_name] = {}

        if shard_file not in file_headers:
            shard_path = os.path.join(model_path, shard_file)
            with open(shard_path, "rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_size))
            file_headers[shard_file] = (header, header_size)

        header, h_size = file_headers[shard_file]
        tensor_info = header[key]
        abs_start = tensor_info["data_offsets"][0] + 8 + h_size
        abs_end   = tensor_info["data_offsets"][1] + 8 + h_size
        raw_index[layer_idx][proj_name][part_name] = {
            "file": shard_file,
            "abs_start": abs_start,
            "abs_end": abs_end,
            "shape": tensor_info["shape"],
        }

    if not raw_index:
        raise ValueError(
            f"No expert tensors matched pattern {EXPERT_KEY_PATTERN!r} in "
            f"{index_path}. Check --profile and weight_map keys."
        )

    # -----------------------------------------------------------------------
    # Derive layout from layer 0 (all layers share the same shape)
    # -----------------------------------------------------------------------
    layer0 = raw_index.get("0")
    assert layer0 is not None, "Layer 0 not found in weight_map — cannot derive layout"

    for proj in ("gate_up_proj", "down_proj"):
        assert proj in layer0, (
            f"Missing '{proj}' in layer 0 expert tensors. "
            f"Found: {list(layer0.keys())}"
        )
        for part in ("weight", "scales", "biases"):
            assert part in layer0[proj], (
                f"Missing '{proj}.{part}' in layer 0 expert tensors."
            )

    gu_w_shape = layer0["gate_up_proj"]["weight"]["shape"]
    gu_s_shape = layer0["gate_up_proj"]["scales"]["shape"]
    dn_w_shape = layer0["down_proj"]["weight"]["shape"]
    dn_s_shape = layer0["down_proj"]["scales"]["shape"]

    print(f"  gate_up_proj.weight shape: {gu_w_shape}")
    print(f"  gate_up_proj.scales shape: {gu_s_shape}")
    print(f"  down_proj.weight    shape: {dn_w_shape}")
    print(f"  down_proj.scales    shape: {dn_s_shape}")

    components, expert_size = _derive_fused_expert_layout(
        gu_w_shape, gu_s_shape, dn_w_shape, dn_s_shape
    )

    # Update globals now that we have the real shapes
    EXPERT_SIZE = expert_size
    LAYER_SIZE  = NUM_EXPERTS * EXPERT_SIZE
    COMPONENTS  = components

    print(f"  Derived EXPERT_SIZE: {EXPERT_SIZE:,} bytes")
    print(f"  Derived LAYER_SIZE:  {LAYER_SIZE:,} bytes ({LAYER_SIZE / 1024**3:.2f} GB)")

    # -----------------------------------------------------------------------
    # Build expert_reads index for all layers
    #
    # For fused tensors the "stride" per expert is: total_bytes / num_experts.
    # gate_up_proj is then split into gate (first half of rows) and up (second
    # half) at repack time; the index stores gate_up_proj as a unit so repack
    # can read the full fused block and slice in memory.
    #
    # Schema per component entry (matches 4-bit path):
    #   file, abs_offset, expert_stride, expert_size, total_size, shape
    #
    # For gate_up_proj we store the FULL per-expert block (both gate+up halves)
    # under a temporary key "gate_up_proj.*" — the repack function splits them.
    # -----------------------------------------------------------------------
    expert_reads = {}
    n_exp = gu_w_shape[0]
    assert n_exp == NUM_EXPERTS, (
        f"Tensor num_experts ({n_exp}) != profile NUM_EXPERTS ({NUM_EXPERTS})"
    )

    for layer_str, projs in sorted(raw_index.items(), key=lambda x: int(x[0])):
        layer_entry = {}
        for proj_name, parts in projs.items():
            for part_name, info in parts.items():
                total_bytes = info["abs_end"] - info["abs_start"]
                per_expert  = total_bytes // n_exp
                assert total_bytes % n_exp == 0, (
                    f"Layer {layer_str} {proj_name}.{part_name}: total size {total_bytes} "
                    f"not divisible by num_experts {n_exp}"
                )
                comp_key = f"{proj_name}.{part_name}"
                layer_entry[comp_key] = {
                    "file": info["file"],
                    "abs_offset": info["abs_start"],
                    "expert_stride": per_expert,
                    "expert_size": per_expert,
                    "total_size": total_bytes,
                    "shape": info["shape"],
                }
        expert_reads[layer_str] = layer_entry

    expected_layers = set(str(i) for i in range(NUM_LAYERS))
    found_layers = set(expert_reads.keys())
    missing = expected_layers - found_layers
    if missing:
        missing_sorted = sorted(int(x) for x in missing)
        raise ValueError(
            f"expert_reads is missing {len(missing)} layers: {missing_sorted}. "
            f"Download may be incomplete."
        )

    print(f"Generated expert index (fused): {len(expert_reads)} layers, "
          f"{len(next(iter(expert_reads.values())))} tensors/layer")
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
    """Verify that component sizes in the index match expected sizes.

    For 'fused' profiles the COMPONENTS list is populated after index generation,
    so this check is meaningful (it cross-validates the derived layout against the
    raw index). For empty COMPONENTS (should not happen after fused index gen)
    the check is skipped with a warning.
    """
    if not COMPONENTS:
        print("WARNING: COMPONENTS list is empty — skipping size verification "
              "(expected for 'fused' profile before index generation)")
        return True
    expected = {c['name']: c['size'] for c in COMPONENTS}
    for layer_key, comps in expert_reads.items():
        for comp_name, info in comps.items():
            if comp_name not in expected:
                # Fused index stores gate_up_proj.* keys which are split at repack time
                if EXPERT_KEY_TYPE == "fused" and comp_name.startswith("gate_up_proj."):
                    continue
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


def repack_layer_fused(layer_idx, expert_reads, fds, output_dir, dry_run=False):
    """Repack all experts for one layer — fused 3D tensor path (8-bit MLX).

    For Qwen3.6-35B-A3B the safetensors layout is:
      gate_up_proj.weight  [num_experts, 2*moe_inter, hidden/8_u32]  -- gate then up
      gate_up_proj.scales  [num_experts, 2*moe_inter, hidden/gs]
      gate_up_proj.biases  [num_experts, 2*moe_inter, hidden/gs]
      down_proj.weight     [num_experts, hidden, moe_inter/8_u32]
      down_proj.scales     [num_experts, hidden, moe_inter/gs]
      down_proj.biases     [num_experts, hidden, moe_inter/gs]

    For each expert E we:
      1. Read the full per-expert slice of gate_up_proj (both gate and up halves).
      2. Split at the row midpoint: gate = rows 0:half_rows, up = rows half_rows:
      3. Write 9 components in canonical order (same as 4-bit path):
           gate_proj.{weight,scales,biases}, up_proj.{weight,scales,biases},
           down_proj.{weight,scales,biases}

    Returns (bytes_written, elapsed_seconds).
    """
    layer_key = str(layer_idx)
    if layer_key not in expert_reads:
        print(f"  Layer {layer_idx}: NOT FOUND in index, skipping")
        return 0, 0.0

    layer_info = expert_reads[layer_key]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    # Validate required keys are present
    required_keys = [
        "gate_up_proj.weight", "gate_up_proj.scales", "gate_up_proj.biases",
        "down_proj.weight", "down_proj.scales", "down_proj.biases",
    ]
    for k in required_keys:
        assert k in layer_info, (
            f"Layer {layer_idx}: missing '{k}' in expert_reads. "
            f"Found keys: {list(layer_info.keys())}"
        )

    # Derive row-split sizes from COMPONENTS (already set by generate_expert_index_fused)
    # COMPONENTS order: gate.w, gate.s, gate.b, up.w, up.s, up.b, down.w, down.s, down.b
    comp_by_name = {c["name"]: c for c in COMPONENTS}
    gate_w_sz  = comp_by_name["gate_proj.weight"]["size"]
    gate_s_sz  = comp_by_name["gate_proj.scales"]["size"]
    gate_b_sz  = comp_by_name["gate_proj.biases"]["size"]
    up_w_sz    = comp_by_name["up_proj.weight"]["size"]
    up_s_sz    = comp_by_name["up_proj.scales"]["size"]
    up_b_sz    = comp_by_name["up_proj.biases"]["size"]
    down_w_sz  = comp_by_name["down_proj.weight"]["size"]
    down_s_sz  = comp_by_name["down_proj.scales"]["size"]
    down_b_sz  = comp_by_name["down_proj.biases"]["size"]

    # Per-expert sizes from the fused index (each = total_per_expert for the fused proj)
    gu_w_stride = layer_info["gate_up_proj.weight"]["expert_stride"]
    gu_s_stride = layer_info["gate_up_proj.scales"]["expert_stride"]
    gu_b_stride = layer_info["gate_up_proj.biases"]["expert_stride"]
    dn_w_stride = layer_info["down_proj.weight"]["expert_stride"]
    dn_s_stride = layer_info["down_proj.scales"]["expert_stride"]
    dn_b_stride = layer_info["down_proj.biases"]["expert_stride"]

    # The fused gate_up_proj stride must equal gate + up for each part
    assert gu_w_stride == gate_w_sz + up_w_sz, (
        f"Layer {layer_idx}: gate_up_proj.weight stride {gu_w_stride} != "
        f"gate_w_sz + up_w_sz = {gate_w_sz + up_w_sz}"
    )
    assert gu_s_stride == gate_s_sz + up_s_sz, (
        f"Layer {layer_idx}: gate_up_proj.scales stride {gu_s_stride} != "
        f"gate_s_sz + up_s_sz = {gate_s_sz + up_s_sz}"
    )
    assert gu_b_stride == gate_b_sz + up_b_sz, (
        f"Layer {layer_idx}: gate_up_proj.biases stride {gu_b_stride} != "
        f"gate_b_sz + up_b_sz = {gate_b_sz + up_b_sz}"
    )

    if dry_run:
        for expert_idx in range(NUM_EXPERTS):
            _ = expert_idx * EXPERT_SIZE  # verify EXPERT_SIZE is set
        print(f"  Layer {layer_idx:2d}: DRY RUN OK — would write {LAYER_SIZE:,} bytes to {out_path}")
        return LAYER_SIZE, 0.0

    t0 = time.monotonic()

    # Pre-allocate output file
    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, LAYER_SIZE)

    bytes_written = 0

    for expert_idx in range(NUM_EXPERTS):
        dst_base = expert_idx * EXPERT_SIZE

        # Helper: read one component from the fused source
        def read_fused(comp_key, offset_within_stride, size):
            info = layer_info[comp_key]
            src_fd = fds[info["file"]]
            src_off = info["abs_offset"] + expert_idx * info["expert_stride"] + offset_within_stride
            data = os.pread(src_fd, size, src_off)
            assert len(data) == size, (
                f"Short read: expert {expert_idx} {comp_key}+{offset_within_stride}: "
                f"expected {size}, got {len(data)}"
            )
            return data

        # gate_up_proj split: first half = gate, second half = up
        # Each "half" is contiguous in the fused tensor row dimension
        gate_w_data = read_fused("gate_up_proj.weight", 0,         gate_w_sz)
        up_w_data   = read_fused("gate_up_proj.weight", gate_w_sz, up_w_sz)
        gate_s_data = read_fused("gate_up_proj.scales", 0,         gate_s_sz)
        up_s_data   = read_fused("gate_up_proj.scales", gate_s_sz, up_s_sz)
        gate_b_data = read_fused("gate_up_proj.biases", 0,         gate_b_sz)
        up_b_data   = read_fused("gate_up_proj.biases", gate_b_sz, up_b_sz)
        down_w_data = read_fused("down_proj.weight",    0,         down_w_sz)
        down_s_data = read_fused("down_proj.scales",    0,         down_s_sz)
        down_b_data = read_fused("down_proj.biases",    0,         down_b_sz)

        # Write in canonical order matching COMPONENTS layout
        dst = dst_base
        for data in (gate_w_data, gate_s_data, gate_b_data,
                     up_w_data,   up_s_data,   up_b_data,
                     down_w_data, down_s_data, down_b_data):
            os.pwrite(fd_out, data, dst)
            dst += len(data)
            bytes_written += len(data)

    os.close(fd_out)
    elapsed = time.monotonic() - t0
    return bytes_written, elapsed


def verify_layer(layer_idx, expert_reads, fds, output_dir):
    """Read back several experts from packed file and compare to originals.

    For 'fused' profiles (Qwen3.6-35B-A3B), verification re-reads the fused
    gate_up_proj and down_proj tensors and verifies each split half matches
    the corresponding region in the packed file.

    For all other profiles the standard component-by-component comparison is used.
    """
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

    if EXPERT_KEY_TYPE == "fused":
        # Fused path: COMPONENTS has split gate_proj/up_proj entries;
        # source data lives in gate_up_proj.* keys.
        comp_by_name = {c["name"]: c for c in COMPONENTS}
        gate_w_comp = comp_by_name["gate_proj.weight"]
        gate_s_comp = comp_by_name["gate_proj.scales"]
        gate_b_comp = comp_by_name["gate_proj.biases"]
        up_w_comp   = comp_by_name["up_proj.weight"]
        up_s_comp   = comp_by_name["up_proj.scales"]
        up_b_comp   = comp_by_name["up_proj.biases"]
        down_w_comp = comp_by_name["down_proj.weight"]
        down_s_comp = comp_by_name["down_proj.scales"]
        down_b_comp = comp_by_name["down_proj.biases"]

        gate_w_sz = gate_w_comp["size"]
        gate_s_sz = gate_s_comp["size"]
        gate_b_sz = gate_b_comp["size"]

        for expert_idx in check_indices:
            def read_packed(comp):
                off = expert_idx * EXPERT_SIZE + comp["offset"]
                return os.pread(fd_packed, comp["size"], off)

            def read_fused_src(comp_key, offset_within_stride, size):
                info = layer_info[comp_key]
                src_fd = fds[info["file"]]
                src_off = info["abs_offset"] + expert_idx * info["expert_stride"] + offset_within_stride
                return os.pread(src_fd, size, src_off)

            checks = [
                ("gate_proj.weight", gate_w_comp, read_fused_src("gate_up_proj.weight", 0, gate_w_sz)),
                ("gate_proj.scales", gate_s_comp, read_fused_src("gate_up_proj.scales", 0, gate_s_sz)),
                ("gate_proj.biases", gate_b_comp, read_fused_src("gate_up_proj.biases", 0, gate_b_sz)),
                ("up_proj.weight",   up_w_comp,   read_fused_src("gate_up_proj.weight", gate_w_sz, up_w_comp["size"])),
                ("up_proj.scales",   up_s_comp,   read_fused_src("gate_up_proj.scales", gate_s_sz, up_s_comp["size"])),
                ("up_proj.biases",   up_b_comp,   read_fused_src("gate_up_proj.biases", gate_b_sz, up_b_comp["size"])),
                ("down_proj.weight", down_w_comp, read_fused_src("down_proj.weight", 0, down_w_comp["size"])),
                ("down_proj.scales", down_s_comp, read_fused_src("down_proj.scales", 0, down_s_comp["size"])),
                ("down_proj.biases", down_b_comp, read_fused_src("down_proj.biases", 0, down_b_comp["size"])),
            ]
            for name, comp, original in checks:
                packed = read_packed(comp)
                if original != packed:
                    print(f"  MISMATCH: layer {layer_idx}, expert {expert_idx}, {name}")
                    mismatches += 1

    else:
        # Standard path: COMPONENTS entries match layer_info keys 1:1
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


def write_expert_index(model_path, expert_reads, index_out_path):
    """Write a complete expert_index.json covering all layers.

    Fixes issue #17 (upstream ships a layer-0-only index). The generated file
    covers all NUM_LAYERS layers with the full expert_reads structure.

    Structure:
      {
        "model_path": "<model_path>",
        "num_layers": <int>,
        "num_experts": <int>,
        "expert_reads": {
          "0": { <comp_name>: {file, abs_offset, expert_stride, ...}, ... },
          ...
          "<num_layers-1>": { ... }
        }
      }
    """
    index_doc = {
        "model_path": model_path,
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "expert_reads": expert_reads,
    }
    with open(index_out_path, "w") as f:
        json.dump(index_doc, f, indent=2)
    size_kb = os.path.getsize(index_out_path) / 1024
    print(f"Wrote expert_index.json: {index_out_path} ({size_kb:.1f} KB, {len(expert_reads)} layers)")


def main():
    parser = argparse.ArgumentParser(
        description="Repack expert weights into contiguous per-layer binary files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Qwen3.5-397B (4-bit, original flow)
  python repack_experts.py --model ~/qwen35-397b-4bit

  # Qwen3-Coder-480B (4-bit packed)
  python repack_experts.py --model ~/qwen3-coder-480b-4bit --profile qwen3-coder

  # Qwen3.6-35B-A3B (8-bit fused — generates complete 40-layer index, fixes issue #17)
  python repack_experts.py --model ~/qwen36-35b-a3b-8bit --profile qwen3.6-35b-a3b

  # Write the expert index JSON without repacking (useful for debugging)
  python repack_experts.py --model ~/qwen36-35b-a3b-8bit --profile qwen3.6-35b-a3b --write-index

  # Process a subset of layers
  python repack_experts.py --model ~/qwen35-397b-4bit --layers 0-4

  # Dry run (verify offsets only, no writes)
  python repack_experts.py --model ~/qwen35-397b-4bit --dry-run

  # Verify packed layer against originals
  python repack_experts.py --model ~/qwen35-397b-4bit --verify-only 0
        """)
    default_index = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'expert_index.json')
    parser.add_argument('--model', required=True,
                        help='Path to model directory containing safetensors shards')
    parser.add_argument('--profile', default='qwen35-397b',
                        choices=list(MODEL_PROFILES.keys()),
                        help='Model profile (default: qwen35-397b)')
    parser.add_argument('--index', default=default_index,
                        help='Path to expert_index.json (input for 4-bit path, '
                             'output path when --write-index is used)')
    parser.add_argument('--write-index', action='store_true',
                        help='Generate and write expert_index.json (all layers) '
                             'without repacking; fixes issue #17. '
                             'For fused profiles (e.g. qwen3.6-35b-a3b) this is '
                             'the primary way to produce the index.')
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

    # -----------------------------------------------------------------------
    # Load or generate expert index
    # -----------------------------------------------------------------------
    if EXPERT_KEY_TYPE == "fused":
        # Qwen3.6-35B-A3B: fused 3D tensors, 8-bit.
        # EXPERT_SIZE and COMPONENTS are unknown until we read the headers.
        print(f"Profile '{args.profile}': fused 3D expert tensor layout (8-bit MLX)")
        print("Generating expert index from safetensors headers...")
        expert_reads = generate_expert_index_fused(model_path)
        print(f"Expert size: {EXPERT_SIZE:,} bytes, Experts/layer: {NUM_EXPERTS}, Layers: {NUM_LAYERS}")

    elif EXPERT_KEY_TYPE == "packed":
        # Qwen3-Coder style: 3D packed tensors, 4-bit.
        print(f"Profile '{args.profile}': packed 3D expert tensor layout (4-bit MLX)")
        print(f"Expert size: {EXPERT_SIZE:,} bytes, Experts/layer: {NUM_EXPERTS}, Layers: {NUM_LAYERS}")
        print("Generating expert index from safetensors headers...")
        expert_reads = generate_expert_index(model_path)

    else:
        # Individual tensors per expert (Qwen3.5-397B default).
        print(f"Profile '{args.profile}': per-expert tensor layout (4-bit MLX)")
        print(f"Expert size: {EXPERT_SIZE:,} bytes, Experts/layer: {NUM_EXPERTS}, Layers: {NUM_LAYERS}")
        if not os.path.isfile(index_path):
            print(f"ERROR: expert index not found: {index_path}", file=sys.stderr)
            sys.exit(1)
        print("Loading expert index...")
        expert_reads = load_index(index_path)
        print(f"Index path: {index_path}")

    print(f"Model path: {model_path}")
    print(f"Layers in index: {len(expert_reads)}")

    # --write-index: generate the complete JSON and exit (issue #17 fix)
    if args.write_index:
        write_expert_index(model_path, expert_reads, index_path)
        return

    # Verify component sizes (for non-fused profiles or after fused derives layout)
    if EXPERT_KEY_TYPE != "fused":
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

    # Write complete expert_index.json covering all layers (issue #17)
    write_expert_index(model_path, expert_reads, index_path)

    # Repack each layer — dispatch to fused or standard path
    t_start = time.monotonic()
    total_written = 0

    for i, layer_idx in enumerate(layers):
        if EXPERT_KEY_TYPE == "fused":
            bytes_written, elapsed = repack_layer_fused(
                layer_idx, expert_reads, fds, output_dir, dry_run=args.dry_run
            )
        else:
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
