#!/usr/bin/env python3
"""
extract_weights.py — Extract all non-expert weights from MLX quantized models
into a single binary file that the C inference engine can mmap.

Supports:
  - Qwen3.5-397B-A17B-4bit (default, has 'language_model.' prefix)
  - Qwen3-Coder-480B-A35B-Instruct-4bit (no prefix, no linear attn weights)

Outputs:
  - model_weights.bin: binary blob containing all non-expert weight tensors
  - model_weights.json: manifest describing each tensor's location, shape, dtype

Usage:
    python extract_weights.py [--model PATH] [--output DIR]
    python extract_weights.py --model ~/qwen3-coder-480b-4bit --profile qwen3-coder
"""

import json
import struct
import sys
import os
import argparse
import time
from pathlib import Path
from collections import defaultdict
import re
import numpy as np


def parse_safetensors_header(filepath):
    """Parse a safetensors file header. Returns (header_dict, data_start_offset)."""
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def main():
    # Model profile configs for the JSON manifest
    PROFILES = {
        "qwen35-397b": {
            "weight_prefix": "language_model.",
            "config": {
                "hidden_size": 4096,
                "num_hidden_layers": 60,
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "vocab_size": 248320,
                "rms_norm_eps": 1e-6,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "moe_intermediate_size": 1024,
                "shared_expert_intermediate_size": 1024,
                "full_attention_interval": 4,
                "linear_num_value_heads": 64,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000.0,
            },
        },
        "qwen3-coder": {
            "weight_prefix": "",
            "config": {
                "hidden_size": 6144,
                "num_hidden_layers": 62,
                "num_attention_heads": 96,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
                "num_experts": 160,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 2560,
                "shared_expert_intermediate_size": 0,
                "full_attention_interval": 1,
                "partial_rotary_factor": 1.0,
                "rope_theta": 10000000.0,
            },
        },
    }

    parser = argparse.ArgumentParser(description='Extract non-expert weights to binary')
    parser.add_argument('--model', type=str,
                        default=os.path.expanduser(
                            '~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit'
                            '/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3'),
                        help='Path to model directory')
    parser.add_argument('--profile', type=str, default='qwen35-397b',
                        choices=list(PROFILES.keys()),
                        help='Model profile (default: qwen35-397b)')
    parser.add_argument('--output', type=str, default='.',
                        help='Output directory for model_weights.bin and .json')
    parser.add_argument('--include-experts', action='store_true',
                        help='Also extract expert weights (huge, not recommended)')
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    prefix = profile["weight_prefix"]
    num_layers = profile["config"]["num_hidden_layers"]
    full_attn_interval = profile["config"]["full_attention_interval"]

    model_path = Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the weight index
    index_path = model_path / 'model.safetensors.index.json'
    if not index_path.exists():
        print(f"ERROR: {index_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(index_path) as f:
        idx = json.load(f)

    weight_map = idx['weight_map']

    # Filter: keep only language_model weights, skip vision_tower
    # Also skip expert weights (switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases})
    # unless --include-experts is set
    expert_pattern = re.compile(r'\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$')
    vision_pattern = re.compile(r'^(vision_tower|model\.visual)')

    tensors_to_extract = {}  # name -> filename
    skipped_expert = 0
    skipped_vision = 0

    for name, filename in weight_map.items():
        if vision_pattern.match(name):
            skipped_vision += 1
            continue
        if not args.include_experts and expert_pattern.search(name):
            skipped_expert += 1
            continue
        tensors_to_extract[name] = filename

    print(f"Model: {model_path}")
    print(f"Total weights in index: {len(weight_map)}")
    print(f"Skipped vision: {skipped_vision}")
    print(f"Skipped expert: {skipped_expert}")
    print(f"Extracting: {len(tensors_to_extract)} tensors")

    # Group by shard file for sequential I/O
    by_file = defaultdict(list)
    for name, filename in tensors_to_extract.items():
        by_file[filename].append(name)

    # Parse headers and plan layout
    print("\nParsing safetensors headers...")
    header_cache = {}
    for filename in sorted(by_file.keys()):
        filepath = model_path / filename
        header_cache[filename] = parse_safetensors_header(str(filepath))

    # Sanitize tensor names: remove weight prefix for the C engine
    def sanitize_name(name):
        if prefix and name.startswith(prefix):
            return name[len(prefix):]
        return name

    # Plan the output layout
    # Sort tensors for deterministic output
    all_tensors = []  # (sanitized_name, original_name, filename)
    for name in sorted(tensors_to_extract.keys()):
        san_name = sanitize_name(name)
        all_tensors.append((san_name, name, tensors_to_extract[name]))

    # Write binary file
    bin_path = output_dir / 'model_weights.bin'
    manifest = {
        "model": str(model_path),
        "num_tensors": len(all_tensors),
        "tensors": {},
        # Model config for the C engine
        "config": profile["config"].copy(),
    }

    # Layer type map
    layer_types = []
    for i in range(num_layers):
        if (i + 1) % full_attn_interval == 0:
            layer_types.append("full_attention")
        else:
            layer_types.append("linear_attention")
    manifest["config"]["layer_types"] = layer_types

    print(f"\nWriting {bin_path}...")
    t0 = time.time()
    offset = 0
    total_bytes = 0

    ALIGN = 64  # 64-byte alignment for Metal buffers

    with open(bin_path, 'wb') as out_f:
        for i, (san_name, orig_name, filename) in enumerate(all_tensors):
            filepath = model_path / filename
            header, data_start = header_cache[filename]

            if orig_name not in header:
                print(f"  WARNING: {orig_name} not found in {filename}, skipping")
                continue

            meta = header[orig_name]
            tensor_offsets = meta['data_offsets']
            byte_len = tensor_offsets[1] - tensor_offsets[0]
            shape = meta['shape']
            dtype = meta['dtype']

            # Align offset
            if offset % ALIGN != 0:
                pad = ALIGN - (offset % ALIGN)
                out_f.write(b'\x00' * pad)
                offset += pad

            # Read tensor data from safetensors
            with open(filepath, 'rb') as sf:
                sf.seek(data_start + tensor_offsets[0])
                data = sf.read(byte_len)

            out_f.write(data)

            manifest["tensors"][san_name] = {
                "offset": offset,
                "size": byte_len,
                "shape": shape,
                "dtype": dtype,
            }

            offset += byte_len
            total_bytes += byte_len

            if (i + 1) % 100 == 0 or i == len(all_tensors) - 1:
                print(f"  [{i+1}/{len(all_tensors)}] {total_bytes / 1e9:.2f} GB written")

    elapsed = time.time() - t0
    throughput = total_bytes / elapsed / 1e9

    print(f"\nDone: {total_bytes / 1e9:.2f} GB in {elapsed:.1f}s ({throughput:.1f} GB/s)")
    print(f"Binary: {bin_path} ({os.path.getsize(bin_path) / 1e9:.2f} GB)")

    # Write manifest
    json_path = output_dir / 'model_weights.json'
    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {json_path}")

    # Print summary by category
    categories = defaultdict(lambda: {"count": 0, "bytes": 0})
    for san_name, info in manifest["tensors"].items():
        if "embed_tokens" in san_name:
            cat = "embedding"
        elif "norm.weight" in san_name and "layers." not in san_name:
            cat = "final_norm"
        elif "lm_head" in san_name:
            cat = "lm_head"
        elif "input_layernorm" in san_name or "post_attention_layernorm" in san_name:
            cat = "layer_norms"
        elif "linear_attn" in san_name:
            cat = "linear_attention"
        elif "self_attn" in san_name:
            cat = "full_attention"
        elif "mlp.gate." in san_name:
            cat = "routing_gate"
        elif "shared_expert." in san_name:
            cat = "shared_expert"
        elif "shared_expert_gate" in san_name:
            cat = "shared_expert_gate"
        elif "switch_mlp" in san_name:
            cat = "routed_experts"
        else:
            cat = "other"
        categories[cat]["count"] += 1
        categories[cat]["bytes"] += info["size"]

    print("\nWeight categories:")
    for cat in sorted(categories.keys()):
        info = categories[cat]
        print(f"  {cat:25s}: {info['count']:4d} tensors, {info['bytes']/1e6:8.1f} MB")


if __name__ == '__main__':
    main()
