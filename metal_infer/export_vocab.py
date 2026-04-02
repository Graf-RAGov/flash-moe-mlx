#!/usr/bin/env python3
"""Export vocab.bin from MODEL/tokenizer.json with byte-level BPE decoding."""

import argparse
import json
import os
import struct
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Export vocab.bin with byte-level BPE decoding')
    parser.add_argument('--model', required=True,
                        help='Path to model directory containing tokenizer.json')
    parser.add_argument('--output', default='vocab.bin',
                        help='Output path for vocab.bin')
    args = parser.parse_args()

    model_path = os.path.abspath(os.path.expanduser(args.model))
    tok_path = os.path.join(model_path, 'tokenizer.json')
    out_path = os.path.abspath(os.path.expanduser(args.output))

    if not os.path.isdir(model_path):
        print(f'ERROR: model directory not found: {model_path}', file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(tok_path):
        print(f'ERROR: tokenizer.json not found: {tok_path}', file=sys.stderr)
        sys.exit(1)

    with open(tok_path, 'r', encoding='utf-8') as f:
        t = json.load(f)

    vocab = t['model']['vocab']
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    # Build byte-level BPE decode table (GPT-style)
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_decoder = {chr(c): bytes([b]) for b, c in zip(bs, cs)}

    with open(out_path, 'wb') as f:
        f.write(struct.pack('<I', len(sorted_vocab)))
        f.write(struct.pack('<I', sorted_vocab[-1][1]))
        for token_str, token_id in sorted_vocab:
            try:
                decoded = b''.join(byte_decoder.get(c, c.encode('utf-8')) for c in token_str)
            except UnicodeEncodeError:
                decoded = token_str.encode('utf-8')
            f.write(struct.pack('<H', len(decoded)))
            f.write(decoded)

    print(f'Rebuilt {out_path}: {len(sorted_vocab)} entries (byte-level BPE decoded)')


if __name__ == '__main__':
    main()
