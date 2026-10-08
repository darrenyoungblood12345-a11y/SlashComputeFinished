"""Tiny GGUF writer for tests: builds a valid header (and optionally zero tensor data)."""
from __future__ import annotations

import struct

from slashcompute.inference.gguf import parse_header, tensor_nbytes

F32, F16, Q4_K, Q8_0 = 0, 1, 12, 8


def _str(s: str) -> bytes:
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


def _value(v) -> tuple[int, bytes]:
    if isinstance(v, bool):
        return 7, struct.pack('<?', v)
    if isinstance(v, int):
        return 4, struct.pack('<I', v)
    if isinstance(v, float):
        return 6, struct.pack('<f', v)
    if isinstance(v, str):
        return 8, _str(v)
    if isinstance(v, list):
        if all(isinstance(x, bool) for x in v):
            return 9, struct.pack('<IQ', 7, len(v)) + b''.join(struct.pack('<?', x) for x in v)
        if all(isinstance(x, int) for x in v):
            return 9, struct.pack('<IQ', 4, len(v)) + b''.join(struct.pack('<I', x) for x in v)
        if all(isinstance(x, str) for x in v):
            return 9, struct.pack('<IQ', 8, len(v)) + b''.join(_str(x) for x in v)
    raise TypeError(f'unsupported value {v!r}')


def write_gguf(kv: dict, tensors: list[tuple[str, tuple[int, ...], int]], with_data: bool = False,
               alignment: int = 32) -> bytes:
    out = bytearray(b'GGUF')
    out += struct.pack('<IQQ', 3, len(tensors), len(kv))
    for k, v in kv.items():
        vtype, payload = _value(v)
        out += _str(k) + struct.pack('<I', vtype) + payload
    offset = 0
    sizes = []
    for name, dims, ty in tensors:
        n = 1
        for d in dims:
            n *= d
        nbytes = tensor_nbytes(ty, n)
        out += _str(name) + struct.pack('<I', len(dims)) + b''.join(struct.pack('<Q', d) for d in dims)
        out += struct.pack('<IQ', ty, offset)
        sizes.append(nbytes)
        offset += (nbytes + alignment - 1) // alignment * alignment
    if with_data:
        pad = (-len(out)) % alignment
        out += b'\0' * pad + b'\0' * offset
    return bytes(out)


def dense_model(n_layers: int = 4, emb: int = 256, ff: int = 512, heads: int = 8, kv_heads: int = 2,
                tied: bool = False, arch: str = 'llama', vocab: int = 1000) -> bytes:
    kv = {
        'general.architecture': arch,
        f'{arch}.block_count': n_layers,
        f'{arch}.embedding_length': emb,
        f'{arch}.attention.head_count': heads,
        f'{arch}.attention.head_count_kv': kv_heads,
        f'{arch}.context_length': 4096,
    }
    hd = emb // heads
    tensors = [('token_embd.weight', (emb, vocab), Q8_0)]
    for i in range(n_layers):
        tensors += [
            (f'blk.{i}.attn_norm.weight', (emb,), F32),
            (f'blk.{i}.attn_q.weight', (emb, emb), Q8_0),
            (f'blk.{i}.attn_k.weight', (emb, kv_heads * hd), Q8_0),
            (f'blk.{i}.attn_v.weight', (emb, kv_heads * hd), Q8_0),
            (f'blk.{i}.attn_output.weight', (emb, emb), Q8_0),
            (f'blk.{i}.ffn_up.weight', (emb, ff), Q8_0),
            (f'blk.{i}.ffn_down.weight', (ff, emb), Q8_0),
        ]
    tensors.append(('output_norm.weight', (emb,), F32))
    if not tied:
        tensors.append(('output.weight', (emb, vocab), Q8_0))
    return write_gguf(kv, tensors)


def moe_model(n_layers: int = 4, emb: int = 256, ff: int = 128, experts: int = 8, used: int = 2,
              arch: str = 'qwen3moe', vocab: int = 1000) -> bytes:
    kv = {
        'general.architecture': arch,
        f'{arch}.block_count': n_layers,
        f'{arch}.embedding_length': emb,
        f'{arch}.attention.head_count': 8,
        f'{arch}.attention.head_count_kv': 2,
        f'{arch}.expert_count': experts,
        f'{arch}.expert_used_count': used,
    }
    tensors = [('token_embd.weight', (emb, vocab), Q8_0)]
    for i in range(n_layers):
        tensors += [
            (f'blk.{i}.attn_q.weight', (emb, emb), Q8_0),
            (f'blk.{i}.attn_k.weight', (emb, 64), Q8_0),
            (f'blk.{i}.attn_v.weight', (emb, 64), Q8_0),
            (f'blk.{i}.attn_output.weight', (emb, emb), Q8_0),
            (f'blk.{i}.ffn_gate_inp.weight', (emb, experts), F32),
            (f'blk.{i}.ffn_up_exps.weight', (emb, ff, experts), Q8_0),
            (f'blk.{i}.ffn_down_exps.weight', (ff, emb, experts), Q8_0),
        ]
    tensors.append(('output.weight', (emb, vocab), Q8_0))
    return write_gguf(kv, tensors)


def tiny_gguf(n_layers: int = 2) -> bytes:
    """A small but complete GGUF (header + zeroed tensor data)."""
    h = parse_header(dense_model(n_layers=n_layers, emb=64, ff=128, heads=4, kv_heads=2, vocab=100))
    tensors = [(t.name, t.dims, t.ggml_type) for t in h.tensors]
    return write_gguf(dict(h.kv), tensors, with_data=True)
