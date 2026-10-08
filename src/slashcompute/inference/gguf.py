"""Minimal GGUF header reader that works on raw bytes.

Works on a prefix of the file (HTTP Range fetches): if the buffer ends before the header does,
`parse_header` raises `NeedMoreBytes` and the caller fetches a bigger prefix. Only metadata and the
tensor table are read, never tensor data.
"""
from __future__ import annotations

import glob
import re
import struct
from dataclasses import dataclass
from pathlib import Path

MAGIC = b'GGUF'

# ggml type id -> (name, elements per block, bytes per block)
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ('F32', 1, 4), 1: ('F16', 1, 2), 2: ('Q4_0', 32, 18), 3: ('Q4_1', 32, 20),
    6: ('Q5_0', 32, 22), 7: ('Q5_1', 32, 24), 8: ('Q8_0', 32, 34), 9: ('Q8_1', 32, 36),
    10: ('Q2_K', 256, 84), 11: ('Q3_K', 256, 110), 12: ('Q4_K', 256, 144), 13: ('Q5_K', 256, 176),
    14: ('Q6_K', 256, 210), 15: ('Q8_K', 256, 292), 16: ('IQ2_XXS', 256, 66), 17: ('IQ2_XS', 256, 74),
    18: ('IQ3_XXS', 256, 98), 19: ('IQ1_S', 256, 50), 20: ('IQ4_NL', 32, 18), 21: ('IQ3_S', 256, 110),
    22: ('IQ2_S', 256, 82), 23: ('IQ4_XS', 256, 136), 24: ('I8', 1, 1), 25: ('I16', 1, 2),
    26: ('I32', 1, 4), 27: ('I64', 1, 8), 28: ('F64', 1, 8), 29: ('IQ1_M', 256, 56),
    30: ('BF16', 1, 2), 34: ('TQ1_0', 256, 54), 35: ('TQ2_0', 256, 66), 39: ('MXFP4', 32, 17),
}
FLOAT_TYPES = {'F32', 'F16', 'BF16', 'F64'}

# metadata value types
_SCALARS = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
T_STRING, T_ARRAY = 8, 9
MAX_INLINE_ARRAY = 4096  # longer arrays (tokenizer vocab etc.) are summarised, not stored

SHARD_RE = re.compile(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$')


class NeedMoreBytes(Exception):
    """The buffer ended inside the header; fetch a longer prefix and try again."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    offset: int
    nbytes: int

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.ggml_type, (f'type{self.ggml_type}', 0, 0))[0]

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


@dataclass(frozen=True)
class GGUFHeader:
    version: int
    kv: dict
    tensors: tuple[TensorInfo, ...]
    data_start: int

    @property
    def arch(self) -> str:
        return self.kv.get('general.architecture', '?')

    def get(self, key: str, default=None):
        """Architecture-scoped metadata, e.g. get('block_count') -> llama.block_count."""
        return self.kv.get(f'{self.arch}.{key}', default)

    def to_json(self) -> dict:
        return {
            'version': self.version,
            'kv': self.kv,
            'data_start': self.data_start,
            'tensors': [[t.name, list(t.dims), t.ggml_type, t.offset, t.nbytes] for t in self.tensors],
        }

    @classmethod
    def from_json(cls, d: dict) -> 'GGUFHeader':
        tensors = tuple(TensorInfo(n, tuple(dims), ty, off, nb) for n, dims, ty, off, nb in d['tensors'])
        return cls(version=d['version'], kv=d['kv'], tensors=tensors, data_start=d['data_start'])


class _Reader:
    def __init__(self, buf: bytes):
        self.buf = memoryview(buf)
        self.pos = 0

    def take(self, n: int) -> memoryview:
        if self.pos + n > len(self.buf):
            raise NeedMoreBytes(f'header continues past byte {len(self.buf)}')
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def scalar(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = self.scalar('<Q')
        if n > 1 << 30:
            raise ValueError('bad GGUF string length')
        return bytes(self.take(n)).decode('utf-8', 'replace')

    def value(self, vtype: int):
        if vtype in _SCALARS:
            return self.scalar(_SCALARS[vtype])
        if vtype == T_STRING:
            return self.string()
        if vtype == T_ARRAY:
            item_type = self.scalar('<I')
            n = self.scalar('<Q')
            if item_type in _SCALARS:
                fmt = _SCALARS[item_type]
                raw = self.take(struct.calcsize(fmt) * n)
                if n > MAX_INLINE_ARRAY:
                    return {'array_of': item_type, 'len': n}
                return list(struct.unpack(f'<{n}{fmt[1]}', raw))
            if item_type == T_STRING:
                if n > MAX_INLINE_ARRAY:
                    for _ in range(n):
                        self.take(self.scalar('<Q'))
                    return {'array_of': 'string', 'len': n}
                return [self.string() for _ in range(n)]
            return [self.value(item_type) for _ in range(n)]
        raise ValueError(f'unknown GGUF value type {vtype}')


def tensor_nbytes(ggml_type: int, n_elements: int) -> int | None:
    info = GGML_TYPES.get(ggml_type)
    if info is None:
        return None
    _, block, size = info
    return n_elements // block * size


def parse_header(buf: bytes, file_size: int | None = None) -> GGUFHeader:
    r = _Reader(buf)
    if bytes(r.take(4)) != MAGIC:
        raise ValueError('not a GGUF file')
    version = r.scalar('<I')
    n_tensors = r.scalar('<Q')
    n_kv = r.scalar('<Q')
    kv = {}
    for _ in range(n_kv):
        key = r.string()
        kv[key] = r.value(r.scalar('<I'))
    raw = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.scalar('<I')
        dims = tuple(r.scalar('<Q') for _ in range(n_dims))
        raw.append((name, dims, r.scalar('<I'), r.scalar('<Q')))
    align = kv.get('general.alignment', 32) or 32
    data_start = (r.pos + align - 1) // align * align

    # Unknown types: size = gap to the next tensor (or to the end of the file for the last one).
    by_offset = sorted(range(len(raw)), key=lambda i: raw[i][3])
    next_offset = {}
    for a, b in zip(by_offset, by_offset[1:]):
        next_offset[a] = raw[b][3]
    tensors = []
    for i, (name, dims, ty, off) in enumerate(raw):
        n = 1
        for d in dims:
            n *= d
        nbytes = tensor_nbytes(ty, n)
        if nbytes is None:
            end = next_offset.get(i, (file_size - data_start) if file_size else off)
            nbytes = max(end - off, 0)
        tensors.append(TensorInfo(name, dims, ty, off, nbytes))
    return GGUFHeader(version=version, kv=kv, tensors=tuple(tensors), data_start=data_start)


def read_header_file(path: str | Path, start: int = 1 << 20, limit: int = 1 << 30) -> GGUFHeader:
    """Read just enough of a local file to parse its header."""
    path = Path(path)
    size = path.stat().st_size
    n = start
    with path.open('rb') as fh:
        while True:
            fh.seek(0)
            buf = fh.read(min(n, size))
            try:
                return parse_header(buf, file_size=size)
            except NeedMoreBytes:
                if n >= size or n >= limit:
                    raise
                n *= 2


def shard_paths(path: str | Path) -> list[Path]:
    """All shards of a split model given any shard (or the single file)."""
    path = Path(path)
    m = SHARD_RE.match(path.name)
    if not m:
        return [path]
    pattern = glob.escape(str(path.parent / m.group(1))) + f'-*-of-{m.group(3)}.gguf'
    # the glob's * also matches another model's shards ("x-big-00001-of-00003.gguf" for "x"): keep exact ones
    return sorted(p for p in map(Path, glob.glob(pattern))
                  if (s := SHARD_RE.match(p.name)) and s.group(1) == m.group(1))


def is_first_shard_or_single(name: str) -> bool:
    m = SHARD_RE.match(name)
    return not m or m.group(2) == '00001'


def plain_gguf(name) -> bool:
    """A bare GGUF file name (no folder, not hidden): the only kind a removal may touch."""
    return (isinstance(name, str) and name.endswith('.gguf') and not name.startswith('.')
            and '/' not in name and '\\' not in name and '\0' not in name)
