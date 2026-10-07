"""Dataset loading and deterministic batching (batch for step ``s`` depends
only on the seed and ``s``, so a resumed job sees the same data)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np


@dataclass
class Example:
    tokens: list[int]
    loss_start: int  # first token index (in ``tokens``) that contributes to loss


@dataclass
class Batch:
    inputs: mx.array   # (B, T) int32
    targets: mx.array  # (B, T) int32
    mask: mx.array     # (B, T) float32, 1 where the target counts toward loss
    ntoks: int

    def microbatch(self, j: int, size: int) -> "Batch":
        sl = slice(j * size, (j + 1) * size)
        m = self.mask[sl]
        return Batch(self.inputs[sl], self.targets[sl], m, int(m.sum().item()))


def check_dataset(path: str | Path) -> int:
    """Check every row's shape before a job is queued, so a malformed file is refused at
    submit with its line number instead of failing on a Mac after the model has loaded.
    Returns the number of rows."""
    rows = 0
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as e:
                raise ValueError(f"dataset line {lineno} is not JSON ({e.msg}); the file needs "
                                 "one JSON object per line") from None
            if not isinstance(row, dict):
                raise ValueError(f"dataset line {lineno} is not a JSON object")
            ok = (isinstance(row.get("tokens"), list) or isinstance(row.get("text"), str)
                  or (isinstance(row.get("prompt"), str) and isinstance(row.get("completion"), str))
                  or isinstance(row.get("messages"), list))
            if not ok:
                raise ValueError(f"dataset line {lineno} has keys {sorted(row)}; expected \"text\", "
                                 "\"prompt\" + \"completion\", \"messages\" or \"tokens\"")
            rows += 1
    if not rows:
        raise ValueError("the dataset has no rows")
    return rows


def load_examples(path: str | Path, model_path: Optional[str | Path], max_seq_len: int) -> list[Example]:
    tokenizer = None
    vocab_size = None
    examples = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "tokens" in row:
            toks, start = list(row["tokens"]), int(row.get("loss_start", 0))
            if vocab_size is None:
                from slashcompute.pipeline.model_profile import profile_model

                vocab_size = profile_model(str(model_path)).vocab_size
            # an out-of-range id would silently index past the embedding table
            bad = [t for t in toks if type(t) is not int or not 0 <= t < vocab_size]
            if bad:
                raise ValueError(f"dataset line {lineno}: token id {bad[0]!r} is outside the "
                                 f"model vocabulary [0, {vocab_size})")
        else:
            if tokenizer is None:
                from mlx_lm.utils import load_tokenizer

                from slashcompute.pipeline.model_profile import resolve_model_path

                tokenizer = load_tokenizer(resolve_model_path(str(model_path)))
            toks, start = _tokenize(tokenizer, row)
        toks = toks[: max_seq_len + 1]
        # drop rows whose completion was truncated away, else prompt tokens get trained on
        if len(toks) >= 2 and start < len(toks):
            examples.append(Example(toks, start))
    if not examples:
        raise ValueError(f"no usable examples in {path} (need a completion token within max_seq_len={max_seq_len})")
    return examples


def _tokenize(tokenizer, row: dict) -> tuple[list[int], int]:
    if "text" in row:
        return tokenizer.encode(row["text"]), 0
    if "prompt" in row and "completion" in row:
        prompt = tokenizer.encode(row["prompt"])
        full = tokenizer.encode(row["prompt"] + row["completion"])
        return full, len(prompt)
    if "messages" in row:
        msgs = row["messages"]
        prompt = tokenizer.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True)
        full = tokenizer.apply_chat_template(msgs, tokenize=True)
        return list(full), len(prompt)
    raise ValueError(f"unrecognised dataset row keys: {sorted(row)}")


def make_batch(examples: list[Example], step: int, batch_size: int, seed: int) -> Batch:
    rng = np.random.default_rng([seed, step])
    idx = rng.choice(len(examples), size=batch_size, replace=len(examples) < batch_size)
    chosen = [examples[i] for i in idx]
    T = max(len(e.tokens) for e in chosen) - 1
    inputs = np.zeros((batch_size, T), np.int32)
    targets = np.zeros((batch_size, T), np.int32)
    mask = np.zeros((batch_size, T), np.float32)
    for b, e in enumerate(chosen):
        t = np.asarray(e.tokens, np.int32)
        n = len(t) - 1
        inputs[b, :n] = t[:-1]
        targets[b, :n] = t[1:]
        # target position p predicts token p+1
        mask[b, max(e.loss_start - 1, 0):n] = 1.0
    return Batch(mx.array(inputs), mx.array(targets), mx.array(mask), int(mask.sum()))
