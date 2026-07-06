"""Tokenizers.

Two options, per the plan's "cut" rule (the BPE is not the core of the project):

* :class:`ByteTokenizer` — a zero-dependency byte-level tokenizer, perfect for the
  Phase-1 smoke tests and TinyStories-scale char models.
* :class:`BPETokenizer` — a minimal, correct byte-level BPE trainer (merge most-frequent
  adjacent pair, repeat). Educational; slower than production tokenizers.
* :func:`gpt2_tokenizer` — the documented shortcut: GPT-2 BPE via ``tiktoken``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class ByteTokenizer:
    """Identity byte-level tokenizer. vocab_size == 256."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


class BPETokenizer:
    """Minimal byte-level BPE. Not fast — correctness and legibility over speed."""

    def __init__(self, merges: list[tuple[int, int]] | None = None, vocab_size: int = 256):
        self.merges = merges or []
        self.vocab_size = vocab_size

    @classmethod
    def train(cls, text: str, vocab_size: int) -> BPETokenizer:
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        merges: list[tuple[int, int]] = []
        rank: dict[tuple[int, int], int] = {}
        next_id = 256
        while next_id < vocab_size:
            pairs = Counter(zip(ids, ids[1:], strict=False))
            if not pairs:
                break
            top = max(pairs, key=pairs.get)
            merges.append(top)
            rank[top] = next_id
            ids = _merge(ids, top, next_id)
            next_id += 1
        tok = cls(merges, next_id)
        tok._rank = rank
        return tok

    @property
    def _ranks(self) -> dict[tuple[int, int], int]:
        if not hasattr(self, "_rank"):
            self._rank = {pair: 256 + i for i, pair in enumerate(self.merges)}
        return self._rank

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        ranks = self._ranks
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:], strict=False))
            # merge the pair with the lowest (earliest) rank first
            candidate = min(pairs, key=lambda p: ranks.get(p, float("inf")))
            if candidate not in ranks:
                break
            ids = _merge(ids, candidate, ranks[candidate])
        return ids

    def decode(self, ids: list[int]) -> str:
        # expand merges back down to bytes
        inv = {v: k for k, v in self._ranks.items()}
        out: list[int] = []
        stack = list(ids)
        while stack:
            i = stack.pop(0)
            if i < 256:
                out.append(i)
            else:
                a, b = inv[i]
                stack = [a, b, *stack]
        return bytes(out).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"vocab_size": self.vocab_size, "merges": self.merges}))

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        d = json.loads(Path(path).read_text())
        return cls([tuple(m) for m in d["merges"]], d["vocab_size"])


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    a, b = pair
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def gpt2_tokenizer():
    """The documented shortcut. Requires ``pip install tiktoken``."""
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")

    class _Wrap:
        vocab_size = enc.n_vocab

        def encode(self, text: str) -> list[int]:
            return enc.encode(text)

        def decode(self, ids: list[int]) -> str:
            return enc.decode(ids)

    return _Wrap()
