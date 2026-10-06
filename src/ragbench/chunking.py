"""Chunking strategies for splitting documents into retrievable passages.

Two strategies, deliberately simple ones we can explain in an interview:
  - fixed:    split on whitespace tokens into fixed-size windows with
              overlap. Cheap, predictable, ignores sentence boundaries.
  - sentence: greedily pack whole sentences into a window up to
              chunk_size tokens. Respects sentence boundaries, which
              should help retrieval precision (fewer half-sentence
              fragments) at the cost of variable chunk sizes.

We measure the effect of this choice in the ablation study rather than
just asserting one is "better" -- that's the whole point of the project.
"""
from __future__ import annotations

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s for s in sentences if s]


def chunk_fixed(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Whitespace-token sliding window with overlap."""
    tokens = text.split()
    if not tokens:
        return []
    step = max(1, chunk_size - chunk_overlap)
    chunks = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(tokens):
            break
    return chunks


def chunk_sentence(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Pack whole sentences into windows up to ~chunk_size tokens.

    `chunk_overlap` here means "carry the last N tokens' worth of
    sentences into the next chunk" so consecutive chunks share context,
    mirroring what fixed-window overlap gives us.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent.split())
        if current and current_len + sent_len > chunk_size:
            chunks.append(" ".join(current))
            # carry trailing sentences whose combined length <= overlap
            carried: list[str] = []
            carried_len = 0
            for s in reversed(current):
                l = len(s.split())
                if carried_len + l > chunk_overlap:
                    break
                carried.insert(0, s)
                carried_len += l
            current = carried
            current_len = carried_len
        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(text: str, strategy: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if strategy == "fixed":
        return chunk_fixed(text, chunk_size, chunk_overlap)
    if strategy == "sentence":
        return chunk_sentence(text, chunk_size, chunk_overlap)
    raise ValueError(f"Unknown chunk_strategy: {strategy!r}")
