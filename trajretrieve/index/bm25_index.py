"""Sparse BM25 segment index for hybrid scoring.

A thin wrapper over ``rank_bm25.BM25Okapi`` providing the same ``id -> score``
interface as the dense index, so the two can be linearly combined. The inverted
index adds only modest overhead on top of the dense vectors.
"""

from __future__ import annotations

import re
from typing import Optional

try:
    from rank_bm25 import BM25Okapi  # type: ignore

    _HAS_BM25 = True
except Exception:  # pragma: no cover - optional dep
    _HAS_BM25 = False

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class BM25Index:
    """BM25 over candidate render texts, keyed by segment id."""

    def __init__(self, texts: list[str], ids: list[str]) -> None:
        if not _HAS_BM25:
            raise ImportError("rank_bm25 is required for BM25Index (pip install rank_bm25)")
        if len(texts) != len(ids):
            raise ValueError("texts and ids length mismatch")
        self.ids = ids
        self._pos = {sid: i for i, sid in enumerate(ids)}
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])

    def scores(self, query: str) -> dict[str, float]:
        """Return ``{segment_id: bm25_score}`` for all indexed segments."""
        raw = self._bm25.get_scores(tokenize(query))
        return {self.ids[i]: float(raw[i]) for i in range(len(self.ids))}

    def score_for(self, query: str, segment_id: str) -> Optional[float]:
        raw = self._bm25.get_scores(tokenize(query))
        i = self._pos.get(segment_id)
        return float(raw[i]) if i is not None else None
