"""Online asymmetric retrieval pipeline.

Three steps per stuck-state query q:

  Step 1  Vector recall with hybrid scoring
          score(q,s) = alpha * dense_hat(q,s) + (1-alpha) * bm25_hat(q,s)
          (dense via the HNSW index, BM25 via the inverted index; both
           query-wise min-max normalized).
  Step 2  Provenance filtering
          drop candidates sharing a provenance link with the current task.
  Step 3  Cross-encoder reranking over the top-K (K=50) survivors.

The cross-encoder is the default reranker; an LLM listwise judge can be
substituted via the ``reranker`` argument. Returns the final top-k segments
with their structured records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

from ..models import Provenance, StructuredRecord, StuckState
from ..index.dense_index import DenseIndex
from ..index.bm25_index import BM25Index
from .templates import render_candidate, render_query


class Encoder(Protocol):
    def __call__(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...


class Reranker(Protocol):
    """Scores (query_text, candidate_text) pairs; higher = more relevant."""

    def __call__(self, query_text: str, candidate_texts: list[str]) -> list[float]: ...


@dataclass
class Candidate:
    segment_id: str
    record: StructuredRecord
    provenance: Provenance
    candidate_text: str = ""


@dataclass
class RetrievalResult:
    segment_id: str
    score: float
    record: StructuredRecord
    provenance: Provenance


def _minmax(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}
    lo = min(d.values())
    hi = max(d.values())
    if hi - lo < 1e-12:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


class HybridRetriever:
    def __init__(
        self,
        encoder: Encoder,
        dense_index: DenseIndex,
        candidates: dict[str, Candidate],
        bm25_index: Optional[BM25Index] = None,
        reranker: Optional[Reranker] = None,
        alpha: float = 0.7,
        recall_width: int = 200,
        rerank_k: int = 50,
        candidate_fields=("phase", "summary", "targets", "error", "resolution"),
    ) -> None:
        self.encoder = encoder
        self.dense = dense_index
        self.candidates = candidates
        self.bm25 = bm25_index
        self.reranker = reranker
        self.alpha = alpha
        self.recall_width = recall_width
        self.rerank_k = rerank_k
        self.candidate_fields = candidate_fields

    # ------------------------------------------------------------------ #

    def retrieve(self, q: StuckState, top_k: int = 10) -> list[RetrievalResult]:
        query_text = render_query(q)
        q_vec = self.encoder([query_text], is_query=True)

        # Step 1: hybrid dense + BM25
        dense_hits = self.dense.search(q_vec, k=self.recall_width)
        dense_scores = {sid: sc for sid, sc in dense_hits}
        if self.bm25 is not None:
            full_bm25 = self.bm25.scores(query_text)
            bm25_scores = {sid: full_bm25.get(sid, 0.0) for sid in dense_scores}
        else:
            bm25_scores = {}

        dense_n = _minmax(dense_scores)
        bm25_n = _minmax(bm25_scores)
        a = self.alpha if self.bm25 is not None else 1.0
        hybrid = {
            sid: a * dense_n.get(sid, 0.0) + (1 - a) * bm25_n.get(sid, 0.0)
            for sid in dense_scores
        }

        # Step 2: provenance filtering
        survivors = [
            sid for sid in hybrid if self._keep(self.candidates.get(sid), q)
        ]
        survivors.sort(key=lambda s: (-hybrid[s], s))  # ties broken by segment id
        survivors = survivors[: self.rerank_k]

        # Step 3: cross-encoder rerank (optional but default)
        if self.reranker is not None and survivors:
            cand_texts = [self._text(sid) for sid in survivors]
            re_scores = self.reranker(query_text, cand_texts)
            order = sorted(range(len(survivors)), key=lambda i: (-re_scores[i], survivors[i]))
            ranked = [(survivors[i], float(re_scores[i])) for i in order]
        else:
            ranked = [(sid, hybrid[sid]) for sid in survivors]

        out: list[RetrievalResult] = []
        for sid, score in ranked[:top_k]:
            c = self.candidates[sid]
            out.append(RetrievalResult(sid, score, c.record, c.provenance))
        return out

    # ------------------------------------------------------------------ #

    def _text(self, sid: str) -> str:
        c = self.candidates[sid]
        if c.candidate_text:
            return c.candidate_text
        return render_candidate(c.record, self.candidate_fields)

    @staticmethod
    def _keep(c: Optional[Candidate], q: StuckState) -> bool:
        """Exclude candidates sharing a provenance link with the current task."""
        if c is None:
            return False
        if q.task_id is None:
            return True
        p = c.provenance
        if p.task_id == q.task_id:
            return False
        # sibling AppWorld task variant (e.g. abc1234_2 vs abc1234_1)
        q_stem = Provenance(q.task_id, "", "", (0, 0)).parent_stem()
        if q_stem and p.parent_stem() == q_stem:
            return False
        return True
