"""Dense segment index with HNSW approximate nearest-neighbour search.

Implements the dense-recall component of the retriever. Embeddings are
L2-normalized so inner-product search is equivalent to cosine similarity.

Backends, in priority order:
  1. faiss  ``IndexHNSWFlat``      (sub-millisecond ANN; supports add/search)
  2. hnswlib ``Index('cosine')``   (sub-millisecond ANN; supports add/search)
  3. brute-force numpy matmul      (exact; fallback when neither is installed)

All three expose the same ``add`` / ``search`` API and support incremental
``add`` without a full rebuild. With the default 1024-d BGE-M3 encoder the
float32 vectors occupy ~4 KB per segment, i.e. ~4 GB at 1e6 segments, plus the
HNSW graph.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:  # backend 1
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover - optional dep
    _HAS_FAISS = False

try:  # backend 2
    import hnswlib  # type: ignore

    _HAS_HNSWLIB = True
except Exception:  # pragma: no cover - optional dep
    _HAS_HNSWLIB = False


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class DenseIndex:
    """HNSW dense index over segment embeddings.

    Parameters
    ----------
    dim:
        Embedding dimensionality.
    backend:
        ``"auto"`` (default) | ``"faiss"`` | ``"hnswlib"`` | ``"bruteforce"``.
    space:
        Always cosine (vectors are normalized on add/search).
    M, ef_construction, ef_search:
        Standard HNSW hyper-parameters.
    """

    def __init__(
        self,
        dim: int,
        backend: str = "auto",
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        max_elements: int = 1_000_000,
    ) -> None:
        self.dim = dim
        self.ef_search = ef_search
        self.ids: list[str] = []
        self._vectors: list[np.ndarray] = []  # kept for bruteforce / introspection

        if backend == "auto":
            backend = "faiss" if _HAS_FAISS else ("hnswlib" if _HAS_HNSWLIB else "bruteforce")
        self.backend = backend

        if backend == "faiss":
            index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search
            self._index = index
        elif backend == "hnswlib":
            index = hnswlib.Index(space="cosine", dim=dim)
            index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
            index.set_ef(ef_search)
            self._index = index
        elif backend == "bruteforce":
            self._index = None
            self._matrix: Optional[np.ndarray] = None
        else:
            raise ValueError(f"unknown backend {backend!r}")

    # ------------------------------------------------------------------ #

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        """Incrementally add normalized vectors with their segment ids."""
        vecs = _normalize(vectors)
        if len(ids) != vecs.shape[0]:
            raise ValueError("ids and vectors length mismatch")
        offset = len(self.ids)
        self.ids.extend(ids)
        self._vectors.append(vecs)

        # single-threaded build
        if self.backend == "faiss":
            threads = faiss.omp_get_max_threads()
            faiss.omp_set_num_threads(1)
            try:
                self._index.add(vecs)
            finally:
                faiss.omp_set_num_threads(threads)
        elif self.backend == "hnswlib":
            self._index.add_items(vecs, np.arange(offset, offset + vecs.shape[0]), num_threads=1)
        else:  # bruteforce
            self._matrix = vecs if self._matrix is None else np.vstack([self._matrix, vecs])

    def search(self, query: np.ndarray, k: int = 20) -> list[tuple[str, float]]:
        """Return the top-k (segment_id, cosine_score) for a query vector."""
        q = _normalize(query)
        k = min(k, len(self.ids)) or 0
        if k == 0:
            return []
        if self.backend == "faiss":
            self._index.hnsw.efSearch = max(self.ef_search, k)
            scores, idx = self._index.search(q, k)
            return [(self.ids[i], float(scores[0][r])) for r, i in enumerate(idx[0]) if i >= 0]
        if self.backend == "hnswlib":
            self._index.set_ef(max(self.ef_search, k))
            labels, dist = self._index.knn_query(q, k=k)
            return [(self.ids[int(i)], 1.0 - float(dist[0][r])) for r, i in enumerate(labels[0])]
        # bruteforce exact cosine
        sims = (self._matrix @ q[0]).astype(np.float32)
        top = np.argsort(-sims, kind="stable")[:k]
        return [(self.ids[int(i)], float(sims[int(i)])) for i in top]

    # ------------------------------------------------------------------ #

    def bytes_per_segment(self) -> int:
        """Dense vector footprint per segment (float32), excluding metadata."""
        return self.dim * 4

    def __len__(self) -> int:
        return len(self.ids)
