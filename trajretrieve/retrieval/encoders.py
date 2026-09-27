"""Concrete encoder + cross-encoder reranker wrappers.

Thin, optional-dependency adapters so the retriever stays model-agnostic. The
embedding model and reranker ids are read from configuration (see
``configs/config.template.yaml``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class BGEEncoder:
    """sentence-transformers bi-encoder (default: BAAI/bge-m3).

    ``query_prefix`` is the BGE retrieval instruction; pass ``None`` to disable.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        query_prefix: Optional[str] = "Represent this sentence for searching relevant passages: ",
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # lazy

        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = query_prefix
        self.batch_size = batch_size

    def __call__(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if is_query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        emb = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)


class CrossEncoderReranker:
    """Pretrained cross-encoder reranker (default: BAAI/bge-reranker-base).

    This is the default Step-3 reranker. Inference cost is independent of total
    index size: it scores at most the top-K (K=50) candidates per query.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: Optional[str] = None) -> None:
        from sentence_transformers import CrossEncoder  # lazy

        self.model = CrossEncoder(model_name, device=device)

    def __call__(self, query_text: str, candidate_texts: list[str]) -> list[float]:
        pairs = [[query_text, c] for c in candidate_texts]
        return [float(s) for s in self.model.predict(pairs)]


class LLMJudgeReranker:
    """Optional listwise LLM judge reranker (alternative to the cross-encoder).

    ``client`` is an OpenAI-compatible chat client; ``model`` names the judge LLM.
    """

    def __init__(self, client, model: str, instruction: Optional[str] = None) -> None:
        self.client = client
        self.model = model
        self.instruction = instruction or (
            "Score 0-1 how directly each candidate repair helps resolve the query. "
            "Return one float per candidate, comma-separated."
        )

    def __call__(self, query_text: str, candidate_texts: list[str]) -> list[float]:
        listing = "\n".join(f"[{i}] {c}" for i, c in enumerate(candidate_texts))
        msg = f"Query:\n{query_text}\n\nCandidates:\n{listing}\n\n{self.instruction}"
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": msg}],
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        scores: list[float] = []
        for tok in text.replace("\n", ",").split(","):
            tok = tok.strip()
            try:
                scores.append(float(tok))
            except ValueError:
                continue
        # pad / truncate to candidate count
        if len(scores) < len(candidate_texts):
            scores += [0.0] * (len(candidate_texts) - len(scores))
        return scores[: len(candidate_texts)]
