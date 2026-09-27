"""Retrieval evaluation + candidate-side field selection.

Loads an index built by ``scripts/build_index.py``, runs the online
``HybridRetriever`` over a set of stuck-state queries, and reports
Recall@k / Precision@k / nDCG@k / MRR@k against graded relevance labels.

By default Step 3 uses the pretrained ``CrossEncoderReranker``; pass
``--reranker llm`` to substitute the listwise ``LLMJudgeReranker`` (reads the
optional LLM endpoint from ``--config``), or ``--reranker none`` to rank by the
hybrid score alone.

File formats
------------
Both files are JSON Lines (one object per line), joined on ``query_id``.

``--queries queries.jsonl`` -- one stuck-state query q per line::

    {
      "query_id":       "query_017",         # required, joins to gold
      "task_id":        "abc1234_1",             # optional, for provenance filter
      "goal":           "send 20 USD to ...",     # q_goal
      "phase":          "execution",              # q_phase (optional)
      "error":          "venmo.create_transaction 401", # q_err (optional)
      "failed_actions": ["apis.venmo.create_transaction(...)"], # q_fail (optional)
      "subgoals":       ["log in to venmo first"], # q_sub (optional)
      "task_type":      "playlist_edit"            # optional, --stratify-key for field selection
    }

``--gold qrels.jsonl`` -- one graded label per (query, segment), as in
``samples/<env>/qrels.jsonl``::

    {"query_id": "query_017", "segment_id": "def5678_2::seg_004", "relevance_grade": 3}

Labels range from -1 to 3. A grouped form
``{"query_id": ..., "relevant": {"segment_id": label}}`` is also accepted, and a
plain list of segment ids counts each as label 2. A gold segment that is not in
the index may carry its ``trajectory_id`` and inclusive ``step_range``.

Hit rule
--------
Segments with label >= 2 are relevant. A retrieved candidate is a hit when its
steps overlap a relevant segment of the same trajectory by at least 0.3 of the
candidate's length. Recall@k is the fraction of relevant segments covered by
the top-k, Precision@k the fraction of the top-k that are hits, MRR@k the
reciprocal rank of the first hit, and nDCG@k uses the gain 2^label - 1 of the
best relevant segment a candidate hits, counting each relevant segment once.

Field selection
---------------
``--field-selection greedy`` runs forward greedy selection
(``trajretrieve.schema.field_selection.greedy_selection``) over the candidate
fields, scoring each subset by mean nDCG@k of dense-only retrieval on a
validation split (20% of the queries by default; ``--stratify-key task_type``
draws it per task type, matching the paper's stratified split). It re-renders + re-encodes the
candidate pool per subset and prints the selected fields and per-step gains.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trajretrieve.config import build_llm_client, load_config
from trajretrieve.index import BM25Index, DenseIndex
from trajretrieve.models import Provenance, StructuredRecord, StuckState
from trajretrieve.retrieval import Candidate, HybridRetriever, render_candidate
from trajretrieve.retrieval.encoders import BGEEncoder, CrossEncoderReranker, LLMJudgeReranker
from trajretrieve.schema.field_selection import DEFAULT_FIELDS, greedy_selection, validation_split


# --------------------------------------------------------------------------- #
# Loading a built index.                                                       #
# --------------------------------------------------------------------------- #


def _load_index_dir(index_dir: Path):
    """Return (meta, candidate_dicts, embeddings) from a build_index.py output."""
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    cands = [
        json.loads(line)
        for line in (index_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    embeddings = np.load(index_dir / "embeddings.npy").astype(np.float32)
    if embeddings.shape[0] != len(cands):
        raise SystemExit("embeddings.npy and candidates.jsonl are out of sync")
    return meta, cands, embeddings


def _to_record(c: dict) -> StructuredRecord:
    p = c["provenance"]
    prov = Provenance(
        task_id=p.get("task_id"),
        trajectory_id=p["trajectory_id"],
        segment_id=p["segment_id"],
        step_range=tuple(p["step_range"]),
        source=p.get("source"),
    )
    return StructuredRecord(
        phase=c.get("phase"),
        summary=c.get("summary"),
        targets=list(c.get("targets") or []),
        error=c.get("error"),
        resolution=c.get("resolution"),
        provenance=prov,
    )


def _build_candidates(cands: list[dict]) -> dict[str, Candidate]:
    out: dict[str, Candidate] = {}
    for c in cands:
        rec = _to_record(c)
        out[c["segment_id"]] = Candidate(
            segment_id=c["segment_id"],
            record=rec,
            provenance=rec.provenance,
            candidate_text=c.get("candidate_text", ""),
        )
    return out


# --------------------------------------------------------------------------- #
# Queries / gold.                                                              #
# --------------------------------------------------------------------------- #


def _read_jsonl(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _to_stuck_state(q: dict) -> StuckState:
    return StuckState(
        failed_actions=list(q.get("failed_actions") or []),
        goal=q.get("goal", ""),
        error=q.get("error", ""),
        subgoals=list(q.get("subgoals") or []),
        phase=q.get("phase"),
        task_id=q.get("task_id"),
    )


REL_THRESHOLD = 2      # labels >= 2 are relevant
MIN_HIT_OVERLAP = 0.3  # overlap / candidate length needed for a hit


def _load_gold(path: str) -> dict[str, dict[str, dict]]:
    """query_id -> {segment_id: {"label": float[, "trajectory_id", "step_range"]}}."""
    gold: dict[str, dict[str, dict]] = {}
    for row in _read_jsonl(path):
        q = gold.setdefault(row["query_id"], {})
        if "segment_id" in row:
            entry = {"label": float(row.get("relevance_grade", REL_THRESHOLD))}
            for key in ("trajectory_id", "step_range"):
                if key in row:
                    entry[key] = row[key]
            q[row["segment_id"]] = entry
            continue
        rel = row.get("relevant") or row.get("relevant_segments") or []
        if isinstance(rel, dict):
            q.update({sid: {"label": float(v)} for sid, v in rel.items()})
        else:
            q.update({sid: {"label": float(REL_THRESHOLD)} for sid in rel})
    return gold


# --------------------------------------------------------------------------- #
# Metrics.                                                                     #
# --------------------------------------------------------------------------- #


def _span(trajectory_id: str, step_range) -> tuple:
    """(trajectory_id, start, end) with ``end`` exclusive; step_range is inclusive."""
    start, end = step_range
    return (trajectory_id, int(start), int(end) + 1)


def _spans(cands: list[dict]) -> dict[str, tuple]:
    return {
        c["segment_id"]: _span(c["provenance"]["trajectory_id"], c["provenance"]["step_range"])
        for c in cands
    }


def _gold_spans(rel: dict[str, dict], spans: dict[str, tuple]) -> list[tuple]:
    """Relevant gold segments as [(span, label)]."""
    out = []
    for sid, g in rel.items():
        if g["label"] < REL_THRESHOLD:
            continue
        if "trajectory_id" in g and "step_range" in g:
            span = _span(g["trajectory_id"], g["step_range"])
        else:
            span = spans.get(sid)
        if span is not None:
            out.append((span, g["label"]))
    return out


def _hit(cand: tuple, gold: tuple) -> bool:
    (ct, cs, ce), (gt, gs, ge) = cand, gold
    if ct != gt or ce <= cs:
        return False
    return max(0, min(ce, ge) - max(cs, gs)) / (ce - cs) >= MIN_HIT_OVERLAP


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _metrics(ranked: list[tuple], gold: list[tuple], k: int) -> dict[str, float]:
    """Recall / Precision / nDCG / MRR at k of ranked candidate spans."""
    covered, credited, hits, gains, mrr = set(), set(), 0, [], 0.0
    for rank, cand in enumerate(ranked[:k], 1):
        hit = [i for i, (span, _) in enumerate(gold) if _hit(cand, span)]
        covered.update(hit)
        if hit:
            hits += 1
            mrr = mrr or 1.0 / rank
        fresh = [i for i in hit if i not in credited]
        best = max(fresh, key=lambda i: gold[i][1]) if fresh else None
        if best is not None:
            credited.add(best)
        gains.append(2 ** max(gold[best][1], 0.0) - 1 if best is not None else 0.0)
    idcg = _dcg(sorted((2 ** max(label, 0.0) - 1 for _, label in gold), reverse=True)[:k])
    return {
        "recall": len(covered) / len(gold),
        "precision": hits / k if k else 0.0,
        "ndcg": _dcg(gains) / idcg if idcg > 0 else 0.0,
        "mrr": mrr,
    }

def evaluate(retriever: HybridRetriever, queries: list[dict],
             gold: dict[str, dict[str, dict]], spans: dict[str, tuple],
             k: int) -> dict[str, float]:
    totals = {"recall": 0.0, "precision": 0.0, "ndcg": 0.0, "mrr": 0.0}
    n = 0
    for q in queries:
        rel = _gold_spans(gold.get(q["query_id"], {}), spans)
        if not rel:
            continue
        results = retriever.retrieve(_to_stuck_state(q), top_k=k)
        ranked = [spans[r.segment_id] for r in results]
        for name, value in _metrics(ranked, rel, k).items():
            totals[name] += value
        n += 1
    if n == 0:
        raise SystemExit("no queries matched gold; check query_id alignment")
    out = {f"{name}@{k}": value / n for name, value in totals.items()}
    out["n_queries"] = n
    return out

# --------------------------------------------------------------------------- #
# Field selection.                                                            #
# --------------------------------------------------------------------------- #


def _ndcg_scorer(encoder, cands: list[dict], val_queries: list[dict],
                 gold: dict[str, dict[str, dict]], k: int):
    """Return a scorer(subset)->mean nDCG@k via dense-only retrieval.

    For each candidate-field subset the candidate pool is re-rendered and
    re-encoded, then queries are scored by cosine against the fresh pool. This
    isolates the contribution of the rendered fields, matching the selection
    objective G_c* = argmax_{G_c} nDCG@k(G_c; V).
    """
    ids = [c["segment_id"] for c in cands]
    records = [_to_record(c) for c in cands]
    spans = _spans(cands)
    # Encode queries once; only candidates change with the subset.
    q_texts = [_query_text(q) for q in val_queries]
    q_vecs = encoder(q_texts, is_query=True) if q_texts else np.zeros((0, 1))

    def scorer(subset: frozenset) -> float:
        fields = [f for f in DEFAULT_FIELDS if f in subset]
        texts = [render_candidate(r, fields) for r in records]
        cand_vecs = encoder(texts)
        index = DenseIndex(int(cand_vecs.shape[1]))
        index.add(cand_vecs, ids)
        total, n = 0.0, 0
        for q, qv in zip(val_queries, q_vecs):
            rel = _gold_spans(gold.get(q["query_id"], {}), spans)
            if not rel:
                continue
            ranked = [spans[sid] for sid, _ in index.search(qv[None, :], k=k)]
            total += _metrics(ranked, rel, k)["ndcg"]
            n += 1
        return total / n if n else 0.0

    return scorer


def _query_text(q: dict) -> str:
    from trajretrieve.retrieval import render_query  # local import keeps top clean

    return render_query(_to_stuck_state(q))


def run_field_selection(args, encoder, cands, queries, gold, k: int) -> None:
    # Validation split: 20% of the queries by default; ``--stratify-key`` draws
    # the hold-out per stratum (e.g. task type) as in the paper's stratified split.
    _, val_queries = validation_split(queries, args.val_frac, args.stratify_key, args.split_seed)
    how = f"stratified by {args.stratify_key!r}" if args.stratify_key else "tail of query list"
    print(f"[field-selection] validation split: {len(val_queries)} of {len(queries)} queries ({how})", flush=True)

    scorer = _ndcg_scorer(encoder, cands, val_queries, gold, k)
    selected, score, path = greedy_selection(scorer)
    print(f"[field-selection] selected fields: {sorted(selected)}", flush=True)
    print(f"[field-selection] nDCG@{k} = {score:.4f}", flush=True)
    for field, after in path:
        print(f"    + {field:<12} -> nDCG@{k} {after:.4f}", flush=True)


# --------------------------------------------------------------------------- #
# Reranker wiring.                                                             #
# --------------------------------------------------------------------------- #


def _build_reranker(kind: str, cfg: dict, device=None):
    if kind == "none":
        return None
    if kind == "cross":
        return CrossEncoderReranker(model_name=cfg["reranker_model"], device=device)
    if kind == "llm":
        client, model = build_llm_client(cfg)
        if client is None:
            raise SystemExit("--reranker llm requires a configured LLM endpoint in --config")
        return LLMJudgeReranker(client=client, model=model)
    raise SystemExit(f"unknown reranker {kind!r}")


# --------------------------------------------------------------------------- #
# Entry point.                                                                 #
# --------------------------------------------------------------------------- #


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    index_dir = Path(args.index)
    meta, cands, embeddings = _load_index_dir(index_dir)
    k = args.k

    encoder = BGEEncoder(model_name=meta["embedding_model"], device=args.device)
    queries = _read_jsonl(args.queries)
    gold = _load_gold(args.gold)

    if args.field_selection:
        run_field_selection(args, encoder, cands, queries, gold, k)
        return

    # Rebuild the dense (+ optional BM25) index over the persisted pool.
    dense = DenseIndex(int(embeddings.shape[1]))
    dense.add(embeddings, [c["segment_id"] for c in cands])
    bm25 = None
    if meta.get("bm25", True) and not args.no_bm25:
        texts = [c.get("candidate_text", "") for c in cands]
        bm25 = BM25Index(texts, [c["segment_id"] for c in cands])

    alpha = args.alpha if args.alpha is not None else meta.get("alpha", cfg["alpha"])
    retriever = HybridRetriever(
        encoder=encoder,
        dense_index=dense,
        candidates=_build_candidates(cands),
        bm25_index=bm25,
        reranker=_build_reranker(args.reranker, cfg, args.device),
        alpha=alpha,
        rerank_k=cfg["rerank_k"],
    )

    metrics = evaluate(retriever, queries, gold, _spans(cands), k)
    metrics["alpha"] = alpha
    print(json.dumps(metrics, indent=2))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", required=True, help="index directory from build_index.py")
    p.add_argument("--queries", required=True, help="queries JSONL (see module docstring)")
    p.add_argument("--gold", required=True, help="gold relevance JSONL (see module docstring)")
    p.add_argument("--config", default="configs/config.yaml",
                   help="YAML config for reranker model + optional LLM endpoint")
    p.add_argument("--k", type=int, default=10, help="cutoff k for recall/precision/nDCG/MRR")
    p.add_argument("--reranker", choices=("cross", "llm", "none"), default="cross",
                   help="Step-3 reranker (default: cross-encoder)")
    p.add_argument("--no-bm25", action="store_true", help="dense-only (skip BM25)")
    p.add_argument("--alpha", type=float, default=None,
                   help="hybrid weight alpha; default: the value stored with the index")
    p.add_argument("--field-selection", choices=("greedy",), default=None,
                   help="run candidate-side field selection instead of evaluation")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="validation fraction held out for --field-selection (paper: 0.2)")
    p.add_argument("--stratify-key", default=None,
                   help="query field to stratify the validation split by (e.g. task_type)")
    p.add_argument("--split-seed", type=int, default=0,
                   help="seed for the stratified validation split")
    p.add_argument("--device", default=None, help="device for the encoder and cross-encoder (cuda/cpu); default: auto")
    return p.parse_args(argv)


if __name__ == "__main__":
    main()
