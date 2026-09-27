"""Offline index construction.

Turns a corpus of raw agent trajectories into a searchable segment index:

    trajectories  ->  segment  ->  schema fields R(s)  ->  encode  ->  index

Concretely, for ``--env {appworld,alfworld,swe}`` it

  (a) loads trajectories from ``--trajectories`` via
      ``adapters.<env>.load_trajectories``;
  (b) segments each trajectory with ``Segmenter(get_mapper(env))``;
  (c) extracts a ``StructuredRecord`` per segment via ``FieldExtractor`` -- the
      offline LLM tier (summary / error / resolution) is enabled with
      ``--annotate`` and read from ``configs/config.yaml``; otherwise only the
      rule-derived fields (phase / targets / provenance) are produced;
  (d) renders the candidate text with ``render_candidate``;
  (e) encodes candidates with ``BGEEncoder`` and builds a ``DenseIndex``
      (plus, unless ``--no-bm25``, a ``BM25Index``);
  (f) saves the embedding pool + candidate metadata to ``--out``.

Output layout (consumed by ``scripts/evaluate.py``)::

    <out>/embeddings.npy     float32 [N, dim] candidate pool (row i <-> id i)
    <out>/candidates.jsonl   one JSON record per segment (id, fields, provenance)
    <out>/meta.json          env, embedding_model, dim, alpha, counts
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Ensure the package root is importable when run as a bare script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trajretrieve.config import load_config
from trajretrieve.retrieval import render_candidate
from trajretrieve.retrieval.encoders import BGEEncoder
from trajretrieve.index import BM25Index, DenseIndex
from trajretrieve.schema.action_category import get_mapper
from trajretrieve.schema.fields import FieldExtractor
from trajretrieve.segmenter import Segmenter

# Map the CLI env name onto its dataset adapter package.
_ENV_MODULES = {
    "appworld": "adapters.appworld",
    "swe": "adapters.swe",
    "swepro": "adapters.swe",
    "alfworld": "adapters.alfworld",
}


def _load_adapter(env: str):
    """Return the ``adapters.<env>`` module (load_trajectories + SegmentAnnotator)."""
    mod_name = _ENV_MODULES.get(env)
    if mod_name is None:
        raise SystemExit(f"unknown env {env!r}; choose from {sorted(_ENV_MODULES)}")
    return importlib.import_module(mod_name)


def _build_annotator(adapter, config_path: str):
    """Instantiate the env's offline SegmentAnnotator (no-op without creds)."""
    # Every adapter exposes SegmentAnnotator(config_path=...) and degrades to a
    # no-op when the config has no usable key, so we can construct it uniformly.
    return adapter.SegmentAnnotator(config_path=config_path)


def _candidate_record(segment, record, text: str) -> dict:
    """Serialize one candidate (segment_id, record fields, provenance, text)."""
    prov = asdict(segment.provenance)
    # step_range is a tuple -> list after asdict; keep JSON-friendly.
    prov["step_range"] = list(segment.provenance.step_range)
    return {
        "segment_id": segment.provenance.segment_id,
        "phase": record.phase,
        "summary": record.summary,
        "targets": list(record.targets),
        "error": record.error,
        "resolution": record.resolution,
        "provenance": prov,
        "candidate_text": text,
    }


def build(args: argparse.Namespace) -> None:
    adapter = _load_adapter(args.env)
    mapper = get_mapper(args.env)
    segmenter = Segmenter(mapper, max_segment_length=args.max_segment_length)

    # Offline LLM field generation is opt-in.
    annotator = _build_annotator(adapter, args.config) if args.annotate else None
    if annotator is None:
        extractor = FieldExtractor(env=args.env)
    else:
        # one annotator call per segment serves the LLM fields and E_llm(s)
        last: dict = {}

        def annotate_once(segment):
            sid = segment.provenance.segment_id
            if last.get("sid") != sid:
                last.update(sid=sid, out=annotator(segment) or {})
            return last["out"]

        def completion(segment, rule_targets):
            out = annotate_once(segment)
            return [t for t in (out.get("targets") or []) if isinstance(t, str)]

        extractor = FieldExtractor(llm=annotate_once, target_completion=completion, env=args.env)

    candidates: list[dict] = []
    n_traj = 0
    seen_ids: dict[str, int] = {}
    for traj in adapter.load_trajectories(args.trajectories):
        n_traj += 1
        # make trajectory ids unique
        base = traj.trajectory_id
        k = seen_ids.get(base, 0) + 1
        while k > 1 and f"{base}#{k}" in seen_ids:
            k += 1
        seen_ids[base] = k
        if k > 1:
            traj.trajectory_id = f"{base}#{k}"
            seen_ids[traj.trajectory_id] = 1
        for segment in segmenter.segment(traj):
            record = extractor.extract(segment)
            text = (render_candidate(record, args.candidate_fields)
                    if args.candidate_fields else render_candidate(record))
            candidates.append(_candidate_record(segment, record, text))

    if not candidates:
        raise SystemExit(
            f"no segments produced from {args.trajectories!r}; check the path "
            f"and that it matches adapters.{args.env}.load_trajectories's format."
        )

    print(f"[build] {n_traj} trajectories -> {len(candidates)} segments", flush=True)

    # (e) encode candidate texts -> dense pool. BGE handles empty strings; a few
    # rule-only candidates may render empty when no LLM fields are present.
    encoder = BGEEncoder(model_name=args.embedding_model, device=args.device)
    texts = [c["candidate_text"] for c in candidates]
    ids = [c["segment_id"] for c in candidates]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate segment ids in the candidate pool")
    embeddings = encoder(texts)
    dim = int(embeddings.shape[1])

    # Validate the dense index round-trips (also surfaces the active backend).
    index = DenseIndex(dim)
    index.add(embeddings, ids)
    print(f"[build] dense index: {len(index)} vectors, dim={dim}, "
          f"backend={index.backend}", flush=True)

    if not args.no_bm25:
        BM25Index(texts, ids)  # validate sparse index builds from the same texts

    # (f) persist the pool + metadata. Embeddings are saved as a plain float32
    # matrix aligned row-for-row with candidates.jsonl; the indices are rebuilt
    # at evaluation time (cheap, and keeps backend choice flexible).
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", embeddings.astype(np.float32))
    with (out / "candidates.jsonl").open("w", encoding="utf-8") as fh:
        for c in candidates:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    meta = {
        "env": args.env,
        "embedding_model": args.embedding_model,
        "dim": dim,
        "n_trajectories": n_traj,
        "n_segments": len(candidates),
        "alpha": args.alpha,
        "annotated": bool(args.annotate),
        "bm25": not args.no_bm25,
        "max_segment_length": args.max_segment_length,
        "candidate_fields": list(args.candidate_fields) if args.candidate_fields else None,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[build] wrote index to {out}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", required=True, choices=sorted(_ENV_MODULES),
                   help="environment whose adapter + action->category mapper to use")
    p.add_argument("--trajectories", required=True,
                   help="path to the raw trajectory dump (file or directory)")
    p.add_argument("--out", required=True, help="output index directory")
    p.add_argument("--annotate", action="store_true",
                   help="enable the offline LLM (summary/error/resolution); reads --config")
    p.add_argument("--config", default="configs/config.yaml",
                   help="YAML config for model ids + optional LLM endpoint")
    p.add_argument("--no-bm25", action="store_true",
                   help="skip building/validating the sparse BM25 index")
    p.add_argument("--max-segment-length", type=int, default=None,
                   help="L_max for short-segment merging (default: none)")
    p.add_argument("--device", default=None,
                   help="encoder device (e.g. cuda, cpu); default: auto")
    p.add_argument("--candidate-fields", default=None,
                   help="comma-separated candidate-side field subset "
                        "(overrides candidate_fields in --config)")
    p.add_argument("--embedding-model", default=None,
                   help="override config embedding_model (HuggingFace id)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.embedding_model is None:
        args.embedding_model = cfg["embedding_model"]
    args.alpha = cfg["alpha"]
    if args.candidate_fields:
        args.candidate_fields = tuple(
            f.strip() for f in args.candidate_fields.split(",") if f.strip())
    elif cfg.get("candidate_fields"):
        args.candidate_fields = tuple(cfg["candidate_fields"])
    return args


if __name__ == "__main__":
    build(parse_args())
