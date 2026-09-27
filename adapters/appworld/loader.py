"""AppWorld trajectory loader.

Parses an AppWorld raw trajectory dump into the unified
:class:`trajretrieve.models.Trajectory` / :class:`~trajretrieve.models.Step`
model so the rest of the pipeline (segmenter -> schema -> index -> retrieval)
is environment-agnostic.

Expected input format
---------------------
A single JSON file (one experiment / model run) shaped as::

    {
      "source":          str,            # optional, default provenance source
      "trajectories": [
        {
          "task_id":          str,       # e.g. "abc1234_1"
          "task_instruction": str,       # natural-language goal d
          "success":          bool,      # episode outcome r
          "source":           str,       # optional provenance source
          "metadata":         dict,      # optional, passed through
          "method":           str,       # optional -> metadata
          "llm":              str,       # optional -> metadata
          "steps": [
            {
              "index":       int,        # 0-based step ordinal t
              "action_raw":  str,        # the executed code / API call a_t
              "api_apps":    list[str],  # apps touched (kept in raw)
              "observation": str,        # execution result / error o_t
              "reward":      float,      # optional per-step reward
              "thought":     str | None  # optional agent reasoning
            },
            ...
          ]
        },
        ...
      ]
    }

A bare ``list`` of trajectory objects (no top-level wrapper) is also accepted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from trajretrieve.models import Step, Trajectory

# Execution-error detection is left to the segmenter's ``error_fn``; the raw
# observation is kept verbatim here so no signal is lost.

_OUTCOME = {True: "success", False: "fail"}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _iter_records(payload: Any) -> Iterator[dict]:
    """Yield per-trajectory dicts from either the wrapper or a bare list."""
    if isinstance(payload, dict):
        yield from payload.get("trajectories", [])
    elif isinstance(payload, list):
        yield from payload
    # anything else: nothing to iterate


def _build_step(raw: dict, fallback_index: int) -> Step:
    return Step(
        index=fallback_index if raw.get("index") is None else int(raw["index"]),
        action=raw.get("action_raw") or "",
        observation=raw.get("observation") or "",
        metadata={
            "api_apps": raw.get("api_apps") or [],
            "reward": raw.get("reward"),
            "thought": raw.get("thought"),
        },
        raw=raw,
    )


def _build_trajectory(record: dict, source: str | None, fallback_id: str = "traj_0") -> Trajectory:
    task_id = record.get("task_id")
    steps = [
        _build_step(s, i) for i, s in enumerate(record.get("steps") or [])
    ]
    outcome = _OUTCOME.get(record.get("success"), "unknown")
    return Trajectory(
        trajectory_id=str(task_id) if task_id is not None else fallback_id,
        task_instruction=record.get("task_instruction") or "",
        steps=steps,
        metadata={
            "source": record.get("source") or source,
            "method": record.get("method"),
            "llm": record.get("llm"),
            **(record.get("metadata") or {}),
        },
        outcome=outcome,
        task_id=str(task_id) if task_id is not None else None,
    )


def load_trajectories(path: str | Path) -> Iterator[Trajectory]:
    """Stream :class:`Trajectory` objects from an AppWorld dump at ``path``.

    ``path`` may point at a single JSON file or a directory of such files; in
    the directory case every ``*.json`` file is loaded in sorted order.
    """
    p = Path(path)
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    for fp in files:
        payload = _read_json(fp)
        source = payload.get("source") if isinstance(payload, dict) else None
        for n, record in enumerate(_iter_records(payload)):
            yield _build_trajectory(record, source, f"{fp.stem}_{n}")
