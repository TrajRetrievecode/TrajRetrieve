"""Role-specific (asymmetric) templating.

Queries describe an unresolved error state; candidates record a verified repair
path. Their information structures differ, so we render them with *different*
templates and feed both into the same encoder. ``render_candidate`` uses the
field subset G_c* chosen by field selection; ``render_query`` extracts fields from
the live stuck state q. Empty fields are skipped so the same template handles
both densely- and sparsely-populated records.
"""

from __future__ import annotations

from typing import Iterable

from ..models import StructuredRecord, StuckState

# human-readable labels for candidate fields (provenance is never rendered)
_CANDIDATE_LABELS = {
    "phase": "Phase",
    "summary": "Action",
    "targets": "Targets",
    "error": "Error",
    "resolution": "Fix",
}

DEFAULT_CANDIDATE_FIELDS = ("phase", "summary", "targets", "error", "resolution")


def _fmt(value) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def render_candidate(
    record: StructuredRecord,
    fields: Iterable[str] = DEFAULT_CANDIDATE_FIELDS,
) -> str:
    """render_c(s; G_c*): compose the selected candidate fields, skip empties."""
    parts: list[str] = []
    for f in fields:
        if f == "provenance":
            continue
        if not record.non_empty(f):
            continue
        label = _CANDIDATE_LABELS.get(f, f.capitalize())
        parts.append(f"{label}: {_fmt(record.get(f))}")
    return "\n".join(parts)


def render_query(q: StuckState) -> str:
    """render_q(q): compose the query-side execution requirement."""
    parts: list[str] = []
    if q.goal:
        parts.append(f"Task: {q.goal}")
    if q.phase:
        parts.append(f"Phase: {q.phase}")
    if q.failed_actions:
        parts.append(f"Stuck: {_fmt(q.failed_actions)}")
    if q.error:
        parts.append(f"Error: {q.error}")
    if q.subgoals:
        parts.append(f"Should: {_fmt(q.subgoals)}")
    return "\n".join(parts)


def render_raw_step_text(steps, max_chars: int = 2000) -> str:
    """Raw-text rendering: concatenated (action, observation) steps.

    Used for the Dense-Raw baseline and the raw-text ablation.
    """
    chunks = []
    for s in steps:
        chunks.append(f"Step {getattr(s, 'index', '?')}: {s.action}\n-> {s.observation}")
    return "\n".join(chunks)[:max_chars]
