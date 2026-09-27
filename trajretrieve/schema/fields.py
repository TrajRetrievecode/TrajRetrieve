"""Schema-based field extraction R(s).

Each segment is turned into a six-field structured record
``(phase, summary, targets, error, resolution, provenance)``. Fields fall into
three derivation tiers:

  * deterministic rule  -- ``phase`` (from the action-category + error sequence)
                           and ``provenance`` (carried from segmentation).
  * rule + LLM completion -- ``targets`` (regex entities, optionally completed).
  * offline LLM         -- ``summary``, ``error``, ``resolution``.

The LLM tier is injected as a callable so the package runs without any API
access (the three LLM fields are then left ``None`` and skipped by templates).
All field values are computed in a single offline pass -- no online cost.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Optional, Protocol

from ..models import ActionCategory, Segment, StructuredRecord

# --------------------------------------------------------------------------- #
# Phase derivation: a deterministic instance of the unified taxonomy.         #
# --------------------------------------------------------------------------- #


def derive_phase(segment: Segment) -> str:
    """phase(s): a deterministic function of [f(a_i..j)] and [err(o_i..j)]."""
    cats = segment.categories
    if not cats:
        return "unknown"
    has_error = any(s.is_error for s in segment.steps)
    has_recovery = any(s.is_recovery for s in segment.steps)

    if ActionCategory.SUBMIT in cats:
        return "submission"
    if ActionCategory.CREDENTIAL in cats:
        return "auth"
    # an error, or recovery from one, with modification/verification = repair
    if (has_error or has_recovery) and any(
        c in (ActionCategory.EDIT, ActionCategory.ACTION, ActionCategory.TEST) for c in cats
    ):
        return "repair"
    dominant, _ = Counter(c.value for c in cats).most_common(1)[0]
    return {
        "data_fetch": "exploration",
        "navigation": "navigation",
        "edit": "edit",
        "test": "test",
        "action": "execution",
    }.get(dominant, dominant)


# --------------------------------------------------------------------------- #
# Target-entity extraction: rule side E_rule(s).                              #
# --------------------------------------------------------------------------- #

# shared regex rules for target entities
_TARGET_RES = (
    re.compile(r"apis\.[a-z_]+\.[a-z_]+"),          # AppWorld API symbols
    re.compile(r"\b[\w/]+\.(?:py|js|ts|java|go|rb|c|cpp|h)\b"),  # file paths
    re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\([^)]*\)"),           # call-like tokens
    re.compile(r"'([^']{2,40})'"),                  # quoted identifiers
)

# AppWorld: the API called in each step, e.g. ``spotify.login``.
_AW_API_RE = re.compile(r"apis\.([a-z_]+\.[a-z_]+)")

# ALFWorld: the objects named in each text command, e.g. ``pen 1``, ``shelf 5``.
_ALF_OBJ_RE = re.compile(r"\b[a-z]+ \d+\b")


def _shared_targets(step) -> list[str]:
    out = []
    for rx in _TARGET_RES:
        for m in rx.findall(f"{step.action} {step.observation}"):
            tok = (m if isinstance(m, str) else m[0]).strip()
            if 2 <= len(tok) <= 80:
                out.append(tok)
    return out


def _appworld_targets(step) -> list[str]:
    return _AW_API_RE.findall(step.action)


def _alfworld_targets(step) -> list[str]:
    return _ALF_OBJ_RE.findall(step.action.lower())


# Per-environment rules; other environments use the shared rules.
_TARGET_RULES = {"appworld": _appworld_targets, "alfworld": _alfworld_targets}


def extract_targets_rule(segment: Segment, env: Optional[str] = None) -> list[str]:
    """E_rule(s): regex-extracted structured entities (deduplicated, ordered)."""
    rule = _TARGET_RULES.get((env or "").strip().lower(), _shared_targets)
    seen: dict[str, None] = {}
    for step in segment.steps:
        for tok in rule(step):
            seen.setdefault(tok, None)
    return list(seen)[:16]


class SegmentLLM(Protocol):
    """Offline LLM that generates summary / error / resolution for a segment.

    Implementations receive the raw step sequence and return a mapping with any
    of the keys ``summary``, ``error``, ``resolution``. See
    ``adapters/*/annotate.py`` for the per-environment prompt wiring.
    """

    def __call__(self, segment: Segment) -> dict[str, Optional[str]]: ...


class FieldExtractor:
    """Builds ``StructuredRecord`` objects R(s) for a stream of segments."""

    def __init__(
        self,
        llm: Optional[SegmentLLM] = None,
        target_completion: Optional[Callable[[Segment, list[str]], list[str]]] = None,
        env: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.target_completion = target_completion
        self.env = env

    def extract(self, segment: Segment) -> StructuredRecord:
        targets = extract_targets_rule(segment, self.env)
        if self.target_completion is not None:           # E_rule(s) u E_llm(s)
            extra = self.target_completion(segment, targets)
            for t in extra:
                if t not in targets:
                    targets.append(t)

        summary = error = resolution = None
        if self.llm is not None:
            out = self.llm(segment) or {}
            summary = out.get("summary")
            error = out.get("error")
            resolution = out.get("resolution")

        record = StructuredRecord(
            phase=derive_phase(segment),
            summary=summary,
            targets=targets,
            error=error,
            resolution=resolution,
            provenance=segment.provenance,
        )
        segment.record = record
        return record
