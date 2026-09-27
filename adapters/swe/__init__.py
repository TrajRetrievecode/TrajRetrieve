"""SWE-Bench Pro dataset adapter for TrajRetrieve.

Thin adapter that turns mini-swe-agent trajectory dumps into the
unified :mod:`trajretrieve` data model and supplies the environment-specific
offline LLM annotator for schema field extraction.

  * :func:`load_trajectories` -- parse raw dumps into ``Trajectory`` objects.
  * :class:`SegmentAnnotator` -- the ``SegmentLLM`` producing
    ``summary`` / ``error`` / ``resolution`` for a segment.
"""

from __future__ import annotations

from .annotate import SegmentAnnotator
from .loader import load_trajectories

__all__ = ["load_trajectories", "SegmentAnnotator"]
