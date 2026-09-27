"""ALFWorld dataset adapter for TrajRetrieve.

Thin, environment-specific glue (loader + offline LLM annotator) over the
unified ``trajretrieve`` pipeline. See :mod:`.loader` for the expected input
format and :mod:`.annotate` for the field-generation prompt.
"""

from __future__ import annotations

from .annotate import SegmentAnnotator
from .loader import load_trajectories

__all__ = ["load_trajectories", "SegmentAnnotator"]
