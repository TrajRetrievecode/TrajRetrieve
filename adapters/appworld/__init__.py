"""AppWorld dataset adapter for TrajRetrieve.

Thin adapter that maps AppWorld raw trajectory dumps onto the
unified pipeline:

  * :func:`load_trajectories` -- raw dump -> ``Trajectory`` / ``Step`` stream.
  * :class:`SegmentAnnotator` -- offline ``SegmentLLM`` producing the
    summary / error / resolution schema fields.

Action categorization is supplied by ``trajretrieve.schema.action_category``
via ``get_mapper('appworld')``; this adapter adds only the I/O and the
environment-specific annotation prompt.
"""

from .annotate import SegmentAnnotator
from .loader import load_trajectories

__all__ = ["load_trajectories", "SegmentAnnotator"]
