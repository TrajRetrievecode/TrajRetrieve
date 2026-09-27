"""TrajRetrieve: segment-level structured retrieval over agent trajectories.

Pipeline:

  segmenter  -> schema -> index -> retrieval
"""

from .models import (
    ActionCategory,
    Provenance,
    Segment,
    Step,
    StructuredRecord,
    StuckState,
    Trajectory,
)

__all__ = [
    "ActionCategory",
    "Provenance",
    "Segment",
    "Step",
    "StructuredRecord",
    "StuckState",
    "Trajectory",
]

__version__ = "0.1.0"
