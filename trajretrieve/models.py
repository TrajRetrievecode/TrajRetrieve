"""Core data models for TrajRetrieve.

These dataclasses define the core objects:

  * ``Step``          -- one (action, observation) pair a_t, o_t with metadata.
  * ``Trajectory``    -- tau = (d, [(a_1,o_1),...,(a_T,o_T)], m, r).
  * ``Segment``       -- s = (tau, [i, j], phi); the basic retrieval unit.
  * ``StructuredRecord`` -- R(s) = (phase, summary, targets, error, resolution, phi).
  * ``StuckState``    -- q = (q_fail, q_goal, q_err, q_sub, q_phase).
  * ``ActionCategory``-- the seven-label alphabet F used by the mapping f.

The module is dependency-free (stdlib only) so it can be imported by every
stage of the pipeline without pulling in torch / sentence-transformers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# AppWorld task variants share a family stem: "<hex>_<n>".
_VARIANT_RE = re.compile(r"([0-9a-f]{6,})_\d+")


class ActionCategory(str, Enum):
    """The finite alphabet F of action categories.

    Seven functional groups observed across the three target environments:
    information gathering, state modification (two kinds), traversal,
    verification, authentication, and goal closure.
    """

    DATA_FETCH = "data_fetch"   # information gathering (queries, reads, lookups)
    ACTION = "action"           # state modification via API / world interaction
    NAVIGATION = "navigation"   # environment traversal (go to, cd, locate)
    EDIT = "edit"               # source/file modification
    TEST = "test"               # verification (run tests, authoring checks)
    CREDENTIAL = "credential"   # authentication / secret retrieval
    SUBMIT = "submit"           # goal closure / task completion

    @classmethod
    def from_value(cls, value: str) -> "ActionCategory":
        return cls(value)


# Categories that close a (sub)goal (the submit term of B(t)).
TERMINAL_CATEGORIES = frozenset({ActionCategory.SUBMIT})


@dataclass
class Step:
    """A single execution step: action a_t and its observation o_t.

    ``category`` is filled by the action->category mapping f; ``is_error`` /
    ``is_recovery`` carry the observation-level signals used during boundary
    insertion. ``raw`` keeps the environment-specific payload verbatim so no
    information is lost between stages.
    """

    index: int
    action: str
    observation: str = ""
    category: Optional[ActionCategory] = None
    is_error: bool = False
    is_recovery: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """tau = (d, [(a_1,o_1),...,(a_T,o_T)], m, r)."""

    trajectory_id: str
    task_instruction: str
    steps: list[Step] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    outcome: str = "unknown"  # "success" | "fail" | "unknown"
    task_id: Optional[str] = None

    @property
    def length(self) -> int:
        return len(self.steps)


@dataclass
class Provenance:
    """phi: the provenance metadata carried by every segment.

    Preserved end-to-end so downstream modules can trace evidence sources and
    so the online provenance filter can exclude
    candidates that share a provenance link with the current task.
    """

    task_id: Optional[str]
    trajectory_id: str
    segment_id: str
    step_range: tuple[int, int]
    source: Optional[str] = None  # e.g. originating model / split / repo

    def parent_stem(self) -> Optional[str]:
        """Task-family stem of an AppWorld task variant (``"abc1234_2" -> "abc1234"``)."""
        tid = self.task_id
        if not isinstance(tid, str):
            return None
        m = _VARIANT_RE.fullmatch(tid)
        return m.group(1) if m else tid


@dataclass
class Segment:
    """s = (tau, [i, j], phi). The basic retrieval unit."""

    trajectory_id: str
    step_range: tuple[int, int]
    provenance: Provenance
    steps: list[Step] = field(default_factory=list)
    record: Optional["StructuredRecord"] = None

    @property
    def length(self) -> int:
        return self.step_range[1] - self.step_range[0] + 1

    @property
    def categories(self) -> list[ActionCategory]:
        return [s.category for s in self.steps if s.category is not None]


@dataclass
class StructuredRecord:
    """R(s) = (phase, summary, targets, error, resolution, phi).

    Empty / unavailable fields are left as ``None`` (or an empty list) so the
    template stage can skip them; the same schema therefore accommodates both
    richly- and sparsely-populated records without injecting noise.
    """

    phase: Optional[str] = None
    summary: Optional[str] = None
    targets: list[str] = field(default_factory=list)
    error: Optional[str] = None
    resolution: Optional[str] = None
    provenance: Optional[Provenance] = None

    # canonical field order used by field-subset selection
    FIELDS: tuple[str, ...] = (
        "phase",
        "summary",
        "targets",
        "error",
        "resolution",
        "provenance",
    )

    def get(self, name: str) -> Any:
        return getattr(self, name)

    def non_empty(self, name: str) -> bool:
        v = getattr(self, name)
        return bool(v) if not isinstance(v, Provenance) else True


@dataclass
class StuckState:
    """q = (q_fail, q_goal, q_err, q_sub, q_phase).

    The online query side. Fields are extracted from the live agent state when
    the stuck-state detector fires.
    """

    failed_actions: list[str] = field(default_factory=list)  # q_fail
    goal: str = ""                                            # q_goal
    error: str = ""                                           # q_err
    subgoals: list[str] = field(default_factory=list)         # q_sub
    phase: Optional[str] = None                               # q_phase
    task_id: Optional[str] = None  # for provenance filtering
