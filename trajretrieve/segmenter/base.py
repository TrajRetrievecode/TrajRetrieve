"""Execution-segment construction.

Given a trajectory and an action->category mapping ``f``, the segmenter:

  1. categorizes every step  (f(a_t))
  2. inserts boundaries via the predicate B(t)
  3. greedily merges short, same-category segments

producing non-overlapping segments that tile [1, T]. The construction is
environment-agnostic; the only environment-specific input is the mapper ``f``
(see :mod:`trajretrieve.schema.action_category`).
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from ..models import (
    ActionCategory,
    Provenance,
    Segment,
    Step,
    Trajectory,
)
from ..schema.action_category import Mapper

# default observation-level error / recovery detection ---------------------- #

_ERROR_RE = re.compile(
    r"(traceback|error|exception|keyerror|typeerror|valueerror|attributeerror"
    r"|failed|assertionerror|fatal|cannot|not found|nothing happens)",
    re.IGNORECASE,
)

ErrorFn = Callable[[str], bool]


def default_error_fn(observation: str) -> bool:
    """err(o_t): does the observation carry an explicit error signal?"""
    return bool(_ERROR_RE.search(observation or ""))


class Segmenter:
    """Action-category change-point segmenter.

    Parameters
    ----------
    mapper:
        The action->category function f.
    max_segment_length:
        Optional constraint L_max on merging: a same-category merge is rejected
        once the merged length would exceed it (useful in long-segment
        environments). It bounds merge growth, not the length of a homogeneous
        run produced by boundary insertion alone.
    error_fn:
        Observation-level error detector err(.). Defaults to a regex.
    """

    def __init__(
        self,
        mapper: Mapper,
        max_segment_length: Optional[int] = None,
        error_fn: ErrorFn = default_error_fn,
    ) -> None:
        self.mapper = mapper
        self.max_segment_length = max_segment_length
        self.error_fn = error_fn

    # ------------------------------------------------------------------ #

    def categorize(self, trajectory: Trajectory) -> None:
        """Fill ``step.category`` / ``step.is_error`` / ``step.is_recovery``."""
        prev_error = False
        for step in trajectory.steps:
            if step.category is None:
                step.category = self.mapper(step.action, step.observation)
            step.is_error = self.error_fn(step.observation)
            # recov(o_t): transition from an error state back to normal.
            step.is_recovery = prev_error and not step.is_error
            prev_error = step.is_error

    def boundary(self, steps: list[Step], t: int) -> bool:
        """B(t): insert a boundary *after* step index t (0-based here).

        B(t) = 1[f(a_t) != f(a_{t+1})]
             v 1[err(o_t) v recov(o_{t+1})]
             v 1[f(a_t) == submit]
        """
        cur, nxt = steps[t], steps[t + 1]
        if cur.category != nxt.category:
            return True
        if cur.is_error or nxt.is_recovery:
            return True
        if cur.category == ActionCategory.SUBMIT:
            return True
        return False

    # ------------------------------------------------------------------ #

    def segment(self, trajectory: Trajectory) -> list[Segment]:
        steps = trajectory.steps
        if not steps:
            return []
        self.categorize(trajectory)

        # 1) boundary insertion -> raw [start, end] index ranges (0-based)
        ranges: list[tuple[int, int]] = []
        start = 0
        for t in range(len(steps) - 1):
            if self.boundary(steps, t):
                ranges.append((start, t))
                start = t + 1
        ranges.append((start, len(steps) - 1))

        # 2) greedy left-to-right short-segment merging
        merged: list[tuple[int, int]] = [ranges[0]]
        for rng in ranges[1:]:
            li, lj = merged[-1]
            if self._should_merge(steps, (li, lj), rng):
                merged[-1] = (li, rng[1])
            else:
                merged.append(rng)

        return [self._make_segment(trajectory, idx, r) for idx, r in enumerate(merged)]

    def _should_merge(
        self, steps: list[Step], left: tuple[int, int], right: tuple[int, int]
    ) -> bool:
        """Merge(s_m, s_{m+1}) = 1[f(a_{i_m}) = f(a_{i_{m+1}})] and 1[not beta].

        beta is true when an error boundary or a submission boundary separates
        the two segments. The boundary between them was inserted by B(t), so it
        came from a category change, an error, a recovery, or a submission. Only
        the recovery case may be merged back: it is a recovery transition
        splitting an otherwise homogeneous action run.
        """
        li, lj = left
        ri, rj = right
        # 1[f(a_{i_m}) = f(a_{i_{m+1}})]: same category at the two segment heads.
        if steps[li].category != steps[ri].category:
            return False
        # beta: submission boundary.
        if steps[lj].category == ActionCategory.SUBMIT:
            return False
        # The split must be a recovery transition. An error step also sets
        # is_recovery on its successor, so require the recovery marker and
        # treat any other error-induced split as an error boundary (beta).
        if not steps[ri].is_recovery:
            return False
        if self.max_segment_length is not None and (rj - li + 1) > self.max_segment_length:
            return False
        return True

    @staticmethod
    def _make_segment(trajectory: Trajectory, idx: int, rng: tuple[int, int]) -> Segment:
        i, j = rng
        seg_steps = trajectory.steps[i : j + 1]
        seg_id = f"{trajectory.trajectory_id}::seg_{idx:03d}"
        prov = Provenance(
            task_id=trajectory.task_id,
            trajectory_id=trajectory.trajectory_id,
            segment_id=seg_id,
            step_range=(i, j),
            source=trajectory.metadata.get("source"),
        )
        return Segment(
            trajectory_id=trajectory.trajectory_id,
            step_range=(i, j),
            provenance=prov,
            steps=seg_steps,
        )
