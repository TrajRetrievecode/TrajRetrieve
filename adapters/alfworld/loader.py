"""ALFWorld trajectory loader for TrajRetrieve.

Parses an ALFWorld trajectory dump into the unified
:class:`trajretrieve.models.Trajectory` / :class:`~trajretrieve.models.Step`
model so the shared segmentation / schema / retrieval pipeline can consume it
without any environment-specific branching downstream.

Expected input format
---------------------
A JSON file (or a directory of them) shaped like the AgentInstruct ALFWorld corpus::

    {
      "metadata": {...},                 # optional, ignored
      "trajectories": [
        {
          "task_instance_id": "example_0",          # -> Trajectory.task_id
          "task_description": "find two laptop ...",  # -> task_instruction
          "total_steps": 14,                          # optional
          "source": "agentinstruct",                  # optional -> metadata
          "state_action_pairs": [
            {"step_id": 1,
             "state":  "You are in the middle of a room ...",  # seen before the action
             "action": "go to diningtable 1"}                  # -> action
          ]
        }
      ]
    }

The following field aliases are accepted:

  * step list:   ``state_action_pairs`` | ``steps``
  * observation: ``state`` | ``observation`` | ``obs``
  * action:      ``action`` | ``command``
  * id:          ``task_instance_id`` | ``trajectory_id`` | ``id``

The file is read from a caller-supplied ``path``.
ALFWorld AgentInstruct demonstrations are successful by construction, so
``outcome`` is set to ``"success"`` (overridable via a per-record ``outcome``
field if one is present).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from trajretrieve.models import Step, Trajectory

_STEP_KEYS = ("state_action_pairs", "steps")
_OBS_KEYS = ("state", "observation", "obs")
_ACTION_KEYS = ("action", "command")
_ID_KEYS = ("task_instance_id", "trajectory_id", "id")


def _first(record: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return default


def _build_steps(raw_steps: list[dict[str, Any]], task_instruction: str) -> list[Step]:
    # o_t of a_t is the next pair's state (empty after the last action)
    states = [str(_first(raw, _OBS_KEYS)).strip() for raw in raw_steps]
    before = bool(raw_steps) and "state" in raw_steps[0]
    steps: list[Step] = []
    for index, raw in enumerate(raw_steps):
        action = str(_first(raw, _ACTION_KEYS)).strip()
        if before:
            observation = states[index + 1] if index + 1 < len(states) else ""
        else:
            observation = states[index]
        steps.append(
            Step(
                index=index,
                action=action,
                observation=observation,
                # task_instruction is stamped per-step so a Segment (which only
                # carries Step objects) can recover its task context offline.
                metadata={
                    "step_id": raw.get("step_id", index + 1),
                    "task_instruction": task_instruction,
                },
                raw=dict(raw),
            )
        )
    return steps


def load_trajectories(path: str) -> Iterator[Trajectory]:
    """Yield unified :class:`Trajectory` objects from an ALFWorld dump at ``path``.

    ``path`` is a JSON file or a directory of them (searched recursively).
    """
    p = Path(path)
    records: list = []
    for fp in (sorted(p.rglob("*.json")) if p.is_dir() else [p]):
        data = json.loads(fp.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("trajectories", [])
        records.extend(r for r in data if isinstance(r, dict))

    for offset, record in enumerate(records):
        trajectory_id = str(_first(record, _ID_KEYS, default=f"alfworld_{offset}"))
        raw_steps = _first(record, _STEP_KEYS, default=[]) or []
        task_instruction = str(record.get("task_description") or "").strip()
        yield Trajectory(
            trajectory_id=trajectory_id,
            task_instruction=task_instruction,
            steps=_build_steps(raw_steps, task_instruction),
            metadata={"source": record.get("source", "agentinstruct")},
            outcome=str(record.get("outcome") or "success"),
            task_id=trajectory_id,
        )
