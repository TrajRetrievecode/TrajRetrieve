"""SWE-Bench Pro trajectory loader for TrajRetrieve.

Parses a SWE-Bench Pro trajectory dump into the unified
:class:`trajretrieve.models.Trajectory` / :class:`~trajretrieve.models.Step`
model so the shared segmenter and schema-based field extractor can run
unchanged across environments.

Expected input record format
----------------------------
A *single* SWE-Bench Pro trajectory is one JSON object (one ``*.json`` file).
The relevant fields are::

    {
      "metadata": {
        "instance_id": "instance_owner__repo-<commit>",  # task id; repo "owner/repo"
        "exit_status": "Submitted",                # agent termination status
        "scores": {"resolved": 1}                  # 1 = patch resolved the issue
      },
      "transcripts": [
        {
          "messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "<pr_description>...</pr_description>"},
            {"role": "assistant", "content": "...<bash_code>CMD</bash_code>",
                                  "tool_calls": [...]},   # optional, tool-call encoding
            {"role": "user",      "content": "<returncode>0</returncode>"
                                             "<output>...</output>",
                                  "tool_call_id": "..."}, # or role == "tool"
            ...
          ]
        }
      ]
    }

Files with top-level ``messages``, ``instance_id`` and ``info`` are read the
same way.

Two on-disk action encodings are supported transparently:

  * **inline**: the assistant emits the shell command inside
    ``<bash_code>...</bash_code>`` or a fenced ``bash`` block, and the following
    ``user`` turn carries ``<returncode>N</returncode><output>...</output>``.
  * **tool-call**: the assistant emits ``tool_calls`` with a
    ``command`` argument and the observation arrives as a ``role: "tool"``
    message matched by ``tool_call_id`` (``Exit code: N`` prefix).

Mapping into the unified model
-------------------------------
  * ``Step.action``      <- the shell command of the turn.
  * ``Step.observation`` <- the command's stdout/stderr (+ exit code line).
  * ``Step.metadata``    <- ``{"exit_code": int|None, "task_instruction": str}``.
  * ``Trajectory.task_id``  <- ``metadata.instance_id``.
  * ``Trajectory.outcome``  <- ``"success"`` if ``scores.resolved`` truthy,
                               ``"fail"`` if explicitly unresolved, else
                               ``"unknown"``.

``path`` may be a single ``*.json`` file or a directory, searched recursively.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Iterator, Optional

from trajretrieve.models import Step, Trajectory

# inline encoding: <bash_code>...</bash_code> or a fenced bash block
_BASH_RE = re.compile(r"<bash_code>(.*?)</bash_code>|```bash\s*\n(.*?)```", re.DOTALL)
_RETURNCODE_RE = re.compile(r"<returncode>(-?\d+)</returncode>", re.DOTALL)
_OUTPUT_RE = re.compile(r"<output>(.*?)</output>", re.DOTALL)
# tool-call encoding: observation carries an "Exit code: N" prefix line.
_EXIT_PREFIX_RE = re.compile(r"Exit code:\s*(-?\d+)")


def _as_text(content) -> str:
    """Coerce a message ``content`` (str or structured list) to plain text."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _repo_from_instance(instance_id: Optional[str]) -> Optional[str]:
    """Derive ``owner/repo`` from an ``[instance_]<owner>__<repo>-<suffix>`` id."""
    if not instance_id or "__" not in instance_id:
        return None
    owner, rest = re.sub(r"^instance_", "", instance_id).split("__", 1)
    rest = re.sub(r"-[0-9a-f]{7,40}(?:-v[0-9a-z]+)?$", "", rest)
    repo = re.sub(r"-\d+$", "", rest)
    return f"{owner}/{repo}" if repo else owner


def _task_instruction(messages: list[dict]) -> str:
    """First user turn holds the PR description / problem statement."""
    for m in messages:
        if m.get("role") == "user":
            return _as_text(m.get("content")).strip()
    return ""


def _outcome(metadata: dict) -> str:
    resolved = (metadata.get("scores") or {}).get("resolved")
    if resolved in (1, True):
        return "success"
    if resolved in (0, False):
        return "fail"
    return "unknown"


def _extract_steps(messages: list[dict]) -> list[Step]:
    """Extract (command, observation, exit_code) steps from a message list.

    Handles both the inline encoding (``<bash_code>`` or a fenced ``bash``
    block) and the ``tool_calls`` + ``role: "tool"`` encoding.
    """
    # index tool responses by tool_call_id for the tool-call encoding.
    tool_responses: dict[str, dict] = {}
    for m in messages:
        if m.get("role") != "tool":
            continue
        tcid = m.get("tool_call_id")
        if not tcid:
            continue
        body = _as_text(m.get("content"))
        ec = None
        mm = _EXIT_PREFIX_RE.match(body.strip())
        if mm:
            ec = int(mm.group(1))
        tool_responses[tcid] = {"observation": body, "exit_code": ec}

    steps: list[Step] = []
    for idx, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue

        # tool-call encoding (each parallel call -> one step).
        tool_calls = m.get("tool_calls") or []
        if tool_calls:
            for tc in tool_calls:
                args = tc.get("arguments") or {}
                cmd = args.get("command", "") if isinstance(args, dict) else str(args)
                resp = tool_responses.get(tc.get("id"), {})
                steps.append(_make_step(
                    len(steps), cmd,
                    resp.get("observation", ""), resp.get("exit_code"),
                ))
            continue

        # inline encoding: command in <bash_code>, observation in next user turn.
        content = _as_text(m.get("content"))
        bm = _BASH_RE.search(content)
        if not bm:
            continue
        cmd = (bm.group(1) or bm.group(2) or "").strip()
        obs, ec = "", None
        nxt = messages[idx + 1] if idx + 1 < len(messages) else None
        if nxt and nxt.get("role") in ("user", "tool"):
            ntext = _as_text(nxt.get("content"))
            if "<returncode>" not in ntext and "EXACTLY ONE action" in ntext:
                continue  # format-error reply
            om = _OUTPUT_RE.search(ntext)
            obs = om.group(1).strip() if om else ntext.strip()
            rm = _RETURNCODE_RE.search(ntext)
            if rm:
                ec = int(rm.group(1))
        steps.append(_make_step(len(steps), cmd, obs, ec))
    return steps


def _make_step(index: int, command: str, observation: str, exit_code) -> Step:
    return Step(
        index=index,
        action=command,
        observation=observation,
        metadata={"exit_code": exit_code},
    )


def _load_one(record: dict, source: Optional[str] = None) -> Optional[Trajectory]:
    transcripts = record.get("transcripts") or []
    messages = (transcripts[0].get("messages") if transcripts else record.get("messages")) or []
    metadata = record.get("metadata") or record.get("info") or {}
    instance_id = metadata.get("instance_id") or record.get("instance_id") or record.get("id")

    steps = _extract_steps(messages)
    if not steps:
        return None
    instruction = _task_instruction(messages)
    for step in steps:
        step.metadata["task_instruction"] = instruction
    repo = _repo_from_instance(instance_id)

    return Trajectory(
        trajectory_id=str(instance_id),
        task_instruction=instruction,
        steps=steps,
        task_id=str(instance_id) if instance_id is not None else None,
        outcome=_outcome(metadata),
        metadata={
            "repo": repo,
            "exit_status": metadata.get("exit_status"),
            "source": repo or source,
        },
    )


def _iter_paths(path: str) -> Iterator[str]:
    if os.path.isdir(path):
        for fp in sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True)):
            yield fp
    else:
        yield path


def load_trajectories(path: str) -> Iterator[Trajectory]:
    """Yield unified :class:`Trajectory` objects from a SWE-Bench Pro dump.

    ``path`` is a single ``*.json`` trajectory file or a directory (searched
    recursively).
    Files without a usable transcript are skipped.
    """
    for fp in _iter_paths(path):
        try:
            with open(fp, encoding="utf-8") as fh:
                record = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or not ("transcripts" in record or "messages" in record):
            continue
        source = os.path.basename(os.path.dirname(fp)) or None
        traj = _load_one(record, source=source)
        if traj is not None:
            yield traj
