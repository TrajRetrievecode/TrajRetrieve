"""Offline LLM annotator for SWE-Bench Pro segments.

Implements the :class:`trajretrieve.schema.fields.SegmentLLM` interface: given a
segment of shell-command steps, it returns the free-text schema fields
``{"summary", "error", "resolution"}`` and LLM-completed ``targets``
(``phase`` and ``provenance`` are rule-derived).

The prompt asks an OpenAI-compatible chat model to read a contiguous run of
``$ command -> (exit code) output`` lines and emit a compact JSON record:

  * ``summary``    -- one verb-led sentence naming the file path and the
                      class/function/method touched (no vague "fixed config").
  * ``targets``    -- concrete symbols (file paths, function/class names,
                      identifiers) for retrieval; merged into the rule targets.
  * ``error``      -- tool + error type + key message, or ``null`` if none.
  * ``resolution`` -- the recovery action taken in this segment, or ``null``.

These mirror the structured-record fields used for failure-anchored retrieval:
a future stuck agent matches its error/goal against ``summary``/``error``/
``resolution``. The client (api key, base url, model id) is read from a YAML
config; when no client is configured the annotator is a no-op that returns
all-``None`` so the wider pipeline runs without any API access.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from trajretrieve.config import build_annotator_client
from trajretrieve.models import Segment
from trajretrieve.schema.fields import derive_phase

# --------------------------------------------------------------------------- #
# Prompt.                                                                     #
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = (
    "You are a trajectory analyst for an autonomous software-repair agent. "
    "You label a contiguous run of shell commands and their outputs with a "
    "compact structured record. These labels are later retrieved when a "
    "different repair task gets stuck, so they must be specific and grounded "
    "strictly in the commands and outputs shown."
)

_USER_PROMPT = """# Task context
Repository: {repo}
Problem statement (truncated):
{problem_statement}

# Current segment ({n_steps} step(s), rule-inferred phase = {phase}, contains failure = {has_failure})
{segment_body}

# Output (return ONE JSON object, no code fence, no commentary)
Fill these fields, grounded ONLY in the commands and outputs above:

1. "summary": one sentence (<= 80 words). MUST contain a verb + the concrete
   file path + the class/function/method name + key arguments. Example:
   "Replaced the dict access on `_host_states` with a `setdefault` call in
   `lib/executor/play_iterator.py`." Do NOT write vague phrases like "modified
   config" or "ran tests".
2. "targets": object with three string arrays, extracted verbatim (do not
   invent), deduplicated:
     - "files": file paths read/written/operated on in this segment.
     - "symbols": function, class and method names involved (no parentheses).
     - "identifiers": module names, config keys, command names, argument or
       variable names.
3. "error": tool + error type + key error string, e.g.
   "pytest raised ModuleNotFoundError: package.module_utils.urls".
   Use null if this segment has no failure.
4. "resolution": the recovery action taken later in this segment, e.g.
   "inserted /app/lib at the front of sys.path". Use null if there was no
   recovery attempt.

Return exactly:
{{"summary": "...", "targets": {{"files": ["..."], "symbols": ["..."], "identifiers": ["..."]}}, "error": "..." or null, "resolution": "..." or null}}"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction tolerant of code fences / surrounding prose."""
    s = (text or "").strip()
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # string-aware brace matching to find the largest balanced object.
    depth, start, best, in_str, esc = 0, None, None, False, False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                blk = s[start:i + 1]
                if best is None or len(blk) > len(best):
                    best = blk
                start = None
    for cand in filter(None, [best, *_JSON_BLOCK_RE.findall(s)]):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def _render_segment(segment: Segment, max_chars: int = 1600) -> str:
    """Render the segment as ``$ command -> (exit code) output`` lines."""
    lines = []
    for step in segment.steps:
        cmd = (step.action or "").strip().split("\n", 1)[0]
        if len(cmd) > 240:
            cmd = cmd[:240] + "..."
        obs = (step.observation or "").strip().replace("\n", " \\n ")
        if len(obs) > 240:
            obs = obs[:240] + "..."
        ec = step.metadata.get("exit_code")
        ec_tag = f"(exit={ec}) " if ec is not None else ""
        lines.append(f"$ {cmd}\n  -> {ec_tag}{obs!r}")
    body = "\n".join(lines)
    return body if len(body) <= max_chars else body[:max_chars] + "...[truncated]"


def _flatten_targets(targets) -> list[str]:
    """Flatten the {files, symbols, identifiers} object into a unique list."""
    out: list[str] = []
    if isinstance(targets, dict):
        for key in ("files", "symbols", "identifiers"):
            for tok in targets.get(key) or []:
                if isinstance(tok, str) and tok.strip() and tok not in out:
                    out.append(tok.strip())
    elif isinstance(targets, list):
        for tok in targets:
            if isinstance(tok, str) and tok.strip() and tok not in out:
                out.append(tok.strip())
    return out


class SegmentAnnotator:
    """SWE-Bench Pro offline segment annotator (a ``SegmentLLM``).

    Parameters
    ----------
    client:
        A pre-built OpenAI-compatible client (must expose
        ``chat.completions.create``). If ``None`` and ``config_path`` is given,
        a client is built from ``config_path``.
    model:
        Chat model id. Falls back to the annotator endpoint's model.
    config_path:
        YAML config with the annotator endpoint (see ``configs/config.template.yaml``).
        Never hard-code secrets; they are read here.
    max_tokens / temperature / timeout:
        Generation controls; ``timeout`` is the per-request timeout in seconds.

    With no client and no usable config the annotator becomes a no-op that
    returns ``{"summary": None, "error": None, "resolution": None}``.
    """

    def __init__(
        self,
        client=None,
        model: Optional[str] = None,
        config_path: Optional[str] = None,
        max_tokens: int = 1500,
        temperature: float = 0.0,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self._client = client

        if self._client is None and config_path:
            self._client, self.model = build_annotator_client(config_path, model)

    @property
    def enabled(self) -> bool:
        return self._client is not None and bool(self.model)

    # ------------------------------------------------------------------ #

    def _build_messages(self, segment: Segment) -> list[dict]:
        prov = segment.provenance
        repo = prov.source if prov is not None and prov.source else "?"
        problem = segment.steps[0].metadata.get("task_instruction", "") if segment.steps else ""
        phase = derive_phase(segment)
        has_failure = any(s.is_error for s in segment.steps)
        user = _USER_PROMPT.format(
            repo=repo,
            problem_statement=(problem or "(not provided)")[:600],
            n_steps=len(segment.steps),
            phase=phase or "unknown",
            has_failure="yes" if has_failure else "no",
            segment_body=_render_segment(segment),
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def __call__(self, segment: Segment) -> dict[str, Optional[str]]:
        empty = {"summary": None, "error": None, "resolution": None}
        if not self.enabled:
            return empty

        messages = self._build_messages(segment)
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception:
            return empty

        parsed = _parse_json(raw)
        if not isinstance(parsed, dict):
            return empty

        summary = parsed.get("summary")
        error = parsed.get("error")
        resolution = parsed.get("resolution")
        targets = _flatten_targets(parsed.get("targets"))
        return {
            "summary": summary if isinstance(summary, str) else None,
            "error": error if isinstance(error, str) else None,
            "resolution": resolution if isinstance(resolution, str) else None,
            "targets": targets,
        }
