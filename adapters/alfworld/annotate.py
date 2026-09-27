"""ALFWorld offline segment annotator.

Implements the :class:`trajretrieve.schema.fields.SegmentLLM` interface: given a
:class:`~trajretrieve.models.Segment`, it returns the three LLM-derived schema
fields ``{"summary", "error", "resolution"}``. The other record fields
(``phase``, ``targets``, ``provenance``) are filled by
:class:`~trajretrieve.schema.fields.FieldExtractor`, so this class only owns the
ALFWorld-specific prompt.

ALFWorld AgentInstruct trajectories are successful household demonstrations, so
``error`` is essentially always ``None`` and ``resolution`` describes the
sub-goal a segment closes (e.g. "Heated the apple using microwave 1").

The model is read from a YAML config (default ``configs/config.yaml``; see
``configs/config.template.yaml``). Without a usable client ``__call__`` returns
empty fields.

Config keys (``annotator:``, else ``llm:``)::

    annotator:
      base_url: "https://.../v1"   # OpenAI-compatible base URL
      api_key:  "..."              # read from config, never committed
      model:    "..."              # chat-completions model id
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from trajretrieve.config import build_annotator_client
from trajretrieve.models import Segment

DEFAULT_CONFIG_PATH = "configs/config.yaml"
_EMPTY = {"summary": None, "error": None, "resolution": None}

_PROMPT = """You are annotating one segment of a successful ALFWorld household-task trajectory.

Task: {task}
Segment phase: {phase}
Steps ({n}):
{steps}

Return ONLY a JSON object with exactly these keys:
{{
  "summary": "20-45 words: what the agent did in this segment. Name specific objects and locations (e.g. 'apple 1', 'countertop 1').",
  "error": null,
  "resolution": "the concrete sub-goal completed (e.g. 'Heated the apple using microwave 1'), or null for pure navigation/search."
}}

Rules:
- summary is REQUIRED and must be specific.
- error is almost always null: these are successful demonstrations.
- resolution must be a real sub-goal closure, or null if the segment only navigates/searches.
Output only the JSON object, no prose."""


def _format_steps(segment: Segment) -> str:
    lines = []
    for step in segment.steps:
        obs = (step.observation or "").replace("\n", " ")[:200]
        lines.append(f"- action: {step.action}; obs: {obs}")
    return "\n".join(lines)


def _phase_label(segment: Segment) -> str:
    cats = segment.categories
    if not cats:
        return "unknown"
    return cats[0].value if len(set(cats)) == 1 else "mixed"


def _parse_json_object(text: str) -> Optional[dict[str, Any]]:
    body = (text or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", body, re.DOTALL)
    if fence:
        body = fence.group(1).strip()
    start, depth = None, 0
    for i, ch in enumerate(body):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    parsed = json.loads(body[start:i + 1])
                except ValueError:
                    start = None
                    continue
                if isinstance(parsed, dict):
                    return parsed
    return None


class SegmentAnnotator:
    """Offline LLM annotator producing ``summary`` / ``error`` / ``resolution``.

    Parameters
    ----------
    client, model:
        An explicit OpenAI-compatible client and chat model id. If omitted, both
        are loaded from ``config_path``.
    config_path:
        YAML config providing ``annotator.{base_url,api_key,model}``.
    timeout, max_retries:
        Per-call request timeout (seconds) and retry budget.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        model: Optional[str] = None,
        config_path: str = DEFAULT_CONFIG_PATH,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        if client is None:
            client, model = build_annotator_client(config_path, model)
        self.client = client
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def enabled(self) -> bool:
        """``True`` when a live LLM client is configured."""
        return self.client is not None and self.model is not None

    def __call__(self, segment: Segment) -> dict[str, Optional[str]]:
        if not self.enabled:
            return dict(_EMPTY)

        prompt = _PROMPT.format(
            task=segment.steps[0].metadata.get("task_instruction", "") if segment.steps else "",
            phase=_phase_label(segment),
            n=len(segment.steps),
            steps=_format_steps(segment),
        )
        parsed: Optional[dict[str, Any]] = None
        for _ in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=500,
                    timeout=self.timeout,
                )
                parsed = _parse_json_object(response.choices[0].message.content or "")
                if parsed and parsed.get("summary"):
                    break
            except Exception:
                parsed = None
        if not parsed or not parsed.get("summary"):
            return dict(_EMPTY)
        return {
            "summary": parsed.get("summary"),
            "error": parsed.get("error"),
            "resolution": parsed.get("resolution"),
        }
