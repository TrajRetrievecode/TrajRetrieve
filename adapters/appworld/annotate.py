"""AppWorld offline segment annotator.

Implements the :class:`trajretrieve.schema.fields.SegmentLLM` interface: given a
:class:`~trajretrieve.models.Segment`, it returns the three LLM-derived fields
of the structured record ``R(s)`` -- ``summary``, ``error``, ``resolution``.

The prompt encodes AppWorld-specific conventions:

  * actions are Python that calls ``apis.<app>.<method>(...)``;
  * a segment's *intent* is best summarized by its terminal business API rather
    than the scaffolding ``apis.api_docs.*`` / ``apis.supervisor.*`` lookups;
  * failures surface as ``Execution failed`` / ``Traceback`` / ``status code
    is 4xx`` in the observation, and the actionable error key is the
    ``app.method`` of the first failing API call (the "stuck API").

The annotator is an *offline* component: it runs once over the candidate pool.
When no OpenAI-compatible client is configured it degrades to a no-op that
returns empty fields, so the package remains runnable without API access.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from trajretrieve.config import build_annotator_client
from trajretrieve.models import Segment

# --------------------------------------------------------------------------- #
# AppWorld surface patterns (grounding only; categories come from the package  #
# mapper get_mapper('appworld')).                                              #
# --------------------------------------------------------------------------- #

_API_CALL_RE = re.compile(r"apis\.(\w+)\.(\w+)\s*\(")
_ERROR_MARKERS = (
    "Execution failed",
    "Traceback",
    "Exception:",
    "status code is 4",
    "Syntax error",
)

# Default client configuration.
_CONFIG_PATH = Path("configs/config.yaml")


def _is_error_observation(observation: str) -> bool:
    return any(m in (observation or "") for m in _ERROR_MARKERS)


def _stuck_api(segment: Segment) -> Optional[str]:
    """The ``app.method`` of the first failing API call in the segment."""
    for step in segment.steps:
        if not _is_error_observation(step.observation):
            continue
        m = _API_CALL_RE.search(step.action or "")
        if m:
            return f"{m.group(1)}.{m.group(2)}"
    return None


# --------------------------------------------------------------------------- #
# Prompt.                                                                       #
# --------------------------------------------------------------------------- #

_SYSTEM = (
    "You label a contiguous slice of an AppWorld agent trajectory. Actions are "
    "Python that calls business APIs of the form apis.<app>.<method>(...). "
    "Return ONLY a JSON object with keys \"summary\", \"error\", \"resolution\".\n\n"
    "Guidance:\n"
    "- summary: one short sentence (<= 25 words) naming the segment's INTENT. "
    "Judge intent by the terminal business API, not the apis.api_docs.* or "
    "apis.supervisor.* scaffolding lookups; cover every key business API in "
    "order. Keep app names (e.g. venmo, gmail, phone, amazon) but never include "
    "concrete values (tokens, emails, person names, amounts, item names).\n"
    "- error: if the segment contains a failure (Execution failed / Traceback / "
    "HTTP 4xx), state the failing API and the cause in one phrase; otherwise "
    "use null.\n"
    "- resolution: if a later step recovers from that failure, state in one "
    "phrase what change fixed it (e.g. corrected argument, added login, used a "
    "different API); otherwise use null.\n\n"
    "Output strictly: {\"summary\": \"...\", \"error\": null, \"resolution\": null}"
)

_MAX_STEPS_IN_PROMPT = 20
_MAX_CODE_CHARS = 240
_MAX_OBS_CHARS = 240


def _render_user(segment: Segment) -> str:
    lines = [
        f"Segment steps {segment.step_range[0]}-{segment.step_range[1]} "
        f"({segment.length} steps).",
    ]
    hint = _stuck_api(segment)
    if hint:
        lines.append(f"Detected stuck API (rule hint): {hint}")
    lines.append("")
    lines.append("Steps:")
    for step in segment.steps[:_MAX_STEPS_IN_PROMPT]:
        flag = "[ERR]" if _is_error_observation(step.observation) else "     "
        code = (step.action or "").strip().replace("\n", " ")[:_MAX_CODE_CHARS]
        obs = (step.observation or "").strip().replace("\n", " ")[:_MAX_OBS_CHARS]
        lines.append(f"  [{step.index:>3}] {flag} {code}")
        if obs:
            lines.append(f"        -> {obs}")
    return "\n".join(lines)


class SegmentAnnotator:
    """Offline LLM annotator for AppWorld segments (``SegmentLLM`` protocol).

    Parameters
    ----------
    client:
        An OpenAI-compatible chat client. If ``None``, one is built from
        ``config_path``; if that also yields nothing, the annotator runs as a
        no-op and returns empty fields.
    model:
        The chat model id. Falls back to the configured model.
    config_path:
        Location of the OpenAI-compatible client configuration.
    """

    def __init__(
        self,
        client: Any = None,
        model: Optional[str] = None,
        config_path: str | Path = _CONFIG_PATH,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> None:
        if client is None:
            client, cfg_model = build_annotator_client(config_path, model)
            model = model or cfg_model
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    # -- SegmentLLM interface ------------------------------------------- #

    def __call__(self, segment: Segment) -> dict[str, Optional[str]]:
        empty: dict[str, Optional[str]] = {
            "summary": None,
            "error": None,
            "resolution": None,
        }
        if self.client is None or not self.model:
            return empty  # no-op fallback

        user = _render_user(segment)
        for _ in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                )
                content = (resp.choices[0].message.content or "").strip()
                parsed = _parse_json(content)
                if parsed is not None:
                    return {
                        "summary": _clean(parsed.get("summary")),
                        "error": _clean(parsed.get("error")),
                        "resolution": _clean(parsed.get("resolution")),
                    }
            except Exception:  # API or parse failure; retry
                pass
        return empty  # retries exhausted


# --------------------------------------------------------------------------- #
# Response parsing.                                                             #
# --------------------------------------------------------------------------- #


def _parse_json(content: str) -> Optional[dict]:
    if not content:
        return None
    if content.startswith("```"):
        content = re.sub(r"^```\w*\n?", "", content)
        content = re.sub(r"\n?```$", "", content).strip()
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a"}:
        return None
    return text
