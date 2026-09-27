"""Action-to-category mapping f.

The mapping ``f : action -> ActionCategory`` is the primary hand-authored
environment-dependent component of TrajRetrieve; the Target-entities
extraction rules in :mod:`trajretrieve.schema.fields` are the other one, and
the candidate-side field subset is selected on a validation set rather than
hand-authored. It projects each environment's native action
vocabulary onto the shared seven-label alphabet ``F`` defined in
:mod:`trajretrieve.models`:

    data_fetch, action, navigation, edit, test, credential, submit

Adapting TrajRetrieve to a new environment starts by defining one such mapper
and registering it. Not every environment exercises all seven labels;
e.g. ALFWorld never produces ``edit``/``test``/``credential``/``submit`` and
AppWorld never produces ``navigation``/``edit``/``test``. Unmatched actions
fall back to ``data_fetch``.

Each mapper is a pure function of the raw action string (and, where useful, the
observation) so categorization is deterministic and side-effect free.
"""

from __future__ import annotations

import re
from typing import Callable

from ..models import ActionCategory

Mapper = Callable[[str, str], ActionCategory]

FALLBACK = ActionCategory.DATA_FETCH


# --------------------------------------------------------------------------- #
# AppWorld: API calls of the form ``apis.<app>.<method>(...)``.                #
# --------------------------------------------------------------------------- #

# write/mutation verbs -> ACTION  (create/send/delete/update/add/...)
_AW_WRITE_RE = re.compile(
    r"\.(create|send|place|delete|update|add|remove|set|move|mark|star|unstar"
    r"|reply|forward|share|cancel|edit)_",
)
_AW_CREDENTIAL_RE = re.compile(r"\.(show_account_passwords|login|signup)\b")
_AW_SUBMIT_RE = re.compile(r"(complete_task|\.submit\b)")
# documentation / supervisor lookups are information gathering
_AW_LOOKUP_RE = re.compile(r"\.(show_api_doc|show_api_descriptions|show_profile|show_)")


def appworld_category(action: str, observation: str = "") -> ActionCategory:
    a = action.strip().lower()
    if _AW_SUBMIT_RE.search(a):
        return ActionCategory.SUBMIT
    if _AW_CREDENTIAL_RE.search(a):
        return ActionCategory.CREDENTIAL
    if _AW_WRITE_RE.search(a):
        return ActionCategory.ACTION
    # api_docs.show* / supervisor.show* / generic reads
    if _AW_LOOKUP_RE.search(a):
        return ActionCategory.DATA_FETCH
    return FALLBACK


# --------------------------------------------------------------------------- #
# ALFWorld: natural-language text-world commands.                             #
# --------------------------------------------------------------------------- #

_ALF_NAVIGATE = ("go to", "goto", "move to")
_ALF_ACTION = (
    "take", "pick up", "put", "open", "close", "clean",
    "heat", "cook", "cool", "use", "toggle", "slice",
)
_ALF_READ = ("look", "examine", "inventory", "read")


def alfworld_category(action: str, observation: str = "") -> ActionCategory:
    a = action.strip().lower()
    if a.startswith(_ALF_NAVIGATE):
        return ActionCategory.NAVIGATION
    if a.startswith(_ALF_READ):
        return ActionCategory.DATA_FETCH
    if a.startswith(_ALF_ACTION):
        return ActionCategory.ACTION
    return FALLBACK


# --------------------------------------------------------------------------- #
# SWE-Bench Pro: shell commands.                                              #
# --------------------------------------------------------------------------- #

_SWE_PREFIX = r"^\s*(?:cd\s+\S+\s*&&\s*)?"
_SWE_SUBMIT_RE = re.compile(r"echo\s+(COMPLETE_TASK|MICRO_SWE_AGENT_FINAL)", re.I)
_SWE_NAVIGATE_RE = re.compile(_SWE_PREFIX + r"(ls|find|tree|which|stat|file|du|pwd)\b")
_SWE_READ_RE = re.compile(_SWE_PREFIX + r"(cat|less|more|head|tail|grep|rg|ag|nl|wc|awk|cut|sed\s+-n)\b")
_SWE_EDIT_RE = re.compile(_SWE_PREFIX + r"(sed\s+-i|patch\b|apply_patch|git\s+apply)")
# files whose name suggests test/repro/verification authoring -> TEST
_SWE_TEST_FILE_RE = re.compile(r"(test|repro|debug|verify|check|reproduce)", re.I)
# running interpreters / test runners / build tools -> TEST (verification)
_SWE_EXEC_RE = re.compile(
    _SWE_PREFIX + r"(python\d?|pytest|nose|tox|node|npx|go|cargo|make|cmake|npm|pip"
    r"|django-admin|manage\.py|gcc|g\+\+|java|ruby|perl|bash\s+\S+\.sh)\b",
)
# writes to a file via redirect / heredoc -> EDIT (or TEST if test-named)
_SWE_WRITE_TARGET_RE = re.compile(r">\s*([\w./\-]+\.\w+)")


def swe_category(action: str, observation: str = "") -> ActionCategory:
    a = action.strip()
    if _SWE_SUBMIT_RE.search(a):
        return ActionCategory.SUBMIT
    if _SWE_EDIT_RE.search(a):
        return ActionCategory.EDIT
    m = _SWE_WRITE_TARGET_RE.search(a)
    if m:
        target = m.group(1)
        return ActionCategory.TEST if _SWE_TEST_FILE_RE.search(target) else ActionCategory.EDIT
    if _SWE_EXEC_RE.search(a):
        return ActionCategory.TEST  # running tests / build = verification
    if _SWE_NAVIGATE_RE.search(a):
        return ActionCategory.NAVIGATION
    if _SWE_READ_RE.search(a):
        return ActionCategory.DATA_FETCH
    return FALLBACK


# --------------------------------------------------------------------------- #
# Registry.                                                                    #
# --------------------------------------------------------------------------- #

_MAPPERS: dict[str, Mapper] = {
    "appworld": appworld_category,
    "alfworld": alfworld_category,
    "swe": swe_category,
    "swebench": swe_category,
    "swe_bench_pro": swe_category,
    "swepro": swe_category,
}


def get_mapper(env: str) -> Mapper:
    """Return the action->category mapper for ``env``."""
    key = env.strip().lower()
    if key not in _MAPPERS:
        raise KeyError(
            f"No action-category mapper registered for environment {env!r}. "
            f"Known: {sorted(_MAPPERS)}. Register one with register_mapper()."
        )
    return _MAPPERS[key]


def register_mapper(env: str, mapper: Mapper) -> None:
    """Register a mapper for a new environment."""
    _MAPPERS[env.strip().lower()] = mapper
