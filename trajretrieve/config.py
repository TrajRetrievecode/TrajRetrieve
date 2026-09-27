"""Configuration loader (maps ``configs/config.template.yaml`` to runtime objects).

A thin, dependency-light reader that resolves the model ids and retrieval
hyper-parameters used by ``scripts/build_index.py`` and ``scripts/evaluate.py``,
and lazily constructs an OpenAI-compatible client for the offline LLM
(schema-field annotation) or LLM-judge reranker.

No secret is ever stored in code: ``api_key`` / ``base_url`` / ``model`` come
exclusively from the YAML file. When the key is absent or left at its template
placeholder, :func:`build_llm_client` returns ``(None, None)`` so every caller can fall
back to a no-op and the pipeline still runs end-to-end without API access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

# Defaults mirror configs/config.template.yaml so the package is runnable even
# if a key is omitted from a user's config.yaml.
DEFAULTS: dict[str, Any] = {
    "embedding_model": "BAAI/bge-m3",
    "reranker_model": "BAAI/bge-reranker-base",
    "rerank_k": 50,
    "alpha": 0.7,
}

# Values that mean "not filled in" -- treated as absent so the LLM no-ops.
_PLACEHOLDERS = {"", "YOUR_KEY_HERE", "your-model-id",
                 "https://your-openai-compatible-endpoint/v1", None}


def load_config(path: str | Path) -> dict[str, Any]:
    """Read ``path`` (YAML) and return a flat config dict with defaults applied.

    Resolves the convenience scalars used throughout the scripts:
    ``embedding_model``, ``reranker_model``, ``alpha``, ``rerank_k``.
    Nested blocks (``llm:``, ``annotator:``, ``bm25:``) are kept as-is;
    ``bm25.alpha`` is also exposed as ``alpha``. Missing file -> defaults only.
    """
    cfg: dict[str, Any] = {}
    p = Path(path)
    if p.exists():
        import yaml  # lazy: optional dependency

        loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            cfg = loaded

    out = dict(DEFAULTS)
    out.update({k: v for k, v in cfg.items() if v is not None})

    # alpha lives under bm25: in the template; hoist it for convenience.
    bm25 = cfg.get("bm25") or {}
    if isinstance(bm25, dict) and bm25.get("alpha") is not None:
        out["alpha"] = bm25["alpha"]

    return out


def _is_set(value: Any) -> bool:
    return value not in _PLACEHOLDERS


def _client(block: dict[str, Any], api_key: str) -> Optional[Any]:
    try:
        from openai import OpenAI  # lazy: optional dependency
    except ImportError:
        return None
    base_url = block.get("base_url")
    return OpenAI(
        api_key=api_key,
        base_url=base_url if _is_set(base_url) else None,
        timeout=block.get("timeout") or 180,
    )


def _resolve(
    config: dict[str, Any], blocks: tuple, model: Optional[str] = None
) -> tuple[Optional[Any], Optional[str]]:
    """First block with a usable key and model -> ``(client, model)``.

    A block without its own ``base_url`` uses the one in ``llm:``.
    """
    llm = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    for block in blocks:
        if not isinstance(block, dict):
            continue
        api_key = block.get("api_key")
        model_id = model if _is_set(model) else block.get("model")
        if not (_is_set(api_key) and _is_set(model_id)):
            continue
        if not _is_set(block.get("base_url")):
            block = {**block, "base_url": llm.get("base_url")}
        client = _client(block, api_key)
        return (client, model_id) if client is not None else (None, None)
    return None, None


def build_llm_client(config: dict[str, Any]) -> tuple[Optional[Any], Optional[str]]:
    """Client + model id for the LLM-judge reranker.

    Uses ``llm:``, else the flat keys; ``(None, None)`` when no usable key is
    configured or ``openai`` is not installed.
    """
    return _resolve(config, (config.get("llm"), config))


def build_annotator_client(
    config_path: str | Path, model: Optional[str] = None
) -> tuple[Optional[Any], Optional[str]]:
    """Client + model id for the offline annotators (``--annotate``).

    Uses ``annotator:``, else ``llm:``, else the flat keys; ``(None, None)`` if
    none is filled in. ``model`` overrides the configured model id.
    """
    config = load_config(config_path)
    return _resolve(config, (config.get("annotator"), config.get("llm"), config), model)
