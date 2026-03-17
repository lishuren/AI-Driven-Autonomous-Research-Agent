"""
config_loader.py – Loads and merges filter configuration for the research agent.

The bundled defaults live in ``research-agent/config/filters.json``.  A
custom ``filters.json`` placed in a user-supplied *config_dir* (e.g. the
``config/`` sub-folder of a topic directory) will be **merged** on top of the
defaults: only the keys that appear in the custom file are overridden; all
other keys retain their bundled values.

Usage::

    from src.config_loader import load_filters_config, get_filters_config

    # Called once at startup (optional – defaults are loaded lazily otherwise)
    load_filters_config(config_dir="/path/to/topic/config")

    # Called anywhere to retrieve the active configuration
    cfg = get_filters_config()
    stopwords: set[str] = cfg["stopwords"]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = "filters.json"

# Module-level cache; populated on first call to get_filters_config() or
# explicitly via load_filters_config().
_filters_config: Optional[dict[str, Any]] = None


def bundled_config_dir() -> Path:
    """Return the repository's bundled config directory."""
    return Path(__file__).resolve().parent.parent / "config"


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file and return its contents as a dict."""
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    # Strip the meta-comment key that is only there for human readers
    data.pop("_comment", None)
    return data


def _to_set(value: Any) -> set[str]:
    """Convert a list (from JSON) to a set of strings."""
    if isinstance(value, set):
        return value
    return set(value) if value is not None else set()


def _to_frozenset(value: Any) -> frozenset[str]:
    """Convert a list (from JSON) to a frozenset of strings."""
    return frozenset(_to_set(value))


def _to_tuple(value: Any) -> tuple[str, ...]:
    """Convert a list (from JSON) to a tuple of strings."""
    if isinstance(value, tuple):
        return value
    return tuple(value) if value is not None else ()


def _build_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise raw JSON data into the typed structures expected by consumers."""
    return {
        "stopwords": _to_set(raw.get("stopwords", [])),
        "filler_words": _to_set(raw.get("filler_words", [])),
        "allowed_query_helpers": _to_set(raw.get("allowed_query_helpers", [])),
        "captcha_url_markers": _to_tuple(raw.get("captcha_url_markers", [])),
        "transient_error_keywords": _to_set(raw.get("transient_error_keywords", [])),
        "permanent_error_keywords": _to_set(raw.get("permanent_error_keywords", [])),
        "hub_path_segments": _to_frozenset(raw.get("hub_path_segments", [])),
        "hub_title_keywords": _to_frozenset(raw.get("hub_title_keywords", [])),
        "link_exclude_substrings": _to_frozenset(raw.get("link_exclude_substrings", [])),
    }


def load_filters_config(config_dir: Optional[str] = None) -> None:
    """Load (and cache) the filters configuration.

    Reads the bundled ``config/filters.json``.  When *config_dir* is given
    and contains a ``filters.json`` file, its keys are merged on top of the
    bundled defaults so that only the sections present in the custom file are
    overridden.

    This function is idempotent when called with the same *config_dir*; call
    ``reset_filters_config()`` first if you need to force a reload.
    """
    global _filters_config

    bundled_path = bundled_config_dir() / _CONFIG_FILENAME
    if not bundled_path.is_file():
        raise FileNotFoundError(
            f"Bundled filters config not found: {bundled_path}"
        )
    raw = _load_json(bundled_path)

    if config_dir:
        override_path = Path(config_dir) / _CONFIG_FILENAME
        if override_path.is_file():
            override_raw = _load_json(override_path)
            raw.update(override_raw)
            logger.info("Loaded filters config override from %s", override_path)
        else:
            logger.debug(
                "No filters.json found in config_dir %s; using bundled defaults.",
                config_dir,
            )

    _filters_config = _build_config(raw)


def get_filters_config() -> dict[str, Any]:
    """Return the active filters configuration, loading defaults if necessary."""
    global _filters_config
    if _filters_config is None:
        load_filters_config()
    return _filters_config  # type: ignore[return-value]


def reset_filters_config() -> None:
    """Clear the cached configuration (mainly for use in tests)."""
    global _filters_config
    _filters_config = None
