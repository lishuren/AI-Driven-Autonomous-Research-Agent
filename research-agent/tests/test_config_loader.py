"""
Tests for src.config_loader – filter configuration loading and override.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config_loader import (
    bundled_config_dir,
    get_filters_config,
    load_filters_config,
    reset_filters_config,
)


@pytest.fixture(autouse=True)
def reset_config():
    """Ensure each test starts with a fresh config cache."""
    reset_filters_config()
    yield
    reset_filters_config()


class TestBundledConfigDir:
    def test_returns_path_object(self):
        path = bundled_config_dir()
        assert isinstance(path, Path)

    def test_points_to_config_directory(self):
        path = bundled_config_dir()
        assert path.name == "config"

    def test_filters_json_exists(self):
        filters = bundled_config_dir() / "filters.json"
        assert filters.is_file(), f"Expected filters.json at {filters}"


class TestGetFiltersConfig:
    def test_returns_dict(self):
        cfg = get_filters_config()
        assert isinstance(cfg, dict)

    def test_contains_required_keys(self):
        cfg = get_filters_config()
        required = {
            "stopwords",
            "filler_words",
            "allowed_query_helpers",
            "captcha_url_markers",
            "transient_error_keywords",
            "permanent_error_keywords",
            "hub_path_segments",
            "hub_title_keywords",
            "link_exclude_substrings",
        }
        assert required.issubset(cfg.keys())

    def test_stopwords_is_set(self):
        cfg = get_filters_config()
        assert isinstance(cfg["stopwords"], set)
        assert "the" in cfg["stopwords"]
        assert "a" in cfg["stopwords"]

    def test_filler_words_is_set(self):
        cfg = get_filters_config()
        assert isinstance(cfg["filler_words"], set)
        assert "comprehensive" in cfg["filler_words"]
        assert "detailed" in cfg["filler_words"]

    def test_allowed_query_helpers_is_set(self):
        cfg = get_filters_config()
        assert isinstance(cfg["allowed_query_helpers"], set)
        assert "tutorial" in cfg["allowed_query_helpers"]

    def test_captcha_url_markers_is_tuple(self):
        cfg = get_filters_config()
        assert isinstance(cfg["captcha_url_markers"], tuple)
        assert "showcaptcha" in cfg["captcha_url_markers"]

    def test_hub_path_segments_is_frozenset(self):
        cfg = get_filters_config()
        assert isinstance(cfg["hub_path_segments"], frozenset)
        assert "category" in cfg["hub_path_segments"]

    def test_hub_title_keywords_is_frozenset(self):
        cfg = get_filters_config()
        assert isinstance(cfg["hub_title_keywords"], frozenset)
        assert "index" in cfg["hub_title_keywords"]

    def test_link_exclude_substrings_is_frozenset(self):
        cfg = get_filters_config()
        assert isinstance(cfg["link_exclude_substrings"], frozenset)
        assert "login" in cfg["link_exclude_substrings"]

    def test_transient_error_keywords_is_set(self):
        cfg = get_filters_config()
        assert isinstance(cfg["transient_error_keywords"], set)
        assert "timeout" in cfg["transient_error_keywords"]

    def test_permanent_error_keywords_is_set(self):
        cfg = get_filters_config()
        assert isinstance(cfg["permanent_error_keywords"], set)
        assert "404" in cfg["permanent_error_keywords"]

    def test_cached_on_repeated_calls(self):
        cfg1 = get_filters_config()
        cfg2 = get_filters_config()
        assert cfg1 is cfg2


class TestLoadFiltersConfig:
    def test_load_without_override_uses_bundled(self):
        load_filters_config()
        cfg = get_filters_config()
        assert "the" in cfg["stopwords"]

    def test_load_with_nonexistent_config_dir_uses_bundled(self, tmp_path):
        load_filters_config(config_dir=str(tmp_path / "nonexistent"))
        cfg = get_filters_config()
        assert "the" in cfg["stopwords"]

    def test_load_with_empty_config_dir_uses_bundled(self, tmp_path):
        load_filters_config(config_dir=str(tmp_path))
        cfg = get_filters_config()
        assert "the" in cfg["stopwords"]

    def test_override_stopwords(self, tmp_path):
        override = {"stopwords": ["x", "y", "z"]}
        (tmp_path / "filters.json").write_text(json.dumps(override), encoding="utf-8")

        load_filters_config(config_dir=str(tmp_path))
        cfg = get_filters_config()

        assert cfg["stopwords"] == {"x", "y", "z"}
        # Other keys must still have bundled defaults
        assert "comprehensive" in cfg["filler_words"]

    def test_override_filler_words(self, tmp_path):
        override = {"filler_words": ["fancy", "flowery"]}
        (tmp_path / "filters.json").write_text(json.dumps(override), encoding="utf-8")

        load_filters_config(config_dir=str(tmp_path))
        cfg = get_filters_config()

        assert cfg["filler_words"] == {"fancy", "flowery"}
        # Stopwords still from bundled defaults
        assert "the" in cfg["stopwords"]

    def test_override_captcha_url_markers(self, tmp_path):
        override = {"captcha_url_markers": ["blockme", "nobot"]}
        (tmp_path / "filters.json").write_text(json.dumps(override), encoding="utf-8")

        load_filters_config(config_dir=str(tmp_path))
        cfg = get_filters_config()

        assert cfg["captcha_url_markers"] == ("blockme", "nobot")

    def test_override_hub_path_segments(self, tmp_path):
        override = {"hub_path_segments": ["custom_hub", "my_index"]}
        (tmp_path / "filters.json").write_text(json.dumps(override), encoding="utf-8")

        load_filters_config(config_dir=str(tmp_path))
        cfg = get_filters_config()

        assert cfg["hub_path_segments"] == frozenset({"custom_hub", "my_index"})

    def test_override_only_affects_specified_keys(self, tmp_path):
        override = {"stopwords": ["custom_stop"]}
        (tmp_path / "filters.json").write_text(json.dumps(override), encoding="utf-8")

        load_filters_config(config_dir=str(tmp_path))
        cfg = get_filters_config()

        assert cfg["stopwords"] == {"custom_stop"}
        assert "login" in cfg["link_exclude_substrings"]
        assert "timeout" in cfg["transient_error_keywords"]

    def test_comment_key_is_stripped(self, tmp_path):
        override = {"_comment": "My override", "stopwords": ["only"]}
        (tmp_path / "filters.json").write_text(json.dumps(override), encoding="utf-8")

        load_filters_config(config_dir=str(tmp_path))
        cfg = get_filters_config()

        assert "_comment" not in cfg
        assert cfg["stopwords"] == {"only"}


class TestResetFiltersConfig:
    def test_reset_clears_cache(self):
        cfg1 = get_filters_config()
        reset_filters_config()
        cfg2 = get_filters_config()
        # Should be equal in content but different objects
        assert cfg1 is not cfg2
        assert cfg1["stopwords"] == cfg2["stopwords"]

    def test_reset_allows_new_override(self, tmp_path):
        load_filters_config()
        original_stopwords = get_filters_config()["stopwords"].copy()

        reset_filters_config()
        override = {"stopwords": ["only_this"]}
        (tmp_path / "filters.json").write_text(json.dumps(override), encoding="utf-8")
        load_filters_config(config_dir=str(tmp_path))

        new_stopwords = get_filters_config()["stopwords"]
        assert new_stopwords == {"only_this"}
        assert new_stopwords != original_stopwords
