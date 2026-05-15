"""Per-machine config reader (``molbuilder.json``).

Covered:

* nested form is parsed straight through
* flat (legacy) form is folded into the nested shape
* nested wins when both present
* malformed JSON raises ``RuntimeConfigError`` (not silent swallow)
* non-dict top-level / sections raise ``RuntimeConfigError``
* missing file returns ``{}`` (not raise)
* unknown top-level keys are ignored silently (forward compat)
* the convenience accessors filter junk values defensively

Every test ``chdir``s into ``tmp_path`` so the repo-root molbuilder.json
template never leaks into the reader.
"""

from __future__ import annotations

import json

import pytest

from molbuilder.runtime_config import (CONFIG_FILENAME, RuntimeConfigError,
                                         get_envs, get_tls, read_config)


# --------------------------------------------------------------------- #
#  Existence / shape gates                                              #
# --------------------------------------------------------------------- #


def test_missing_file_returns_empty_dict(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert read_config() == {}


def test_empty_json_object_returns_empty_dict(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text("{}")
    assert read_config() == {}


def test_malformed_json_raises_usage_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text("{ this is not json")
    with pytest.raises(RuntimeConfigError, match="invalid JSON"):
        read_config()


def test_non_dict_top_level_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps(["not", "a", "dict"]))
    with pytest.raises(RuntimeConfigError, match="top-level value must be an object"):
        read_config()


def test_non_dict_tls_section_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({"tls": "string"}))
    with pytest.raises(RuntimeConfigError, match="'tls' must be an object"):
        read_config()


def test_non_dict_envs_section_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({"envs": ["a", "b"]}))
    with pytest.raises(RuntimeConfigError, match="'envs' must be an object"):
        read_config()


# --------------------------------------------------------------------- #
#  Nested form (the post-2026-05-14 shape)                              #
# --------------------------------------------------------------------- #


def test_nested_tls_section_parses(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "tls": {"cert": "/c.pem", "key": "/k.pem"},
    }))
    cfg = read_config()
    assert cfg == {"tls": {"cert": "/c.pem", "key": "/k.pem"}}


def test_nested_envs_section_parses(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "envs": {"siesta": "my-siesta", "pyscf": "my-pyscf"},
    }))
    cfg = read_config()
    assert cfg == {"envs": {"siesta": "my-siesta", "pyscf": "my-pyscf"}}


# --------------------------------------------------------------------- #
#  Flat form (backwards compatibility)                                  #
# --------------------------------------------------------------------- #


def test_flat_cert_key_folded_into_tls(monkeypatch, tmp_path):
    """The file shipped before 2026-05-14 had bare top-level cert/key."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "cert": "/c.pem", "key": "/k.pem",
    }))
    cfg = read_config()
    assert cfg == {"tls": {"cert": "/c.pem", "key": "/k.pem"}}


def test_nested_tls_wins_over_flat(monkeypatch, tmp_path):
    """If a user adds the nested section but forgets to remove the
    flat keys, the nested values should take precedence (so migrating
    is non-destructive: the new section reflects the intended state)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "cert": "/old-flat.pem", "key": "/old-flat.key",
        "tls": {"cert": "/new-nested.pem", "key": "/new-nested.key"},
    }))
    cfg = read_config()
    assert cfg["tls"]["cert"] == "/new-nested.pem"
    assert cfg["tls"]["key"]  == "/new-nested.key"


def test_flat_fills_in_when_nested_partial(monkeypatch, tmp_path):
    """nested.cert + flat.key is a valid combination (the flat key fills
    in the slot the nested section omitted)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "key": "/flat.key",
        "tls": {"cert": "/nested.pem"},
    }))
    cfg = read_config()
    assert cfg["tls"] == {"cert": "/nested.pem", "key": "/flat.key"}


# --------------------------------------------------------------------- #
#  Forward compat: unknown keys ignored silently                        #
# --------------------------------------------------------------------- #


def test_unknown_top_level_keys_ignored(monkeypatch, tmp_path):
    """The file should be free to grow new top-level sections in
    future releases without breaking older readers."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "tls":   {"cert": "/c.pem", "key": "/k.pem"},
        "future_section": {"some_key": 42},
        "another_top_level": "ignored",
    }))
    cfg = read_config()
    assert cfg == {"tls": {"cert": "/c.pem", "key": "/k.pem"}}


# --------------------------------------------------------------------- #
#  Accessors                                                            #
# --------------------------------------------------------------------- #


def test_get_tls_on_empty_cfg_returns_empty():
    assert get_tls({}) == {}


def test_get_tls_returns_section_copy():
    cfg = {"tls": {"cert": "/c.pem", "key": "/k.pem"}}
    out = get_tls(cfg)
    out["mutated"] = "should not leak back"
    assert "mutated" not in cfg["tls"]


def test_get_envs_on_missing_section_returns_empty():
    assert get_envs({"tls": {"cert": "/c.pem"}}) == {}


# --------------------------------------------------------------------- #
#  Value-type validation lives in _normalise, not the accessors         #
# --------------------------------------------------------------------- #


def test_read_config_rejects_non_string_envs_value(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "envs": {"siesta": "molbuilder-siesta", "pyscf": 123},
    }))
    with pytest.raises(RuntimeConfigError, match="envs"):
        read_config()


def test_read_config_rejects_non_string_envs_key(monkeypatch, tmp_path):
    # JSON forces string keys at parse time, so we exercise this path
    # via _normalise directly to confirm the validation is in place.
    from molbuilder.runtime_config import _normalise
    with pytest.raises(RuntimeConfigError, match="envs"):
        _normalise({"envs": {42: "x"}})


def test_read_config_rejects_non_string_tls_value(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "tls": {"cert": 123, "key": "/k.pem"},
    }))
    with pytest.raises(RuntimeConfigError, match="tls.cert"):
        read_config()


def test_read_config_rejects_empty_string_envs_value(monkeypatch, tmp_path):
    """An empty env-name override silently breaks dispatch
    (``routed_env`` returns None, falls back to host PATH).  Caught
    at the config boundary instead."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "envs": {"siesta": ""},
    }))
    with pytest.raises(RuntimeConfigError, match="cannot be empty"):
        read_config()
