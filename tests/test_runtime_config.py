"""Per-machine config reader (``molbuilder.json``).

Covered:

* nested form is parsed straight through
* flat (legacy) form is folded into the nested shape
* nested wins when both present
* malformed JSON raises ``RuntimeConfigError`` (not silent swallow)
* non-dict top-level / sections raise ``RuntimeConfigError``
* missing file returns ``{}`` (not raise)
* unknown top-level keys are REFUSED with the known set named (U7,
  2026-08-12 -- "ignored silently" is how admin/rate_limit were dropped)
* the convenience accessors filter junk values defensively

Every test writes its ``molbuilder.json`` into ``tmp_path`` and the fixture
below makes ``tmp_path`` the CONFIG ROOT, so the reader opens exactly the file
the test wrote and nothing else.

That used to be arranged by ``chdir`` -- the reader's first candidate was the
working directory -- with ``HOME`` and ``XDG_CONFIG_HOME`` sandboxed underneath
in case the file was absent.  The working-directory step was deleted on
2026-08-31 (`configuration.md` § 2.1a), and the ``chdir`` calls that remain in
these tests no longer decide anything; the one variable does.
"""

from __future__ import annotations

import json

import pytest

from molbuilder.runtime_config import (CONFIG_FILENAME, RuntimeConfigError,
                                         get_envs, get_tls, read_config)


@pytest.fixture(autouse=True)
def _tmp_path_is_the_config_root(monkeypatch, tmp_path, tmp_path_factory):
    """``tmp_path`` holds this test's machine config, and the reader knows it.

    ONE variable answers for the whole lookup (`configuration.md` § 2.1c), so
    this replaces the HOME/XDG sandboxing that used to guard the fallback --
    which is kept anyway, because a test that reads ``$HOME`` for some other
    reason should still not find the developer's.

    `conftest.config_root` is the general form of this fixture and is what new
    tests should ask for; this one exists because every test in this file
    already writes to ``tmp_path`` by name.
    """
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)


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
#  Unknown keys are refused, never silently ineffective (U7)            #
# --------------------------------------------------------------------- #


def test_unknown_top_level_keys_are_refused_by_name(monkeypatch, tmp_path):
    """This pinned the OPPOSITE until 2026-08-12 ("free to grow new
    sections without breaking older readers") -- and that tolerance is
    exactly how `admin` and `rate_limit`, sections with live getters,
    were silently dropped: the file looked configured and nobody could
    be admin.  A key the loader does not know is a typo or a section
    nobody wired -- both deserve an error naming what IS known."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "tls":   {"cert": "/c.pem", "key": "/k.pem"},
        "future_section": {"some_key": 42},
    }))
    with pytest.raises(RuntimeConfigError, match="future_section"):
        read_config()
    # the refusal teaches: every known section is named
    with pytest.raises(RuntimeConfigError, match="admin"):
        read_config()


def test_admin_emails_survive_the_loader(monkeypatch, tmp_path):
    """THE U7 regression pin: `admin` must reach `get_admin_emails`
    through read_config.  Until 2026-08-12 `_normalise` dropped it (the
    section was absent from its ad-hoc allowlist), so the web layer read
    post-strip config and NOBODY could be admin, silently."""
    from molbuilder.runtime_config import get_admin_emails
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "admin": {"emails": ["Operator@ASU.edu", "  second@asu.edu "]},
    }))
    cfg = read_config()
    assert get_admin_emails(cfg) == frozenset(
        {"operator@asu.edu", "second@asu.edu"})


def test_rate_limit_survives_the_loader(monkeypatch, tmp_path):
    """Same defect family as admin: the tuning block must round-trip."""
    from molbuilder.runtime_config import get_rate_limit
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "rate_limit": {"enabled": False, "allowlist": ["10.0.0.1"]},
    }))
    cfg = read_config()
    assert get_rate_limit(cfg) == {"enabled": False,
                                   "allowlist": ["10.0.0.1"]}


def test_admin_with_a_broken_emails_shape_is_refused(monkeypatch, tmp_path):
    """A mistyped emails list must not fail silently into the
    safe-but-wrong 'nobody'."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "admin": {"emails": "operator@asu.edu"},
    }))
    with pytest.raises(RuntimeConfigError, match="admin.emails"):
        read_config()


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


def test_underscore_keys_are_comments_and_pass(monkeypatch, tmp_path):
    """JSON has no comments; the committed templates use "_comment_*"
    keys.  An explicit underscore marker is not the typo class the
    unknown-key refusal exists for (running-a-job.md § 5)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "_comment_about_this_file": "explains things",
        "tls": {"cert": "/c.pem", "key": "/k.pem"},
    }))
    cfg = read_config()
    assert cfg == {"tls": {"cert": "/c.pem", "key": "/k.pem"}}
