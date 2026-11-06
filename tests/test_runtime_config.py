"""Per-machine config reader (``molbuilder.json``): a section that is not an
object, a value of the wrong type, and `admin` and `rate_limit` reaching their
readers through the loader (U7, 2026-08-12: a section the loader dropped read
as nobody being an admin).

Elsewhere: malformed JSON and a top level that is not an object,
``test_auth_config.py``; a missing file,
``test_machine_config_file.py::test_no_config_files_returns_empty``; an unknown
top-level key, ``data/molbuilder_json.toml``.

Every test writes its ``molbuilder.json`` into ``tmp_path`` and the fixture
below makes ``tmp_path`` the CONFIG ROOT, so the reader opens exactly the file
the test wrote and nothing else.
"""

from __future__ import annotations

import json

import pytest

from molbuilder.runtime_config import (CONFIG_FILENAME, RuntimeConfigError,
                                         read_config)


@pytest.fixture(autouse=True)
def _tmp_path_is_the_config_root(monkeypatch, tmp_path, tmp_path_factory):
    """``tmp_path`` holds this test's machine config, and the reader knows it.

    ONE variable answers for the whole lookup (`configuration.md` § 2.1c);
    conftest's root fixture redirects HOME as well, so a test that reads
    ``$HOME`` for another reason still does not find the developer's.

    `conftest.config_root` is the general form of this fixture and is what new
    tests should ask for; this one exists because every test in this file
    already writes to ``tmp_path`` by name.
    """
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
# --------------------------------------------------------------------- #
#  Existence / shape gates                                              #
# --------------------------------------------------------------------- #


def test_non_dict_tls_section_raises(tmp_path):
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({"tls": "string"}))
    with pytest.raises(RuntimeConfigError, match="'tls' must be an object"):
        read_config()


def test_non_dict_envs_section_raises(tmp_path):
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({"envs": ["a", "b"]}))
    with pytest.raises(RuntimeConfigError, match="'envs' must be an object"):
        read_config()


# --------------------------------------------------------------------- #
#  ONE SPELLING FOR TLS (2026-09-02)                                    #
# --------------------------------------------------------------------- #


def test_the_tls_section_is_read(tmp_path):
    """The one spelling, whole and partial."""
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "tls": {"cert": "/c.pem", "key": "/k.pem"}}))
    assert read_config() == {"tls": {"cert": "/c.pem", "key": "/k.pem"}}

    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "tls": {"cert": "/only.pem"}}))
    assert read_config() == {"tls": {"cert": "/only.pem"}}


# --------------------------------------------------------------------- #
#  Unknown keys are refused, never silently ineffective (U7)            #
# --------------------------------------------------------------------- #


def test_admin_emails_survive_the_loader(tmp_path):
    """THE U7 regression pin: `admin` must reach `get_admin_emails`
    through read_config.  Until 2026-08-12 `_normalise` dropped it (the
    section was absent from its ad-hoc allowlist), so the web layer read
    post-strip config and NOBODY could be admin, silently."""
    from molbuilder.runtime_config import get_admin_emails
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "admin": {"emails": ["Operator@ASU.edu", "  second@asu.edu "]},
    }))
    cfg = read_config()
    assert get_admin_emails(cfg) == frozenset(
        {"operator@asu.edu", "second@asu.edu"})


def test_rate_limit_survives_the_loader(tmp_path):
    """Same defect family as admin: the tuning block must round-trip."""
    from molbuilder.runtime_config import get_rate_limit
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "rate_limit": {"enabled": False, "allowlist": ["10.0.0.1"]},
    }))
    cfg = read_config()
    assert get_rate_limit(cfg) == {"enabled": False,
                                   "allowlist": ["10.0.0.1"]}


def test_admin_with_a_broken_emails_shape_is_refused(tmp_path):
    """A mistyped emails list must not fail silently into the
    safe-but-wrong 'nobody'."""
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "admin": {"emails": "operator@asu.edu"},
    }))
    with pytest.raises(RuntimeConfigError, match="admin.emails"):
        read_config()


# --------------------------------------------------------------------- #
#  Value-type validation lives in _normalise, not the accessors         #
# --------------------------------------------------------------------- #


def test_read_config_rejects_non_string_envs_value(tmp_path):
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "envs": {"siesta": "molbuilder-siesta", "pyscf": 123},
    }))
    with pytest.raises(RuntimeConfigError, match="envs"):
        read_config()


def test_read_config_rejects_non_string_tls_value(tmp_path):
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "tls": {"cert": 123, "key": "/k.pem"},
    }))
    with pytest.raises(RuntimeConfigError, match="tls.cert"):
        read_config()


def test_read_config_rejects_empty_string_envs_value(tmp_path):
    """An empty env-name override silently breaks dispatch
    (``routed_env`` returns None, falls back to host PATH).  Caught
    at the config boundary instead."""
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "envs": {"siesta": ""},
    }))
    with pytest.raises(RuntimeConfigError, match="cannot be empty"):
        read_config()
