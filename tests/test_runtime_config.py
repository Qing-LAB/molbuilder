"""Per-machine config reader (``molbuilder.json``).

Covered:

* nested form is parsed straight through
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
                                         read_config)


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
# --------------------------------------------------------------------- #
#  Existence / shape gates                                              #
# --------------------------------------------------------------------- #


# `test_the_config_molbuilder_SEEDS_reads` retired 2026-10-02 (W54 T21): the
# seeding file's `test_the_seeded_template_loads_and_names_every_section` reads
# the same seeded document through the same reader, and died on every one of the
# 12 informative mutants that killed this (`tools/verify_subsumption.py`).


# `test_missing_file_returns_empty_dict` retired 2026-10-02 (W54): it equals `test_machine_config_file.py::test_no_config_files_returns_empty`.


def test_empty_json_object_returns_empty_dict(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text("{}")
    assert read_config() == {}


# `test_malformed_json_raises_usage_error` retired 2026-10-02 (W54): `test_auth_config.py`'s door test refuses the same file and checks it names the path (R10).


# `test_non_dict_top_level_raises` retired 2026-10-02 (W54): `test_auth_config.py`'s door test refuses the same file and checks it names the path (R10).


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


# `test_nested_tls_section_parses` retired 2026-10-02 (W54): it equals the first half of `test_the_tls_section_is_read`.


def test_nested_envs_section_parses(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "envs": {"siesta": "my-siesta", "pyscf": "my-pyscf"},
    }))
    cfg = read_config()
    assert cfg == {"envs": {"siesta": "my-siesta", "pyscf": "my-pyscf"}}


# --------------------------------------------------------------------- #
#  ONE SPELLING FOR TLS (2026-09-02)                                    #
#                                                                       #
#  Three tests lived here for the flat top-level `cert`/`key`: that it   #
#  folded into `tls`, that a nested value beat it, and that it filled a  #
#  slot the section omitted.  All three described how two spellings get  #
#  along.  There is one spelling now, so there is nothing to get along   #
#  with -- and a fold that quietly accepted either could not tell a      #
#  migrated file from an un-migrated one.  What replaces them is the     #
#  refusal, below, because a person whose file worked yesterday is owed  #
#  the new spelling rather than a generic unknown-key list.              #
# --------------------------------------------------------------------- #


# `test_the_flat_tls_spelling_is_REFUSED_and_the_message_shows_the_new_one` retired 2026-10-02 (W54): a row of `tests/data/molbuilder_json.toml`, down the road, with both sentences.


def test_the_tls_section_is_read(monkeypatch, tmp_path):
    """The one spelling, whole and partial."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "tls": {"cert": "/c.pem", "key": "/k.pem"}}))
    assert read_config() == {"tls": {"cert": "/c.pem", "key": "/k.pem"}}

    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({
        "tls": {"cert": "/only.pem"}}))
    assert read_config() == {"tls": {"cert": "/only.pem"}}


# --------------------------------------------------------------------- #
#  Unknown keys are refused, never silently ineffective (U7)            #
# --------------------------------------------------------------------- #


# `test_unknown_top_level_keys_are_refused_by_name` retired 2026-10-02 (W54): a row of `tests/data/molbuilder_json.toml` (it names the live sections too).


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


# `test_get_tls_on_empty_cfg_returns_empty` retired 2026-10-02 (W54): an accessor's shape, not a result through the road.


# `test_get_tls_returns_section_copy` retired 2026-10-02 (W54): an accessor's shape, not a result through the road.


# `test_get_envs_on_missing_section_returns_empty` retired 2026-10-02 (W54): an accessor's shape, not a result through the road.


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


# `test_read_config_rejects_non_string_envs_key` retired 2026-10-02 (W54): JSON keys are strings, so no file can reach it -- and an unlisted key is refused before its type is read.


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


# `test_underscore_keys_are_comments_and_pass` retired 2026-10-02 (W54): a row of `tests/data/molbuilder_json.toml` (a comment at the top and inside a section).


