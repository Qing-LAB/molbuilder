"""Tests for the unified-data-model API (docs/execution/running-a-job.md § 5):
``read_effective_config``, ``write_config_scope``, ``get_script_generation``.

Pinned contracts:
  * server-wide lookup chain: cwd ``molbuilder.json`` first, XDG
    fallback (``$XDG_CONFIG_HOME/molbuilder/molbuilder.json`` or
    ``~/.config/molbuilder/molbuilder.json``) second.
  * project-scope lookup: ``<project_dir>/.molbuilder.json``.
  * deep-merge rules: scalars + arrays = project replaces, objects =
    recurse; project wins on conflict.
  * ``script_generation.preactivate`` CONCATENATES server-wide ++
    project (per § 3.6, not the generic replace rule) -- pinned in
    ``test_preactivate_concatenates_across_scopes``.
  * ``script_generation.autodetect_conda`` and ``preactivate_format``
    follow the standard replace rule.
  * ``write_config_scope`` produces files mode 0600 and preserves
    keys outside the patch.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from molbuilder.runtime_config import (
    CONFIG_FILENAME,
    PROJECT_CONFIG_FILENAME,
    RuntimeConfigError,
    get_script_generation,
    read_effective_config,
    write_config_scope,
)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A clean cwd + isolated $HOME + cleared XDG_CONFIG_HOME so
    read_effective_config + write_config_scope land in tmp_path.

    Yields the tmp_path (which becomes the cwd) for tests to write
    server-wide / project files into.
    """
    monkeypatch.chdir(tmp_path)
    # THE SANDBOX IS THE CONFIG ROOT (§ 2.1c).  The cwd step these
    # tests were written against is gone, so without this every
    # config they write is a file nothing reads.
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()
    return tmp_path


# --------------------------------------------------------------------- #
#  read_effective_config -- server-wide lookup chain                    #
# --------------------------------------------------------------------- #


def test_no_config_files_returns_empty(sandbox):
    assert read_effective_config() == {}


def test_cwd_server_wide_is_read(sandbox):
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "envs": {"siesta": "alt-siesta"},
    }))
    cfg = read_effective_config()
    assert cfg.get("envs") == {"siesta": "alt-siesta"}



@pytest.fixture
def xdg_branch(sandbox, monkeypatch):
    """Exercise the XDG step, which the override deliberately outranks.

    `sandbox` names ``MOLBUILDER_CONFIG_DIR`` so the file a test writes is the
    file the reader opens.  A test whose SUBJECT is the XDG fallback has to
    clear it — the override is not a search step, and a test that left it set
    would be asserting the override while claiming to assert XDG
    (`configuration.md` § 2.1c).
    """
    monkeypatch.delenv("MOLBUILDER_CONFIG_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(sandbox / "home" / ".config"))
    return sandbox


def test_xdg_fallback_is_read_when_cwd_absent(xdg_branch, monkeypatch):
    sandbox = xdg_branch
    xdg_dir = sandbox / "home" / ".config" / "molbuilder"
    xdg_dir.mkdir(parents=True)
    (xdg_dir / "molbuilder.json").write_text(json.dumps({
        "envs": {"pyscf": "alt-pyscf"},
    }))
    cfg = read_effective_config()
    assert cfg.get("envs") == {"pyscf": "alt-pyscf"}


def test_cwd_wins_over_xdg_fallback(sandbox, monkeypatch):
    """When both exist, cwd is the server-wide layer; XDG is ignored.
    Per docs/execution/running-a-job.md § 5's lookup order."""
    xdg_dir = sandbox / "home" / ".config" / "molbuilder"
    xdg_dir.mkdir(parents=True)
    (xdg_dir / "molbuilder.json").write_text(json.dumps({
        "envs": {"pyscf": "xdg-pyscf"},
    }))
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "envs": {"pyscf": "cwd-pyscf"},
    }))
    cfg = read_effective_config()
    assert cfg["envs"]["pyscf"] == "cwd-pyscf"


def test_explicit_xdg_config_home_is_honored(xdg_branch, monkeypatch):
    sandbox = xdg_branch
    xdg_dir = sandbox / "elsewhere"
    (xdg_dir / "molbuilder").mkdir(parents=True)
    (xdg_dir / "molbuilder" / "molbuilder.json").write_text(json.dumps({
        "envs": {"pyscf": "elsewhere-pyscf"},
    }))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_dir))
    cfg = read_effective_config()
    assert cfg["envs"]["pyscf"] == "elsewhere-pyscf"


def test_the_bare_default_read_honours_the_same_fallback(xdg_branch):
    """A-7 (final review, 2026-08-13): the bare ``read_config()`` was
    cwd-only while the section getters honoured the XDG file too, so an
    operator with an XDG-only config got a no-auth, no-TLS server while
    `jobset` honoured the very same file.

    The cwd step is gone (2026-08-31) and the split-brain it enabled with it,
    but the property this pins is the one that outlived it: the default read
    and the section getters share ONE lookup."""
    sandbox = xdg_branch
    from molbuilder.runtime_config import get_tls, read_config
    xdg_dir = sandbox / "home" / ".config" / "molbuilder"
    xdg_dir.mkdir(parents=True)
    (xdg_dir / "molbuilder.json").write_text(json.dumps({
        "tls": {"cert": "c.pem", "key": "k.pem"}}))
    assert get_tls(read_config()) == {"cert": "c.pem", "key": "k.pem"}
    # A file in the working directory changes NOTHING -- it is not read
    # (§ 2.1a), which is the half of this that inverted.
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "tls": {"cert": "cwd.pem", "key": "cwd-k.pem"}}))
    assert read_config()["tls"]["cert"] == "c.pem", (
        "a working-directory file reached the reader")


# --------------------------------------------------------------------- #
#  read_effective_config -- project scope + deep merge                  #
# --------------------------------------------------------------------- #


def test_project_overlay_replaces_scalar(sandbox):
    # `execution` here, not `envs`: the merge mechanics need a section a
    # BUNDLE may legitimately carry, and since U7 the registry refuses
    # machine-only sections in project scope (envs is one -- every
    # consumer reads it through read_config()).
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "execution": {"mode": "submit"},
    }))
    proj = sandbox / "myproject"
    proj.mkdir()
    (proj / PROJECT_CONFIG_FILENAME).write_text(json.dumps({
        "execution": {"mode": "direct"},
    }))
    cfg = read_effective_config(project_dir=proj)
    assert cfg["execution"]["mode"] == "direct"


def test_project_overlay_deep_merges_objects(sandbox):
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "scheduler": {"kind": "slurm",
                      "defaults": {"time": "0-01:00:00", "mem": "8G"}},
    }))
    proj = sandbox / "myproject"
    proj.mkdir()
    (proj / PROJECT_CONFIG_FILENAME).write_text(json.dumps({
        "scheduler": {"defaults": {"time": "0-04:00:00"}},
    }))
    cfg = read_effective_config(project_dir=proj)
    # mem preserved from server, time overridden -- objects recurse.
    assert cfg["scheduler"]["kind"] == "slurm"
    assert cfg["scheduler"]["defaults"] == {
        "time": "0-04:00:00",
        "mem":  "8G",
    }


def test_project_only_returns_project_layer(sandbox):
    """Server-wide absent + project present -> project values come
    through alone."""
    proj = sandbox / "myproject"
    proj.mkdir()
    (proj / PROJECT_CONFIG_FILENAME).write_text(json.dumps({
        "execution": {"mode": "direct"},
    }))
    cfg = read_effective_config(project_dir=proj)
    assert cfg["execution"]["mode"] == "direct"


def test_project_dir_none_returns_server_layer_unchanged(sandbox):
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "envs": {"siesta": "server-siesta"},
    }))
    cfg = read_effective_config(project_dir=None)
    assert cfg.get("envs") == {"siesta": "server-siesta"}


# --------------------------------------------------------------------- #
#  get_script_generation -- new v2 schema (preamble + activation)        #
# --------------------------------------------------------------------- #


def test_get_script_generation_defaults_when_all_empty(sandbox):
    sg = get_script_generation()
    assert sg["preamble"] == ""
    assert sg["activation"] is None
    assert sg["preamble_chunks"] == []


def test_preamble_concatenates_across_scopes(sandbox):
    """Per docs/execution/running-a-job.md § 5: preamble concatenates server-then-
    project (server first, project after), joined by ``\\n``."""
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "script_generation": {
            "preamble":   "module load mamba",
            "activation": "source activate",
        },
    }))
    proj = sandbox / "myproject"
    proj.mkdir()
    (proj / PROJECT_CONFIG_FILENAME).write_text(json.dumps({
        "script_generation": {"preamble": "export FOO=bar"},
    }))
    sg = get_script_generation(project_dir=proj)
    assert "module load mamba" in sg["preamble"]
    assert "export FOO=bar" in sg["preamble"]
    server_ix = sg["preamble"].find("module load mamba")
    project_ix = sg["preamble"].find("export FOO=bar")
    assert server_ix < project_ix
    assert [c[0] for c in sg["preamble_chunks"]] == ["server", "project"]


def test_activation_uses_replace_rule_project_wins(sandbox):
    """activation: project wins; else server-wide; else None."""
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "script_generation": {"activation": "source activate"},
    }))
    proj = sandbox / "myproject"
    proj.mkdir()
    (proj / PROJECT_CONFIG_FILENAME).write_text(json.dumps({
        "script_generation": {"activation": "conda activate"},
    }))
    sg = get_script_generation(project_dir=proj)
    assert sg["activation"] == "conda activate"


def test_activation_falls_back_to_server_when_project_silent(sandbox):
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "script_generation": {"activation": "source activate"},
    }))
    proj = sandbox / "myproject"
    proj.mkdir()
    (proj / PROJECT_CONFIG_FILENAME).write_text(json.dumps({
        "script_generation": {"preamble": "export X=1"},
    }))
    sg = get_script_generation(project_dir=proj)
    assert sg["activation"] == "source activate"


def test_activation_default_is_none(sandbox):
    """Per docs/execution/running-a-job.md § 5: activation has NO default; the generator
    refuses to emit a wrapper when it isn't set in either scope."""
    sg = get_script_generation()
    assert sg["activation"] is None


# --------------------------------------------------------------------- #
#  Validation                                                            #
# --------------------------------------------------------------------- #


def test_invalid_preamble_type_rejected(sandbox):
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "script_generation": {"preamble": 123},
    }))
    with pytest.raises(RuntimeConfigError, match="preamble.*string"):
        read_effective_config()


def test_invalid_activation_value_rejected(sandbox):
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "script_generation": {"activation": "mamba activate"},
    }))
    with pytest.raises(RuntimeConfigError, match="activation"):
        read_effective_config()


def test_legacy_preactivate_key_warns_and_aliases_to_preamble(sandbox):
    """Per docs/execution/running-a-job.md § 5: the legacy key ``preactivate`` is
    treated as ``preamble`` for one release with a deprecation
    warning."""
    import warnings
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "script_generation": {
            "preactivate": "module load mamba",
            "activation":  "source activate",
        },
    }))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sg = get_script_generation()
    assert sg["preamble"] == "module load mamba"
    assert any("preactivate" in str(w.message) and
                "preamble" in str(w.message)
                for w in caught), [str(w.message) for w in caught]


def test_dropped_keys_warn_but_dont_fail(sandbox):
    """Per docs/execution/running-a-job.md § 5: ``autodetect_conda`` /
    ``preactivate_format`` are silently ignored (with a warning)."""
    import warnings
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "script_generation": {
            "preamble":           "module load mamba",
            "activation":         "source activate",
            "autodetect_conda":   True,
            "preactivate_format": "shell",
        },
    }))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sg = get_script_generation()
    assert sg["preamble"] == "module load mamba"
    assert "autodetect_conda" not in sg
    warned = [str(w.message) for w in caught]
    assert any("autodetect_conda" in m for m in warned)
    assert any("preactivate_format" in m for m in warned)


# --------------------------------------------------------------------- #
#  write_config_scope                                                    #
# --------------------------------------------------------------------- #


def test_write_server_wide_when_cwd_file_exists_writes_to_cwd(sandbox):
    """When the cwd molbuilder.json exists, a server-wide write lands
    there (per docs/execution/running-a-job.md § 5: writes to the highest-precedence
    EXISTING location)."""
    (sandbox / "molbuilder.json").write_text("{}\n")
    target = write_config_scope(project_dir=None, patch={
        "script_generation": {"preamble": "module load mamba"},
    })
    assert target == sandbox / "molbuilder.json"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    cfg = json.loads(target.read_text())
    assert cfg["script_generation"]["preamble"] == "module load mamba"


def test_write_server_wide_creates_xdg_when_cwd_absent(xdg_branch):
    sandbox = xdg_branch
    """When NO server-wide file exists, the write lands at the XDG
    path (per docs/execution/running-a-job.md § 5 last sentence)."""
    target = write_config_scope(project_dir=None, patch={
        "script_generation": {"preamble": "module load mamba"},
    })
    assert target == sandbox / "home" / ".config" / "molbuilder" / "molbuilder.json"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    cfg = json.loads(target.read_text())
    assert cfg["script_generation"]["preamble"] == "module load mamba"


def test_write_project_scope_creates_hidden_file(sandbox):
    proj = sandbox / "myproject"
    proj.mkdir()
    target = write_config_scope(project_dir=proj, patch={
        "script_generation": {"autodetect_conda": True},
    })
    assert target == proj / ".molbuilder.json"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    cfg = json.loads(target.read_text())
    assert cfg["script_generation"]["autodetect_conda"] is True


def test_write_preserves_existing_unrelated_keys(sandbox):
    """A patch only touches the keys it carries.  Sister sections
    survive."""
    # `secret_key_file` was the second sister section here until 2026-08-31.
    # A config carrying it is now REFUSED, so seeding one would test the
    # refusal rather than the merge (`configuration.md` § 2.1e).
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "envs": {"siesta": "molbuilder-siesta"},
        "execution": {"mode": "direct"},
    }))
    write_config_scope(project_dir=None, patch={
        "script_generation": {"preamble": "module load mamba"},
    })
    cfg = json.loads((sandbox / "molbuilder.json").read_text())
    assert cfg["envs"]["siesta"] == "molbuilder-siesta"
    assert cfg["execution"]["mode"] == "direct"
    assert cfg["script_generation"]["preamble"] == "module load mamba"


def test_write_validates_before_writing(sandbox):
    """A patch with an invalid value is rejected -- the file is NOT
    written.  Otherwise the next read_effective_config call would fail
    even though the user thought their write succeeded."""
    with pytest.raises(RuntimeConfigError, match="activation"):
        write_config_scope(project_dir=None, patch={
            "script_generation": {"activation": "wrong-form"},
        })
    # File never created.
    assert not (sandbox / "molbuilder.json").exists()


def test_write_refuses_to_overwrite_a_corrupt_file(sandbox):
    """R10 (review-4 G7): this test pinned the OPPOSITE -- 'overwrites
    it with the patch... documented behaviour' -- and that tolerance
    silently destroyed whatever a hand-edit broke: a config carrying
    auth providers and TLS paths is exactly the file a user cannot
    afford to lose to a typo plus any later --write.  Corrupt now
    REFUSES, naming the path and the way out."""
    import pytest
    from molbuilder.runtime_config import RuntimeConfigError
    (sandbox / "molbuilder.json").write_text("not valid json {{{")
    with pytest.raises(RuntimeConfigError, match="refusing to overwrite"):
        write_config_scope(project_dir=None, patch={
            "script_generation": {"preamble": "module load mamba"},
        })
    # the broken content is untouched -- nothing was destroyed
    assert (sandbox / "molbuilder.json").read_text() == "not valid json {{{"


def test_machine_sections_are_refused_in_project_scope(sandbox):
    """The registry's scope rule (U7): a machine-only section in a
    bundle's .molbuilder.json is REFUSED, never silently unread -- S1c's
    argument generalised beyond checkpoint.  `admin` in a bundle would
    otherwise look effective while the web layer never sees it."""
    import pytest
    from molbuilder.runtime_config import (RuntimeConfigError,
                                           read_effective_config)
    proj = sandbox / "proj"
    proj.mkdir(exist_ok=True)
    (proj / ".molbuilder.json").write_text(json.dumps({
        "admin": {"emails": ["x@y.edu"]},
        "script_generation": {"activation": "conda activate"},
    }))
    with pytest.raises(RuntimeConfigError) as e:
        read_effective_config(project_dir=proj)
    assert "admin" in str(e.value)
    assert "machine sections have one home" in str(e.value)


def test_write_config_scope_refuses_machine_sections_for_a_bundle(sandbox):
    """Refusing at WRITE time beats producing a file every later read
    refuses (U7 -- the same registry row drives both)."""
    import pytest
    from molbuilder.runtime_config import (RuntimeConfigError,
                                           write_config_scope)
    proj = sandbox / "proj2"
    proj.mkdir(exist_ok=True)
    with pytest.raises(RuntimeConfigError, match="admin"):
        write_config_scope(proj, {"admin": {"emails": ["x@y.edu"]}})
    assert not (proj / ".molbuilder.json").exists()
    # a bundle section still writes fine
    out = write_config_scope(proj, {"execution": {"mode": "direct"}})
    assert out.is_file()
