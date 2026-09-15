"""Where every configured file sits, and what mode it has when we made it.

`configuration.md` § 3.1 draws the tree and § 2.1b states the mode rule, and
until 2026-09-13 both were prose a reviewer had to re-verify.  Three of that
page's own statements were measured false: the config root was said to be 0700
and nothing created it that way, the serve log was said to be 0600 and `serve
status` made it 0664, and a row claimed the page owned "the mode and the
durability of every file listed here" while several had no stated mode at all.

`placement.places()` is the same facts as a table, and `placement.findings()`
audits them.  These tests hold the two properties that matter:

  * a tree THIS PROGRAM creates is never loose -- asserted end to end, through
    the real doors, not by checking that a helper was called;
  * something that ARRIVES loose is reported, with the exact `chmod`.

The audit found seven real ones on the developer's own machine the first time
it ran, including four serve logs at 0664 that carry a provider's
`client_secret` (`web/auth_providers/oauth.py` routes one there deliberately,
to keep it out of a user-visible response).
"""
from __future__ import annotations

import os
import stat

import pytest

from molbuilder import config_dir as CD
from molbuilder import placement as P


@pytest.fixture
def isolated_tree(tmp_path, monkeypatch):
    """All three roots inside tmp_path, so nothing here can see -- or touch --
    the developer's real installation."""
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path


def test_every_row_of_the_table_resolves(isolated_tree):
    """A row whose resolver is wrong is a file the audit silently never checks,
    which is the failure this table exists to prevent."""
    seen = []
    for place in P.places():
        path = place.resolve()
        seen.append(place.what)
        roots = (CD.config_dir(), CD.state_dir(), CD.runtime_dir())
        assert any(path == r or r in path.parents for r in roots), (
            f"{place.what} -> {path} is under none of the three roots")
    assert len(seen) == len(set(seen)), f"duplicate rows: {seen}"


def test_every_credential_door_in_config_dir_has_a_row(isolated_tree):
    """THE TABLE MUST NOT HAVE A HOLE, and nothing checked that until now.

    `placement` is the audit for a file that **arrives** loose -- restored
    from a backup, copied off another machine, or left by an older
    molbuilder -- so a door missing from the table is a credential nothing
    watches.  On 2026-09-15 the four newest doors were all missing:
    `jupyter_log` (which `serve_daemon` calls "a SECRET SINK", measured at
    0664 with fourteen `token=` lines), `jupyter_runtime` (the token
    itself), `jupyter_pidfile` and `jupyter_lab_home`.  Every writer was
    careful; nothing was WATCHING (`plan.md` § 5n.8).

    So this asserts COVERAGE rather than any one row: every path-returning
    door `config_dir` exports is either in the table or named here as
    deliberately outside it, with the reason.  Adding a door and forgetting
    the row now fails, which is the whole point.
    """
    # Doors that are deliberately NOT rows, each with its reason.  A door
    # added to `config_dir.__all__` lands in neither list and fails.
    NOT_POLICED = {
        # A per-port name the `serve-*.log` / `jupyter-*.log` patterns
        # already cover as families; a row per port is impossible.
        "serve_log", "serve_stacks_log", "jupyter_log",
        "serve_pidfile", "jupyter_pidfile", "jupyter_runtime",
        # Not a path door.
        "ports_with_pidfile",
    }
    rows = {p.what for p in P.places()}
    doors = [n for n in CD.__all__
             if callable(getattr(CD, n, None)) and n not in NOT_POLICED]
    missing = []
    for name in doors:
        fn = getattr(CD, name)
        try:
            path = fn()                      # only the zero-argument doors
        except TypeError:
            continue                         # takes a port: a family, above
        if not any(p.resolve() == path for p in P.places()):
            missing.append(f"{name}() -> {path}")
    assert not missing, (
        "config_dir doors with no row in `placement.places()`, so nothing "
        "audits them: " + "; ".join(missing) + f".  Rows today: {sorted(rows)}")

    # AND the per-port families are covered by a pattern, not by luck.
    patterns = {p.pattern for p in P.places() if p.pattern}
    for family in ("serve-*.log", "serve-*.pid",
                   "jupyter-*.log", "jupyter-*.pid", "jupyter-*.json"):
        assert family in patterns, f"no row covers the {family} family"


def test_a_tree_this_program_creates_is_never_loose(isolated_tree):
    """End to end, through the real doors: seed a fresh installation, write a
    config, run the supervisor's log roll -- then ask the audit.

    This is the property, and it is stronger than "the creator was called":
    every one of those paths is made by a different surface, and A4/A2/I5/B3
    were each a surface that made one of them with a bare `mkdir` or `open`.
    """
    from molbuilder.envs import initconfig
    from molbuilder.runtime_config import write_config_scope
    from molbuilder.serve_daemon import LogRoll

    initconfig.init_config("conda activate", probe=False)
    write_config_scope(None, {"envs": {"siesta": "molbuilder-siesta"}})
    roll = LogRoll(CD.serve_log(9999), max_bytes=10_000, keep=1)
    roll.write(b"a line\n")
    roll.close()
    CD.ensure_private_dir(CD.runtime_dir(), tighten=True)

    assert P.findings() == [], (
        "a tree molbuilder made itself has loose modes:\n"
        + "\n".join(P.findings()))
    # and the things it made really are there, not merely un-flagged
    assert CD.config_dir().is_dir() and CD.secrets_dir().is_dir()
    assert CD.serve_log(9999).is_file()


def test_the_probe_makes_the_config_directory_private(isolated_tree):
    """A4, and it is the FIRST command a person runs on a target: the seeded
    `environments/README` tells them to.  `jobset probe --write` made the
    directory every later secret lands in with a bare `mkdir` and no mode, and
    a later `envs init-config` reported it *"kept"* -- so `secret_key`,
    `google_client_secret`, `notify_keys` and the TLS key the `secrets/README`
    invites all sat 0600 inside a directory anyone on a shared login node could
    list and traverse."""
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group

    result = CliRunner().invoke(jobset_group, ["probe", "--write", "--yes"])

    assert result.exit_code == 0, result.output
    from molbuilder.scheduler import machine_scope_path
    assert machine_scope_path().is_file(), result.output
    assert stat.S_IMODE(CD.config_dir().stat().st_mode) == 0o700
    assert P.findings() == [], "\n".join(P.findings())


def test_what_arrives_loose_is_reported_with_its_chmod(isolated_tree):
    """The case no writer can control: copied from another machine, restored
    from a backup, unpacked from an archive that dropped its modes.  A warning,
    never a refusal -- the fix is one command and the line carries it."""
    from molbuilder.envs import initconfig

    initconfig.init_config("conda activate", probe=False)
    assert P.findings() == []

    os.chmod(CD.secrets_dir(), 0o755)
    out = P.findings()

    assert len(out) == 1, out
    assert "secrets/" in out[0]
    assert f"chmod 0700 {CD.secrets_dir()}" in out[0]
    assert "0755" in out[0], "the mode it actually has is part of the report"


def test_the_config_write_is_private_when_it_is_the_first_writer(isolated_tree):
    """D11.  `write_config_scope` went through the one atomic writer and then
    `chmod`-ed: `write_bytes` widened the temp to 0644 **with the content in
    it** and the chmod closed the window afterwards -- which is the exact
    sequence `write_bytes`' own `mode=` parameter exists to make impossible,
    and its docstring already claimed the property.

    First writer, deliberately: when `init-config` has already made the file
    0600, a rewrite PRESERVES that mode and the defect is invisible.  It was
    invisible to the first version of this test, which is how it got here.
    """
    from molbuilder.runtime_config import write_config_scope

    target = write_config_scope(None, {"envs": {"siesta": "molbuilder-siesta"}})

    assert stat.S_IMODE(target.stat().st_mode) == 0o600, (
        "the machine config was written world-readable by its own writer")
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert P.findings() == [], "\n".join(P.findings())


def test_tighter_than_required_is_not_a_finding(isolated_tree):
    """An operator who went further than we ask has not made a mistake."""
    from molbuilder.envs import initconfig

    initconfig.init_config("conda activate", probe=False)
    os.chmod(CD.secrets_dir(), 0o500)

    assert P.findings() == []


def test_an_absent_file_is_not_a_finding(isolated_tree):
    """Most of this tree is optional: a workstation with no HTTPS and no
    sign-in has an empty `secrets/` and no `notify` at all."""
    CD.ensure_private_dir(CD.config_dir())
    assert not CD.session_key().exists()
    assert P.findings() == []


class TestTheOneCreator:

    def test_it_makes_a_directory_private(self, tmp_path):
        d = CD.ensure_private_dir(tmp_path / "a" / "b")
        assert stat.S_IMODE(d.stat().st_mode) == 0o700

    def test_every_missing_ancestor_is_private_too(self, tmp_path):
        """`Path.mkdir(parents=True, mode=)` gives the mode to the leaf and
        the umask to the parents it creates -- so `environments/` made under a
        fresh config root left the ROOT at 0755 around every secret written
        into it later (review C-L1, measured 2026-09-14)."""
        d = CD.ensure_private_dir(tmp_path / "root" / "mid" / "leaf")
        for p in (d, d.parent, d.parent.parent):
            assert stat.S_IMODE(p.stat().st_mode) == 0o700, p

    def test_it_does_not_police_what_is_already_there(self, tmp_path):
        """Seeding seeds.  On a cluster `XDG_CONFIG_HOME=/scratch/$USER` is how
        a person keeps tokens off an NFS `$HOME`, and silently re-moding what
        they set up is the program deciding for them.  `envs doctor` reports
        it instead."""
        d = tmp_path / "a"
        d.mkdir(mode=0o755)
        CD.ensure_private_dir(d)
        assert stat.S_IMODE(d.stat().st_mode) == 0o755

    def test_it_tightens_when_the_caller_owns_the_directory(self, tmp_path):
        """The supervisor's log directory is ours and is about to hold a log
        carrying a provider's `client_secret`.  It was measured at 0775."""
        d = tmp_path / "a"
        d.mkdir(mode=0o755)
        CD.ensure_private_dir(d, tighten=True)
        assert stat.S_IMODE(d.stat().st_mode) == 0o700
