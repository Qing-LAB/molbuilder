"""One directory, one function — the callers cannot disagree about it.

`configuration.md` M-4 gave ``environment.json`` one home for its FILENAME,
which "was a string literal in three modules".  The DIRECTORY that filename
sits in stayed spelled three times:

    runtime_config._machine_config_file      -> molbuilder.json
    scheduler/record.machine_scope_path      -> environment.json, environments/
    auth_setup.default_secret_dir            -> secret_key

They agreed, and two of them said so in prose -- *"Mirrors
auth_setup.default_secret_dir's convention"* and *"mirrored rather than
imported"*.  **A comment is not a mechanism.**  Fixed 2026-08-23 by
``molbuilder/config_dir.py``; this pins it.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "molbuilder"


@pytest.fixture
def moved(monkeypatch, tmp_path):
    """One operator action -- set ``XDG_CONFIG_HOME`` -- moves everything."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "elsewhere"))
    return tmp_path / "elsewhere" / "molbuilder"


def _all_four():
    # Asked of `config_dir` itself.  These went through
    # `auth_setup.secret_key_path` / `default_secret_dir` until 2026-09-13 --
    # pass-throughs that were this module's point made backwards: a second
    # public name for a door `config_dir` owns (I8).
    from molbuilder.config_dir import config_dir, session_key
    from molbuilder.runtime_config import _machine_config_file
    from molbuilder.scheduler import environments_dir, machine_scope_path
    return {
        "secret_key":     session_key(),
        "secret_dir":     config_dir(),
        "environment":    machine_scope_path(),
        "environments":   environments_dir(),
        "molbuilder.json": _machine_config_file(),
    }


def test_every_per_user_file_sits_under_the_one_directory(moved):
    """The property that matters to an operator: move the variable, and the
    secrets, the machine record and the config all move TOGETHER."""
    for name, p in _all_four().items():
        assert moved in p.parents or p == moved, (
            f"{name} -> {p} is not under {moved}")


def test_the_directory_is_read_at_call_time_not_captured_at_import(
        monkeypatch, tmp_path):
    """Captured at import, a test (or an operator) could move the variable
    and have half the callers keep the old answer -- which is precisely how
    the suite came to read the developer's real `~/.config/molbuilder`."""
    from molbuilder.config_dir import config_dir
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "a"))
    first = config_dir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "b"))
    assert config_dir() != first


def test_no_module_spells_the_rule_a_second_time():
    """**The guard that would have caught the original drift.**

    Any module joining ``XDG_CONFIG_HOME`` to a path itself is a fourth copy
    growing.  ``config_dir.py`` is the one place allowed to read it.
    """
    offenders = []
    for py in _SRC.rglob("*.py"):
        if py.name == "config_dir.py":
            continue
        src = py.read_text(encoding="utf-8")
        if "XDG_CONFIG_HOME" not in src:
            continue
        for node in ast.walk(ast.parse(src)):
            # os.environ.get("XDG_CONFIG_HOME") / os.environ["XDG_CONFIG_HOME"]
            if isinstance(node, ast.Constant) and node.value == "XDG_CONFIG_HOME":
                offenders.append(str(py.relative_to(_SRC)))
                break
    assert not offenders, (
        "these modules read XDG_CONFIG_HOME directly instead of calling "
        f"config_dir(): {sorted(set(offenders))}")


# ---------------------------------------------------------------------------
# The override — `configuration.md` § 2.1c, which carries the rule the
# 2026-09-01 access plan decided.  (The comment cited the PLAN until
# 2026-09-02; a plan records how a decision was reached, the contract
# records what is true now, and only one of them is maintained.)
# ---------------------------------------------------------------------------

class TestTheRootCanBeNamedOutright:

    def test_it_is_used_exactly_as_given(self, monkeypatch, tmp_path):
        """No ``molbuilder`` component is appended, and that asymmetry with
        ``XDG_CONFIG_HOME`` is the design.

        ``XDG_CONFIG_HOME`` names a root shared by every application, so ours
        must add its own name under it.  ``MOLBUILDER_CONFIG_DIR`` names OUR
        directory; appending to it would put the files somewhere the person
        did not ask for.
        """
        from molbuilder.config_dir import config_dir, CONFIG_DIR_ENV
        monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path / "here"))
        assert config_dir() == tmp_path / "here"

    def test_it_beats_xdg_and_does_not_fall_back_past_it(
            self, monkeypatch, tmp_path):
        """An override, not a search step.

        A fallback here would recreate exactly the shadowing
        `configuration.md` § 2.1a exists to warn about: one setting, two
        files, one of them silently winning.
        """
        from molbuilder.config_dir import config_dir, CONFIG_DIR_ENV
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path / "named"))
        assert config_dir() == tmp_path / "named"

    def test_empty_is_not_set(self, monkeypatch, tmp_path):
        """``MOLBUILDER_CONFIG_DIR=`` is how a shell unsets a variable it
        cannot unset; treating it as a root would put the config at the
        filesystem root."""
        from molbuilder.config_dir import config_dir, CONFIG_DIR_ENV
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        monkeypatch.setenv(CONFIG_DIR_ENV, "")
        assert config_dir() == tmp_path / "xdg" / "molbuilder"

    def test_it_moves_every_per_user_file_together(self, monkeypatch, tmp_path):
        """The property the variable exists for -- the same one
        ``XDG_CONFIG_HOME`` has, asserted for the new door too."""
        from molbuilder.config_dir import CONFIG_DIR_ENV
        monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path / "one"))
        for name, path in _all_four().items():
            assert (tmp_path / "one") in path.parents \
                or path == tmp_path / "one", f"{name} -> {path}"

    def test_no_module_spells_the_override_a_second_time(self):
        """Same guard as `XDG_CONFIG_HOME`'s, for the same reason.

        Matched against the SYNTAX TREE, not the text.  A first version
        matched the name anywhere and flagged `envs/_cli.py`, whose only
        mention is a docstring explaining why its log directory moved -- the
        same trap the second-root pin fell into, and the same answer: a test
        that cannot tell a path from a sentence about one punishes the
        documentation we want.
        """
        offenders = []
        for py in _SRC.rglob("*.py"):
            if py.name == "config_dir.py":
                continue
            src = py.read_text(encoding="utf-8")
            if "MOLBUILDER_CONFIG_DIR" not in src:
                continue
            for node in ast.walk(ast.parse(src)):
                # os.environ.get("MOLBUILDER_CONFIG_DIR") / os.environ[...]
                if isinstance(node, (ast.Call, ast.Subscript)) \
                        and "MOLBUILDER_CONFIG_DIR" in {
                            n.value for n in ast.walk(node)
                            if isinstance(n, ast.Constant)
                            and isinstance(n.value, str)}:
                    offenders.append(str(py.relative_to(_SRC)))
                    break
        assert not offenders, (
            "these modules read MOLBUILDER_CONFIG_DIR directly instead of "
            f"calling config_dir(): {sorted(set(offenders))}")


# ---------------------------------------------------------------------------
# The second root — RETIRED 2026-08-31
# ---------------------------------------------------------------------------
#
# `_SECOND_ROOT_HOLDOUTS` listed the three modules that still computed
# `~/.molbuilder/...` themselves, and its own failure message said: "when the
# list empties, delete the list and this test with it: the root is gone and
# there is nothing to shrink."  Step 2 emptied it, so it is deleted rather than
# left standing as an empty allowance.
#
# What replaces it is not a smaller allow-list but a stricter question, below:
# no module may compute a per-user root AT ALL.
# ---------------------------------------------------------------------------
# Operational state — archive/2026-09-01-config-access-plan.md § 3.2
# ---------------------------------------------------------------------------

class TestOperationalStateFollowsXdg:
    """`$XDG_STATE_HOME` entered the Base Directory spec in 0.8 for state that
    persists across restarts but is not portable enough for `$XDG_DATA_HOME`,
    and the spec names LOGS first.  `$XDG_RUNTIME_DIR` is the one for pidfiles.
    """

    def test_state_follows_its_variable(self, monkeypatch, tmp_path):
        from molbuilder.config_dir import state_dir
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "s"))
        assert state_dir() == tmp_path / "s" / "molbuilder"

    def test_state_defaults_to_the_spec_location(self, monkeypatch, tmp_path):
        from molbuilder.config_dir import state_dir
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert state_dir() == tmp_path / ".local" / "state" / "molbuilder"

    def test_runtime_prefers_its_own_variable(self, monkeypatch, tmp_path):
        from molbuilder.config_dir import runtime_dir
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "r"))
        assert runtime_dir() == tmp_path / "r" / "molbuilder"

    def test_runtime_falls_back_to_state_and_not_to_a_temp_dir(
            self, monkeypatch, tmp_path):
        """XDG_RUNTIME_DIR is cleared when the session ends, and is not always
        set (cron, a detached ssh, some containers).  A supervisor's pidfile
        that vanished under it would leave a running server nothing can find,
        so the fallback persists."""
        from molbuilder.config_dir import runtime_dir, state_dir
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "s"))
        assert runtime_dir() == state_dir() / "run"

    def test_state_is_not_under_the_config_root(self, monkeypatch, tmp_path):
        """Configuration is edited and backed up; logs grow and are deleted.
        A person who wants them together says so with `paths`."""
        from molbuilder.config_dir import config_dir, state_dir
        monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path / "cfg"))
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        assert config_dir() not in state_dir().parents
        assert state_dir() != config_dir()


class TestOperationalStateFollowsTheVariablesOnly:
    """`paths.logs` / `paths.run` / `paths.reports` are RETIRED.

    They existed for a day.  `$XDG_STATE_HOME` and `$XDG_RUNTIME_DIR` already
    move these directories, so the keys were a second way to say one thing --
    and being a second way is what put the answer out of reach of the layer
    that needs it: `serve_daemon` is L1, the config reader is L2, and a
    supervisor must be able to write its log before any config is read.

    Deleting them removed the inversion instead of working around it with an
    injection point (`archive/2026-09-01-config-access-plan.md` § 5.3).
    """

    @pytest.fixture()
    def cfg(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path / "cfg"))
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        (tmp_path / "cfg").mkdir(parents=True, exist_ok=True)
        return tmp_path

    def _write(self, cfg, obj):
        import json
        (cfg / "cfg" / "molbuilder.json").write_text(json.dumps(obj))

    def test_the_variables_move_them(self, cfg):
        from molbuilder.config_dir import logs_dir, reports_dir, runtime_dir
        assert logs_dir() == cfg / "state" / "molbuilder" / "logs"
        assert reports_dir() == cfg / "state" / "molbuilder" / "reports"
        assert runtime_dir() == cfg / "state" / "molbuilder" / "run"

    @pytest.mark.parametrize("key", ["logs", "run", "reports"])
    def test_the_retired_key_is_refused(self, cfg, key):
        from molbuilder.runtime_config import RuntimeConfigError, read_config
        self._write(cfg, {"paths": {key: "/somewhere"}})
        with pytest.raises(RuntimeConfigError, match="no longer configured"):
            read_config()

    def test_the_refusal_names_the_variable_that_replaces_it(self, cfg):
        """A refusal that does not name the replacement is a dead end."""
        from molbuilder.runtime_config import RuntimeConfigError, read_config
        self._write(cfg, {"paths": {"logs": "/somewhere"}})
        with pytest.raises(RuntimeConfigError) as e:
            read_config()
        assert "XDG_STATE_HOME" in str(e.value)

    def test_it_does_not_read_as_a_typo(self, cfg):
        from molbuilder.runtime_config import RuntimeConfigError, read_config
        self._write(cfg, {"paths": {"run": "/somewhere"}})
        with pytest.raises(RuntimeConfigError) as e:
            read_config()
        assert "unknown key" not in str(e.value)

    def test_projects_stays(self, cfg):
        """Data rather than operational state, no XDG equivalent, and
        `$MOLBUILDER_PROJECTS` is its documented override."""
        from molbuilder.runtime_config import read_config
        self._write(cfg, {"paths": {"projects": "/data/projects"}})
        assert read_config()["paths"]["projects"] == "/data/projects"

    def test_a_key_nothing_reads_is_still_refused(self, cfg):
        from molbuilder.runtime_config import RuntimeConfigError, read_config
        self._write(cfg, {"paths": {"cache": "/tmp/x"}})
        with pytest.raises(RuntimeConfigError, match="cache"):
            read_config()
# ---------------------------------------------------------------------------
# One API — archive/2026-09-01-config-access-plan.md § 5
# ---------------------------------------------------------------------------

#: THE DIVISION A11 DRAWS: the module that owns a FORMAT owns its NAME, and
#: `config_dir` owns the DIRECTORY.  So a file with a format owner keeps its
#: name there and that owner exposes the path function; what lives in
#: `config_dir` is the files with no format to own.
#:
#: Pulling `environment.json` and `notify` into `config_dir` was tried and
#: reverted the same day — it took a name from its format owner, and
#: `test_architecture_rules`' A11 said so before any of this shipped.
class TestEveryFileHasADoor:

    def test_the_ported_ones_take_the_port(self, monkeypatch, tmp_path):
        from molbuilder.config_dir import serve_log, serve_pidfile
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "r"))
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "s"))
        assert serve_pidfile(8888).name == "serve-8888.pid"
        assert serve_log(8888).name == "serve-8888.log"

    def test_every_door_moves_with_the_one_variable(self, monkeypatch,
                                                    tmp_path):
        """The property the whole change exists for: name one directory and
        every configuration file is under it — the format owners' doors
        included, since they ask this module for the directory."""
        from molbuilder.config_dir import (config_dir, google_client_secret,
                                           session_key)
        from molbuilder.monitor import default_notify_path, notify_keys_path
        from molbuilder.runtime_config import machine_config_path
        from molbuilder.scheduler.record import (environments_dir,
                                                 machine_scope_path)
        root = tmp_path / "named"
        monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(root))
        doors = {
            "session_key": session_key(),
            "google_client_secret": google_client_secret(),
            "machine config": machine_config_path(),
            "environment record": machine_scope_path(),
            "environments/": environments_dir(),
            "notify": default_notify_path(),
            "notify_keys": notify_keys_path(),
        }
        assert config_dir() == root
        for name, got in doors.items():
            assert root in got.parents or got == root, f"{name} -> {got}"

        # AND EVERY CREDENTIAL IS IN `secrets/`, asked of the audit rather
        # than listed here.  `placement.misplaced()` reads the same table
        # `envs doctor` prints from, so this pins the rule where it is
        # DEFINED -- a credential added to that table is covered the moment
        # it is added, and a test listing names would have to be remembered.
        #
        # Measured 2026-09-20, before that function existed: pointing
        # `session_key()` back at the config root left 222 tests green,
        # because every test and every audit row asks the same resolver and
        # moves with it.  Nothing compared the answer against the rule.
        from molbuilder.placement import misplaced
        assert misplaced() == [], (
            "a credential store resolves outside secrets/:\n"
            + "\n".join(misplaced()))


class TestNoModuleNamesOneOfThoseFilesItself:
    """**The rule, asked the strict way round.**

    Not *"is everything using the API"* — a module can pass that by doing
    nothing — but *"does anything still spell one of these filenames"*.  Seven
    modules each held one and joined it onto a directory; each join was correct
    and together they were seven places that had to agree with nothing making
    them.  `configuration.md` M-4 recorded exactly this for `environment.json`
    ("a string literal in three modules"), fixed that one file, and did not
    generalise — so the other six grew back.
    """

    #: The format owner of each, per A11.  Nobody else may join it.
    _OWNERS = {
        "config_dir.py": "the files with no format owner",
        "runtime_config.py": "molbuilder.json's schema",
        "scheduler/record.py": "environment.json's schema",
        "monitor.py": "the notify exchange",
    }

    #: `"notify"` joined the list on 2026-09-20.  It was the one credential
    #: filename the guard did not cover -- `notify_keys` was here and its
    #: neighbour was not -- so a module could have built that path by hand and
    #: nothing would have said.  Verified clean before adding: no module
    #: outside `monitor.py` joins it.
    @pytest.mark.parametrize("literal", [
        '"molbuilder.json"', '"secret_key"', '"google_client_secret"',
        '"environment.json"', '"notify"', '"notify_keys"',
        # the DIRECTORY too, not just the files in it: `config_dir` owns
        # SECRETS_DIRNAME, and a module joining `/ "secrets"` by hand would
        # pin the layout in a second place.  Verified clean when added.
        '"secrets"',
    ])
    def test_the_literal_appears_only_where_it_is_owned(self, literal):
        """**Read as CODE, not as text** *(rewritten 2026-09-20)*.

        It matched the literal and a `/`, `Path(` or `join` ON THE SAME LINE.
        Four shapes walked through it, each measured by appending them to a
        non-owner module and watching the suite stay green:

            config_dir() / 'secrets' / 'secret_key'     # single quotes
            base = config_dir(); base / _SEC / _NAME    # name on another line
            Path(f"{config_dir()}/secrets/notify")      # f-string, no quote
            os.path.join(str(config_dir()), _SEC, _KEYS)

        All four built a real credential path in a module with no business
        doing so.  The sibling test above already walks the AST for
        `XDG_CONFIG_HOME`, with a docstring saying why text matching was
        wrong; this one had not been given the same treatment.

        The AST sees a string wherever it appears -- quote style, line
        breaks and f-string pieces are all gone by then -- so what remains is
        one question: does this module contain the filename as a string
        constant at all, outside a docstring?  A module that legitimately
        needs the file asks its owner and never names it.
        """
        name = literal.strip('"')

        def joined_into_a_path(tree):
            """Line numbers where `name` is BUILT INTO a path in this tree."""
            # Module-level `_NAME = "secret_key"` then `base / _NAME` was one
            # of the measured evasions, so a simple alias map comes first.
            # Deliberately NOT dataflow analysis: one level of
            # `Name = Constant`, which is the shape that actually appeared.
            alias = {t.id for n in ast.walk(tree)
                     if isinstance(n, ast.Assign)
                     for t in n.targets
                     if isinstance(t, ast.Name)
                     and isinstance(n.value, ast.Constant)
                     and n.value.value == name}

            def is_it(node):
                return ((isinstance(node, ast.Constant) and node.value == name)
                        or (isinstance(node, ast.Name) and node.id in alias))

            hits = []
            for node in ast.walk(tree):
                # `config_dir() / "secrets" / "secret_key"` -- quote style and
                # line breaks are gone by the time the AST exists, which is
                # the whole reason this is not a text match any more.
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                    if is_it(node.left) or is_it(node.right):
                        hits.append(node.lineno)
                # `os.path.join(...)` and `Path(...).joinpath(...)`.
                # Spelled out rather than "any .join", because `", ".join(x)`
                # is a string method and has nothing to do with paths.
                elif isinstance(node, ast.Call):
                    fn = node.func
                    joins = isinstance(fn, ast.Attribute) and (
                        fn.attr == "joinpath"
                        or (fn.attr == "join"
                            and isinstance(fn.value, ast.Attribute)
                            and fn.value.attr == "path"))
                    if joins and any(is_it(a) for a in node.args):
                        hits.append(node.lineno)
                # `f"{config_dir()}/secrets/notify"` -- no quote character
                # precedes the name, so the old text match could not see it.
                # The shape that makes it a PATH is a "/" directly after an
                # interpolated base; merely containing the word is prose, and
                # every error message in this package would trip on that.
                elif isinstance(node, ast.JoinedStr):
                    vals = node.values
                    for i, v in enumerate(vals[:-1]):
                        nxt = vals[i + 1]
                        if not (isinstance(v, ast.FormattedValue)
                                and isinstance(nxt, ast.Constant)
                                and isinstance(nxt.value, str)
                                and nxt.value.startswith("/")):
                            continue
                        tail = nxt.value.split()[0]     # stop at prose
                        if name in tail.strip("/").split("/"):
                            hits.append(node.lineno)
                            break
            return hits

        offenders = {}
        for py in _SRC.rglob("*.py"):
            rel = str(py.relative_to(_SRC))
            if rel in self._OWNERS:
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except SyntaxError:                       # not ours to police
                continue
            hits = joined_into_a_path(tree)
            if hits:
                offenders[rel] = sorted(set(hits))
        assert not offenders, (
            f"{literal} is built into a path outside the module that owns "
            f"it: {offenders}.  Ask the owner for the path instead "
            f"(archive/2026-09-01-config-access-plan.md § 5)")



class TestAFilesResolverIsNotAWayToReachTheRoot:
    """A11, corrected 2026-09-12.

    Its elaboration used to say *"a per-user config path is
    `environment.machine_scope_path`'s"* -- naming ONE FILE's resolver as the
    owner of the root -- so a reader following the rule as written was sent to
    `machine_scope_path().parent`, and two sites went there.

    `test_every_door_moves_with_the_one_variable` above cannot catch that: a
    climb off a resolver that itself follows the variable gives the RIGHT answer
    today.  What it gets wrong is WHOSE answer it is.
    """

    def test_moving_the_machine_record_does_not_move_the_directory_beside_it(
            self, monkeypatch, tmp_path):
        """The drift, made visible.  `environments_dir()` reached the config
        root through `environment.json`'s resolver, so it followed that FILE
        rather than the root: move the machine record and the named-target
        directory moved with it, silently, on whichever machine did that."""
        from molbuilder.scheduler import record as R
        before = R.environments_dir()

        monkeypatch.setattr(
            R, "machine_scope_path",
            lambda: tmp_path / "somewhere" / "else" / "environment.json")

        assert R.environments_dir() == before, (
            "the named-target directory moved because the machine record "
            "moved; it is the config ROOT's child, not that file's sibling")
        assert R.named_environment_path("sol") == before / "sol.json"

    def test_the_provenance_display_reports_the_record_s_OWN_door(
            self, monkeypatch, tmp_path):
        """`config_provenance` spelled `Path(project_dir) / FILENAME` itself --
        in the one function whose whole job is to tell a reader which file
        answered, and against a façade that exported `FILENAME` while hiding
        `calculation_record`.  A reader is TOLD this path, so it has to be the
        path the reader would be read from."""
        import molbuilder.scheduler as S
        from molbuilder import runtime_config as RC

        marker = tmp_path / "bundle" / "a-different-shape.json"
        # Patched on the FAÇADE, which is where the display asks for it -- and
        # the façade is the structural half of this finding: it exported
        # `FILENAME` and did not export `calculation_record`, so it offered the
        # filename and hid the door.
        monkeypatch.setattr(S, "calculation_record", lambda d: marker)

        report = RC.config_provenance(project_dir=str(tmp_path / "bundle"))
        reported = [r["path"] for r in report["sources"]
                    if r.get("scope") == "environment"
                    and r.get("via") == "calculation"]

        assert reported == [str(marker)], (
            f"the display spells the record's path itself: {reported}")


def test_a_printed_remedy_names_the_resolved_directory(monkeypatch, tmp_path):
    """I3.  `prep` refused a remote target whose record states no activation and
    told the person to *"copy the record into `~/.config/molbuilder/environments/`
    here"* -- a literal, on the machine where `MOLBUILDER_CONFIG_DIR` is the
    whole point of the variable.  Following it put the record where `prep` does
    not look, and `prep` then refused again with the same message.

    This is the defect `notify-token --keys-file` was DELETED for on the same
    day; the sweep missed this site.
    """
    import pytest as _pytest

    root = tmp_path / "named"
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(root))
    from molbuilder.jobset.prep import PrepError, _require_remote_activation
    from molbuilder.scheduler.record import Environment

    with _pytest.raises(PrepError) as excinfo:
        _require_remote_activation("sol", Environment(scheduler="slurm"))

    message = str(excinfo.value)
    assert str(root / "environments") in message, message
    assert "~/.config/molbuilder" not in message, (
        "the remedy still hard-codes the default root:\n" + message)
