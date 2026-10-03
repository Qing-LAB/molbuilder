"""A machine that has never run molbuilder can be made to work by one command.

`execution/running-a-job.md` § 5.2 names the failure these tests are about:

    `activation` ... has **no default**: a target whose record carries none
    is refused at prep ... On a fresh install that would bite a workstation
    first.

Nothing created the config directory or this machine's record, so that was
the state every new machine started in.  ``molbuilder envs init-config`` --
run at the end of ``bootstrap`` -- is what ends it: it asks the activation and
writes it into ``molbuilder.json`` as ``env_init`` -- the one fact that file
keeps about this machine (`configuration.md` § 4) -- and seeds this machine's
record through the probe, which copies it there.

**These drive the API and assert the OUTCOME.**  Not one of them reads the
source of anything: the question is always *what does the program do now that
it did not do before*, asked through the same doors the wrapper generator and
the probe use.  A test that grepped ``initconfig.py`` for a string would pass
just as happily against a module that wrote its file to the wrong directory.
"""
from __future__ import annotations

import json
from pathlib import Path
import os
import stat

import pytest

from molbuilder.envs import initconfig
from molbuilder.runtime_config import ACTIVATION_FORMS, CONFIG_FILENAME


def _the_gate():
    """What prep asks of this machine's record before it renders a wrapper
    (`jobset.prep._require_activation`) -- the record read the way every
    prep reads it.  Returns the record."""
    from molbuilder.jobset.prep import _require_activation
    from molbuilder.scheduler import machine_for
    env = machine_for()
    _require_activation(None, env)
    return env


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """A machine that has never run molbuilder -- returns its config dir.

    Isolated by ``XDG_CONFIG_HOME``, which the suite-wide
    ``config_root_is_never_the_developers`` fixture deliberately leaves working
    (it clears the ``MOLBUILDER_CONFIG_DIR`` override rather than pinning it,
    so a test's own arrangement still decides).
    """
    root = tmp_path / "xdg"
    root.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(root))
    from molbuilder.config_dir import config_dir
    where = config_dir()
    assert not where.exists(), "the fixture must start from nothing"
    return where


# ══ THE SYMPTOM THE CONTRACT NAMES ═════════════════════════════════════════

def test_a_fresh_machine_refuses_to_render_any_wrapper(fresh):
    """The baseline.  Without this failing first, nothing below means anything.
    """
    from molbuilder.jobset.prep import PrepError
    with pytest.raises(PrepError) as exc:
        _the_gate()
    assert "env_init" in str(exc.value)


@pytest.mark.parametrize("activation", sorted(ACTIVATION_FORMS))
def test_seeding_removes_the_refusal(fresh, activation):
    """The outcome, asked through the gate every prep goes through: the
    record this machine is read by carries the activation asked for -- a
    pass here is a wrapper rendering, not a file merely existing on disk.
    """
    initconfig.init_config(activation, probe=True)
    assert _the_gate().env_init["activation"] == activation


def test_the_seeded_file_is_read_by_the_real_reader(fresh):
    """A seeded file that the validator refuses is worse than no file.

    ``molbuilder.json`` REFUSES unknown top-level keys, and the guidance this
    file carries is written as ``_``-prefixed comment keys.  This is the test
    that those two facts agree -- the read goes through the server's own
    reader -- and that the answers asked at install reach the record whole.
    """
    from molbuilder.runtime_config import read_config
    initconfig.init_config("conda activate", preamble="module load mamba",
                           probe=True)
    read_config(fresh / CONFIG_FILENAME)            # refuses -> raises
    assert _the_gate().env_init == {
        "activation": "conda activate", "preamble": "module load mamba"}




# ══ IT NEVER OVERWRITES ════════════════════════════════════════════════════

def test_seeding_twice_changes_nothing(fresh):
    """Idempotent, so ``bootstrap`` may call it unconditionally."""
    initconfig.init_config("conda activate", probe=True)
    before = {p: p.read_bytes() for p in sorted(fresh.rglob("*")) if p.is_file()}
    assert before, "the first run must have written something"

    steps = initconfig.init_config("source activate", probe=True)
    assert [s.action for s in steps] == ["kept"] * len(steps)
    after = {p: p.read_bytes() for p in sorted(fresh.rglob("*")) if p.is_file()}
    assert after == before


def test_an_existing_config_is_left_exactly_as_it_is(fresh):
    """A person's own file is never merged into, never reformatted -- and
    keeping it is not staying quiet about it: one written before a section
    was retired no longer reads, and the note says why, in the reader's own
    words.
    """
    fresh.mkdir(parents=True)
    mine = fresh / CONFIG_FILENAME
    mine.write_text('{"execution": {"mode": "direct"}}')
    original = mine.read_bytes()

    step = initconfig.seed_machine_config("conda activate")

    assert step.action == "kept"
    assert mine.read_bytes() == original
    assert "does not read" in step.note, step.note
    assert "'execution' is now 'launch'" in step.note, step.note

def test_a_config_without_env_init_gets_the_one_asked_and_nothing_else(fresh):
    """``init-config`` asks how this machine enters an environment, so the
    answer is written -- into a file that already exists and declares none,
    that section alone; what the person wrote stays as it was.  It asked and
    dropped the answer until 2026-10-02 (W54 R17)."""
    from click.testing import CliRunner
    from molbuilder.envs._cli import envs_group
    fresh.mkdir(parents=True)
    (fresh / CONFIG_FILENAME).write_text('{"launch": {"mode": "direct"}}')
    result = CliRunner().invoke(
        envs_group, ["init-config", "--activation", "source activate",
                     "--no-probe", "--yes"])
    assert result.exit_code == 0, result.output
    doc = json.loads((fresh / CONFIG_FILENAME).read_text())
    assert doc["env_init"] == {"activation": "source activate"}, doc
    assert doc["launch"] == {"mode": "direct"}, doc


def test_a_seed_the_loader_would_refuse_is_not_written(fresh, monkeypatch):
    """The seed goes through the one writer of molbuilder.json, which
    validates before a byte lands.  `init-config` had a writer of its own,
    so a seed the loader refused would have been written, reported
    "created", and refused by every later read (review C-Y1)."""
    from molbuilder.runtime_config import RuntimeConfigError, machine_config_path
    monkeypatch.setattr(initconfig, "seed_document",
                        lambda *a, **k: {"bogus": {"x": 1}})
    with pytest.raises(RuntimeConfigError, match="unknown top-level"):
        initconfig.seed_machine_config("conda activate")
    assert not machine_config_path().exists()



# ══ WHAT IT WRITES, AND WHAT IT REFUSES TO WRITE ═══════════════════════════

def test_no_probe_seeds_the_config_without_a_record(fresh):
    """``--no-probe`` writes the config and no record: `jobset probe --write`
    makes the record later, copying the activation the config declares."""
    initconfig.init_config("conda activate", probe=False)
    assert (fresh / CONFIG_FILENAME).is_file()
    from molbuilder.scheduler import machine_scope_path, environments_dir
    assert not machine_scope_path().exists()
    assert environments_dir().is_dir()


def test_the_directory_and_the_config_are_not_world_readable(fresh):
    """It sits beside the session key and the OAuth secret."""
    initconfig.init_config("conda activate", probe=False)
    assert stat.S_IMODE(fresh.stat().st_mode) == 0o700
    assert stat.S_IMODE((fresh / CONFIG_FILENAME).stat().st_mode) == 0o600


def test_an_unwritable_root_is_refused_with_the_sentence_that_names_the_fix(
        tmp_path, monkeypatch):
    """D1 -- the *"a remedy the program prints that it then refuses to run"*
    class, from the inside.

    `bootstrap` prints `init-config` as the remedy, and `seeding_blockers()`
    has held the exact sentence for an unwritable config root since it was
    written -- while `init_config` answered the same condition with a raw
    `PermissionError` traceback out of `mkdir`, which names a system call
    rather than the thing to change.
    """
    import stat as _stat

    parent = tmp_path / "readonly"
    parent.mkdir()
    parent.chmod(0o500)
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(parent / "cfg"))

    with pytest.raises(RuntimeError) as excinfo:
        initconfig.init_config("conda activate", probe=False)

    message = str(excinfo.value)
    assert "not writable" in message, message
    assert "MOLBUILDER_CONFIG_DIR" in message, (
        "the refusal must name the way out", message)
    assert not (parent / "cfg").exists(), "it wrote despite refusing"
    parent.chmod(0o700)          # so tmp_path can be cleaned up


def test_a_preamble_is_never_written_for_a_hook_that_is_not_there(tmp_path):
    """Checked on disk, not derived and hoped for.

    A preamble naming a file that does not exist fails later, on the cluster,
    inside a job -- which is strictly worse than no preamble at all.
    micromamba ships no ``conda.sh``, so this is a real installation, not a
    hypothetical one.
    """
    assert initconfig.conda_hook(None) is None
    assert initconfig.conda_hook(str(tmp_path / "bin" / "micromamba")) is None

    root = tmp_path / "conda"
    hook = root / "etc" / "profile.d" / "conda.sh"
    hook.parent.mkdir(parents=True)
    hook.touch()
    assert initconfig.conda_hook(
        str(_manager_answering(tmp_path / "conda" / "bin" / "conda", root))
    ) == f"source {hook}"


def test_the_root_is_the_manager_s_ANSWER_not_its_binary_s_grandparent(tmp_path):
    """`installation.md` M2, and ASU Sol is the case.

    The root used to be `Path(conda_binary).resolve().parent.parent`, which
    happens to be right for ``<root>/bin/conda`` and ``<root>/condabin/conda``
    and is wrong for everything else -- a distro ``/usr/bin/conda`` derives
    ``/usr``, and the manager a cluster module puts on PATH is a shell WRAPPER
    whose own location says nothing about where the installation is.  The value
    this computes is written into the user's `molbuilder.json`, so the guess does
    not fail here: it fails in a job, on a cluster, weeks later.

    Here the wrapper and the installation are deliberately in different places,
    which is what a derivation cannot see and an answer can.
    """
    installation = tmp_path / "packages" / "apps" / "mamba" / "2.6.2"
    hook = installation / "etc" / "profile.d" / "conda.sh"
    hook.parent.mkdir(parents=True)
    hook.touch()
    wrapper = _manager_answering(tmp_path / "usr" / "local" / "bin" / "mamba",
                                installation)

    assert initconfig.conda_hook(str(wrapper)) == f"source {hook}", (
        "the hook was looked for beside the wrapper instead of inside the "
        "installation the manager names")


def _manager_answering(path, root_prefix):
    """A manager binary that answers `info --json` with `root_prefix`, which is
    the only thing `conda_hook` asks it."""
    import json
    import stat as _stat
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"root_prefix": str(root_prefix)})
    path.write_text("#!/bin/sh\necho '" + payload + "'\n")
    path.chmod(path.stat().st_mode | _stat.S_IXUSR)
    return path


def test_source_activate_carries_no_conda_hook_preamble(fresh):
    """``source activate`` is a script on PATH; it needs no hook sourced.

    Driven through the CLI's own decision, because that is where the pairing
    lives -- a preamble offered alongside the recommendation must be dropped
    when the person picks the other form.
    """
    from click.testing import CliRunner
    from molbuilder.envs._cli import envs_group

    result = CliRunner().invoke(
        envs_group, ["init-config", "--activation", "source activate",
                     "--yes"])

    assert result.exit_code == 0, result.output
    assert _the_gate().env_init == {"activation": "source activate"}


# ══ THE WARNING THAT COULD NOT FIRE ════════════════════════════════════════

def test_a_machine_declaring_no_activation_is_warned(fresh, monkeypatch):
    """`diagnostics.local_facts` returns the note whenever no activation was
    declared -- and NOT only when there is nothing else either.

    It used to be gated on the env list being empty as well, which is a fact
    it has nothing to do with: every machine that can run a calculation has
    conda envs, so the warning was suppressed on exactly the machines it was
    written for.  Then the probe assigned it to a variable nothing read.
    """
    from molbuilder import diagnostics
    from molbuilder.scheduler import resolve_environment

    monkeypatch.setattr(diagnostics, "get_capabilities",
                        lambda: diagnostics.Capabilities(
                            runtime_config={}, conda_binary="/x/bin/conda",
                            conda_binary_source="test",
                            conda_envs=frozenset({"molbuilder"})))

    env, note = diagnostics.local_facts(resolve_environment())

    assert env.conda_envs == ["molbuilder"], "the envs still travel"
    assert note and "declares no `env_init.activation`" in note


def test_source_activate_is_not_reported_as_a_missing_hook(fresh):
    """The absence of a preamble is correct for ``source activate``.

    It was reported as *"no conda.sh hook found to source"* -- on machines
    that had one and simply did not need it.  A note that states a false
    reason is worse than no note: it sends someone to look for a file that is
    already there.  Said on the step that declares the activation, this
    machine's molbuilder.json (`configuration.md` § 4).
    """
    step = _the_config_step(initconfig.init_config("source activate"))
    assert '"source activate"' in step.note, step.note
    assert "found" not in step.note and "hook" not in step.note, step.note


def test_conda_activate_without_a_hook_says_what_will_break(fresh):
    """The same absence IS worth flagging for the other form: `conda
    activate` is a shell function a non-interactive shell has never
    defined, so no preamble means a wrapper that fails inside the job."""
    step = _the_config_step(initconfig.init_config("conda activate"))
    assert "needs conda's hook" in step.note, step.note


def _the_config_step(steps):
    """The step that wrote this machine's molbuilder.json -- where the
    activation is declared, so where what is said about it is said."""
    from molbuilder.runtime_config import machine_config_path
    return next(s for s in steps if s.path == machine_config_path())


# ══ THE SEEDED FILE MUST NOT BLOCK THE SIGN-IN WIZARD ══════════════════════
#
# Seeding `molbuilder.json` at install time put a file where `auth-setup` had
# always found none.  Its early guard refused any existing file -- so the one
# change meant to make a fresh machine work would have stopped the sign-in
# wizard on every fresh machine.  These two are the pair: the wizard runs, and
# what it is actually guarding still gets guarded.

def _auth_setup(*args):
    from click.testing import CliRunner
    from molbuilder.cli import cli
    return CliRunner().invoke(cli, ["auth-setup", *args])


def test_the_seeded_template_loads_and_names_every_section(fresh):
    """The template must be a LOADABLE file, not a commented-out example.

    The point of seeding every section as an empty stub is that a person fills
    one in rather than inventing the key name.  That only works if the file
    with all the stubs present actually reads -- and one section cannot be a
    stub: an empty `auth` is REFUSED ("must be a non-empty list of provider
    entries"), which is why it is comment-only.
    """
    from molbuilder.runtime_config import read_config
    initconfig.init_config("conda activate", probe=False)
    cfg = read_config(fresh / CONFIG_FILENAME)      # refuses -> raises
    doc = json.loads((fresh / CONFIG_FILENAME).read_text())

    assert "auth" not in doc, (
        "an empty `auth` is refused, so it must be comment-only")
    # Every other live section is present for someone to fill in.
    for section in ("launch", "paths", "tls", "admin", "rate_limit", "envs",
                    "checkpoint"):
        assert section in doc, f"{section} should be a fillable stub"
    # ...and no retired one: each is refused by name (`configuration.md`
    # § 4), and the activation is `env_init`'s.
    for retired in ("execution", "script_generation", "scheduler"):
        assert retired not in doc and retired not in cfg, retired


def test_a_declared_projects_root_is_written_and_reported(fresh):
    """`paths.projects` is written when declared, and the resolver then names
    the config as its source -- which is what every surface prints."""
    from molbuilder.projects import projects_root_with_source
    initconfig.init_config("conda activate", probe=False,
                           projects=Path("/scratch/someone/mb"))

    doc = json.loads((fresh / CONFIG_FILENAME).read_text())
    assert doc["paths"]["projects"] == "/scratch/someone/mb"
    resolved = projects_root_with_source()
    assert resolved.path == Path("/scratch/someone/mb")
    assert "paths.projects" in resolved.source, (
        f"the source must name the config, not just the path: {resolved}")


def test_no_declared_projects_root_leaves_the_default_and_says_so(fresh):
    """The default is the right answer on a workstation -- but the source must
    say it IS the default, so nobody has to believe it."""
    from molbuilder.projects import projects_root_with_source
    initconfig.init_config("conda activate", probe=False)

    doc = json.loads((fresh / CONFIG_FILENAME).read_text())
    assert doc["paths"] == {}, "left empty, not guessed at"
    assert "default" in projects_root_with_source().source


# RETIRED 2026-09-14 (review D): `test_the_environments_readme_says_the_probe
# _runs_on_the_target` grepped a README this program writes for a phrase.  The
# README that earns a test is the one below, whose mock channels are handed to
# the real reader.


def test_the_secrets_readme_mock_channels_are_a_shape_that_parses(fresh):
    """The notify examples must be the real shape, or they teach a wrong one.

    Parsed by `monitor.load_channels` itself -- the same reader the monitor
    uses -- so a drift in the channel format fails here rather than silently
    leaving a README that documents a format nothing accepts.
    """
    import json
    import re
    from molbuilder.monitor import default_notify_path, load_channels

    initconfig.init_config("conda activate", probe=False)
    from molbuilder.config_dir import secrets_dir
    readme = (secrets_dir() / "README").read_text(encoding="utf-8")
    block = re.search(r"(\{\n        \"channels\".*?\n      \})", readme, re.S)
    assert block, "the mock channel block should be in the README"
    doc = json.loads(block.group(1))
    # AT THE FILE'S ONE HOME: `load_channels` took a `path=` until
    # 2026-09-14 and this handed it a temp file.  `fresh` is the config
    # directory, so the documented example is read exactly where a real one
    # would be.
    default_notify_path().write_text(json.dumps(doc), encoding="utf-8")
    got = load_channels()

    assert set(got) == {"local", "team-slack", "team-discord"}, (
        f"the loader did not accept the documented shape: {sorted(got)}")
    # The two shapes differ in WHERE the credential is, which is the point
    # the README makes -- so the examples must actually differ that way.
    assert got["local"].get("key"), "a molbuilder listener signs with a key"
    assert not got["team-slack"].get("key"), (
        "Slack's credential is in the URL -- a key would misteach that")


def test_the_secrets_directory_is_tight_and_explains_itself(fresh):
    """0700, with a README that does not tell a lie.

    THE SECOND HALF OF THIS TEST WAS RETIRED 2026-09-20 rather than repaired.
    It asserted the README carried `../secret_key`, `../google_client_secret`,
    `../notify` and `../notify_keys` -- and said why: *"the `../` form is the
    point: it says the file is one level UP, not here."*  Those four moved
    INTO this directory, so the rule that assertion existed to protect no
    longer exists.  A test kept alive by rewriting its expected strings would
    assert nothing; what it now checks is the rule that replaced it -- the
    README names the files that really are here, and names the FUNCTION each
    is reached through, since the whole point is that nobody builds the path.
    """
    from molbuilder.config_dir import secrets_dir

    initconfig.init_config("conda activate", probe=False)
    d = secrets_dir()

    assert d.is_dir()
    assert oct(d.stat().st_mode)[-3:] == "700", "it holds every credential"
    readme = (d / "README").read_text(encoding="utf-8")
    assert "0600" in readme, "the mode rule is the point"
    # `"secret_key" in readme` passed on the SUBSTRING of `secret_key_file`,
    # which the README carries in an unrelated sentence -- measured
    # 2026-09-20 by deleting the whole `secret_key` row and watching 25 tests
    # stay green.  The rows are what must be there, so match the row.
    for here in ("secret_key ", "google_client_secret ", "notify_keys "):
        assert here in readme, (
            f"the README has no row for {here.strip()}, so a reader opening "
            f"this directory is not told the file lives here")
    assert "../secret_key" not in readme, (
        "the `../` form said the file was one level up; it is not, and a "
        "README that still said so would be the lie this test exists to catch")
    for door in ("config_dir.session_key()", "monitor.default_notify_path()",
                 "monitor.notify_keys_path()", "config_dir.secrets_dir()"):
        assert door in readme, (
            f"the README must name {door} -- a reader who cannot see the "
            f"resolver will build the path by hand, which is the defect")
    # `assert "may be empty" in readme` stood here and was RETIRED with the
    # rest, 2026-09-20.  That reassurance existed because the directory
    # normally WAS empty -- it held only what `molbuilder.json` named, and a
    # workstation with no HTTPS and no sign-in named nothing.  The session key
    # now appears here on first server run, so the sentence would be false and
    # the assertion was protecting it.


def test_environments_is_not_looser_than_its_parent(fresh):
    """It sits inside a 0700 directory holding secrets; 0775 on a child of
    that is the same mistake one level down (it was the umask default until
    2026-09-12)."""
    initconfig.init_config("conda activate", probe=False)
    assert oct((fresh / "environments").stat().st_mode)[-3:] == "700"


def test_the_sign_in_wizard_runs_on_a_seeded_config(fresh):
    """Seed, then wire up sign-in -- and keep both halves."""
    initconfig.init_config("conda activate", probe=False)

    result = _auth_setup("--provider", "asu", "--asurite", "someone")

    assert result.exit_code == 0, result.output
    doc = json.loads((fresh / CONFIG_FILENAME).read_text())
    assert doc["auth"]["providers"], "the wizard wrote its block"
    # THE RULE, not one key's name.  This asserted `_comment_signin`
    # specifically and broke when the seeded template was rewritten
    # 2026-09-12 -- the subject was always "the wizard preserves what the
    # seed wrote", so assert that: every key the seed wrote, the `_`-prefixed
    # guidance and the sections alike, is still there.
    seeded = set(initconfig.seed_document("conda activate"))
    assert {k for k in seeded if k.startswith("_")}, (
        "the seed writes guidance keys")
    assert seeded <= set(doc), (
        f"the wizard dropped seeded keys: {sorted(seeded - set(doc))}")


def test_an_existing_auth_block_is_still_not_replaced_silently(fresh):
    """What the guard is for.  Replacing sign-in config needs --force."""
    initconfig.init_config("conda activate", probe=False)
    assert _auth_setup("--provider", "asu", "--asurite", "someone").exit_code == 0

    again = _auth_setup("--provider", "asu", "--asurite", "other")

    assert again.exit_code == 2
    assert "auth" in again.output
