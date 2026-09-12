"""A machine that has never run molbuilder can be made to work by one command.

`execution/running-a-job.md` § 5.2 names the failure these tests are about:

    ``activation`` ... has **no default** -- if it is unset in every scope,
    rendering **any** wrapper refuses ... On a fresh install that is the
    *"the ``.fdf`` saved but no ``.run.sh`` appeared"* symptom, and it bites a
    workstation first.

Nothing created the config directory, so that was the state every new machine
started in.  ``molbuilder envs init-config`` -- run at the end of
``bootstrap`` -- is what ends it.

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
from molbuilder.runtime_config import (ACTIVATION_FORMS, CONFIG_FILENAME,
                                       RuntimeConfigError,
                                       get_script_generation,
                                       require_activation)


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
    with pytest.raises(RuntimeConfigError) as exc:
        require_activation()
    assert "activation" in str(exc.value)


@pytest.mark.parametrize("activation", sorted(ACTIVATION_FORMS))
def test_seeding_removes_the_refusal(fresh, activation):
    """The outcome, asked through the generator's own gate.

    ``require_activation`` is what every wrapper goes through -- it is called
    from ``render_run_wrapper`` -- so a pass here is the wrapper rendering, not
    a file merely existing on disk.
    """
    initconfig.init_config(activation, probe=False)
    assert require_activation() == activation


def test_the_seeded_file_is_read_by_the_real_reader(fresh):
    """A seeded file that the validator refuses is worse than no file.

    ``molbuilder.json`` REFUSES unknown top-level keys, and the guidance this
    file carries is written as ``_``-prefixed comment keys.  This is the test
    that those two facts agree: the read goes through ``get_script_generation``,
    which parses the whole document on the way.
    """
    initconfig.init_config("conda activate", preamble="module load mamba",
                           probe=False)
    sg = get_script_generation(project_dir=None)
    assert sg["activation"] == "conda activate"
    assert "module load mamba" in sg["preamble"]




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
    """A person's own file is never merged into, never reformatted.

    The one thing worth adding -- ``activation`` -- is exactly what they may
    have left for a project-scope ``.molbuilder.json`` to supply (§ 5.1:
    project wins), so an installer forming an opinion about it would be
    overruling a deliberate choice.
    """
    fresh.mkdir(parents=True)
    mine = fresh / CONFIG_FILENAME
    mine.write_text('{"script_generation": {"activation": "source activate"}}')
    original = mine.read_bytes()

    step = initconfig.seed_machine_config("conda activate")

    assert step.action == "kept"
    assert mine.read_bytes() == original
    assert require_activation() == "source activate"


def test_a_kept_config_that_states_no_activation_says_so(fresh):
    """Keeping the file is not the same as staying quiet about it.

    The machine is still in the state the contract calls a refusal, and the
    operator has to learn that from somewhere.
    """
    fresh.mkdir(parents=True)
    (fresh / CONFIG_FILENAME).write_text('{"execution": {}}')

    step = initconfig.seed_machine_config("conda activate")

    assert step.action == "kept"
    assert "UNSET" in step.note


# ══ WHAT IT WRITES, AND WHAT IT REFUSES TO WRITE ═══════════════════════════

def test_no_probe_seeds_the_config_without_a_record(fresh):
    """For a build host or an image baked once and copied."""
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


def test_a_preamble_is_never_written_for_a_hook_that_is_not_there(tmp_path):
    """Checked on disk, not derived and hoped for.

    A preamble naming a file that does not exist fails later, on the cluster,
    inside a job -- which is strictly worse than no preamble at all.
    micromamba ships no ``conda.sh``, so this is a real installation, not a
    hypothetical one.
    """
    assert initconfig.conda_hook(None) is None
    assert initconfig.conda_hook(str(tmp_path / "bin" / "micromamba")) is None

    real = tmp_path / "conda" / "bin" / "conda"
    real.parent.mkdir(parents=True)
    real.touch()
    hook = tmp_path / "conda" / "etc" / "profile.d" / "conda.sh"
    hook.parent.mkdir(parents=True)
    hook.touch()
    assert initconfig.conda_hook(str(real)) == f"source {hook}"


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
                     "--no-probe", "--yes"])

    assert result.exit_code == 0, result.output
    doc = json.loads((fresh / CONFIG_FILENAME).read_text())
    assert doc["script_generation"] == {"activation": "source activate"}


# ══ THE WARNING THAT COULD NOT FIRE ════════════════════════════════════════

def test_a_machine_stating_no_script_generation_is_warned(fresh, monkeypatch):
    """`diagnostics.local_facts` returns the note whenever there is no
    ``script_generation`` -- and NOT only when there is nothing else either.

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
    assert note and "no script_generation" in note


def test_source_activate_is_not_reported_as_a_missing_hook(fresh):
    """The absence of a preamble is correct for ``source activate``.

    It was reported as *"no conda.sh hook found to source"* -- on machines
    that had one and simply did not need it.  A note that states a false
    reason is worse than no note: it sends someone to look for a file that is
    already there.
    """
    step = initconfig.seed_machine_config("source activate")
    assert "no preamble" in step.note
    assert "found" not in step.note


def test_conda_activate_without_a_hook_says_what_will_break(fresh):
    """The same absence IS worth flagging for the other form."""
    step = initconfig.seed_machine_config("conda activate")
    assert "needs conda's hook" in step.note


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
    for section in ("execution", "script_generation", "scheduler", "paths",
                    "tls", "admin", "rate_limit", "envs", "checkpoint"):
        assert section in doc, f"{section} should be a fillable stub"
    # And the one value with no default is filled.
    assert cfg["script_generation"]["activation"] == "conda activate"


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


def test_the_secrets_directory_is_tight_and_explains_itself(fresh):
    """0700, with a README that does not tell a lie.

    Two secrets have ONE fixed home each directly in the config dir and the
    config deliberately cannot name them, so a README pointing a person at
    `secrets/` for those would be actively wrong.
    """
    initconfig.init_config("conda activate", probe=False)
    d = fresh / "secrets"

    assert d.is_dir()
    assert oct(d.stat().st_mode)[-3:] == "700", "it sits beside the session key"
    readme = (d / "README").read_text(encoding="utf-8")
    assert "0600" in readme, "the mode rule is the point"
    for fixed in ("secret_key", "google_client_secret"):
        assert fixed in readme, f"{fixed} cannot live here and must say so"
    assert "may be empty" in readme, (
        "an empty secrets/ is normal and should not read as half-done")


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
    assert require_activation() == "conda activate", "and did not eat ours"
    # THE RULE, not one key's name.  This asserted `_comment_signin`
    # specifically and broke when the seeded template was rewritten
    # 2026-09-12 -- the subject was always "the wizard preserves the guidance",
    # so assert that: every `_`-prefixed key the seed wrote is still there.
    seeded_comments = {k for k in initconfig.seed_document("conda activate")
                       if k.startswith("_")}
    assert seeded_comments, "the seed writes guidance keys"
    assert seeded_comments <= set(doc), (
        f"the wizard dropped guidance keys: "
        f"{sorted(seeded_comments - set(doc))}")


def test_an_existing_auth_block_is_still_not_replaced_silently(fresh):
    """What the guard is for.  Replacing sign-in config needs --force."""
    initconfig.init_config("conda activate", probe=False)
    assert _auth_setup("--provider", "asu", "--asurite", "someone").exit_code == 0

    again = _auth_setup("--provider", "asu", "--asurite", "other")

    assert again.exit_code == 2
    assert "auth" in again.output
