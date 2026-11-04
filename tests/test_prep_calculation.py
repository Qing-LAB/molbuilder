"""`prep`, entire — what only a described calculation that `molbuilder describe` did
not make can show.

Contract: ``docs/execution/project-layout.md`` § 2.3.1 (the five steps) ·
``docs/execution/job-contracts.md`` § 2.5a (the pseudopotentials) ·
`job-system.md` § 5.0 (the entry).

Every prep here goes through the one entry, `prep.prep_task` -- what
`jobset prep` and the Task setup tab call.  The rules a road row holds:

* the deck, its run script, the machine's copy, the plan -- every file the
  Task setup card names for a stage, `tests/data/the_catalogue.toml`;
* the stage named is the stage written, a template value no stage varies,
  the retry budget, the reporting policy, a stage the ladder does not hold
  -- `tests/data/prep_protocol.toml`;
* the deck agreeing with its launch -- every launch row that sends; the
  allocation reaching the header -- `tests/data/launch_values.toml`;
* a folder that is not a calculation -- `test_jobset.py`'s
  `test_cli_prep_is_described_only`.

What stays is API-level, and each says why.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder import describe as D
from molbuilder.config.siesta import SiestaConfig
from molbuilder.jobset.errors import PrepError
from molbuilder.jobset.model import Resources
from conftest import prep_the_group_of
from molbuilder.jobset.ladder import engine_ladder
from molbuilder.structure import Structure

from conftest import write_pseudos as _pseudos_for

#: BDT's first atoms -- three species, which the pseudopotential cases need:
#: a library that lacks one of them is the case.
_BDT = Structure(elements=["S", "C", "C", "H"],
                 positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.78],
                                     [1.21, 0.0, 2.48], [2.15, 0.0, 1.94]]),
                 vacuum=(10.0, 10.0, 10.0))


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path_factory):
    """The rest of the sandbox (B-9, 2026-08-13): the file ran with cwd =
    repo root, so the repo's molbuilder.json kept folding into every wrapper
    under test."""
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


def _prep(calc, stage="coarse"):
    """`prep task <stage>`, through the one entry, the launch shape stated
    on the command line."""
    return prep_the_group_of(calc, stage,
                             allocation=Resources(mpi_np=8, cpus_per_task=1))


@pytest.fixture
def calc(tmp_path):
    """A described calculation whose pseudopotentials are in its own folder
    and whose description names no library -- which `molbuilder describe` does not
    make: it names one.

    How a shell enters an environment is read off the MACHINE RECORD --
    conftest's probed record carries it, so the calculation needs no config
    of its own.
    """
    src = tmp_path / "bdt.xyz"
    src.write_text(_BDT.to_xyz())
    dest = tmp_path / "calc"
    desc = D.build_description(
        _BDT, SiestaConfig(system_label="calc", mesh_cutoff=300.0),
        engine_ladder("siesta", "publishable"),
        engine="siesta",
        calculation="optimization",
        name="calc", source=str(src))
    D.write_description(desc, dest)
    _pseudos_for(dest, ["S", "C", "H"])
    return dest


# --------------------------------------------------------------------- #
#  The pseudopotentials checked are the ones the run will open           #
# --------------------------------------------------------------------- #

def test_pseudopotentials_beside_the_calculation_need_no_library(calc):
    """A calculation whose `.psml` files are in its own folder is not told
    `psml_lib` is unset.

    GOAL: prep used the folder's files -- the folder wins -- while the report
    beside the deck said "cfg.psml_lib is not set ... SIESTA will refuse to
    start": the settings gate was never told the folder (2026-09-25).
    CONTRACT: `job-contracts.md` § 2.5a (pseudopotentials already beside the
    calculation are used without this field); `pseudos.psml_sources` (the
    folder first, then the library -- one rule for prep and the gate).

    API-LEVEL: a description naming no library, its files put beside it --
    `molbuilder describe` names one, so the road does not make this calculation.
    """
    _prep(calc)
    report = (calc / "01_coarse" / "calc_01_coarse.validation.txt").read_text()
    assert "[config.psml_lib" not in report, report


def test_the_folder_wins_over_a_library_that_lacks_a_species(
        isolated_projects_root):
    """A library that lacks a species the folder has does not stop prep.

    GOAL: prep's provider takes the folder's files and never asks the library
    for them, while the gate read the library alone -- so a calculation with
    every file beside it was refused over a library it did not need
    (2026-09-25).  CONTRACT: `pseudos.psml_sources` -- the folder first, then
    the library for what the folder lacks.

    API-LEVEL, for the reason above: the road's calculation is H2, one
    species, so no library can lack one the folder has.
    """
    tree = isolated_projects_root
    lib = tree / "pseudopotential"
    lib.mkdir()
    _pseudos_for(lib, ["S"])                       # the library: sulfur only
    src = tree / "bdt.xyz"
    src.write_text(_BDT.to_xyz())
    dest = tree / "P" / "optimization" / "calc"
    D.write_description(D.build_description(
        _BDT, SiestaConfig(system_label="calc", mesh_cutoff=300.0,
                           psml_lib="pseudopotential"),
        engine_ladder("siesta", "publishable"),
        engine="siesta",
        calculation="optimization",
        name="calc", source=str(src)),
        dest)
    _pseudos_for(dest, ["S", "C", "H"])            # the folder: all three
    _prep(dest)
    report = (dest / "01_coarse" / "calc_01_coarse.validation.txt").read_text()
    assert "[config.psml_lib" not in report, report


# --------------------------------------------------------------------- #
#  Refusals                                                              #
# --------------------------------------------------------------------- #

def test_a_structure_that_changed_since_describing_is_refused(calc):
    """§ 6.3's witness earning its place: the description records a formula and
    an atom count, so building a *different* calculation under the same id is
    caught rather than discovered in the results.

    The mutated file is the CALCULATION'S OWN copy — `describe` copies the
    structure in, and that copy is what `prep` reads: the one `task.json`
    records.

    API-LEVEL: a refusal the road cannot reach -- nothing molbuilder does
    writes another structure over the calculation's copy; a hand does."""
    from molbuilder.task import of as task_of
    own = task_of(calc).structure.source
    (calc / own).write_text(
        Structure(elements=["H", "H"],
                  positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
                  vacuum=(10.0, 10.0, 10.0)).to_xyz())
    with pytest.raises(PrepError, match=r"structure has changed"):
        _prep(calc)


# --------------------------------------------------------------------- #
#  The pipeline log's resource line                                      #
# --------------------------------------------------------------------- #

def test_the_decision_log_prints_channel_names_readably():
    """The pipeline log's whole value is that a person can read it, which is what
    `_flat_resources`' own docstring says.

    Every field it renders was a scalar until `notify_channels` (2026-08-31),
    so the default `f"{v}"` put a Python repr into the file -- and a bare
    `()` for the answer that matters most, *send this calculation nowhere*.
    """
    from molbuilder.jobset.prep import _flat_resources

    assert "notify_channels=slack,lab" in _flat_resources(
        Resources(mpi_np=4, notify_channels=("slack", "lab")))
    assert "notify_channels=(none)" in _flat_resources(
        Resources(mpi_np=4, notify_channels=()))
    # absent stays absent: `None` means "not asked for" and is left out
    # rather than printed as a null.
    assert "notify_channels" not in _flat_resources(Resources(mpi_np=4))


#: Where a key a release retired can lie, each with how the molbuilder
#: before it wrote it there: ``(file, how to plant it, the rule the old
#: file is kept under, what migrate says it did, whether the plan's groups
#: are prepared again)`` -- a run setting 2.60.0 retired (2.59 wrote it),
#: and a job's `asked` and `written_at`, which 2.41.0 and 2.42.0 retired
#: and 2.65.7's reader refuses (2.40 wrote a stage's prep with no flag on a
#: machine with no queue as ``{"allocation": {}, "env": null,
#: "emit_sbatch": false}``, and a stage with no relaxation run null).
_RETIRED_WHERE = {
    "a job's resources": ("job-set.json", lambda d: d["jobs"][0]["resources"]
                          .__setitem__("continue_retries", 1), "2.60.0",
                          "continue_retries removed", True),
    "the calculation's run card": ("job-request.json", lambda d: d.setdefault(
        "execution", {}).__setitem__("continue_retries", 2), "2.60.0",
        "continue_retries removed", False),
    "a stage's run card": ("job-request.json", lambda d: d["stages"]["coarse"]
                           .setdefault("execution", {})
                           .__setitem__("continue_retries", 2), "2.60.0",
                           "continue_retries removed", False),
    "a job's asked": ("job-set.json", lambda d: d["jobs"][0]
                      .__setitem__("asked", {"allocation": {}, "env": None,
                                             "emit_sbatch": False}),
                      "2.65.7", "`asked` removed", False),
    "a job's written_at": ("job-set.json", lambda d: d["jobs"][0]
                           .__setitem__("written_at", None), "2.65.7",
                           "`written_at` removed", False),
}


@pytest.mark.parametrize("where", list(_RETIRED_WHERE))
def test_a_retired_run_setting_comes_out_where_it_lies(calc, where):
    """A key a release retired -- the run script's warm retry,
    `continue_retries` (`template.RETIRED_ITEMS`, 2.60.0), a job's `asked`
    and `written_at` (`jobset.model.JOB_RETIRED_KEYS`) -- is refused where
    it lies, naming `jobset migrate`, which takes it out there and keeps the
    rest of the file as it was, the old file beside the new
    (`execution/job-request.md` § 8; `job-system.md`, the job set's
    reader); a plan that held a retired run setting was prepared by an
    older molbuilder, so its groups' prep is named.  THE SILENT FAILURE:
    the key left in one record that every later read refuses, or read past
    unsaid; or a stage prepared for the old run script launched, and
    refused on the node after its queue wait."""
    import json
    from molbuilder.jobset.migrate import migrate_state
    _prep(calc)
    name, plant, rule, removed, again_named = _RETIRED_WHERE[where]
    path = calc / name
    clean = json.loads(path.read_text())
    held = json.loads(path.read_text())
    plant(held)
    path.write_text(json.dumps(held, indent=2))
    with pytest.raises(Exception, match="jobset migrate"):
        _prep(calc)
    said = migrate_state(calc)
    assert json.loads(path.read_text()) == clean, said
    kept = calc / f"{name}.pre-{rule}"
    assert json.loads(kept.read_text()) == held
    assert any(removed in s for s in said), said
    again = [s for s in said if "prepare each group again" in s]
    if again_named:
        assert again and "prep task --group 1" in again[0], said
    else:
        assert not again, said
    _prep(calc)


def test_what_a_stage_was_prepared_with_leaves_run_settings_to_its_card(
        calc):
    """What prep records a stage was prepared with (`prepared-with.json`,
    read through its door) holds the template's settings as resolved for
    the stage; a run setting is its run card's (`template.run_settings`),
    recorded there alone (`job-system.md`, *Changed since prepared*).
    THE SILENT FAILURE: a run setting recorded at its default among the
    settings -- a newer molbuilder's default then reads as the person's
    change to what the stage computes, and a launched group is told to
    start from the beginning."""
    from molbuilder.jobset.prepared_with import read
    from molbuilder.template import run_settings
    _prep(calc)
    (settings,) = [v["settings"] for k, v in read(calc).items()
                   if k == "coarse"]
    assert "mesh_cutoff" in settings, sorted(settings)
    assert not set(settings) & run_settings("siesta"), sorted(
        set(settings) & run_settings("siesta"))


def test_a_calculation_this_molbuilder_prepared_is_current_to_migrate(calc):
    """`jobset migrate` on a calculation this molbuilder described and
    prepared says it is current and rewrites nothing (`migrate.CURRENT`).
    The silent failure: a key a release retired still listed among those an
    older plan lacks, so every current plan is "given" it, rewritten and
    kept beside a copy, on every run."""
    from molbuilder.jobset.migrate import migrate_state
    _prep(calc)
    before = sorted(p.relative_to(calc) for p in calc.rglob("*") if p.is_file())
    # NOTHING SAID IS NOTHING MIGRATED: the verb prints `migrate.CURRENT`
    # for an empty answer (`_cli.migrate_cmd`).
    said = migrate_state(calc)
    assert said == [], said
    after = sorted(p.relative_to(calc) for p in calc.rglob("*") if p.is_file())
    assert after == before, set(after) ^ set(before)


#: A RUN AN OLDER MOLBUILDER NAMED, AS IT LEFT IT -- what each row adds to
#: a run whose env gave it no python, its marker written by its main line
#: (as 2.59's was) and its two files under 2.59's names: ``(what the
#: migration does, its reason)``.  A second start is 2.59's run script
#: started again by hand in the same folder, ``--run 1`` as its refusal
#: asked, failing alike: its marker and its own stamped session log.  A
#: calculation reached through a link is the same run, its folder named
#: by the link -- a projects folder linked into a home.
_OLDER_RUNS = {
    "nothing more": ("renamed", None),
    "a second start": ("left", "started more than once"),
    "reached through a link": ("renamed", None),
}


@pytest.mark.parametrize("how", list(_OLDER_RUNS))
def test_a_run_an_older_molbuilder_named_is_named_once(tmp_path,
                                                        monkeypatch, how):
    """A run an older molbuilder named -- its files by the try, ``-run0``,
    its session log stamped with the clock -- reads `unreadable`, naming
    `jobset migrate`; the migration names a run that started once anew,
    every byte kept and each old name said, and the run reads as its files
    say again (`project-layout.md` § 1.6.1).  A run that started more than
    once is left as it is, said with its reason, and still not read.  THE
    SILENT FAILURE: an older run read *queued* for ever; a second start's
    files folded into one run.  The road: H2's coarse run launched here,
    its env giving it no python -- a marker 2.59's main line wrote alike --
    its files then named as 2.59 named them."""
    from molbuilder.jobset.migrate import migrate_state
    from molbuilder.runs import concluded
    from support.road import jobset, prepared_here
    bundle, run, names = prepared_here(tmp_path, monkeypatch, "true")
    r = jobset("launch", "task", "--stage", "coarse", "--bundle", bundle,
               "--mode", "direct", "--foreground", "--yes")
    assert r.exit_code == 1, r.output
    marker, log = run / names.name(".concluded"), run / names.name(
        ".runwrap.log")
    was = {p.name: p.read_bytes() for p in (marker, log)}
    stem = names.stem
    older = {f"{stem}-run0.concluded": was[marker.name],
             f"{stem}.runwrap-20261031-120000.log": was[log.name]}
    marker.rename(run / f"{stem}-run0.concluded")
    log.rename(run / f"{stem}.runwrap-20261031-120000.log")
    if how == "a second start":
        older[f"{stem}-run1.concluded"] = was[marker.name]
        older[f"{stem}.runwrap-20261031-120500.log"] = was[log.name]
    for name, body in older.items():
        (run / name).write_bytes(body)
    st = concluded(run, names, base=bundle)
    assert st.state == "unreadable" and "jobset migrate" in st.detail, st
    named = bundle
    if how == "reached through a link":
        named = tmp_path / "linked"
        named.symlink_to(bundle, target_is_directory=True)
    said = migrate_state(named)
    does, why = _OLDER_RUNS[how]
    if does == "left":
        assert any(s.startswith("01_coarse/run-0: left as it is")
                   and why in s for s in said), said
        assert all((run / n).read_bytes() == b for n, b in older.items())
        assert concluded(run, names, base=bundle).state == "unreadable"
        return
    assert any(f"{stem}-run0.concluded -> {marker.name}" in s
               for s in said), said
    assert marker.read_bytes() == was[marker.name]
    assert log.read_bytes() == was[log.name]
    assert not any((run / n).exists() for n in older)
    st = concluded(run, names, base=bundle)
    assert st.state == "failed" and "no env python" in st.detail, st


def test_an_older_machine_copy_keeps_how_a_shell_enters_its_environment(
        calc):
    """A calculation's copy of its machine's record written before 2.10.1,
    which named how a shell there enters its environment
    ``script_generation`` (before 2026-10-02): the migration writes it as
    ``env_init``, its values kept, the old file beside the new
    (`configuration.md` § 4).  THE SILENT FAILURE: the activation
    dropped, and every later prep refused for a record carrying none."""
    import json
    from molbuilder.jobset.migrate import migrate_state
    from molbuilder.scheduler.record import calculation_environment
    _prep(calc)
    path = calc / "environment.json"
    rec = json.loads(path.read_text())
    entered = rec.pop("env_init")
    rec.update(schema="molbuilder/environment@2", scheduler="workstation",
               script_generation=entered)
    path.write_text(json.dumps(rec, indent=2))
    said = migrate_state(calc)
    assert any("script_generation -> env_init" in s for s in said), said
    assert calculation_environment(calc).env_init == entered
