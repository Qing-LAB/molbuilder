"""`prep`, entire -- what a calculation's pseudopotentials beside it, a record
prep wrote and the migration of a calculation prep prepared can show.

Contract: ``docs/execution/project-layout.md`` § 2.3.1 (the five steps) ·
``docs/execution/job-contracts.md`` § 2.5a (the pseudopotentials) ·
`job-system.md` § 5.0 (the entry).

Every calculation here is `molbuilder describe`'s and every prep `jobset
prep`'s (`support.road`).  The rules a road row holds:

* the deck, its run script, the machine's copy, the plan -- every file the
  Task setup card names for a stage, `tests/data/the_catalogue.toml`;
* the stage named is the stage written, a template value no stage varies,
  the retry budget, the reporting policy, a stage the ladder does not hold,
  a structure changed since it was described -- `tests/data/prep_protocol.toml`;
* the deck agreeing with its launch -- every launch row that sends; the
  allocation reaching the header -- `tests/data/launch_values.toml`;
* a folder that is not a calculation -- `test_jobset.py`'s
  `test_a_folder_that_is_no_calculation_is_refused_saying_so`.
"""
from __future__ import annotations

from conftest import write_pseudos
from support.road import (CO2, H2, described, group_words, jobset, road_tree,
                          stage_file)


def _prep(calc, stage="coarse"):
    """`jobset prep task` of the group ``stage`` is in, on this machine."""
    r = jobset("prep", "task", *group_words(calc, stage), "--bundle", calc,
               "--target", "this")
    assert r.exit_code == 0, r.output


def _with_its_files_beside_it(tmp_path, monkeypatch, *, structure=H2,
                              library=None):
    """`molbuilder describe` into a folder that already holds its
    pseudopotentials, as a person leaves them there first (`job-system.md`
    § 5.1: covered by *that folder or `--psml-lib`*) -- naming ``library``, the
    one holding only what it holds, or none.  The tree is the road's, the
    calculation the one maker's, one rank."""
    tree = road_tree(tmp_path, monkeypatch, structure)
    folder = tree / "P" / "optimization" / "calc"
    folder.mkdir(parents=True)
    write_pseudos(folder, sorted(set(structure["elements"])))
    return described(tree, "P/optimization/calc", name="calc",
                     psml_lib=library,
                     execution={"mpi_np": 1, "omp_threads": 1})


# --------------------------------------------------------------------- #
#  The pseudopotentials checked are the ones the run will open           #
# --------------------------------------------------------------------- #

def test_pseudopotentials_beside_the_calculation_need_no_library(tmp_path,
                                                                 monkeypatch):
    """A calculation whose `.psml` files are in its own folder is not told
    `psml_lib` is unset.

    GOAL: prep used the folder's files -- the folder wins -- while the report
    beside the deck said "cfg.psml_lib is not set ... SIESTA will refuse to
    start": the settings gate was never told the folder (2026-09-25).
    CONTRACT: `job-contracts.md` § 2.5a (pseudopotentials already beside the
    calculation are used without this field); `pseudos.psml_sources` (the
    folder first, then the library -- one rule for prep and the gate).
    """
    calc = _with_its_files_beside_it(tmp_path, monkeypatch)
    _prep(calc)
    report = stage_file(calc, "coarse", ".validation.txt").read_text()
    assert "[config.psml_lib" not in report, report


def test_the_folder_wins_over_a_library_that_lacks_a_species(tmp_path,
                                                             monkeypatch):
    """A library that lacks a species the folder has does not stop prep.

    GOAL: prep's provider takes the folder's files and never asks the library
    for them, while the gate read the library alone -- so a calculation with
    every file beside it was refused over a library it did not need
    (2026-09-25).  CONTRACT: `pseudos.psml_sources` -- the folder first, then
    the library for what the folder lacks.
    """
    lacking = tmp_path / "projects" / "oxygen-only"
    lacking.mkdir(parents=True)
    write_pseudos(lacking, ["O"])                  # the library: oxygen only
    calc = _with_its_files_beside_it(tmp_path, monkeypatch, structure=CO2,
                                     library="oxygen-only")
    _prep(calc)
    report = stage_file(calc, "coarse", ".validation.txt").read_text()
    assert "[config.psml_lib" not in report, report


# --------------------------------------------------------------------- #
#  What prep records, and what migrate then finds                        #
# --------------------------------------------------------------------- #

def test_what_a_stage_was_prepared_with_leaves_run_settings_to_its_card(
        tmp_path, monkeypatch):
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
    calc = described(road_tree(tmp_path, monkeypatch))
    _prep(calc)
    (settings,) = [v["settings"] for k, v in read(calc).items()
                   if k == "coarse"]
    assert "mesh_cutoff" in settings, sorted(settings)
    assert not set(settings) & run_settings("siesta"), sorted(
        set(settings) & run_settings("siesta"))


def test_an_entry_an_older_molbuilder_wrote_reads_as_not_recorded(
        tmp_path, monkeypatch):
    """A stage's entry in `prepared-with.json` that lacks a value this
    molbuilder records -- its points' folders, as one written before
    them -- is compared with nothing and said in the one spelling
    (`job-system.md`, *Changed since prepared*); an entry that is not an
    object is a record that does not read, said naming the file.
    PREVENTS: an older record's words compared as today's, naming a
    change nobody made, or a hand-broken entry read as quietly
    unrecorded."""
    import json
    import pytest
    from molbuilder.calculation import open as open_calculation
    from molbuilder.jobset.prepared_with import (UNRECORDED_SAID, changes,
                                                 path_of)
    calc = described(road_tree(tmp_path, monkeypatch))
    _prep(calc)
    record = path_of(calc)
    held = json.loads(record.read_text())
    del held["stages"]["coarse"]["point_folders"]
    record.write_text(json.dumps(held))
    assert changes(open_calculation(calc), "coarse") == (False, [])
    r = jobset("status", "--bundle", calc)
    assert UNRECORDED_SAID in r.output, r.output
    held["stages"]["coarse"] = "x"
    record.write_text(json.dumps(held))
    with pytest.raises(ValueError, match="prepared-with.json"):
        changes(open_calculation(calc), "coarse")


def test_status_names_the_warm_files_a_run_left(tmp_path, monkeypatch):
    """`status <stage>`'s *warm files* row names each file of the
    calculation's kind's warm list in effect that the stage's run left --
    what a next stage could take (`job-contracts.md` § 4.2a: the list in
    effect, asked with the calculation's folder) -- and no file of another
    kind's list, and no person's file.
    PREVENTS: the row read from every kind's list, naming a file this kind
    never carries; or from the stage's own job, empty for a stage that
    starts clean though its run left what the next one takes."""
    from molbuilder.runfiles import compose
    from molbuilder.task import of as task_of
    from molbuilder.warmfiles import warm_list
    calc = described(road_tree(tmp_path, monkeypatch))
    _prep(calc)
    run = next(p for p in calc.rglob("run-0") if "coarse" in str(p))
    label = task_of(calc).label
    own = warm_list("siesta", "optimization", calc).carry
    other = [s for s in warm_list("siesta", "transport", calc).carry
             if s not in own]
    left = compose(label, own[0])
    (run / left).write_text("restart\n")
    (run / compose(label, other[0])).write_text("another kind's\n")
    (run / "notes.txt").write_text("the person's\n")
    r = jobset("status", "coarse", "--bundle", calc)
    assert r.exit_code == 0, r.output
    (row,) = [ln for ln in r.output.splitlines()
              if ln.strip().startswith("warm files")]
    assert left in row, row
    assert compose(label, other[0]) not in row and "notes.txt" not in row, row


def test_a_calculation_this_molbuilder_prepared_is_current_to_migrate(
        tmp_path, monkeypatch):
    """`jobset migrate` on a calculation this molbuilder described and
    prepared says it is current and rewrites nothing (`migrate.CURRENT`).
    The silent failure: a key a release retired still listed among those an
    older plan lacks, so every current plan is "given" it, rewritten and
    kept beside a copy, on every run."""
    from molbuilder.jobset.migrate import CURRENT
    calc = described(road_tree(tmp_path, monkeypatch))
    _prep(calc)
    before = sorted(p.relative_to(calc) for p in calc.rglob("*") if p.is_file())
    r = jobset("migrate", "--bundle", calc)
    assert r.exit_code == 0, r.output
    assert CURRENT in r.output, r.output
    after = sorted(p.relative_to(calc) for p in calc.rglob("*") if p.is_file())
    assert after == before, set(after) ^ set(before)


def test_a_template_that_does_not_read_is_refused_by_migrate(tmp_path,
                                                             monkeypatch):
    """`job-request.md` § 8: a template that does not read is refused by name
    and nothing is written.  THE SILENT FAILURE: a Python traceback, its file
    unchanged and no way on said, as a retired word left by hand in a template
    of today's shape ended in."""
    from molbuilder.calculation import open as open_calculation
    calc = described(road_tree(tmp_path, monkeypatch))
    template = open_calculation(calc).template.path
    text = template.read_text()
    item = "[item.spin_treatment]\n"
    assert item in text
    edited = text.replace(item, item + 'value = "polarized"\n')
    template.write_text(edited)
    r = jobset("migrate", "--bundle", calc)
    assert r.exit_code != 0, r.output
    assert "does not read, so nothing was" in r.output, r.output
    assert template.read_text() == edited


def test_a_pyscf_vibration_stating_the_old_push_is_migrated_to_the_probes_nodes(
        tmp_path, monkeypatch):
    """`engines/vibration.md` § 6.11 § 9 item 1, `engines/template.md` § 3:
    a PySCF vibration's template stating the probe's fixed push,
    `displacement_amplitude_ang` -- as every one 2.88.0 wrote does, here
    typed into today's described template -- is refused by prep naming
    `jobset migrate`; the migration takes it out at its value (0.1 Å, no
    value of it running as before), writes the probe's temperature from the
    thermochemistry's (the person's 300 K) and three nodes, says each, keeps
    the file as it was as the release's, and prep then runs.  THE SILENT
    FAILURE: a described probe taking a temperature nobody stated -- the
    class's -- or a template no prep can read and no way on said."""
    from molbuilder.calculation import open as open_calculation
    from molbuilder.template import one, read_template
    from support.road import set_by_the_person
    calc = described(road_tree(tmp_path, monkeypatch, engine="pyscf"),
                     calculation="vibration", engine="pyscf")
    set_by_the_person(calc, {"temperature_K": 300.0})
    template = open_calculation(calc).template.path
    # THE ITEM AS 2.88.0 WROTE IT, its keys the reader requires
    # (`engines/template.md` § 5), set by the person.
    older = template.read_text() + (
        '\n[item.displacement_amplitude_ang]\nkind = "deck"\n'
        'category = ["accuracy"]\n'
        'engine_key = "(molbuilder: finite-difference step amplitude)"\n'
        'expands = ["mode displacement step"]\ntype = "float"\nvalue = 0.1\n'
        'source = "person"\ndefault = 0.02\nunit = "Å"\n'
        'label = "Displacement amplitude"\n'
        'help = "How far atoms are pushed along each mode."\n')
    template.write_text(older)
    r = jobset("prep", "task", *group_words(calc, "freq"), "--bundle", calc,
               "--target", "this")
    assert r.exit_code != 0, r.output
    for words in ("displacement_amplitude_ang", "jobset migrate"):
        assert words in r.output, (words, r.output)
    r = jobset("migrate", "--bundle", calc)
    assert r.exit_code == 0, r.output
    for words in ("displacement_amplitude_ang = 0.1 removed",
                  "probe_temperature_K = 300", "probe_nodes = 3"):
        assert words in r.output, (words, r.output)
    read = read_template(template.read_text())
    assert one(read, "probe_temperature_K").value == 300.0
    assert one(read, "probe_nodes").value == 3
    assert template.with_name(template.name + ".pre-2.89.0").read_text() \
        == older
    _prep(calc, "freq")


def test_a_prep_takes_away_a_bundle_no_deck_of_its_carries(tmp_path,
                                                            monkeypatch):
    """`job-system.md` § 5.0, checkpoint 6: every file prep lays beside a
    deck is prep's, and one an earlier prep wrote that this prep does not is
    taken away -- the bundle of a finish the job no longer has or of
    another engine among them (every bundle, `runwrap.BUNDLES`), never left
    to be sent, run or read as this prep's.  The suite cannot prepare one
    stage under two engines, so the bundles an earlier prep left are laid
    beside the deck here -- the need: only such a folder shows the sweep.
    The bundle this deck carries stays.  THE SILENT FAILURE (to 2.77.20):
    the sweep asked only this engine's and this finish's names, so another's
    bundle stayed beside the deck as if this prep had written it."""
    from molbuilder.runfiles import MONITOR_BUNDLE, PYSCF_BUNDLE, VIBRATION_BUNDLE
    calc = described(road_tree(tmp_path, monkeypatch))
    _prep(calc)
    stage = next(p for p in calc.iterdir() if p.name.endswith("_coarse"))
    for name in (VIBRATION_BUNDLE, PYSCF_BUNDLE):
        (stage / name).write_bytes(b"an earlier prep's bundle")
    _prep(calc)
    assert not (stage / VIBRATION_BUNDLE).exists()
    assert not (stage / PYSCF_BUNDLE).exists()
    assert (stage / MONITOR_BUNDLE).is_file()
