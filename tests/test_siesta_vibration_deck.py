"""The vibration kind on SIESTA: the force-constant deck, the sorted copy
it is rendered from, and the modes derived from what the run leaves.

science/normal-modes.md § 4a.6 and design § 8: SIESTA nudges the free atoms
(`MD.TypeOfRun FC`, one contiguous range) and writes `.FC`; everything after
that is the one path both engines share.  The held-first sort is what makes
a scattered held set a range (`model/overview.md` § 2.2), and the copy's
order record is what puts the answer back in the user order.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig
from molbuilder.parse.fdf import (deck_atoms, fdf_key, parse_fdf_params,
                                  read_deck, read_fdf)
from molbuilder.siesta.vibration_deck import FC_PART
from molbuilder.structure import Structure
from molbuilder.runfiles import RunNames


def _three_h(held):
    return Structure(
        elements=["H", "H", "H"],
        positions=np.array([[5.0, 5.0, 5.741], [5.0, 5.0, 5.0], [7.0, 5.0, 5.0]]),
        regions={"frozen_atoms": list(held)},
        cell=np.diag([10.0, 10.0, 10.0]),
        axis_kind=("isolated",) * 3)


def _lines(text, key):
    return [ln for ln in text.splitlines()
            if key in ln and not ln.lstrip().startswith("#")]


def test_the_force_constant_deck_nudges_the_free_range_only(tmp_path,
                                                           monkeypatch):
    """The road's CO2 -- O O C, the carbon held, in the middle of nothing
    but last in the input: the sorted copy puts it first, the force-constant
    deck prep writes for the `fc` phase names the free atoms as the trailing
    range, and nothing in that deck relaxes.  Read as the engine reads it,
    its coordinates block inlined from the file beside it (`read_deck`)."""
    from support.road import CO2, described, group_words, jobset, road_tree
    bundle = described(road_tree(tmp_path, monkeypatch, CO2), name="CO2",
                       calculation="vibration")
    r = jobset("prep", "task", *group_words(bundle, "freq"),
               "--bundle", bundle, "--target", "this")
    assert r.exit_code == 0, r.output
    names = RunNames("CO2", "01_freq")
    text = read_deck(bundle / "01_freq" / FC_PART / names.name(".fdf"))
    assert _lines(text, "MD.TypeOfRun") == ["MD.TypeOfRun      FC"]
    assert _lines(text, "FC.First") == ["FC.First          2"]
    assert _lines(text, "FC.Last") == ["FC.Last           3"]
    assert any(ln.startswith("FC.Displacement") and ln.rstrip().endswith("Bohr")
               for ln in _lines(text, "FC.Displacement"))
    assert _lines(text, "position ") == ["position 1"]
    assert not _lines(text, "MD.Steps") and not _lines(text, "MD.MaxForceTol")
    # The start state is the kind's: the density is read, the geometry is
    # declined out loud (an FC run leaves its last displacement in .XV),
    # and there is no optimizer history to speak of.
    assert _lines(text, "DM.UseSaveDM") == ["DM.UseSaveDM      .true."]
    assert _lines(text, "MD.UseSaveXV") == ["MD.UseSaveXV      .false."]
    assert not _lines(text, "MD.UseSaveCG")
    # Every atom's partial charge at every step, by both partitions, as
    # SIESTA reads the deck -- the first occurrence of each key
    # (`engines/vibration.md` § 5.3).
    scalars, _ = read_fdf(text)
    for key in ("Charge.Hirshfeld", "Charge.Voronoi"):
        assert scalars.get(fdf_key(key)) == ["geometry"], key


def test_a_deck_with_its_coordinates_beside_it_reads_as_one_deck(tmp_path,
                                                                monkeypatch):
    """`engines/vibration.md` § 5.11 § 4; `parse/fdf.read_deck`: the
    force-constant deck prep writes reads its coordinates from
    `coordinates.fdf` beside it (libfdf's `%block Label < Filename`), and
    through the deck's one file reader it reads as one deck -- its atoms, their
    places and every setting after the include.  THE SILENT FAILURE: the include
    line read as an empty block that swallows every line after it, so every
    reader of the deck sees no atoms and no later setting.  The text alone
    cannot give the rows, so `read_fdf` refuses it naming the file and the
    reader; a file no longer beside the deck is refused by name."""
    from support.road import CO2, described, group_words, jobset, road_tree
    bundle = described(road_tree(tmp_path, monkeypatch, CO2), name="CO2",
                       calculation="vibration")
    r = jobset("prep", "task", *group_words(bundle, "freq"),
               "--bundle", bundle, "--target", "this")
    assert r.exit_code == 0, r.output
    path = bundle / "01_freq" / FC_PART / RunNames("CO2", "01_freq").name(".fdf")
    raw = path.read_text()
    include = next(ln for ln in raw.splitlines() if "< coordinates.fdf" in ln)
    beside = path.parent / "coordinates.fdf"
    rows = [ln.split() for ln in beside.read_text().splitlines() if ln.strip()]
    whole = read_deck(path)
    assert len(deck_atoms(whole).labels) == len(rows) == 3
    assert [[float(c) for c in row[:3]] for row in rows] == [
        pytest.approx(list(p)) for p in parse_fdf_params(whole).coords_ang]
    # every setting after the include still reads
    after, _ = read_fdf(raw[raw.index(include) + len(include):])
    assert after
    scalars, _ = read_fdf(whole)
    assert all(key in scalars for key in after), after
    with pytest.raises(ValueError, match="coordinates.fdf.*read_deck"):
        read_fdf(raw)
    beside.unlink()
    with pytest.raises(ValueError, match="coordinates.fdf.*not beside"):
        read_deck(path)


def test_each_phase_s_record_lists_what_its_role_reads(tmp_path,
                                                       monkeypatch):
    """A phase is one role of its step's run, and what its record shows is
    what that role reads (`template.reads`, the phase's role;
    `running-a-job.md` row 16): the relaxation's step count in the
    relaxation's, the displacement in the force constants'.  Read as the
    Results tab reads it, through the road -- the run door's `record` at
    prep, from the deck each phase was written with.  What each phase's
    fence lists at launch is `test_effective_parameters_record.py`'s, by
    its role (the phase rows)."""
    from molbuilder.runs import folder_answer
    from support.road import CO2, described, group_words, jobset, road_tree
    bundle = described(road_tree(tmp_path, monkeypatch, CO2), name="CO2",
                       calculation="vibration")
    r = jobset("prep", "task", *group_words(bundle, "freq"),
               "--bundle", bundle, "--target", "this")
    assert r.exit_code == 0, r.output
    for phase, listed, never in (("relax", "relax_steps", "fc_displacement"),
                                 (FC_PART, "fc_displacement", "relax_steps")):
        folder = bundle / "01_freq" / "run-0" / phase
        record = folder_answer(folder)["record"]
        said = {row["item"] for row in record["setup"]["rows"]}
        assert listed in said and never not in said, (phase, sorted(said))


#: (the person's values, the charges' keywords in the fc deck, an optical
#: phase prepared) -- `engines/vibration.md` § 3.1, § 5.3, § 5.11 § 9.
_SWITCH_ROWS = [
    ({}, True, False),                         # the defaults: IR on, Raman off
    ({"compute_ir": False}, False, False),
    ({"compute_raman": True, "raman_modes": "range",
      "raman_freq_min_cm1": 1000.0}, True, True),
]


@pytest.mark.parametrize("values,charges,optical", _SWITCH_ROWS)
def test_compute_ir_and_compute_raman_reach_the_decks(
        tmp_path, monkeypatch, values, charges, optical):
    """The two settings both engines share, translated on SIESTA: *Compute
    IR* writes every atom's charges at every force-constant step, by both
    partitions, and leaves them out -- saying so -- when off; *Compute
    Raman* prepares the optical phase, and none when off.  Read as the
    engine reads the deck.  MUTATIONS THIS MUST FAIL AGAINST: the charges
    written whatever Compute IR says (the run computes a population nobody
    asked for, and its result claims an infrared); the optical phase
    switched by anything but Compute Raman."""
    from support.road import (CO2, described, group_words, jobset,
                              road_tree, set_by_the_person)
    bundle = described(road_tree(tmp_path, monkeypatch, CO2), name="CO2",
                       calculation="vibration")
    set_by_the_person(bundle, values)
    r = jobset("prep", "task", *group_words(bundle, "freq"),
               "--bundle", bundle, "--target", "this")
    assert r.exit_code == 0, r.output
    names = RunNames("CO2", "01_freq")
    text = read_deck(bundle / "01_freq" / FC_PART / names.name(".fdf"))
    scalars, _ = read_fdf(text)
    got = {key: scalars.get(fdf_key(key))
           for key in ("Charge.Hirshfeld", "Charge.Voronoi")}
    assert got == ({k: ["geometry"] for k in got} if charges
                   else {k: None for k in got}), got
    if not charges:
        assert "Compute IR is off" in text
    assert (bundle / "01_freq" / "optical").is_dir() is optical


def test_every_atom_held_is_refused_by_the_gate():
    from molbuilder.validation import validate
    s = _three_h(held=[0, 1, 2])
    issues = validate(s, SiestaConfig(system_label="h3"),
                      calculation="vibration")
    assert any(i.severity == "error" and "every atom is held" in i.message
               for i in issues)


@pytest.mark.parametrize("step", ["plann", "loads", "finish"])
def test_a_kind_step_its_finish_does_not_run_is_refused_at_prep(
        tmp_path, monkeypatch, step):
    """`execution/runs-of-points.md` § 2.1: a kind's steps are the steps its
    finish's module runs (`runwrap.verbs`) -- one it does not run is refused
    at prep, naming the steps there are, never met on the node; ``loads``
    is the wrapper's question before the engine, never a step (it would
    answer 0 and do nothing), and ``finish`` is the analysis by hand, which
    takes a deck and its output alone (as any step it would stop on its
    usage line).  No shipped kind names such a
    step, so the vibration's record is given one here -- the need: only a
    record can reach this refusal."""
    import dataclasses
    from molbuilder.jobset import points
    from support.road import H2, described_and_prepared, road_structure
    shipped = points.record_of
    monkeypatch.setattr(points, "record_of", lambda task: dataclasses.replace(
        shipped(task), steps={**shipped(task).steps, "after": step}))
    _dest, _stage, said = described_and_prepared(
        tmp_path, monkeypatch, road_structure(H2), "siesta",
        calculation="vibration", refused=True)
    assert (f"its kind's steps name {step} -- no step mb_vibration.pyz runs; "
            f"the steps it runs are plan, record, finish-run"
            in said), said


def test_a_run_keeps_its_own_copy_of_what_its_optical_calls_run_from(
        tmp_path, monkeypatch):
    """`engines/vibration.md` § 5.11 § 2, *A run keeps its files* (the user,
    2026-11-21): the optical calls are opened while the run goes, so the
    run's `optical/` container holds its own copy of the stage's
    `optical/` files -- the deck, the run script, the coordinates, both
    bundles and each species' pseudopotential -- made when the run is
    opened (the calls filled from it on the node, and what the data file
    states read from it: the SIESTA vibration e2e's
    `test_the_optical_calls_are_filled_from_the_runs_own_copy` and
    `test_a_prep_while_the_run_goes_never_reaches_its_optical_calls`).  READ
    BACK BY THE RESULTS TAB (`runs.folder_answer`): the container opens
    nothing -- its copy holds no result -- and lists each file by its
    catalogue row.  THE SILENT FAILURE (to 2.77.20): the container was
    empty and each optical call took its deck, run script and bundles from
    the stage folder on the node, so a prep while the run was queued gave
    its later calls another release's files."""
    from pathlib import Path
    from molbuilder.runfiles import MONITOR_BUNDLE, VIBRATION_BUNDLE
    from molbuilder.runs import folder_answer
    from support.road import (CO2, described, group_words, jobset,
                              road_tree, set_by_the_person)
    bundle = described(road_tree(tmp_path, monkeypatch, CO2), name="CO2",
                       calculation="vibration")
    set_by_the_person(bundle, {"compute_raman": True})
    r = jobset("prep", "task", *group_words(bundle, "freq"),
               "--bundle", bundle, "--target", "this")
    assert r.exit_code == 0, r.output
    names = RunNames("CO2", "01_freq")
    stage, held = (bundle / "01_freq" / "optical",
                   bundle / "01_freq" / "run-0" / "optical")
    wanted = {names.name(".fdf"), names.run_script, "coordinates.fdf",
              MONITOR_BUNDLE, VIBRATION_BUNDLE, "C.psml", "O.psml"}
    got = {p.name for p in held.iterdir() if p.is_file()}
    assert wanted <= got, sorted(wanted - got)
    for name in wanted - {"C.psml", "O.psml"}:
        assert (held / name).read_bytes() == (stage / name).read_bytes(), name
    answer = folder_answer(held)
    assert answer["place"]["role"] == "container", answer["place"]
    assert answer["openable"] is None, answer["attempts"]
    roles = {Path(row["path"]).name: row["role"] for row in answer["files"]}
    assert roles[names.name(".fdf")] == ".fdf", roles


def test_the_run_scripts_finish_check_starts_the_bundle_isolated(
        tmp_path, monkeypatch):
    """`configuration.md` § 2.3, *Each bundle starts isolated*: before the
    engine starts, the run script asks the finish whether it loads on the
    job's python -- with ``-I``, so a ``PYTHONPATH`` the person's
    environment carries, here one holding a `numpy` that cannot load ahead
    of the env's own, never reaches it, and the job is not stopped as one
    that cannot finish itself.  Run as the run script runs it: its check,
    under bash, in the force constants' folder.  On 2.77.20 the check ran
    the bundle without it: the decoy took numpy's place and the job said it
    could not finish."""
    import os
    import subprocess
    import sys
    from support.road import CO2, described, group_words, jobset, road_tree
    bundle = described(road_tree(tmp_path, monkeypatch, CO2), name="CO2",
                       calculation="vibration")
    r = jobset("prep", "task", *group_words(bundle, "freq"),
               "--bundle", bundle, "--target", "this")
    assert r.exit_code == 0, r.output
    script = (bundle / "01_freq" / FC_PART
              / RunNames("CO2", "01_freq").run_script)
    text = script.read_text()
    begin = text.index("# --- Can the job finish itself?")
    block = text[begin:text.index("\nfi\n", begin) + len("\nfi\n")]
    decoy = tmp_path / "elsewhere" / "numpy"
    decoy.mkdir(parents=True)
    (decoy / "__init__.py").write_text(
        "raise ImportError('a numpy from elsewhere')\n")
    done = subprocess.run(
        ["bash", "-c", f'_mb_py="{sys.executable}"\n'
                       '_runwrap_log=/dev/null\n'
                       '_log() { echo "$2" >&2; }\n' + block],
        cwd=script.parent, capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": str(decoy.parent)})
    assert done.returncode == 0 and "cannot finish itself" not in (
        done.stdout + done.stderr), done.stdout + done.stderr
