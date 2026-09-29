"""The vibration kind on SIESTA: the force-constant deck, the sorted copy
it is rendered from, and the modes derived from what the run leaves.

science/normal-modes.md § 4a.6 and design § 8: SIESTA nudges the free atoms
(`MD.TypeOfRun FC`, one contiguous range) and writes `.FC`; everything after
that is the one path both engines share.  The held-first sort is what makes
a scattered held set a range (`model/overview.md` § 2.2), and the recorded
permutation is what puts the answer back in the input order.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig
from molbuilder.projects import PROJECTS_ROOT_ENV
from molbuilder.script_emit import render_deck
from molbuilder.siesta.input import spec_for
from molbuilder.structure import Structure
from molbuilder.atom_permutation import read_permutation
from molbuilder.transport.sort import sort_by, write_permutation

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def in_tree_psml(monkeypatch):
    """The fixture pseudopotential library, made 'inside the projects tree'
    the way the validator requires."""
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(FIXTURES))
    return "psml"


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


def test_the_force_constant_deck_nudges_the_free_range_only(in_tree_psml):
    """Held atom in the middle of the input; the sorted copy puts it first,
    the deck names the free atoms as the trailing range, and nothing in the
    deck relaxes."""
    s = _three_h(held=[1])
    sorted_s = sort_by(s, "held-first").structure
    cfg = SiestaConfig(system_label="h3", psml_lib=in_tree_psml,
                       already_relaxed=True)
    text = render_deck(spec_for(sorted_s, cfg, calculation="vibration",
                                stage_token="01_fc"), sorted_s, cfg,
                       verbose=False)
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


def test_an_unsorted_copy_is_refused_by_name(in_tree_psml):
    """A deck written from the input order would nudge the wrong atoms;
    the writer refuses rather than guessing a range."""
    s = _three_h(held=[1])
    cfg = SiestaConfig(system_label="h3", psml_lib=in_tree_psml,
                       already_relaxed=True)
    with pytest.raises(ValueError, match="held-first"):
        render_deck(spec_for(s, cfg, calculation="vibration",
                             stage_token="01_fc"), s, cfg, verbose=False)


def test_every_atom_held_is_refused_by_the_gate():
    from molbuilder.validation import validate
    s = _three_h(held=[0, 1, 2])
    issues = validate(s, SiestaConfig(system_label="h3"),
                      calculation="vibration")
    assert any(i.severity == "error" and "every atom is held" in i.message
               for i in issues)


def test_the_modes_come_back_in_the_input_order(tmp_path):
    """The analysis's return leg, at its own API (`spectra.vibrational_analysis`,
    `engines/vibration.md` § 5.5), on the measured unrelaxed `.FC`: held atom
    LAST in the input, so the sort really reorders; the block, masses and
    geometry are the sorted copy's, and every row comes back in the input's
    numbering through the recorded permutation.  A measured fixture, not the
    road -- the road's own run is the e2e test."""
    from molbuilder.chemistry import atomic_mass
    from molbuilder.parse.engines.siesta_fc import hessian_from_fc, read_fc
    from molbuilder.spectra.vibrational_analysis import vibrational_analysis
    s = Structure(elements=["H", "H"],
                  positions=np.array([[5.0, 5.0, 5.741], [5.0, 5.0, 5.0]]),
                  regions={"frozen_atoms": [1]},
                  cell=np.diag([10.0, 10.0, 10.0]),
                  axis_kind=("isolated",) * 3)
    sort = sort_by(s, "held-first")
    assert sort.sorted_to_original == (1, 0) and sort.key == "held-first"
    write_permutation(tmp_path, sort)
    perm = read_permutation(tmp_path)
    assert perm.key == "held-first"
    copy = sort.structure
    r = vibrational_analysis(
        hessian_from_fc(read_fc(FIXTURES / "siesta_fc" / "h2.FC"), [1]),
        [atomic_mass(e) for e in copy.elements], copy.positions,
        copy.elements, [0], axis_kind=copy.axis_kind, cell=copy.cell,
        permutation=perm, label="h2", engine="siesta", temperature_K=298.15)
    assert r.engine == "siesta"
    assert r.free_atom_idxs == [0] and r.frozen_atom_idxs == [1]
    assert [round(m.frequency_cm1, 1) for m in r.modes] == [3354.6]
    assert r.removed_motions["count"] == 2
    assert r.hessian_scope == "free" and r.n_atoms_in_hessian == 1
    assert r.equilibrium_mo_energies_eh is None
    assert all(m.raman_activity_a4_amu is None and m.ir_intensity_km_mol is None
               for m in r.modes)
    assert r.thermo["regime"] == "vibrational-only"
    # no forces handed in: nothing judged, nothing invented
    assert r.relaxation["converged"] is None
    assert r.relaxation["max_force_eh_bohr"] is None


def _relaxed_h2():
    """The relaxed fixture's structure, held atom LAST in the input so the
    sort really reorders (`fixtures/siesta_fc/README.md`)."""
    return Structure(elements=["H", "H"],
                     positions=np.array([[5.0, 5.0, 5.77446], [5.0, 5.0, 5.0]]),
                     regions={"frozen_atoms": [1]},
                     cell=np.diag([10.0, 10.0, 10.0]),
                     axis_kind=("isolated",) * 3)


def _fc_attempt(root, *, criterion=0.02):
    """A finished force-constant attempt built from the measured relaxed H2
    fixtures -- what a job's finish finds beside itself
    (`engines/vibration.md` § 5.5): the measured deck (`h2.fdf`, the held
    atom first) with the two blocks the finish reads written by their own
    writers, the run's output and force constants, and the permutation of a
    held-last input written by the sort.  Returns ``(deck, output)``."""
    import shutil
    from types import SimpleNamespace

    from molbuilder.script_emit import emit_engine_offset, emit_vibration_record
    from molbuilder.spectra.siesta_vibration import vibration_record
    root.mkdir(parents=True, exist_ok=True)
    fx = FIXTURES / "siesta_fc"
    frame = SimpleNamespace(applied_offset=[0.0, 0.0, 0.0],
                            cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                  [0.0, 0.0, 10.0]], stated=True)
    rec = vibration_record(stage="freq", force_criterion_ev_ang=criterion,
                           already_relaxed=True, relaxation=None,
                           temperature_K=298.15, molbuilder_version="fixture")
    (root / "h2.fdf").write_text(
        (fx / "h2.fdf").read_text() + "\n"
        + emit_engine_offset(frame, ["isolated"] * 3) + "\n"
        + emit_vibration_record(rec) + "\n")
    shutil.copy2(fx / "h2_fc.out", root / "h2-run0.out")
    shutil.copy2(fx / "h2_relaxed.FC", root / "h2.FC")
    write_permutation(root, sort_by(_relaxed_h2(), "held-first"))
    return root / "h2.fdf", root / "h2-run0.out"


def test_the_finish_judges_the_reference_forces_by_the_decks_criterion(tmp_path):
    """R5 on this route (`engines/vibration.md` § 5.5): the finish reads the
    forces SIESTA evaluated at its FC step 0 from the run's output, judges the
    largest component over the FREE atoms against the criterion the deck's
    `vibration` block carries, and writes the verdict with the number -- on
    the measured relaxed fixture, then with a criterion the same forces cannot
    meet.  The whole finish, `siesta_vibration.finish`, on a fixture attempt."""
    import json

    from molbuilder.constants import HARTREE_BOHR_EV_ANGSTROM_ASE
    from molbuilder.parse.engines.siesta_fc import fc_block_asymmetry, read_fc
    from molbuilder.spectra.siesta_vibration import finish
    # an axial block: the off-diagonals vanish by symmetry, so it shows nothing
    assert fc_block_asymmetry(
        read_fc(FIXTURES / "siesta_fc" / "h2_relaxed.FC"), [1]) < 1e-9
    d = json.loads(finish(*_fc_attempt(tmp_path / "a", criterion=0.02))
                   .read_text())
    assert [round(m["frequency_cm1"], 1) for m in d["modes"]] == [3022.3]
    assert d["free_atom_idxs"] == [0] and d["frozen_atom_idxs"] == [1]
    rx = d["relaxation"]
    assert rx["converged"] is True and rx["already_relaxed"] is True
    # the free atom's force, not the held one's (the deck's order: held first)
    assert abs(rx["max_force_eh_bohr"]
               - 0.000057 / HARTREE_BOHR_EV_ANGSTROM_ASE) < 1e-12
    assert abs(rx["max_force_all_atoms_eh_bohr"]
               - 0.003787 / HARTREE_BOHR_EV_ANGSTROM_ASE) < 1e-12
    assert d["engine_metadata"]["reference_force_criterion_ev_ang"] == 0.02
    assert d["engine_metadata"]["fc_range_1based"] == [2, 2]
    assert "Head1997" in d["bibliography_keys"]
    # the same forces against a criterion they cannot meet: the verdict flips
    d2 = json.loads(finish(*_fc_attempt(tmp_path / "b", criterion=1e-5))
                    .read_text())
    assert d2["relaxation"]["converged"] is False


def test_the_mass_calibrated_displacement_rides_every_mode(tmp_path):
    """`engines/vibration.md` § 6.3: the zero-point amplitude and the
    displacement at it are derived at serialisation from the frequency and
    the canonical vector -- the number a vibration-coupled transport step
    displaces along -- and are absent for an imaginary mode."""
    import math

    from molbuilder.constants import ZERO_POINT_Q2_AMU_ANG2_CM1
    from molbuilder.spectra.siesta_vibration import (read_force_constant_run,
                                                     result_of)
    r = result_of(read_force_constant_run(*_fc_attempt(tmp_path)))
    row = r.to_dict()["modes"][0]
    nu = row["frequency_cm1"]
    q = row["zero_point_amplitude_amu12_ang"]
    # the constant is physics, not a fit: the H-H zero-point r.m.s. bond
    # amplitude, sqrt(hbar / 2 mu omega) with mu = 0.504 amu at 4401 cm-1,
    # is the textbook 0.087 A
    assert abs(math.sqrt(ZERO_POINT_Q2_AMU_ANG2_CM1 / 4401.0) / math.sqrt(0.504)
               - 0.0873) < 5e-4
    disp = np.asarray(row["zero_point_displacement_ang"])
    canon = np.asarray(row["eigenvector_canonical"])
    # the rule of § 6.3: the displacement is the CANONICAL vector at Q_zp (the
    # display vector would put the same hydrogen at 0.075 A and pass the bound
    # below; this is what tells the two apart)
    assert np.allclose(disp, q * canon)
    # one hydrogen at 3022 cm-1 swings 0.074 A at its zero point
    assert 0.070 < abs(disp[0, 2]) < 0.078
    # an imaginary mode has no amplitude
    r.modes[0].frequency_cm1 = -abs(nu)
    r.modes[0].has_imag = True
    row = r.to_dict()["modes"][0]
    assert row["zero_point_amplitude_amu12_ang"] is None
    assert row["zero_point_displacement_ang"] is None


def test_the_finish_runs_where_molbuilder_is_not(tmp_path):
    """I23 (`engines/vibration.md` § 7): the finish's bundle, `mb_vibration.pyz`,
    run by the SIESTA job env's own python in a fixture attempt, with the
    environment emptied so molbuilder cannot be imported -- it loads, answers
    `loads`, and writes the same spectrum the package's finish writes.

    MUTATION THIS MUST FAIL AGAINST: a member dropped from
    `runwrap.VIBRATION_COMPANIONS`, or one importing a non-member at load.
    Skipped where the SIESTA env is not installed.
    """
    import json
    import subprocess

    from _road import env_available, env_bin
    from molbuilder.runwrap import VIBRATION_BUNDLE, vibration_bundle
    from molbuilder.spectra.siesta_vibration import finish
    if not env_available("molbuilder-siesta"):
        pytest.skip("needs the molbuilder-siesta env")
    py = env_bin("molbuilder-siesta") / "python"
    deck, output = _fc_attempt(tmp_path / "job")
    (deck.parent / VIBRATION_BUNDLE).write_bytes(vibration_bundle())
    bare = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    probe = subprocess.run([str(py), "-c", "import molbuilder"],
                           cwd=deck.parent, env=bare, capture_output=True)
    assert probe.returncode != 0, "molbuilder is importable: not a bare job"
    loads = subprocess.run([str(py), VIBRATION_BUNDLE, "loads"],
                           cwd=deck.parent, env=bare, capture_output=True,
                           text=True)
    assert loads.returncode == 0, loads.stderr
    ran = subprocess.run([str(py), VIBRATION_BUNDLE, deck.name, output.name],
                         cwd=deck.parent, env=bare, capture_output=True,
                         text=True)
    assert ran.returncode == 0, ran.stderr
    beside = json.loads((deck.parent / "h2.spectra.json").read_text())
    host = json.loads(finish(*_fc_attempt(tmp_path / "host")).read_text())
    for d in (beside, host):
        d.pop("timestamp")
    assert beside == host


# --------------------------------------------------------------------- #
#  The structure's own evidence (vibration.md § 2.2, the record table)   #
# --------------------------------------------------------------------- #

_RELAX_RUN = FIXTURES / "siesta_relax" / "01_relax" / "run-0"


def _relaxed_h2_with_record(*, shift_z: float = 0.0):
    """The geometry the measured relaxation left (held atom first, as the
    run had it), carrying the record the run directory answers for
    itself -- the pair the Results tab would export.

    WHY API-LEVEL: the record table's rows (vibration.md § 2.2) are verdicts
    of the gate on a structure's metadata, and the metadata comes from the
    measured fixture `tests/fixtures/siesta_relax` through the composer the
    Results tab uses -- no engine runs and nothing is invented.  The road
    that produced the fixture is the SIESTA e2e test."""
    from molbuilder.parse.dirs.run_info import run_info_for_dir
    s = Structure(elements=["H", "H"],
                  positions=np.array([[5.0, 5.0, 5.0],
                                      [5.0, 5.0, 5.774583 + shift_z]]),
                  regions={"frozen_atoms": [0]},
                  cell=np.diag([10.0, 10.0, 10.0]),
                  axis_kind=("isolated",) * 3)
    s.apply_info_dict(run_info_for_dir(_RELAX_RUN))
    return s


def _record_findings(struct, cfg):
    from molbuilder.validation import validate
    # everything the gate says on the box's card except the statement's own
    # finding (the ticked warning / the unticked info of § 5.8)
    return [i for i in validate(struct, cfg, calculation="vibration")
            if i.where == "config.already_relaxed"
            and not i.message.startswith(("You stated",
                                          "The structure is not stated"))]


def test_a_matching_record_answers_the_ticked_box_with_its_numbers(
        in_tree_psml):
    """Ticked, record present for these coordinates, within this
    calculation's tolerance at the same level and held set: one info line
    with the engine, the tolerance and the largest remaining force; no
    warning about the record."""
    s = _relaxed_h2_with_record()
    cfg = SiestaConfig(system_label="h2", psml_lib=in_tree_psml,
                       already_relaxed=True, relax_force_tol=0.01)
    found = _record_findings(s, cfg)
    assert len(found) == 1 and found[0].severity == "info", found
    assert "relaxed on siesta to 0.01 eV/Å" in found[0].message
    assert "0.0010 eV/Å, within this calculation's tolerance of 0.01" \
        in found[0].message


def test_a_record_looser_than_this_calculation_warns_when_ticked(
        in_tree_psml):
    """The record's largest force above THIS calculation's tolerance: a
    warning when ticked, information when unticked (the record table,
    rows 3 and 5; measured fixture, see `_relaxed_h2_with_record`)."""
    s = _relaxed_h2_with_record()
    cfg = SiestaConfig(system_label="h2", psml_lib=in_tree_psml,
                       already_relaxed=True, relax_force_tol=0.0005)
    found = _record_findings(s, cfg)
    assert len(found) == 1 and found[0].severity == "warn", found
    assert "above this calculation's tolerance of 0.0005" in found[0].message
    # unticked, the same fact is information: the ladder relaxes anyway
    cfg2 = SiestaConfig(system_label="h2", psml_lib=in_tree_psml,
                        already_relaxed=False, relax_force_tol=0.0005)
    found2 = _record_findings(s, cfg2)
    assert len(found2) == 1 and found2[0].severity == "info", found2


def test_a_record_for_another_geometry_does_not_vouch(in_tree_psml):
    """Another frame of the run, or an edit since: the fingerprint differs
    and the record is set aside with a warning (ticked)."""
    s = _relaxed_h2_with_record(shift_z=0.01)
    cfg = SiestaConfig(system_label="h2", psml_lib=in_tree_psml,
                       already_relaxed=True)
    found = _record_findings(s, cfg)
    assert len(found) == 1 and found[0].severity == "warn", found
    assert "different geometry" in found[0].message


def test_a_record_at_another_level_of_theory_warns(in_tree_psml):
    """The recorded contract (DZP, GGA/PBE, 300 Ry, 300 K) against a form
    that runs SZ: a different level of theory, named field by field."""
    s = _relaxed_h2_with_record()
    cfg = SiestaConfig(system_label="h2", psml_lib=in_tree_psml,
                       already_relaxed=True, basis_size="SZ")
    found = [i for i in _record_findings(s, cfg) if i.severity == "warn"]
    assert len(found) == 1, found
    assert "basis_size: relaxed with DZP, this run SZ" in found[0].message
    # and a different ENGINE is the same fact at the engine level
    from molbuilder.config.pyscf import PySCFConfig
    pfound = _record_findings(s, PySCFConfig(job_name="h2",
                                             already_relaxed=True))
    assert any(i.severity == "warn"
               and "relaxed on siesta; this is a pyscf calculation"
               in i.message for i in pfound), pfound


def test_a_record_at_another_charge_warns(in_tree_psml):
    """ES7 (`science/chemistry-correctness.md` § 2a): the electronic state is
    part of the level of theory the record is compared against, RESOLVED --
    a frequency at a geometry relaxed at another charge is usually a
    mistake.  H2 at -2 keeps the electron count even, so the parity rule
    does not answer first."""
    s = _relaxed_h2_with_record()
    cfg = SiestaConfig(system_label="h2", psml_lib=in_tree_psml,
                       already_relaxed=True, net_charge=-2)
    found = [i for i in _record_findings(s, cfg) if i.severity == "warn"]
    assert any("net_charge: relaxed with 0, this run -2" in i.message
               for i in found), found


def test_a_good_record_under_an_unticked_box_offers_the_skip(in_tree_psml):
    """Unticked with a record that already meets the criterion: the info
    line offers the skip (the record table, row 4; measured fixture)."""
    s = _relaxed_h2_with_record()
    cfg = SiestaConfig(system_label="h2", psml_lib=in_tree_psml,
                       already_relaxed=False, relax_force_tol=0.01)
    found = _record_findings(s, cfg)
    assert len(found) == 1 and found[0].severity == "info", found
    assert "the box may be ticked and the relaxation skipped" in found[0].message


def test_no_record_under_a_ticked_box_is_accepted_with_a_hint(in_tree_psml):
    """Ticked with no record: accepted, with the hint that the statement
    stands on its own (the record table, row 1; the fixture's structure
    with its metadata stripped)."""
    s = _relaxed_h2_with_record()
    s.apply_info_dict(None)
    cfg = SiestaConfig(system_label="h2", psml_lib=in_tree_psml,
                       already_relaxed=True)
    found = _record_findings(s, cfg)
    assert len(found) == 1 and found[0].severity == "info", found
    assert "No relaxation record travels with this structure" \
        in found[0].message

