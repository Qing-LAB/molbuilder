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
from molbuilder.transport.sort import (read_permutation, sort_by,
                                       write_permutation)

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
    """The whole return leg on the measured fixture: held atom LAST in the
    input, so the sort really reorders; the derivation reads the record and
    speaks the input's numbering."""
    from molbuilder.spectra.from_siesta import spectra_results_from_fc
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
    r = spectra_results_from_fc(s, sort.structure, perm,
                                FIXTURES / "siesta_fc" / "h2.FC",
                                label="h2", displacement_bohr=0.04)
    assert r.engine == "siesta"
    assert r.free_atom_idxs == [0] and r.frozen_atom_idxs == [1]
    assert [round(m.frequency_cm1, 1) for m in r.modes] == [3354.6]
    assert r.removed_motions["count"] == 2
    assert r.hessian_scope == "free" and r.n_atoms_in_hessian == 1
    assert r.equilibrium_mo_energies_eh is None
    assert all(m.raman_activity_a4_amu is None and m.ir_intensity_km_mol is None
               for m in r.modes)
    assert r.thermo["regime"] == "vibrational-only"
    assert "Head1997" in r.bibliography_keys


def _relaxed_h2():
    """The relaxed fixture's structure, held atom LAST in the input so the
    sort really reorders (`fixtures/siesta_fc/README.md`)."""
    return Structure(elements=["H", "H"],
                     positions=np.array([[5.0, 5.0, 5.77446], [5.0, 5.0, 5.0]]),
                     regions={"frozen_atoms": [1]},
                     cell=np.diag([10.0, 10.0, 10.0]),
                     axis_kind=("isolated",) * 3)


def test_the_reference_forces_are_read_from_the_run_and_stationarity_is_judged(tmp_path):
    """R5 on this route (`engines/vibration.md` § 5.5): the forces SIESTA
    evaluated at its FC step 0 are read from the run's output, the largest
    component over the FREE atoms is judged against the criterion given, and
    the verdict is written with the number -- on the measured relaxed fixture
    (a measured fixture, not the road: the road's own run is the e2e test),
    then with a criterion the same forces cannot meet."""
    from molbuilder.constants import HARTREE_BOHR_EV_ANGSTROM_ASE
    from molbuilder.parse.engines.siesta import SiestaParser
    from molbuilder.parse.engines.siesta_fc import (fc_block_asymmetry,
                                                    read_fc,
                                                    reference_frame_of)
    from molbuilder.spectra.from_siesta import spectra_results_from_fc
    f = np.asarray(reference_frame_of(SiestaParser.parse(
        str(FIXTURES / "siesta_fc" / "h2_fc.out"))).forces, dtype=float)
    # the deck's order: the held atom first, the free one second
    assert abs(f[0, 2] - (-0.003787)) < 1e-9 and abs(f[1, 2] - (-0.000057)) < 1e-9
    fc = read_fc(FIXTURES / "siesta_fc" / "h2_relaxed.FC")
    # an axial block: the off-diagonals vanish by symmetry, so it shows nothing
    assert fc_block_asymmetry(fc, [1]) < 1e-9
    s = _relaxed_h2()
    sort = sort_by(s, "held-first")
    write_permutation(tmp_path, sort)
    perm = read_permutation(tmp_path)
    r = spectra_results_from_fc(s, sort.structure, perm,
                                FIXTURES / "siesta_fc" / "h2_relaxed.FC",
                                label="h2", displacement_bohr=0.04,
                                reference_forces_ev_ang=f,
                                force_criterion_ev_ang=0.02,
                                already_relaxed=True)
    assert [round(m.frequency_cm1, 1) for m in r.modes] == [3022.3]
    rx = r.relaxation
    assert rx["converged"] is True and rx["already_relaxed"] is True
    assert abs(rx["max_force_eh_bohr"] - 0.000057 / HARTREE_BOHR_EV_ANGSTROM_ASE) < 1e-12
    assert abs(rx["max_force_all_atoms_eh_bohr"]
               - 0.003787 / HARTREE_BOHR_EV_ANGSTROM_ASE) < 1e-12
    assert r.engine_metadata["reference_force_criterion_ev_ang"] == 0.02
    # the same forces against a criterion they cannot meet: the verdict flips
    # and the warning names the number and the two ways out
    r2 = spectra_results_from_fc(s, sort.structure, perm,
                                 FIXTURES / "siesta_fc" / "h2_relaxed.FC",
                                 label="h2", displacement_bohr=0.04,
                                 reference_forces_ev_ang=f,
                                 force_criterion_ev_ang=1e-5)
    assert r2.relaxation["converged"] is False
    # no forces given: nothing judged, nothing invented
    r3 = spectra_results_from_fc(s, sort.structure, perm,
                                 FIXTURES / "siesta_fc" / "h2_relaxed.FC",
                                 label="h2", displacement_bohr=0.04)
    assert r3.relaxation["converged"] is None and r3.relaxation["max_force_eh_bohr"] is None


def test_the_mass_calibrated_displacement_rides_every_mode(tmp_path):
    """`engines/vibration.md` § 6.3: the zero-point amplitude and the
    displacement at it are derived at serialisation from the frequency and
    the canonical vector -- the number a vibration-coupled transport step
    displaces along -- and are absent for an imaginary mode."""
    import math
    from molbuilder.constants import ZERO_POINT_Q2_AMU_ANG2_CM1
    from molbuilder.spectra.from_siesta import spectra_results_from_fc
    s = _relaxed_h2()
    sort = sort_by(s, "held-first")
    write_permutation(tmp_path, sort)
    r = spectra_results_from_fc(s, sort.structure, read_permutation(tmp_path),
                                FIXTURES / "siesta_fc" / "h2_relaxed.FC",
                                label="h2", displacement_bohr=0.04)
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

