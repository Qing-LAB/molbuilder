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
    cfg = SiestaConfig(system_label="h3", psml_lib=in_tree_psml)
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
    cfg = SiestaConfig(system_label="h3", psml_lib=in_tree_psml)
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
