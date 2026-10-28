"""The vibration kind on SIESTA: the force-constant deck, the sorted copy
it is rendered from, and the modes derived from what the run leaves.

science/normal-modes.md § 4a.6 and design § 8: SIESTA nudges the free atoms
(`MD.TypeOfRun FC`, one contiguous range) and writes `.FC`; everything after
that is the one path both engines share.  The held-first sort is what makes
a scattered held set a range (`model/overview.md` § 2.2), and the copy's
order record is what puts the answer back in the user order.  And the flat
layout's refusal, which prep makes before any deck is written: a SIESTA
vibration is one run in phases, each SIESTA call in a folder of its own
(`engines/vibration.md` § 5.11).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig
from molbuilder.parse.fdf import fdf_key, read_deck, read_fdf
from molbuilder.siesta.vibration_deck import FC_PART
from molbuilder.structure import Structure
from molbuilder.runfiles import RunNames

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def in_tree_psml(isolated_projects_root):
    """The fixture pseudopotential library, copied inside the projects tree
    the way the validator requires."""
    shutil.copytree(FIXTURES / "psml", isolated_projects_root / "psml")
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


def test_the_force_constant_deck_nudges_the_free_range_only(tmp_path,
                                                           monkeypatch):
    """The road's CO2 -- O O C, the carbon held, in the middle of nothing
    but last in the input: the sorted copy puts it first, the force-constant
    deck prep writes for the `fc` phase names the free atoms as the trailing
    range, and nothing in that deck relaxes.  Read as the engine reads it,
    its coordinates block inlined from the file beside it (`read_deck`)."""
    from support.road import CO2, describe_calculation, group_words, jobset
    bundle = describe_calculation(tmp_path, monkeypatch, name="CO2",
                                  calculation="vibration", structure=CO2)
    r = jobset("prep", "task", *group_words(bundle, "freq"),
               "--bundle", bundle, "--target", "this")
    assert r.exit_code == 0, r.output
    names = RunNames.of("CO2", "01_freq", "hierarchical")
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
    from molbuilder.template import with_values_set
    from molbuilder.template_file import template_path
    from support.road import CO2, describe_calculation, group_words, jobset
    bundle = describe_calculation(tmp_path, monkeypatch, name="CO2",
                                  calculation="vibration", structure=CO2)
    path = template_path(bundle, "CO2")
    path.write_text(with_values_set(path.read_text(), SiestaConfig,
                                    calculation="vibration", values=values,
                                    source="person"))
    r = jobset("prep", "task", *group_words(bundle, "freq"),
               "--bundle", bundle, "--target", "this")
    assert r.exit_code == 0, r.output
    names = RunNames.of("CO2", "01_freq", "hierarchical")
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


