"""The vibration kind on SIESTA: the force-constant deck, the sorted copy
it is rendered from, and the modes derived from what the run leaves.

science/normal-modes.md § 4a.6 and design § 8: SIESTA nudges the free atoms
(`MD.TypeOfRun FC`, one contiguous range) and writes `.FC`; everything after
that is the one path both engines share.  The held-first sort is what makes
a scattered held set a range (`model/overview.md` § 2.2), and the copy's
order record is what puts the answer back in the user order.  And a flat
calculation's refusal of a second force-constant stage, which prep makes
before any deck is written.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig
from molbuilder.script_emit import render_deck
from molbuilder.siesta.input import spec_for
from molbuilder.structure import Structure
from molbuilder.transport.sort import SORT_HELD_FIRST, sort_by
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


def test_the_force_constant_deck_nudges_the_free_range_only(in_tree_psml):
    """Held atom in the middle of the input; the sorted copy puts it first,
    the deck names the free atoms as the trailing range, and nothing in the
    deck relaxes."""
    s = _three_h(held=[1])
    sorted_s = sort_by(s, SORT_HELD_FIRST).structure
    cfg = SiestaConfig(system_label="h3", psml_lib=in_tree_psml,
                       already_relaxed=True)
    text = render_deck(spec_for(sorted_s, cfg, calculation="vibration",
                                names=RunNames.of("h3", "01_fc",
                                                  "hierarchical")),
                       sorted_s, cfg, verbose=False)
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


def test_every_atom_held_is_refused_by_the_gate():
    from molbuilder.validation import validate
    s = _three_h(held=[0, 1, 2])
    issues = validate(s, SiestaConfig(system_label="h3"),
                      calculation="vibration")
    assert any(i.severity == "error" and "every atom is held" in i.message
               for i in issues)


def test_a_flat_calculation_refuses_a_second_force_constant_stage(
        tmp_path, monkeypatch):
    """In the flat layout every stage writes the same `<label>.FC` and
    `<label>.spectra.json`, so two force-constant stages would overwrite each
    other's result: `prep` refuses the description before any sort or deck is
    written, at whichever stage is prepared first (`engines/vibration.md`
    § 5.9).  The road's CO2 described flat, a second force-constant stage
    added on Task setup.

    MUTATION THIS MUST FAIL AGAINST: the check not made.
    """
    from support.road import (CO2, describe_calculation, group_words, jobset,
                              ladder_edited)
    bundle = describe_calculation(tmp_path, monkeypatch, shape="flat",
                                  name="CO2", calculation="vibration",
                                  structure=CO2)
    ladder_edited({"added": [{"name": "freq_half",
                              "overrides": {"fc_displacement": 0.02}}]},
                  bundle)
    r = jobset("prep", "task", *group_words(bundle, "relax"),
               "--bundle", bundle, "--target", "this")
    assert r.exit_code != 0 and "hierarchical layout" in r.output, r.output
    for written in ("*.fdf", "job-set.json"):
        assert not list(bundle.glob(written)), f"{written} before refusing"
