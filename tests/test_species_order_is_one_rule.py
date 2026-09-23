"""The species ORDER — one rule, every engine (`model/chemistry.md` § 3a).

Why this file exists: the order fixes each element's index in
`%block ChemicalSpeciesLabel`, and that index fixes the orbital ordering
inside SIESTA's `.DM` and `.TSHS`.  Two emitters carried two rules until
2026-09-23 -- `siesta/input.py` sorted by atomic number and
`transport/transiesta.py` alphabetically -- so one structure was declared
`H, C, S, Au` by one and `Au, C, H, S` by the other.

**THIS FILE OWNS THE LITERALS, AND IT IS THE ONLY ONE THAT MAY.**  Every
other test asks `chemistry.species_order` instead of spelling an order out
(user, 2026-09-23).  The split is not style:

* here the order IS the subject, so deriving the expectation from the
  function would assert the code equals itself and pin nothing -- the exact
  tautology pulled out of `test_transport_wizard.py` the day before;
* everywhere else the order is incidental -- a deck renders, a message names
  the elements, a sort put the lead first -- so a spelled-out order pins a
  contract that test has no opinion about, and goes red when the rule
  changes.  Two did: `test_psml_anchor` and `test_transport_prep`, both on
  2026-09-23, neither about species order.
"""
from __future__ import annotations

import pytest

from molbuilder.chemistry import species_order


@pytest.mark.parametrize("elements,expected,why", [
    (["Au", "C", "H", "S"], ["C", "H", "S", "Au"],
     "carbon wins over the sulfur -- the Au-BDT-Au junction"),
    (["C", "H", "O"], ["C", "H", "O"],
     "carbon wins over the oxygen -- methanol is CH4O, never HCO"),
    (["H", "O"], ["H", "O"], "group VI puts hydrogen first -- H2O"),
    (["H", "F"], ["H", "F"], "group VII -- HF"),
    (["H", "Cl"], ["H", "Cl"], "group VII -- HCl"),
    (["H", "S"], ["H", "S"], "group VI -- H2S"),
    (["H", "N", "O"], ["H", "N", "O"],
     "the oxygen decides, not the nitrogen -- HNO3"),
    (["H", "N"], ["N", "H"], "no carbon, no VI/VII -- NH3, not H3N"),
    (["B", "H"], ["B", "H"], "anchor is the lightest non-H -- B2H6"),
    (["Si", "H"], ["Si", "H"], "SiH4"),
    (["Au"], ["Au"], "a bulk lead: one species, nothing to order"),
])
def test_the_order_is_the_one_a_chemist_would_write(elements, expected, why):
    assert species_order(elements) == expected, why


def test_numbered_labels_stay_distinct_and_hydrogen_lands_after_the_group():
    """`Au1`/`Au2` are two species a person wrote on purpose, and hydrogen
    belongs after the carbons as a GROUP -- not between `C1` and `C2`."""
    assert species_order(["C1", "H", "C2", "S"]) == \
        ["C1", "C2", "H", "S"]
    assert species_order(["H1", "C", "H2"]) == ["C", "H1", "H2"]
    assert species_order(["Au2", "Au1"]) == ["Au2", "Au1"], \
        "a tie keeps first-seen order, so the file's own order survives"


def test_the_override_reaches_a_transport_deck_too():
    """`species_order` is a catalogue row a person can fill in.  Every
    SIESTA deck honoured it and NO transport deck could see it, because the
    lifted geometry block took no config -- the config was already in scope
    at the call site and simply not passed (`deck.py`).  Global means
    global (user, 2026-09-23).
    """
    import numpy as np
    from molbuilder import script_emit as _sc
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.input import spec_for
    from molbuilder.structure import Structure

    struct = Structure(
        elements=["Au", "C", "H", "S"],
        positions=np.array([[0., 0, 2. * i] for i in range(4)]),
        cell=np.diag([10., 10., 8.]),
        axis_kind=("periodic", "periodic", "transport"),
        regions={"L-electrode": [0], "bridge": [1, 2], "R-electrode": [3]})

    def _declared(order):
        cfg = SiestaConfig(system_label="t", species_order=order)
        deck = _sc.render_deck(
            spec_for(struct, cfg, stage_token="device",
                     calculation="transport"), struct, cfg)
        return [ln.split()[2] for ln in deck.splitlines()
                if ln.startswith("  ") and len(ln.split()) == 3
                and ln.split()[0].isdigit()]

    assert _declared(None) == ["C", "H", "S", "Au"], "the default rule"
    assert _declared(["S", "H", "C", "Au"]) == ["S", "H", "C", "Au"], \
        "the person's own order, honoured verbatim"


def test_both_emitters_ask_the_same_function():
    """The defect this rule closed: two emitters, two orders, one structure.

    Asked through each emitter's own door rather than by reading the source,
    so a third rule appearing anywhere would have to pass here too.
    """
    els = ["Au", "C", "H", "S"]
    import numpy as np
    from molbuilder.structure import Structure
    from molbuilder.transport.transiesta import _emit_geometry
    struct = Structure(
        elements=els,
        positions=np.array([[0., 0, 2. * i] for i in range(4)]),
        cell=np.diag([10., 10., 8.]),
        axis_kind=("periodic", "periodic", "transport"))
    block = "\n".join(_emit_geometry(struct))
    rows = [ln.split() for ln in block.splitlines()
            if ln.startswith("  ") and len(ln.split()) == 3
            and ln.split()[0].isdigit()]
    assert [r[2] for r in rows] == species_order(els), block
