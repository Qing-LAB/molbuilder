"""Tests for :mod:`molbuilder.selection` -- rule evaluator + JSON round-trip.

Three contracts:

  1.  Each primitive rule selects what its docstring says against a
      Structure with predictable shape.
  2.  Boolean operators compose correctly (Or = union, And =
      intersection, Minus = a\\b, Not = complement).
  3.  Every rule round-trips through to_json -> from_json unchanged
      (equality on frozen dataclasses).
"""

from __future__ import annotations

import numpy as np
import pytest

from molbuilder.selection import (
    All, And, ByAtomName, ByChainId, ByClick, ByElement, ByIndexRange,
    ByRegion, ByResidueName, FirstN, Minus, Not, Or,
    SelectionError, evaluate, from_json, to_json,
)
from molbuilder.structure import Structure


# --------------------------------------------------------------------- #
#  Fixture: a small mixed-element / multi-residue structure             #
# --------------------------------------------------------------------- #


@pytest.fixture
def junction_struct() -> Structure:
    """Toy junction: 4 Au + 3 C (bridge) + 4 Au.  Atom indices::

        0..3   = L electrode (Au)
        4..6   = bridge      (C)
        7..10  = R electrode (Au)

    Region labels mirror this so ByRegion tests have something to
    hit.  Atom names / chain ids are populated so the PDB-style
    selectors have meaningful data."""
    elements = ["Au"] * 4 + ["C"] * 3 + ["Au"] * 4
    positions = np.zeros((11, 3))
    return Structure(
        elements=elements,
        positions=positions,
        atom_names=["AuL" if i < 4 else ("C" if i < 7 else "AuR")
                    for i in range(11)],
        residue_ids=[1] * 11,
        residue_names=(["LEL"] * 4 + ["BRG"] * 3 + ["REL"] * 4),
        chain_ids=(["L"] * 4 + ["B"] * 3 + ["R"] * 4),
        regions={
            "L-electrode": list(range(0, 4)),
            "bridge":      list(range(4, 7)),
            "R-electrode": list(range(7, 11)),
        },
    )


# --------------------------------------------------------------------- #
#  Primitive rules                                                       #
# --------------------------------------------------------------------- #


class TestPrimitives:
    def test_all(self, junction_struct):
        assert evaluate(All(), junction_struct) == frozenset(range(11))

    def test_by_element_single(self, junction_struct):
        # All Au atoms across both electrodes:
        sel = evaluate(ByElement(("Au",)), junction_struct)
        assert sel == frozenset([0, 1, 2, 3, 7, 8, 9, 10])

    def test_by_element_multi(self, junction_struct):
        sel = evaluate(ByElement(("Au", "C")), junction_struct)
        assert sel == frozenset(range(11))

    def test_by_element_no_match(self, junction_struct):
        assert evaluate(ByElement(("Pt",)), junction_struct) == frozenset()

    def test_by_residue_name(self, junction_struct):
        sel = evaluate(ByResidueName(("BRG",)), junction_struct)
        assert sel == frozenset([4, 5, 6])

    def test_by_atom_name(self, junction_struct):
        sel = evaluate(ByAtomName(("AuL",)), junction_struct)
        assert sel == frozenset([0, 1, 2, 3])

    def test_by_chain_id(self, junction_struct):
        sel = evaluate(ByChainId(("R",)), junction_struct)
        assert sel == frozenset([7, 8, 9, 10])

    def test_by_region(self, junction_struct):
        sel = evaluate(ByRegion("bridge"), junction_struct)
        assert sel == frozenset([4, 5, 6])

    def test_by_region_unknown_raises(self, junction_struct):
        with pytest.raises(SelectionError, match="no region labelled"):
            evaluate(ByRegion("L_electrode"), junction_struct)

    def test_by_index_range_single_range(self, junction_struct):
        sel = evaluate(ByIndexRange("0-3"), junction_struct)
        assert sel == frozenset([0, 1, 2, 3])

    def test_by_index_range_multi_token(self, junction_struct):
        sel = evaluate(ByIndexRange("0-3, 7, 9-10"), junction_struct)
        assert sel == frozenset([0, 1, 2, 3, 7, 9, 10])

    def test_by_index_range_empty(self, junction_struct):
        assert evaluate(ByIndexRange(""), junction_struct) == frozenset()

    def test_by_index_range_out_of_bounds_raises(self, junction_struct):
        with pytest.raises(SelectionError, match="out of"):
            evaluate(ByIndexRange("0-99"), junction_struct)

    def test_by_index_range_reversed_raises(self, junction_struct):
        with pytest.raises(SelectionError, match="hi < lo"):
            evaluate(ByIndexRange("5-2"), junction_struct)

    def test_by_index_range_malformed_raises(self, junction_struct):
        with pytest.raises(SelectionError, match="invalid token"):
            evaluate(ByIndexRange("abc"), junction_struct)

    def test_by_click(self, junction_struct):
        sel = evaluate(ByClick((3, 5, 8)), junction_struct)
        assert sel == frozenset([3, 5, 8])

    def test_by_click_out_of_range_raises(self, junction_struct):
        with pytest.raises(SelectionError, match="out of range"):
            evaluate(ByClick((42,)), junction_struct)


# --------------------------------------------------------------------- #
#  Boolean operators                                                     #
# --------------------------------------------------------------------- #


class TestBooleanOps:
    def test_or_two_operands(self, junction_struct):
        sel = evaluate(
            Or((ByElement(("C",)), ByIndexRange("0-1"))),
            junction_struct,
        )
        assert sel == frozenset([0, 1, 4, 5, 6])

    def test_or_empty_is_empty_set(self, junction_struct):
        assert evaluate(Or(()), junction_struct) == frozenset()

    def test_and_two_operands(self, junction_struct):
        # All Au atoms intersected with "first half" = L electrode:
        sel = evaluate(
            And((ByElement(("Au",)), ByIndexRange("0-5"))),
            junction_struct,
        )
        assert sel == frozenset([0, 1, 2, 3])

    def test_and_empty_is_full_set(self, junction_struct):
        """Identity for intersection (lattice convention)."""
        assert evaluate(And(()), junction_struct) == frozenset(range(11))

    def test_and_short_circuits_on_empty(self, junction_struct):
        # Doesn't crash on a third operand even though intersection
        # is already empty after two:
        sel = evaluate(
            And((ByElement(("Pt",)), ByElement(("Au",)),
                 ByIndexRange("0-99"))),
            junction_struct,
        )
        assert sel == frozenset()

    def test_minus(self, junction_struct):
        # Everything except the bridge:
        sel = evaluate(Minus(All(), ByRegion("bridge")), junction_struct)
        assert sel == frozenset([0, 1, 2, 3, 7, 8, 9, 10])

    def test_not_is_complement(self, junction_struct):
        # Not(ByElement('Au')) should equal Minus(All(), ByElement('Au'))
        au = ByElement(("Au",))
        assert (evaluate(Not(au), junction_struct)
                == evaluate(Minus(All(), au), junction_struct))

    def test_first_n_picks_lead_side(self, junction_struct):
        # "First 4 Au atoms" = L electrode (not 'all Au' = both leads):
        sel = evaluate(FirstN(ByElement(("Au",)), 4), junction_struct)
        assert sel == frozenset([0, 1, 2, 3])

    def test_first_n_more_than_available(self, junction_struct):
        # n > matches: just returns the matches without error.
        sel = evaluate(FirstN(ByElement(("C",)), 99), junction_struct)
        assert sel == frozenset([4, 5, 6])

    def test_first_n_negative_raises(self, junction_struct):
        with pytest.raises(SelectionError, match="non-negative"):
            evaluate(FirstN(All(), -1), junction_struct)


# --------------------------------------------------------------------- #
#  Realistic composite expression                                       #
# --------------------------------------------------------------------- #


class TestComposites:
    def test_lead_picker_pattern(self, junction_struct):
        """"L electrode = first 4 Au atoms; R electrode = last 4 Au;
        bridge = everything else" -- the standard centred-junction
        recipe expressed as rule trees."""
        au = ByElement(("Au",))
        first_4_au  = FirstN(au, 4)
        not_first_4 = Minus(au, first_4_au)
        bridge      = Minus(All(), Or((first_4_au, not_first_4)))

        assert evaluate(first_4_au, junction_struct) == frozenset([0, 1, 2, 3])
        assert evaluate(not_first_4, junction_struct) == frozenset([7, 8, 9, 10])
        assert evaluate(bridge, junction_struct) == frozenset([4, 5, 6])


# --------------------------------------------------------------------- #
#  JSON round-trip                                                      #
# --------------------------------------------------------------------- #


class TestRoundTrip:
    @pytest.mark.parametrize("rule", [
        All(),
        ByElement(("Au", "C")),
        ByResidueName(("ALA", "GLY")),
        ByAtomName(("CA",)),
        ByChainId(("A", "B")),
        ByIndexRange("0-35, 100, 150-200"),
        ByRegion("L-electrode"),
        ByClick((3, 5, 8)),
        Or((ByElement(("Au",)), ByRegion("bridge"))),
        And((ByElement(("Au",)), ByIndexRange("0-5"))),
        Minus(All(), ByElement(("C",))),
        Not(ByElement(("Au",))),
        FirstN(ByElement(("Au",)), 4),
    ])
    def test_round_trip_equals_original(self, rule):
        assert from_json(to_json(rule)) == rule

    def test_nested_round_trip(self):
        rule = Or((
            FirstN(ByElement(("Au",)), 4),
            Minus(All(), Or((ByRegion("L-electrode"),
                             ByRegion("R-electrode")))),
        ))
        assert from_json(to_json(rule)) == rule

    def test_json_form_uses_op_key(self):
        d = to_json(ByElement(("Au",)))
        assert d["op"] == "by_element"
        assert d["elements"] == ["Au"]   # list, not tuple (JSON-able)

    def test_from_json_unknown_op_raises(self):
        with pytest.raises(SelectionError, match="unknown op"):
            from_json({"op": "no_such_op"})

    def test_from_json_missing_field_raises(self):
        with pytest.raises(SelectionError, match="missing required field"):
            from_json({"op": "by_element"})  # elements missing

    def test_from_json_non_dict_raises(self):
        with pytest.raises(SelectionError, match="expected dict"):
            from_json([1, 2, 3])
