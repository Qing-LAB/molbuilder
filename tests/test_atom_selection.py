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
        """`All` selects EVERY atom, which is what makes `Minus(All(), x)` a true
        complement.

        If this drifts (an off-by-one on `n`, an empty set), every rule built as
        "everything except ..." -- the bridge of a centred junction, the standard
        lead-picker recipe below -- silently loses or gains atoms, and the deck is
        written for the wrong ones. Grammar: `selection.py`; the 0-based internal
        index is `model/overview.md` § 2.
        """
        assert evaluate(All(), junction_struct) == frozenset(range(11))

    def test_by_element_single(self, junction_struct):
        """Selecting one element returns every atom of it, wherever it sits.

        `ByElement(("Au",))` must reach BOTH electrodes -- a matcher that stopped at
        the first contiguous run would give a junction one lead and nothing
        downstream would notice, because every consumer just takes the set it is
        handed. `by_element` is also the kind name the JS filter translates to
        (`structure-annotations.md` § 6).
        """
        # All Au atoms across both electrodes:
        sel = evaluate(ByElement(("Au",)), junction_struct)
        assert sel == frozenset([0, 1, 2, 3, 7, 8, 9, 10])

    def test_by_element_multi(self, junction_struct):
        """Every element in the tuple matches, not just the first.

        `elements` is a tuple because the filter panel lets a person tick several
        species at once. If the evaluator compared against `rule.elements[0]`, the
        single-species tests around it stay green while a two-species filter returns
        half its atoms. Contract: `selection.py` -- `ByElement` is membership in
        `elements`.
        """
        sel = evaluate(ByElement(("Au", "C")), junction_struct)
        assert sel == frozenset(range(11))

    def test_by_element_no_match(self, junction_struct):
        """An element the structure does not contain selects NOTHING, not everything.

        The inverse of an `in` test is `not in`, and that one-word mutation makes a
        filter for an absent species select the whole structure -- which, composed
        into `Minus(All(), ...)` or into a frozen-atom label, freezes or tags every
        atom in the cell. Contract: `selection.py` `ByElement`.
        """
        assert evaluate(ByElement(("Pt",)), junction_struct) == frozenset()

    def test_by_residue_name(self, junction_struct):
        """A PDB residue name selects that residue's atoms and only those.

        `residue_names` is one of four parallel per-atom columns (element / residue
        name / atom name / chain id). A rule reading the wrong column returns a
        plausible-looking set of the WRONG atoms, and nothing downstream can tell --
        the measured form of that is recorded on the `/api/selection/eval` row of
        `web-api.md` (2026-09-07).
        """
        sel = evaluate(ByResidueName(("BRG",)), junction_struct)
        assert sel == frozenset([4, 5, 6])

    def test_by_atom_name(self, junction_struct):
        """`ByAtomName` matches the atom-name column, not the element symbol.

        MEASURED 2026-09-07 (`web-api.md`, the `/api/selection/eval` row): when the
        wire payload carried no `atomName`, `Structure`'s default put the ELEMENT
        SYMBOL in that column, so `by_atom_name "CA"` never matched an alpha carbon
        and the endpoint answered 200 with the wrong atoms rather than refusing.
        This is the L1 half of that contract: the column is read, and it is this one.
        """
        sel = evaluate(ByAtomName(("AuL",)), junction_struct)
        assert sel == frozenset([0, 1, 2, 3])

    def test_by_chain_id(self, junction_struct):
        """`ByChainId` matches the chain column, and a chain selects only its own atoms.

        The sibling half of the same 2026-09-07 defect (`web-api.md`,
        `/api/selection/eval`): with no `chainId` on the wire every atom defaulted to
        chain "A", so `by_chain_id "B"` matched nothing and "A" matched everything --
        both wrong answers delivered as 200.
        """
        sel = evaluate(ByChainId(("R",)), junction_struct)
        assert sel == frozenset([7, 8, 9, 10])

    def test_by_region(self, junction_struct):
        """A region label selects exactly the atoms tagged with it.

        Region labels are what tell a transport calculation which atoms are the left
        electrode, which are the bridge, and which are held frozen
        (`structure-annotations.md` § 5). A `ByRegion` returning the wrong
        membership emits an `.fdf` whose electrodes are the wrong atoms -- a
        calculation that runs to completion and answers a different physical
        question.
        """
        sel = evaluate(ByRegion("bridge"), junction_struct)
        assert sel == frozenset([4, 5, 6])

    def test_by_region_unknown_raises(self, junction_struct):
        """A misspelt region label REFUSES; it never quietly selects nothing.

        `L_electrode` vs `L-electrode` is one keystroke, and an empty selection is
        indistinguishable from "no atoms carry that label" -- so without the raise a
        typo produces a junction with no electrode instead of an error. The
        evaluator lists the labels it does know, and `/api/selection/eval` turns the
        `SelectionError` into a 400 (`web/blueprints/selection.py`), which is the
        only reason the browser ever sees the typo.
        """
        with pytest.raises(SelectionError, match="no region labelled"):
            evaluate(ByRegion("L_electrode"), junction_struct)

    def test_by_index_range_single_range(self, junction_struct):
        """"0-3" selects atoms 0,1,2,3 -- INCLUSIVE at both ends, 0-based.

        Both conventions this could drift to are silent: a half-open upper bound
        drops the last atom of every electrode, and 1-based parsing shifts the whole
        selection by one atom. `model/overview.md` § 2 fixes the internal base
        at 0; the JS filter does the 1-based-to-0-based shift before the rule gets
        here (`structure-annotations.md` § 6).
        """
        sel = evaluate(ByIndexRange("0-3"), junction_struct)
        assert sel == frozenset([0, 1, 2, 3])

    def test_by_index_range_multi_token(self, junction_struct):
        """Comma-separated tokens mix ranges and bare indices, and all of them land.

        The frozen-atom index list is written in this grammar
        (`selection.py::ByIndexRange`), so a parser honouring only the first token
        would freeze a fraction of the atoms a person listed -- the run proceeds and
        the geometry relaxes where it was meant to be held fixed.
        """
        sel = evaluate(ByIndexRange("0-3, 7, 9-10"), junction_struct)
        assert sel == frozenset([0, 1, 2, 3, 7, 9, 10])

    def test_by_index_range_empty(self, junction_struct):
        """An empty expression selects nothing, and is NOT an error.

        The panel's index box is empty until someone types in it, and every keystroke
        round-trips through `/api/selection/eval`. If the empty string were a refusal
        the filter card would show an error for its own resting state; if it were
        `All()` it would flash the entire structure as selected.
        """
        assert evaluate(ByIndexRange(""), junction_struct) == frozenset()

    def test_by_index_range_out_of_bounds_raises(self, junction_struct):
        """An index past the last atom refuses, instead of clamping to a shorter set.

        The expression is written against ONE structure; re-evaluated after atoms are
        deleted, silent clamping hands back a set that means something different from
        what was written. `ByIndexRange` bounds against `len(struct.elements)` --
        which is also what makes the early exit in `And` observable, see
        `test_and_short_circuits_on_empty`.
        """
        with pytest.raises(SelectionError, match="out of"):
            evaluate(ByIndexRange("0-99"), junction_struct)

    def test_by_index_range_reversed_raises(self, junction_struct):
        """"5-2" is refused, not read as an empty range.

        `range(5, 3)` is empty in Python, so without the explicit `hi < lo` check a
        transposed pair selects NOTHING and reads as "those atoms are not there"
        rather than "you typed it backwards".
        """
        with pytest.raises(SelectionError, match="hi < lo"):
            evaluate(ByIndexRange("5-2"), junction_struct)

    def test_by_index_range_malformed_raises(self, junction_struct):
        """A token that is neither `<int>` nor `<int>-<int>` refuses, naming the token.

        The expression arrives from a text box; `_RANGE_TOKEN_RE.fullmatch` is what
        stops `abc` (or `0..3`, or a stray comma) becoming a silently empty
        selection. `SelectionError` is also the one exception type
        `/api/selection/eval` converts to a 400 -- anything else escapes as a 500.
        """
        with pytest.raises(SelectionError, match="invalid token"):
            evaluate(ByIndexRange("abc"), junction_struct)

    def test_by_click(self, junction_struct):
        """Clicked indices are carried through verbatim, in the viewer's own numbering.

        A click set is the one selection with no rule behind it: it is stored as data
        so the panel can round-trip it through `/api/selection/eval` and get the same
        atoms back. Renumbering here would move the highlight off the atom the person
        clicked. 0-based, per `model/overview.md` § 2.
        """
        sel = evaluate(ByClick((3, 5, 8)), junction_struct)
        assert sel == frozenset([3, 5, 8])

    def test_by_click_out_of_range_raises(self, junction_struct):
        """A click index the structure cannot have refuses, rather than being dropped.

        The click set is captured against the structure in the viewer; if the
        structure has since shrunk, silently discarding the out-of-range members
        hands back a partial selection that looks deliberate. The bound check runs
        over EVERY index before any of them is returned.
        """
        with pytest.raises(SelectionError, match="out of range"):
            evaluate(ByClick((42,)), junction_struct)


# --------------------------------------------------------------------- #
#  Boolean operators                                                     #
# --------------------------------------------------------------------- #


class TestBooleanOps:
    def test_or_two_operands(self, junction_struct):
        """`Or` is set UNION -- an atom selected by any operand is in.

        Union and intersection differ by one character in the evaluator and both
        return a subset of the structure, so a swapped operator gives a smaller,
        entirely plausible selection. "All the carbon plus the first two golds" is
        what a person composes in the panel; under intersection it selects nothing.
        """
        sel = evaluate(
            Or((ByElement(("C",)), ByIndexRange("0-1"))),
            junction_struct,
        )
        assert sel == frozenset([0, 1, 4, 5, 6])

    def test_or_empty_is_empty_set(self, junction_struct):
        """An `Or` with no operands is EMPTY -- the lattice identity for union.

        The panel builds an `Or` from ticked rows, so the no-rows state reaches the
        evaluator on every page load. Empty-as-`All()` would mean an untouched filter
        card selects the whole structure, the opposite of what the person sees.
        Paired with `test_and_empty_is_full_set`, which pins the other identity.
        """
        assert evaluate(Or(()), junction_struct) == frozenset()

    def test_and_two_operands(self, junction_struct):
        """`And` is set INTERSECTION, and that is how one lead of a junction is named.

        "Au AND in the first half" expresses "the left electrode" without hard-coding
        indices. If `And` unioned, that expression selects both leads and the bridge
        -- and the calculation still runs, on the wrong atoms.
        """
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
        """Once the intersection is empty, the remaining operands are NOT evaluated.

        Asserted through a side effect rather than a timing measurement: the third
        operand is `ByIndexRange("0-99")` against an 11-atom structure, which RAISES
        if it is ever reached. So deleting `if not acc: break` in `_evaluate` turns
        this test into a `SelectionError`, not a slow pass. Without the early exit, a
        composed filter whose left side already matched nothing reports a typo in a
        clause that can no longer affect the answer.
        """
        # Doesn't crash on a third operand even though intersection
        # is already empty after two:
        sel = evaluate(
            And((ByElement(("Pt",)), ByElement(("Au",)),
                 ByIndexRange("0-99"))),
            junction_struct,
        )
        assert sel == frozenset()

    def test_minus(self, junction_struct):
        """`Minus(a, b)` is asymmetric: `a` without `b`, never `b` without `a`.

        "Everything except the bridge" is how both electrodes are named in one
        expression. A `Minus` that subtracted the wrong way round returns the bridge
        -- exactly the atoms that must NOT be treated as electrode -- and every step
        after it accepts the answer.
        """
        # Everything except the bridge:
        sel = evaluate(Minus(All(), ByRegion("bridge")), junction_struct)
        assert sel == frozenset([0, 1, 2, 3, 7, 8, 9, 10])

    def test_not_is_complement(self, junction_struct):
        """`Not(x)` is the complement over the WHOLE structure, i.e. `Minus(All(), x)`.

        `Not` is the compact spelling the panel emits for an inverted filter row, and
        this is its ONLY evaluation test: if it drifted to complementing within the
        operand's own span, an inverted filter would return a subset of what it was
        meant to invert. Design note, recorded rather than hidden: this asserts the
        two FORMS agree instead of naming the atoms, so a mutation breaking `All()`
        and `Not` in the same way stays green.
        """
        # Not(ByElement('Au')) should equal Minus(All(), ByElement('Au'))
        au = ByElement(("Au",))
        assert (evaluate(Not(au), junction_struct)
                == evaluate(Minus(All(), au), junction_struct))

    def test_first_n_picks_lead_side(self, junction_struct):
        """`FirstN` takes the first n in ASCENDING INDEX order -- which is what makes
        "first 12 Au = left electrode" mean the left electrode.

        The evaluator returns a `frozenset`, which has no order, so the sort has to
        happen before the slice. Slice an unsorted iteration and the "left electrode"
        becomes n arbitrary gold atoms scattered through the junction -- a transport
        calculation that runs and answers nonsense.
        """
        # "First 4 Au atoms" = L electrode (not 'all Au' = both leads):
        sel = evaluate(FirstN(ByElement(("Au",)), 4), junction_struct)
        assert sel == frozenset([0, 1, 2, 3])

    def test_first_n_more_than_available(self, junction_struct):
        """Asking for more than exist returns what exists, and does not refuse.

        `FirstN(ByElement(("C",)), 99)` is what a saved rule looks like after atoms
        were deleted from the structure it was written against
        (`structure-molstruct.md` § 4 keeps the rules for re-evaluation);
        refusing would make such a sidecar unloadable rather than degrade. The
        mechanism is a Python slice, so what this really pins is the DECISION not to
        bound `n` against the match count.
        """
        # n > matches: just returns the matches without error.
        sel = evaluate(FirstN(ByElement(("C",)), 99), junction_struct)
        assert sel == frozenset([4, 5, 6])

    def test_first_n_negative_raises(self, junction_struct):
        """A negative n refuses instead of slicing from the end.

        `ordered[:-1]` is every atom but the last -- so without the explicit check
        `FirstN(rule, -1)` returns an almost-complete selection where the person
        asked for something meaningless, and the mistake surfaces as a wrong
        electrode rather than as a message.
        """
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
        """Every rule class survives `to_json` -> `from_json` unchanged, INCLUDING its
        equality and hash.

        `selection_rules` is persisted in the `.molstruct.json` sidecar so a labelled
        selection can be re-evaluated later (`structure-molstruct.md` § 4). The
        codec is table-driven -- `_RULE_CLASSES`, `_SUBRULE_FIELDS`,
        `_SUBRULE_LIST_FIELDS` -- so a class added without registering it, or a leaf
        list not normalised back to a tuple, breaks re-evaluation of an ALREADY
        SAVED rule long after the commit that caused it. Equality on frozen
        dataclasses is what makes the tuple normalisation observable at all.
        """
        assert from_json(to_json(rule)) == rule

    def test_nested_round_trip(self):
        """A rule tree nested two levels deep round-trips as well as a flat one.

        Guards the recursion in `to_json` / `from_json` at a depth the parametrized
        cases above do not reach (they nest once). RECORDED DOUBT for the section 3b
        review: the codec has no depth-dependent branch -- it recurses through the
        same two tables at every level -- so this may be entirely subsumed by the
        `Or` and `Minus` cases of `test_round_trip_equals_original`.
        """
        rule = Or((
            FirstN(ByElement(("Au",)), 4),
            Minus(All(), Or((ByRegion("L-electrode"),
                             ByRegion("R-electrode")))),
        ))
        assert from_json(to_json(rule)) == rule

    def test_json_form_uses_op_key(self):
        """The wire form is `{"op": "<kind>", ...}` with JSON-able leaf values.

        A CROSS-LANGUAGE contract, which is why it is asserted rather than left to
        review: `_filterToRule` in `_selection-store-impl.js` builds these dicts by
        hand (`structure-annotations.md` § 6 -- `by_element` to `by_element`,
        `by_index` to `by_index_range`, `by_residue` to `by_residue_name`,
        `by_label` to `by_region`), and nothing in Python fails when the two sides
        disagree: the browser simply gets a 400 "unknown op" for a filter it drew
        correctly. The list-not-tuple half is what keeps the dict JSON-encodable.
        """
        d = to_json(ByElement(("Au",)))
        assert d["op"] == "by_element"
        assert d["elements"] == ["Au"]   # list, not tuple (JSON-able)

    def test_from_json_unknown_op_raises(self):
        """An unknown `op` raises `SelectionError` -- the ONE exception type the route
        turns into a 400.

        `_load_rule_from_payload` (`web/blueprints/selection.py`) catches
        `SelectionError` and nothing else, so a `KeyError` or `TypeError` from here
        reaches the browser as a 500 with a stack trace. The rule tree comes straight
        off the wire, so a stale JS build sending a retired kind name is the ordinary
        case, not the adversarial one.
        """
        with pytest.raises(SelectionError, match="unknown op"):
            from_json({"op": "no_such_op"})

    def test_from_json_missing_field_raises(self):
        """A rule dict missing a required field refuses, instead of constructing a rule
        from defaults.

        `cls(**kwargs)` on a dataclass whose field has no default raises `TypeError`,
        which the route does not catch (500). And a field that DID gain a default
        would silently build `ByElement(())` -- a rule matching nothing -- from a
        filter the person filled in.
        """
        with pytest.raises(SelectionError, match="missing required field"):
            from_json({"op": "by_element"})  # elements missing

    @pytest.mark.parametrize("rule,why", [
        ({"op": "or", "operands": 5},                     "a sub-rule LIST is a scalar"),
        ({"op": "by_element", "elements": 5},             "a leaf sequence is a scalar"),
        ({"op": "by_element", "elements": "Au"},          "a leaf sequence is a bare string"),
        ({"op": "by_element", "elements": [1]},           "a str element is a number"),
        ({"op": "by_click", "indices": ["x"]},            "an int index is a string"),
        ({"op": "by_click", "indices": [True]},           "an int index is a bool"),
        ({"op": "first_n", "rule": {"op": "all"}, "n": "x"},   "an int field is a string"),
        ({"op": "first_n", "rule": {"op": "all"}, "n": True},  "an int field is a bool"),
        ({"op": "by_region", "name": 5},                  "a str field is a number"),
    ])
    def test_from_json_refuses_a_leaf_of_the_wrong_type(self, rule, why):
        """A malformed leaf is refused HERE, as a `SelectionError`, and never
        reaches `evaluate` as a bare `TypeError`.

        MEASURED DEFECT (#67, 2026-09-09). `web/blueprints/selection.py:127`
        catches `SelectionError` and only that. Three of these built a rule
        successfully and then raised `TypeError` deep inside the evaluator --
        `tuple(from_json(r) for r in raw)` at `selection.py:461`,
        `set(rule.elements)` at `:228`, `rule.n < 0` at `:306` -- so the route
        answered Flask's HTML **500 page with a stack trace** where
        `web/web-api.md` § 1 requires a JSON 400. The filter panel round-trips
        a rule on EVERY KEYSTROKE in the index box, so this was a live path.

        A bool is refused where an int is wanted because `bool` is an `int`
        subclass: `True` as an index would silently evaluate to atom 1.
        """
        with pytest.raises(SelectionError):
            from_json(rule)

    def test_from_json_still_accepts_every_well_formed_leaf(self):
        """The type gate above must not narrow what a valid rule may say.

        The other half of #67: a check that refuses the malformed is only worth
        having if it passes everything the rules legitimately hold -- a list or
        a tuple for a sequence, and an empty one, which several callers build.
        """
        assert from_json({"op": "by_element", "elements": ["Au", "S"]}).elements \
            == ("Au", "S")
        assert from_json({"op": "by_element", "elements": []}).elements == ()
        assert from_json({"op": "by_click", "indices": [0, 3]}).indices == (0, 3)
        assert from_json({"op": "first_n", "rule": {"op": "all"}, "n": 0}).n == 0
        assert from_json({"op": "by_region", "name": "lead"}).name == "lead"

    def test_from_json_non_dict_raises(self):
        """A non-object `rule` on the wire refuses cleanly rather than crashing the
        route.

        `"op" not in payload` happens to work for a list and a string but raises
        `TypeError` for a number, so without the `isinstance(payload, dict)` guard
        `{"rule": 5}` answers 500 instead of 400. Same door as the two above:
        `web/blueprints/selection.py` converts `SelectionError`, and only
        `SelectionError`.
        """
        with pytest.raises(SelectionError, match="expected dict"):
            from_json([1, 2, 3])
