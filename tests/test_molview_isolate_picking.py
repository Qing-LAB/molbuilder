"""Under isolate the ruler still picks; the selection does not.

`molview.md` § 11.6, user 2026-08-31:

    *"in the isolation mode we do not allow 3dmol view to click and choose atoms
    for selection, but we should allow measurement to pick atoms — the only
    thing we need is a translation layer to map the displayed index to the
    original index such that the measurement results returns the correct atom
    index"*

The window used to drop **every** click while isolating, because the drawn
numbering is not the real one: isolate cuts the drawn list down to the
selection, so everything is renumbered.  The answer to a numbering problem is
the map, not a closed door — and the map already existed (`sourceIndex`, § 6.5,
the reason a label still reads #47 for an atom now drawn third).

Two things are pinned here, and they fail differently.  **The map**: one
definition, shared by the frame calculation and the click entry, because a
second copy would let a click measure the wrong atom while every frame still
drew correctly.  **The split rule**: measuring is allowed under isolate,
selecting is not — and the second is not a numbering matter at all, it is that
isolate draws only the selected atoms, so clicking one to toggle it would make
it vanish under the cursor.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests._node_esm import run_node

REPO = Path(__file__).resolve().parents[1]
ENGINE = REPO / "molbuilder" / "web" / "static" / "lib" / "molview" / "render-engine.js"

PRELUDE = f"""
const E = await import({json.dumps(ENGINE.resolve().as_uri())});
"""


def _js(snippet):
    return run_node([], PRELUDE + snippet)


class TestTheMap:
    """`sourceIndexFor` — drawn seat back to real atom."""

    def test_without_isolate_every_atom_keeps_its_number(self):
        out = _js("console.log(JSON.stringify("
                  "E.sourceIndexFor([1, 3], {isolate: false}, 5)));")
        assert out == [0, 1, 2, 3, 4]

    def test_under_isolate_only_the_selected_are_drawn_in_order(self):
        """Ascending and deduped: the drawn list is a cut-down structure, not a
        record of the order they were picked in."""
        out = _js("console.log(JSON.stringify("
                  "E.sourceIndexFor([7, 2, 2, 5], {isolate: true}, 10)));")
        assert out == [2, 5, 7], (
            "seat 0 is atom 2, seat 1 is atom 5, seat 2 is atom 7")

    def test_isolate_with_nothing_selected_draws_everything(self):
        """Isolate with an empty selection would leave an empty window, so the
        rule is `isolate AND something selected` -- the same condition the
        frame calculation uses."""
        out = _js("console.log(JSON.stringify("
                  "E.sourceIndexFor([], {isolate: true}, 3)));")
        assert out == [0, 1, 2]

    def test_out_of_range_picks_are_dropped(self):
        out = _js("console.log(JSON.stringify("
                  "E.sourceIndexFor([1, 99, -4], {isolate: true}, 5)));")
        assert out == [1]

    def test_the_frame_calculation_uses_THIS_map_and_not_its_own(self):
        """The reason it was lifted out.

        A second copy would disagree the day the isolate rule changed, and a
        click would measure the wrong atom while every frame still drew
        correctly -- silent, and only visible as numbers that are subtly wrong.
        """
        out = _js("""
            const sel = [7, 2, 5];
            const frame = E.processFrame({
                elements: ["C","N","O","F","Ne","Na","Mg","Al","Si","P"],
                positions: Array.from({length: 10}, (_, i) => [i, 0, 0]),
                selection: sel, switches: {isolate: true},
            });
            console.log(JSON.stringify({
                frame: frame.sourceIndex,
                direct: E.sourceIndexFor(sel, {isolate: true}, 10),
            }));
        """)
        assert out["frame"] == out["direct"] == [2, 5, 7]

    def test_the_drawn_atoms_are_the_ones_the_map_names(self):
        """Not a restatement: it pins that the map ORDERS the drawn arrays too,
        which is what makes seat n and `sourceIndex[n]` the same atom."""
        out = _js("""
            const frame = E.processFrame({
                elements: ["C","N","O","F","Ne"],
                positions: [[0,0,0],[1,0,0],[2,0,0],[3,0,0],[4,0,0]],
                selection: [3, 1], switches: {isolate: true},
            });
            console.log(JSON.stringify({
                src: frame.sourceIndex,
                els: frame.elements,
                xs:  frame.positions.map((p) => p[0]),
            }));
        """)
        assert out["src"] == [1, 3]
        assert out["els"] == ["N", "F"], "seat 0 is atom 1, which is N"
        assert out["xs"] == [1, 3]


class TestTheEngineTranslatesAClick:

    def _engine(self, body):
        return _js("""
            const embed = new Proxy({}, {get: () => () => {}});
            const engine = E.createRenderEngine
                ? E.createRenderEngine({embed: embed})
                : E.renderEngine({embed: embed});
            engine.setDataSource({
                structure: () => ({elements: ["C","N","O","F","Ne","Na","Mg","Al"]}),
                frames:    () => [[[0,0,0]]],
                forces:    () => null,
                frame:     () => 0,
                switches:  () => ({isolate: true}),
                selection: () => [5, 1, 3],
                measurement: () => ({active: true, picks: []}),
            });
        """ + body)

    def test_a_click_on_a_drawn_seat_comes_back_as_the_real_atom(self):
        out = self._engine(
            "console.log(JSON.stringify([0,1,2].map(engine.drawnToOriginal)));")
        assert out == [1, 3, 5], (
            "the window reports the seat; the atom is what must reach pickAtom")

    def test_a_seat_that_is_not_on_screen_is_refused(self):
        out = self._engine(
            "console.log(JSON.stringify(engine.drawnToOriginal(7)));")
        assert out is None, "null, so the caller drops it rather than guessing"
