"""MolView § 11.5 — one atom-numbering translation, in one place.

Derived from ``docs/web/molview.md``, not from the source.  The rule:

    Atom numbers are 0-based in code and 1-based on screen, and MolView never writes a bare
    ``+1`` of its own anywhere.  One shared piece of code owns the translation in BOTH
    directions ... Every surface that shows or accepts an atom number goes through it.

§ 13.3 turns that into: *every surface agrees with the shared translation; none computes its
own +1.*  The second half is the one worth thinking about, because a test that asserts
``label == index + 1`` cannot catch a hand-rolled ``+1`` — it agrees with the translation by
arithmetic coincidence.  What separates reuse from re-derivation is DRIFT: change the shared
translation and a surface that reuses it follows, while a surface that re-derived stays put.
So the drift test below moves the translation and asserts the label moved with it.
"""
from pathlib import Path

from _node_esm import run_node

ROOT = Path(__file__).resolve().parents[1]
MOLVIEW = ROOT / "molbuilder/web/static/lib/molview"
ATOM = MOLVIEW / "_atom.js"
ENGINE = MOLVIEW / "render-engine/engine.js"

_BOOT = f"""
    const atom   = await import("file://{ATOM}");
    const engine = await import("file://{ENGINE}");
    const processFrame = engine.process.processFrame;
"""


def _run(snippet: str) -> object:
    return run_node([], _BOOT + snippet)


def test_the_translation_runs_both_ways_and_is_its_own_inverse():
    """§ 11.5 — 0-based in code, 1-based on screen, and back again."""
    out = _run("""
        const N = 500;
        const forward = [], roundTrip = [];
        for (let i = 0; i < N; i++) {
            forward.push(atom.toDisplay(i));
            roundTrip.push(atom.fromDisplay(atom.toDisplay(i)));
        }
        console.log(JSON.stringify({
            firstAtomReadsAs: atom.toDisplay(0),
            forwardIsOffByOne: forward.every((v, i) => v === i + 1),
            roundTripIsIdentity: roundTrip.every((v, i) => v === i),
            typedInputComesBack: atom.fromDisplay(1),
        }));
    """)
    assert out["firstAtomReadsAs"] == 1, "the first atom must read as #1 on screen"
    assert out["forwardIsOffByOne"] is True
    assert out["roundTripIsIdentity"] is True
    assert out["typedInputComesBack"] == 0, "what a user types as 1 is atom 0 in code"


def test_a_typed_index_range_crosses_the_boundary_once():
    """§ 9.5 — the by-atom-index filter row is typed 1-based and sent 0-based."""
    out = _run("""
        console.log(JSON.stringify({
            shifted:   atom.shiftExpression("1-4, 6, 10-11", -1),
            unshifted: atom.shiftExpression("1-4, 6", 0),
            unknown:   atom.shiftExpression("Au, 3", -1),
            notAString: atom.shiftExpression(null, -1),
        }));
    """)
    assert out["shifted"] == "0-3, 5, 9-10", "every bound in the expression shifts, not just the first"
    assert out["unshifted"] == "1-4, 6"
    # An unrecognised token is left for the server to judge; only the numbers move.
    assert out["unknown"] == "Au, 2"
    assert out["notAString"] is None


def test_the_frame_labels_follow_the_translation_rather_than_re_deriving_it():
    """§ 11.5 — the drift test.  Move the shared translation; the label must move with it.

    A surface that reuses the translation reports the new number.  A surface that wrote its
    own ``+1`` reports the old one and is the drift this rule exists to prevent.
    """
    out = _run("""
        const COORDS = [[0,0,0],[1,0,0],[2,0,0]];
        const IDENTITY = { elements: ["C","N","O"] };
        const before = processFrame({ coords: COORDS }, IDENTITY, { showIndex: true });

        // Move the one translation.  Nothing else is touched.
        const original = atom.atomIndexModel.toDisplay;
        atom.atomIndexModel.toDisplay = (i) => i + 1000;
        const after = processFrame({ coords: COORDS }, IDENTITY, { showIndex: true });
        atom.atomIndexModel.toDisplay = original;

        console.log(JSON.stringify({
            before: before.labels.map((l) => l.text),
            after:  after.labels.map((l) => l.text),
        }));
    """)
    assert out["before"] == ["1", "2", "3"], "on screen the first atom is #1"
    assert out["after"] == ["1000", "1001", "1002"], (
        "the atom-number labels re-derived their own +1 instead of going through the "
        "shared translation — § 11.5's single home is not real"
    )
