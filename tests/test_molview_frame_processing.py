"""MolView § 10.3 / § 6.5 — the two per-frame steps, and what comes out of them.

Derived from ``docs/web/molview.md``, not from the source.  § 10.3, for every frame:

    STEP 1 — keep only what is shown ... keep only those atoms, renumber them, and record
    where each came from.   STEP 2 — add the overlays, keyed to the atoms that survived step 1.

and § 6.5, on the result:

    ``selection`` here is content, not styling.  It says *which* atoms are highlighted.  What
    the highlight looks like is a fixed constant owned by the sealed layer ... Keeping the
    appearance out of the per-frame data is what keeps every frame's data identically shaped.

§ 13.3 rows guarded here: the two steps in that order · a label carries the original number ·
frame *f*'s arrows come from frame *f* · the highlight is content, not styling · the cell box
and the axes are not per-frame.
"""
from pathlib import Path

from _node_esm import run_node

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "molbuilder/web/static/lib/molview/render-engine/engine.js"

# Four atoms in a row, so a drawn index and an original index are easy to tell apart.
_BOOT = f"""
    const engine = await import("file://{ENGINE}");
    const processFrame = engine.process.processFrame;
    const COORDS   = [[0,0,0],[1,0,0],[2,0,0],[3,0,0]];
    const IDENTITY = {{ elements: ["C","N","O","H"] }};
"""


def _run(snippet: str) -> object:
    return run_node([], _BOOT + snippet)


def test_selecting_alone_draws_every_atom():
    """§ 6.2 — only selection AND isolate change which atoms are drawn at all."""
    out = _run("""
        const pf = processFrame({ coords: COORDS }, IDENTITY,
                                { selection: [1, 3], isolate: false });
        console.log(JSON.stringify({ positions: pf.positions, sourceIndex: pf.sourceIndex,
                                     elements: pf.elements, selection: pf.selection }));
    """)
    assert out["positions"] == [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]
    assert out["sourceIndex"] == [0, 1, 2, 3], "nothing was cut, so drawn number == original"
    assert out["elements"] == ["C", "N", "O", "H"]
    assert out["selection"] == [1, 3], "the highlight names the drawn atoms to mark"


def test_isolate_cuts_first_and_records_where_each_atom_came_from():
    """§ 10.3 step 1 — cut down, renumber, and keep the map back."""
    out = _run("""
        // Selected out of order on purpose: the drawn set is in ORIGINAL order regardless.
        const pf = processFrame({ coords: COORDS }, IDENTITY,
                                { selection: [3, 1], isolate: true });
        console.log(JSON.stringify({ positions: pf.positions, sourceIndex: pf.sourceIndex,
                                     elements: pf.elements, selection: pf.selection }));
    """)
    assert out["positions"] == [[1, 0, 0], [3, 0, 0]]
    assert out["sourceIndex"] == [1, 3], "the drawn->original map is what everything else needs"
    assert out["elements"] == ["N", "H"], "the overlays are keyed to what survived step 1"
    assert out["selection"] is None, (
        "under isolate every drawn atom IS the selection, so a highlight would say nothing")


def test_a_label_carries_the_original_number_not_its_place_in_the_cut_down_list():
    """§ 10.3 step 2 — the atom-number label is recovered through step 1's map."""
    out = _run("""
        const pf = processFrame({ coords: COORDS }, IDENTITY,
                                { selection: [2, 3], isolate: true, showIndex: true });
        console.log(JSON.stringify({
            text: pf.labels.map((l) => l.text),
            at:   pf.labels.map((l) => l.position),
        }));
    """)
    # Atoms 2 and 3 are drawn 0th and 1st.  On screen they must still read #3 and #4.
    assert out["text"] == ["3", "4"], (
        "the label showed its position in the cut-down list — under isolate that is a "
        "different atom's number")
    assert out["at"] == [[2, 0, 0], [3, 0, 0]], "a label sits on the atom it names"


def test_no_highlight_when_nothing_is_picked():
    """§ 6.5 — `null` means draw no highlight, which is not the same as an empty list."""
    out = _run("""
        const pf = processFrame({ coords: COORDS }, IDENTITY, { selection: [], isolate: false });
        console.log(JSON.stringify({ selection: pf.selection }));
    """)
    assert out["selection"] is None


def test_the_per_frame_data_says_which_atoms_never_what_they_look_like():
    """§ 6.5 — the highlight is content, not styling; and so are the arrows.

    The whole point of the constraint: every frame's data is identically shaped, and the
    appearance lives in one place (the sealed layer) instead of being re-sent per frame.
    """
    out = _run("""
        const FORCES = [[0,1,0],[0,2,0],[0,0,0],[0,0.5,0]];
        const pf = processFrame({ coords: COORDS, forces: FORCES }, IDENTITY,
                                { selection: [1], isolate: false, showIndex: true,
                                  showForces: true, forceScale: 1 });
        const styleWords = ["color", "colour", "radius", "opacity", "style", "width"];
        const found = [];
        const scan = (obj, where) => {
            if (!obj || typeof obj !== "object") return;
            for (const k of Object.keys(obj)) {
                if (styleWords.includes(k.toLowerCase())) found.push(where + "." + k);
            }
        };
        (pf.arrows || []).forEach((a, i) => scan(a, "arrows[" + i + "]"));
        (pf.labels || []).forEach((l, i) => scan(l, "labels[" + i + "]"));
        scan(pf, "frame");
        console.log(JSON.stringify({
            styling: found,
            arrowKeys: [...new Set((pf.arrows || []).flatMap((a) => Object.keys(a)))].sort(),
            frameKeys: Object.keys(pf).sort(),
        }));
    """)
    assert out["styling"] == [], (
        f"per-frame data carries appearance: {out['styling']} — § 6.5 says the sealed layer "
        f"owns what things look like, so this is re-sent on every frame and can drift")
    assert out["arrowKeys"] == ["end", "start"], "an arrow is where it starts and where it ends"
    # § 6.5's table, exactly: nothing more travels down per frame.
    assert out["frameKeys"] == ["arrows", "elements", "labels", "positions", "selection",
                                "sourceIndex"]


def test_the_cell_and_the_axes_are_not_per_frame():
    """§ 10.3 — they are scene-level, worked out once, identical for every frame."""
    out = _run("""
        const pf = processFrame({ coords: COORDS }, IDENTITY,
                                { showCell: true, showAxis: true });
        console.log(JSON.stringify({ keys: Object.keys(pf).sort() }));
    """)
    assert "cell" not in out["keys"] and "cellBox" not in out["keys"]
    assert "axes" not in out["keys"], (
        "recomputing the axes per frame produces an identical answer four hundred times")


def test_an_arrow_ends_where_the_force_takes_it():
    """§ 6.5 — `end = start + force × scale`, at the atom it belongs to."""
    out = _run("""
        const FORCES = [[0,1,0],[0,0,0],[0,0,2],[0,0,0]];
        const at = (scale) => processFrame({ coords: COORDS, forces: FORCES }, IDENTITY,
                                           { showForces: true, forceScale: scale }).arrows;
        console.log(JSON.stringify({ unit: at(1), doubled: at(2) }));
    """)
    # Atoms 1 and 3 carry no force: a zero vector draws no arrow at all.
    assert out["unit"] == [{"start": [0, 0, 0], "end": [0, 1, 0]},
                           {"start": [2, 0, 0], "end": [2, 0, 2]}]
    assert out["doubled"] == [{"start": [0, 0, 0], "end": [0, 2, 0]},
                              {"start": [2, 0, 0], "end": [2, 0, 4]}]


def test_frame_f_gets_frame_f_s_forces():
    """§ 10.3 — getting this wrong shows converged forces on an unconverged frame."""
    out = _run("""
        const early = [[0,4,0],[0,0,0],[0,0,0],[0,0,0]];   // far from converged
        const late  = [[0,0.01,0],[0,0,0],[0,0,0],[0,0,0]];
        const one = (forces) => processFrame({ coords: COORDS, forces }, IDENTITY,
                                             { showForces: true, forceScale: 1 }).arrows[0].end;
        console.log(JSON.stringify({ early: one(early), late: one(late) }));
    """)
    assert out["early"] == [0, 4, 0]
    assert out["late"] == [0, 0.01, 0], "the late frame drew the early frame's forces"


def test_forces_belong_to_the_atoms_that_survived_the_cut():
    """§ 10.3 step 2 — the overlays are keyed to step 1's output, not the original list."""
    out = _run("""
        const FORCES = [[9,0,0],[0,1,0],[0,0,0],[0,0,3]];
        const pf = processFrame({ coords: COORDS, forces: FORCES }, IDENTITY,
                                { selection: [1, 3], isolate: true,
                                  showForces: true, forceScale: 1 });
        console.log(JSON.stringify({ arrows: pf.arrows }));
    """)
    # Atom 0's large force is hidden with atom 0; the drawn arrows start on atoms 1 and 3.
    assert out["arrows"] == [{"start": [1, 0, 0], "end": [1, 1, 0]},
                             {"start": [3, 0, 0], "end": [3, 0, 3]}]


def test_no_forces_no_arrows():
    """§ 10.3 — the overlay is produced when the switch is on AND the data carried forces."""
    out = _run("""
        console.log(JSON.stringify({
            switchOff:  processFrame({ coords: COORDS, forces: [[0,1,0]] }, IDENTITY,
                                     { showForces: false }).arrows,
            noForces:   processFrame({ coords: COORDS }, IDENTITY, { showForces: true }).arrows,
            allZero:    processFrame({ coords: COORDS, forces: [[0,0,0],[0,0,0],[0,0,0],[0,0,0]] },
                                     IDENTITY, { showForces: true }).arrows,
        }));
    """)
    assert out["switchOff"] is None
    assert out["noForces"] is None
    assert out["allZero"] is None, "every force suppressed means no overlay, not empty arrows"
