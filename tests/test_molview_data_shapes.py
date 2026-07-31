"""The data structures and the per-frame calculation — every test derived from
``docs/web/molview.md``, never from the source it checks (§ 13).

Step C of the rebuild (``docs/web/molview-rework-plan.md``). The rows of § 13.3
guarded here:

    § 11.5 one translation, one place
    § 9.5  by atom index crosses the numbering boundary once
    § 6.2  the data holds what the filter enumerates
    § 6.6  MolView interprets no reserved label
    § 10.3 the two steps, in that order
    § 10.3 a label carries the original number
    § 10.3 frame f's arrows come from frame f
    § 10.3 the cell box and the axes are worked out once
    § 10.3 cell geometry and cell visibility travel separately
    § 6.5  the highlight is content, not styling
    § 6.5  the drawn-to-original map holds
    § 6.1  one frame is not a special case

Level 1 of § 13.2: behaviour with no browser — values in, values out.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests._node_esm import run_node
from tests._molview_sources import module_code

REPO = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO / "molbuilder" / "web" / "static" / "lib" / "molview"
ATOM = MODULE_DIR / "_atom.js"
ENGINE = MODULE_DIR / "render-engine.js"

PRELUDE = f"""
const ATOM = await import({json.dumps(ATOM.resolve().as_uri())});
const ENGINE = await import({json.dumps(ENGINE.resolve().as_uri())});

// Eight atoms in a line, so a coordinate names its own atom.
const ELEMENTS = ["C", "O", "N", "H", "C", "O", "N", "H"];
const FRAME = ELEMENTS.map((_, i) => [i, 0, 0]);
const OFF = {{ isolate: false, showIndex: false, showForces: false, forceScale: 1 }};
"""


def _run(snippet: str):
    return run_node([], PRELUDE + snippet)


# ---------------------------------------------------------------------------
# § 11.5 — one translation, one place
# ---------------------------------------------------------------------------

def test_the_first_atom_reads_as_one_and_the_translation_round_trips():
    """§ 11.5: 0-based in code, 1-based on screen, and the first atom reads as #1
    everywhere even though the code sees 0.
    """
    out = _run(
        """
        const roundTrip = [0, 1, 46, 399].every(i => ATOM.fromDisplay(ATOM.toDisplay(i)) === i);
        console.log(JSON.stringify({
            first: ATOM.toDisplay(0),
            typed: ATOM.fromDisplay(1),
            fortySeventh: ATOM.toDisplay(46),
            roundTrip,
        }));
        """
    )
    assert out["first"] == 1, "the first atom must read as #1 on screen"
    assert out["typed"] == 0, "a user typing 1 means the atom the code calls 0"
    assert out["fortySeventh"] == 47
    assert out["roundTrip"] is True


def test_a_typed_range_shifts_once_at_any_size():
    """§ 9.5: "a typed range like 1-4, 6 selects the atoms a user would count off
    on screen, at any structure size, without drifting by one — and the shift
    happens at one point, not at each row."
    """
    out = _run(
        """
        console.log(JSON.stringify({
            example:  ATOM.shiftExpression("1-4, 6, 10-11", -1),
            spacey:   ATOM.shiftExpression("  1 - 4 ,6", -1),
            big:      ATOM.shiftExpression("2000-4000", -1),
            back:     ATOM.shiftExpression(ATOM.shiftExpression("1-4, 6", -1), +1),
            unknown:  ATOM.shiftExpression("1-4, all", -1),
        }));
        """
    )
    assert out["example"] == "0-3, 5, 9-10"
    assert out["spacey"] == "0-3, 5", "whitespace must not change what is matched"
    assert out["big"] == "1999-3999", "the shift must not drift at any size"
    assert out["back"] == "1-4, 6", "the translation must be reversible"
    assert out["unknown"] == "0-3, all", (
        "a token this does not recognise passes through — the server validates "
        "the rule, and dropping it silently would hide a typo"
    )


def test_no_other_file_in_the_module_computes_its_own_shift():
    """§ 11.5: "MolView never writes a bare +1 of its own anywhere."

    The rule is what stops the panel, the labels and the readout drifting apart,
    so it has to be checked across the module rather than inside one file.
    """
    import re

    offenders = {}
    for name, code in module_code().items():
        if name == "_atom.js":
            continue                        # the single home
        # An index arithmetic literal: `something + 1` / `- 1` near an index name.
        hits = re.findall(r"\b(?:index|idx|atom|i|n)\s*[+\-]\s*1\b", code)
        if hits:
            offenders[name] = hits
    assert offenders == {}, (
        f"a bare +/-1 on an atom number outside the one translation: {offenders}"
    )


# ---------------------------------------------------------------------------
# § 6.2 / § 6.6 — the data holds what the filter enumerates
# ---------------------------------------------------------------------------

def test_the_enumerated_channels_are_exactly_what_an_atom_carries():
    """§ 6.2: "every property the filter enumerates from an atom — element,
    labels, residue — is a property the structure actually carries; neither list
    can grow without the other."
    """
    out = _run(
        """
        const annotations = [
            { labels: ["anchor"], residue: "ALA" },
            { labels: [], residue: "ALA" },
            { labels: ["anchor", "tip"] },
        ];
        const kinds = ATOM.channelKinds(["C", "O", "N"], annotations);
        const one = ATOM.atomChannels("C", annotations[0]);
        console.log(JSON.stringify({
            names: kinds.map(k => k.name),
            kinds: kinds.map(k => k.kind),
            one: one,
        }));
        """
    )
    assert out["names"] == ["element", "residue", "anchor", "tip"], (
        "the offered channels must be exactly element, residue and the labels "
        f"the atoms carry, in a stable order — got {out['names']}"
    )
    assert out["kinds"] == ["category", "category", "tag", "tag"]
    assert out["one"]["element"] == {"kind": "category", "value": "C"}, (
        "a category channel carries the value it matches on"
    )
    assert out["one"]["anchor"] == {"kind": "tag"}, (
        "a tag channel is membership, so it carries no value"
    )


def test_a_structure_with_no_residues_offers_no_residue_row():
    """§ 9.5: "which rows are worth offering is read from the structure, not
    hard-coded" — whether a by-residue row makes sense at all comes from looking
    at the atoms.
    """
    out = _run(
        """
        const kinds = ATOM.channelKinds(["C", "O"], [{ labels: [] }, { labels: [] }]);
        console.log(JSON.stringify({ names: kinds.map(k => k.name) }));
        """
    )
    assert out["names"] == ["element"], (
        f"a structure with no residues and no labels offers only element: {out['names']}"
    )


def test_a_reserved_label_is_an_ordinary_label():
    """§ 6.6: a reserved name is "stored, filtered and displayed exactly like any
    other label", and "no code here acts on the name".

    § 13.3: tagging atoms `frozen atoms` changes what is stored and nothing about
    what is drawn.
    """
    out = _run(
        """
        const annotations = [
            { labels: ["frozen atoms"] },
            { labels: ["my region"] },
        ];
        const kinds = ATOM.channelKinds(["C", "C"], annotations);
        const byName = {};
        for (const k of kinds) byName[k.name] = k.kind;

        // ...and it changes nothing about what is drawn.
        const plain = ENGINE.processFrame({
            elements: ["C", "C"], positions: [[0,0,0],[1,0,0]],
            selection: [], switches: OFF,
        });
        const tagged = ENGINE.processFrame({
            elements: ["C", "C"], positions: [[0,0,0],[1,0,0]],
            selection: [], switches: OFF,
        });
        console.log(JSON.stringify({
            byName,
            drawnSame: JSON.stringify(plain) === JSON.stringify(tagged),
        }));
        """
    )
    assert out["byName"]["frozen atoms"] == "tag", (
        "a reserved name must be a tag like any other label, with no kind of "
        f"its own: {out['byName']}"
    )
    assert out["byName"]["my region"] == out["byName"]["frozen atoms"], (
        "a reserved label and an ordinary one must be indistinguishable here"
    )
    assert out["drawnSame"] is True


def test_a_reserved_label_is_written_in_exactly_one_place():
    """§ 6.6: "A reserved meaning costs a NAME and a TRANSLATOR AT THE POINT OF
    USE — nothing else."

    The server has not caught up: it sends the labels and a separate `is_frozen`
    flag, so something must fold the flag into the labels for MolView's end to be
    "one mechanism, no special case". That fold is the one translator, and it
    lives at the inbound boundary.

    What the rule forbids is the ALTERNATIVE § 6.6 rejects — "its own field on
    the structure, its own kind of thing to filter by, its own key in the saved
    file, its own control in the panel". Every one of those is a second place
    naming it, so counting the places is the check.
    """
    import re

    written = {}
    for name, code in module_code().items():
        # The NAME being written — a string literal — not a reference to the
        # constant that holds it. § 9.3's table has a `getFrozen` door, so more
        # than one place legitimately USES the name; what costs is spelling it.
        literals = re.findall(r"""["'][^"'\n]*frozen[^"'\n]*["']""", code, re.I)
        if literals:
            written[name] = literals
    assert list(written) == ["model-jobs.js"], (
        "the reserved name must be SPELLED where the server's shape becomes this "
        f"module's, and nowhere else: {written}"
    )
    assert len(written["model-jobs.js"]) == 1, (
        "one name, written once, as a constant — a value repeated at each use is "
        f"how two spellings drift apart: {written['model-jobs.js']}"
    )


def test_no_code_acts_on_a_reserved_name():
    """§ 6.6: "It never ACTS on the meaning: no code here holds an atom still,
    and tagging atoms `frozen atoms` changes what is stored and nothing about
    what is drawn."

    Knowing a name is not interpreting it. What would be interpreting it is a
    branch that changes behaviour because of the name — so the check is that the
    per-frame calculation, which is the whole of what is drawn, cannot see it.
    """
    engine_code = "\n".join(
        line for line in (MODULE_DIR / "render-engine.js").read_text().splitlines()
        if not line.lstrip().startswith(("*", "//", "/*"))
    )
    assert "frozen" not in engine_code.lower(), (
        "the drawing calculation branches on a reserved label"
    )
    # And behaviourally: the same atoms, tagged and untagged, draw identically.
    out = _run(
        """
        function draw(annotations) {
            return JSON.stringify(ENGINE.processFrame({
                elements: ["C", "C"], positions: [[0,0,0],[1,0,0]],
                annotations, selection: [],
                switches: { isolate: false, showIndex: true, showForces: false, forceScale: 1 },
            }));
        }
        console.log(JSON.stringify({
            same: draw([{ labels: ["frozen_atoms"] }, { labels: [] }])
               === draw([{ labels: [] }, { labels: [] }]),
        }));
        """
    )
    assert out["same"] is True, (
        "tagging an atom with a reserved label changed what is drawn"
    )


# ---------------------------------------------------------------------------
# § 10.3 — the two steps, in that order
# ---------------------------------------------------------------------------

def test_the_isolate_cut_runs_before_the_overlays():
    """§ 13.3: "the isolate cut runs before the overlays, and the overlays are
    keyed to the atoms that survived it."

    Overlays computed first would be keyed to atoms that are no longer drawn.
    """
    out = _run(
        """
        const r = ENGINE.processFrame({
            elements: ELEMENTS, positions: FRAME,
            forces: FRAME.map(() => [1, 0, 0]),
            selection: [2, 5],
            switches: { isolate: true, showIndex: true, showForces: true, forceScale: 1 },
        });
        console.log(JSON.stringify({
            drawn: r.positions.length,
            sourceIndex: r.sourceIndex,
            labels: r.labels.length,
            arrows: r.arrows.length,
            labelText: r.labels.map(l => l.text),
            labelX: r.labels.map(l => l.position[0]),
        }));
        """
    )
    assert out["drawn"] == 2, "isolate keeps only the selected atoms"
    assert out["sourceIndex"] == [2, 5]
    assert out["labels"] == 2 and out["arrows"] == 2, (
        "the overlays are keyed to what survived the cut, not to the whole "
        f"structure: {out['labels']} labels, {out['arrows']} arrows for 2 atoms"
    )
    assert out["labelX"] == [2, 5], "a label sits on the atom it belongs to"


def test_a_label_carries_the_original_number_under_isolate():
    """§ 13.3: "under isolate, a drawn atom's label shows where it came from, not
    its position in the cut-down list."

    § 10.3: this is what lets a label still show #47 for an atom that is now
    third in the list.
    """
    out = _run(
        """
        const r = ENGINE.processFrame({
            elements: ELEMENTS, positions: FRAME,
            selection: [4, 6, 7],
            switches: { isolate: true, showIndex: true, showForces: false, forceScale: 1 },
        });
        console.log(JSON.stringify({ text: r.labels.map(l => l.text) }));
        """
    )
    assert out["text"] == ["5", "7", "8"], (
        "the drawn atoms came from original 4, 6 and 7, so their labels must "
        f"read 5, 7 and 8 — their position in the cut-down list would be 1, 2, 3: {out['text']}"
    )


def test_the_drawn_to_original_map_survives_the_cut():
    """§ 6.5: `sourceIndex[m]` is the original number of drawn atom `m` — "this
    map from drawn back to original is why labels still show the right number
    under isolate". Everything downstream depends on it existing.
    """
    out = _run(
        """
        const iso = ENGINE.processFrame({
            elements: ELEMENTS, positions: FRAME, selection: [7, 2, 2, 99, -1],
            switches: { isolate: true, showIndex: false, showForces: false, forceScale: 1 },
        });
        const all = ENGINE.processFrame({
            elements: ELEMENTS, positions: FRAME, selection: [],
            switches: OFF,
        });
        console.log(JSON.stringify({
            isoSource: iso.sourceIndex,
            isoElements: iso.elements,
            allSource: all.sourceIndex,
        }));
        """
    )
    assert out["isoSource"] == [2, 7], (
        "the map is the surviving atoms in original order, each once, with "
        f"out-of-range entries dropped: {out['isoSource']}"
    )
    assert out["isoElements"] == ["N", "H"], (
        "elements follow the cut, so drawn atom m's element is its own"
    )
    assert out["allSource"] == list(range(8)), (
        "with nothing cut the map is the identity — the same field, not a "
        "special case"
    )


def test_frame_f_arrows_come_from_frame_f():
    """§ 13.3: "arrows on a played trajectory match their own frame's forces."

    § 10.3: getting this wrong shows converged forces on an unconverged frame.
    """
    out = _run(
        """
        const frames = [[[0,0,0]], [[1,0,0]], [[2,0,0]]];
        const forcesPerFrame = [[[9,0,0]], [[3,0,0]], [[0.5,0,0]]];
        const processed = ENGINE.processFrames({
            elements: ["C"], frames, forcesPerFrame, selection: [],
            switches: { isolate: false, showIndex: false, showForces: true, forceScale: 1 },
        });
        console.log(JSON.stringify({
            starts: processed.map(p => p.arrows[0].start[0]),
            ends:   processed.map(p => p.arrows[0].end[0]),
        }));
        """
    )
    assert out["starts"] == [0, 1, 2], "an arrow starts at its own frame's atom"
    assert out["ends"] == [9, 4, 2.5], (
        "frame f's arrow must be frame f's force added to frame f's position — "
        f"got {out['ends']}, expected 0+9, 1+3, 2+0.5"
    )


def test_the_arrow_scale_stretches_the_arrow_and_moves_no_atom():
    """§ 6.5: `end = start + force x scale`. § 10.6: changing the scale re-derives
    arrows and re-bakes them in place, without touching the coordinates.
    """
    out = _run(
        """
        function at(scale) {
            return ENGINE.processFrame({
                elements: ["C"], positions: [[1,0,0]], forces: [[2,0,0]], selection: [],
                switches: { isolate: false, showIndex: false, showForces: true, forceScale: scale },
            });
        }
        const a = at(1), b = at(3);
        console.log(JSON.stringify({
            positions: [a.positions[0][0], b.positions[0][0]],
            ends:      [a.arrows[0].end[0], b.arrows[0].end[0]],
        }));
        """
    )
    assert out["positions"] == [1, 1], "the arrow scale moved an atom"
    assert out["ends"] == [3, 7], f"end = start + force x scale: {out['ends']}"


# ---------------------------------------------------------------------------
# § 6.5 — the highlight is content, not styling
# ---------------------------------------------------------------------------

def test_per_frame_data_carries_no_appearance():
    """§ 13.3: "per-frame data carries no colour, radius or opacity."

    What the highlight and the arrows LOOK like is a constant owned by the sealed
    layer; keeping it out is what keeps every frame's data identically shaped.
    """
    out = _run(
        """
        const processed = ENGINE.processFrames({
            elements: ELEMENTS, frames: [FRAME, FRAME],
            forcesPerFrame: [FRAME.map(() => [1,1,1]), FRAME.map(() => [2,2,2])],
            selection: [1, 3],
            switches: { isolate: false, showIndex: true, showForces: true, forceScale: 2 },
        });
        const blob = JSON.stringify(processed);
        const keys = new Set();
        (function walk(v) {
            if (Array.isArray(v)) return v.forEach(walk);
            if (v && typeof v === "object") {
                for (const k of Object.keys(v)) { keys.add(k); walk(v[k]); }
            }
        })(processed);
        console.log(JSON.stringify({ keys: Array.from(keys).sort(), blob }));
        """
    )
    forbidden = {"color", "colour", "radius", "opacity", "style", "background"}
    leaked = forbidden & set(out["keys"])
    assert leaked == set(), (
        f"per-frame data carries appearance: {leaked} (keys seen: {out['keys']})"
    )


def test_the_highlight_is_empty_under_isolate_and_when_nothing_is_picked():
    """§ 6.5: null means draw no highlight — "which happens both when nothing is
    selected and under isolate, where every drawn atom is selected and a
    highlight would say nothing."
    """
    out = _run(
        """
        function sel(switches, selection) {
            return ENGINE.processFrame({
                elements: ELEMENTS, positions: FRAME, selection, switches,
            }).selection;
        }
        const ON = { isolate: true,  showIndex: false, showForces: false, forceScale: 1 };
        console.log(JSON.stringify({
            nothingPicked: sel(OFF, []),
            picked:        sel(OFF, [1, 3]),
            isolated:      sel(ON,  [1, 3]),
        }));
        """
    )
    assert out["nothingPicked"] is None, "nothing selected means no highlight"
    assert out["picked"] == [1, 3], "with isolate off the highlight names the picked atoms"
    assert out["isolated"] is None, (
        "under isolate the drawn set already IS the selection, so a highlight "
        "would say nothing"
    )


# ---------------------------------------------------------------------------
# § 10.3 — the cell box and the axes are worked out once
# ---------------------------------------------------------------------------

def test_the_cell_box_and_axes_are_not_per_frame_data():
    """§ 6.5: "the cell box and the axes are NOT in here. They are the same for
    every frame unless the cell itself changes."

    § 13.3: they are not recomputed per frame, and playing a trajectory does not
    re-derive them.
    """
    out = _run(
        """
        const processed = ENGINE.processFrames({
            elements: ELEMENTS, frames: [FRAME, FRAME, FRAME], selection: [],
            switches: { isolate: false, showIndex: true, showForces: false, forceScale: 1 },
        });
        const keys = new Set();
        processed.forEach(p => Object.keys(p).forEach(k => keys.add(k)));
        console.log(JSON.stringify({ keys: Array.from(keys).sort() }));
        """
    )
    assert "cellBox" not in out["keys"] and "axes" not in out["keys"], (
        f"scene-level data leaked into the per-frame data: {out['keys']}"
    )
    assert out["keys"] == ["arrows", "elements", "labels", "positions",
                           "selection", "sourceIndex"], (
        f"the processed frame must be exactly § 6.5's fields: {out['keys']}"
    )


def test_cell_geometry_arrives_even_while_the_cell_is_hidden():
    """§ 10.3's callout: geometry is handed down unconditionally, even while the
    cell is hidden; the visibility switch carries only a boolean.

    "If geometry is gated behind the visibility switch, it only ever arrives
    while the cell is already shown — so turning the cell ON AFTER A HIDDEN LOAD
    draws the box from the world origin instead of the structure's corner."
    """
    out = _run(
        """
        const cell = { cell: [[4,0,0],[0,4,0],[0,0,4]], cell_origin: [10, 10, 10] };
        // Every switch off — the cell is hidden.
        const scene = ENGINE.sceneFor(cell);
        console.log(JSON.stringify({
            box: scene.cellBox,
            axisStarts: scene.axes.map(a => a.start),
            axisLabels: scene.axes.map(a => a.label),
        }));
        """
    )
    assert out["box"] is not None, (
        "the cell geometry was gated behind visibility — turning the cell on "
        "after a hidden load would draw the box at the world origin"
    )
    assert out["box"]["origin"] == [10, 10, 10], (
        "the anchor corner must be the structure's, not the world origin"
    )
    assert all(s == [10, 10, 10] for s in out["axisStarts"]), (
        f"the axes must be anchored to the cell's corner: {out['axisStarts']}"
    )
    assert out["axisLabels"] == ["a", "b", "c"], (
        "with a cell the triad follows the lattice vectors and is labelled a/b/c"
    )


def test_with_no_cell_there_is_no_box_and_the_triad_is_cartesian():
    """§ 6.2: `cell` is null when there is no cell. The triad still has a job —
    saying which way is which — so it falls back to x/y/z at the world origin.
    """
    out = _run(
        """
        const scene = ENGINE.sceneFor(null);
        console.log(JSON.stringify({
            box: scene.cellBox,
            labels: scene.axes.map(a => a.label),
            starts: scene.axes.map(a => a.start),
            ends: scene.axes.map(a => a.end),
        }));
        """
    )
    assert out["box"] is None, "no cell means no box to draw"
    assert out["labels"] == ["x", "y", "z"]
    assert all(s == [0, 0, 0] for s in out["starts"])
    assert out["ends"] == [[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]]


# ---------------------------------------------------------------------------
# § 6.1 — one frame is not a special case
# ---------------------------------------------------------------------------

def test_one_frame_goes_through_the_same_calculation_as_many():
    """§ 6.1: "no read, edit, export or save path treats a single frame
    differently from four hundred."
    """
    out = _run(
        """
        const sw = { isolate: false, showIndex: true, showForces: false, forceScale: 1 };
        const one  = ENGINE.processFrames({ elements: ELEMENTS, frames: [FRAME], selection: [1], switches: sw });
        const many = ENGINE.processFrames({
            elements: ELEMENTS, frames: [FRAME, FRAME, FRAME], selection: [1], switches: sw });
        console.log(JSON.stringify({
            oneCount: one.length,
            manyCount: many.length,
            sameShape: JSON.stringify(one[0]) === JSON.stringify(many[0]),
            allIdentical: many.every(f => JSON.stringify(f) === JSON.stringify(many[0])),
        }));
        """
    )
    assert out["oneCount"] == 1 and out["manyCount"] == 3
    assert out["sameShape"] is True, (
        "a one-frame structure produced a differently-shaped frame than a "
        "three-frame one"
    )
    assert out["allIdentical"] is True


def test_the_calculation_reads_no_drawing_setting():
    """§ 9.6 / § 6.2's note: ViewSettings is "held, but handed straight to the
    drawing — the frame calculation never reads it".

    § 13.3: changing style, radius, background or projection re-derives no frame.
    """
    out = _run(
        """
        const sw = { isolate: false, showIndex: true, showForces: false, forceScale: 1 };
        const plain = ENGINE.processFrame({ elements: ELEMENTS, positions: FRAME, selection: [1], switches: sw });
        // The same call with drawing settings smuggled into the switches.
        const dressed = ENGINE.processFrame({
            elements: ELEMENTS, positions: FRAME, selection: [1],
            switches: Object.assign({}, sw, {
                style: "sphere", radius: 9, background: "black", orthographic: true,
            }),
        });
        console.log(JSON.stringify({
            same: JSON.stringify(plain) === JSON.stringify(dressed),
        }));
        """
    )
    assert out["same"] is True, (
        "a drawing setting changed the frame calculation — style, radius, "
        "background and projection go straight to the drawing (§ 9.6)"
    )
