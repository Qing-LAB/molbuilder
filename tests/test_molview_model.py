"""The model — every test derived from ``docs/web/molview.md``, never from the
source it checks (§ 13).

The rows of ``docs/web/molview.md`` § 13.3 guarded here:

    § 6.4  nothing keeps its own copy / master copy, then range, then frame, then notify
    § 9.3  a read cannot be used to write
    § 9.3  one need, one main way in
    § 9.3  the facts a request carries are read together
    § 9.3  one read holds everything a request needs
    § 9.3  a trajectory arrives whole, in one install
    § 10.8 the append doors refuse what they cannot honour
    § 9.4  read-only freezes the core data and nothing else
    § 9.4  one write into an empty viewer, and delivery stays open
    § 9.4  a replacement is refused by default, never impossible
    § 9.4  a read-only viewer has no history
    § 9.6  the camera is fitted on load and on Reset, and at no other moment
    § 6.3  each question goes to the copy that can answer it
    § 6.1  one frame is not a special case
    § 11.1 an edit reaches the route it names
    § 11.1 the count requirement is checked first
    § 11.1 an empty selection means what the table says
    § 11.1 a failed edit changes nothing (its structure half in
           test_molview_refusal; its history half here, in
           test_an_edit_records_itself_and_a_failed_one_records_nothing)

and the rules of § 6.7a (`clear`, the door that means start empty) and
§ 8.4a (the person's metadata edits; the two flags a recorded structure
carries), which § 13.3 states no row for.

THE SERVER IS THE APP'S OWN (`_node_esm.served`, `molview.md` § 13.1): each
structure a pair on disk, written as the road writes one (`support.road`),
opened by path through `/api/build/load`; an edit goes to `/api/modify/*`, a
cell to `/api/structure/periodicity`, and a refused edit is one the route
really refuses; a workspace a test hands in is the real one
(`lib/workspace/dispatcher.js`), in one browser tab on Node's own storage
(`tab_storage`), its points read back through its own door.  Frames and forces
a run delivers come through the doors that deliver them, computed from the
atoms the viewer holds.  The one stand-in is a drawing that records what it is
told: the drawing library needs WebGL, which node cannot run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from support.molview_lane import HOST, KEPT, WS
from support.road import CO2, CO2_HELD_FIRST, H2, WATER
from tests._node_esm import run_node, served, tab_storage

REPO = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO / "molbuilder" / "web" / "static" / "lib" / "molview"
# The pipeline's scene derivation, so a test can ask the DRAWING what cell it
# would draw and compare it with what the Cell page reports (§ 5.2).
ENGINE = MODULE_DIR / "render-engine.js"

#: THE ROAD'S CO2, LABELLED: an ``anchor`` on the first oxygen beside the
#: held carbon, so a label and the reserved one both ride it.  Saved with an
#: origin assigned at :data:`CORNER` by the Cell page's own op
#: (`periodicity_gate.apply_edit`, ``box_corner``, as the road's
#: `water_with_an_assigned_origin` does), so every read of the cell has
#: something to hand out.
LABELLED = dict(CO2, regions={"anchor": [0], **CO2["regions"]})
#: The box corner a person set on it.
CORNER = [1.0, 1.0, 1.0]


def _labelled_assigned():
    """:data:`LABELLED` with :data:`CORNER` set on the Cell page, as its op
    sets it -- the structure saved, and what it states (its offset)."""
    from molbuilder.periodicity_gate import apply_edit
    from support.road import road_structure
    return apply_edit(road_structure(LABELLED), "box_corner", CORNER)[0]


def _moved(spec, frames: int) -> list:
    """``spec``'s atoms at ``frames`` geometries, frame ``f`` moved ``0.01 f``
    Å -- what a run's frames of that structure look like."""
    at = np.asarray(spec["positions"], dtype=float)
    return [(at + 0.01 * f).tolist() for f in range(frames)]


@pytest.fixture(scope="module")
def app_at(isolated_projects_root_module):
    """The app, serving: what answers the viewer and the workspace
    (`support.live_server`)."""
    from support.live_server import serve
    with serve() as base:
        yield base


@pytest.fixture(scope="module")
def pairs(isolated_projects_root_module):
    """The structures the tests open, by name -- the road's own
    (`support.road`), each saved as the road saves one (`structure_saved`)
    and opened by path: its H2 (its first atom held), its water, its CO2
    labelled with its origin assigned (:data:`LABELLED`), water as a
    molecule that states no box (`water_in_vacuum` -- the box the server's
    own), and H2 as frame sets of two and five frames
    (`Structure.with_frames`, the door the road's frame sets go through;
    a frame set opens read-only, § 9.4); its CO2 as a SIESTA vibration's
    decks hold it, its order record on it (`co2_in_engine_order`); and a
    peptide the one build door builds (`builders.build`, the Build tab's
    `/api/build/molecule`) -- its identity columns the builder's own --
    carrying the two kinds of per-atom channel a script sets through the
    Structure API (`Structure.set_channel`, :data:`CHANNELS`)."""
    from molbuilder.builders import build
    from molbuilder.structure import AtomChannel
    from support.road import (co2_in_engine_order, road_structure,
                              structure_saved, water_in_vacuum)
    root = isolated_projects_root_module

    def saved(name, s):
        return str(root / structure_saved(root, s, at=f"P/structure/{name}.xyz"))

    def framed(spec, n):
        return road_structure(spec).with_frames(np.array(_moved(spec, n)))
    peptide = build("peptide", "GA").structure
    for name, (kind, data, said) in CHANNELS.items():
        peptide.set_channel(name, AtomChannel(kind, data, **said))
    named = {"H2": saved("h2", H2), "water": saved("water", WATER),
             "CO2 labelled": saved("co2-labelled", _labelled_assigned()),
             "no box stated": saved("water-no-box", water_in_vacuum()),
             "H2, 2 frames": saved("h2-2-frames", framed(H2, 2)),
             "H2, 5 frames": saved("h2-5-frames", framed(H2, 5)),
             "peptide": saved("peptide-ga", peptide)}
    # AN ENGINE'S RECORD IN ITS OWN ORDER has one door, the codec's
    # ``engine_record`` (`model/overview.md` § 2.2b): a pair is in the user
    # order.
    from molbuilder.workingcopy_structure import StructureCodec
    record = root / "P" / "structure" / "co2-engine-order.xyz"
    StructureCodec().write(co2_in_engine_order(), record, engine_record=True)
    return dict(named, **{"CO2, engine order": str(record)})


#: TWO PER-ATOM CHANNELS, one of each kind (`model/structure-annotations.md`):
#: a tag on atoms 0 and 2 with the colour it is drawn in, and a value on atoms
#: 1 and 2 with the fdf strategy it asks for -- name to (kind, data, the
#: channel's own facts).
CHANNELS = {"spin_up": ("tag", [0, 2], {"color": "#f00"}),
            "charge": ("value", {1: -0.4, 2: 0.1}, {"fdf": "some-strategy"})}


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """THE ROAD'S H2 CARRYING THE RECORD A RUN OF IT LEAVES (`web/molview.md`
    § 8.4a): the calculation described on the road and its first stage
    prepped (`jobset prep`), and the record a run of that deck answers for
    itself (`run_info`, `info.calculation`: the contract the deck states, and
    its `source`) put on the structure by the run's own view (`view_of`,
    `RunView.onto`); the pair saved in the calculation's
    tree, which the app serves for this test (`live_server.serve`: the root
    is read at call time).  NOT RUN: a basic test reaches no engine (the
    suite's tripwire), and the record is the deck's, which prep writes -- a
    run's output would add the `relaxation` block, which these tests do not
    read.  Its path."""
    from support.road import (described, jobset, road_structure, road_tree,
                              stage_file, structure_saved)
    tree = road_tree(tmp_path, monkeypatch)
    bundle = described(tree, stage_strategy="")
    r = jobset("prep", "task", "--group", "1", "--bundle", bundle,
               "--target", "this")
    assert r.exit_code == 0, r.output
    from molbuilder.runs import run_of, view_of
    s = view_of(run_of(stage_file(bundle, "coarse", ".fdf"))).onto(
        road_structure(H2))
    assert s.info["calculation"]["contract"], s.info
    return str(tree / structure_saved(tree, s,
                                      at="P/structure/h2-recorded.xyz"))


def _prelude(pairs) -> str:
    return f"""
const {{ createModel }} = await import({json.dumps(HOST.as_uri())});
const ENGINE = await import({json.dumps(ENGINE.resolve().as_uri())});
await import({json.dumps(KEPT.resolve().as_uri())});
await import({json.dumps(WS.resolve().as_uri())});
/* The workspace a page hands its viewer (workspace.md § 5): the real one,
 * its files on the app's own routes, this run's browser tab its own. */
const ws = globalThis.molbuilder.workspace;
const PAIRS = {json.dumps(pairs)};

/* A structure opened as a tab opens one -- by path, the server reading the
 * pair -- and the wire quiet after it: the load answered and, in an editable
 * viewer handed a workspace, point 0 written. */
async function opened(m, name, extra) {{
    const got = await m.installMolecule(
        Object.assign({{ path: PAIRS[name] }}, extra || {{}}));
    await __quiet();
    return got;
}}

/* A viewer holding `name`, the wire's record starting after it. */
async function loaded(name, opts) {{
    const m = createModel(opts || {{}});
    await opened(m, name);
    __calls.length = 0;
    __answered.length = 0;
    return m;
}}

/* What a run delivers for the atoms a viewer holds: `frame` moved `dz` Å
 * along z, and a force of `fz` along z on each of its atoms. */
const shifted = (frame, dz) => frame.map((p) => [p[0], p[1], p[2] + dz]);
const pushed = (frame, fz) => frame.map(() => [0, 0, fz]);

/* The points the workspace holds for `tag`, read back through its own door. */
async function pointsOf(tag) {{
    const laid = [];
    for (let k = 0; k < 8; k += 1) {{
        if (await ws.readState(tag, k) !== null) laid.push(k);
    }}
    return laid;
}}
"""


def _run(app_at, pairs, snippet: str, **given):
    """The test's script on the page, the app's routes answering and the
    tab's own storage under it; ``given`` -- values a test worked out from
    the road's structures -- as constants of the same names."""
    consts = "".join(f"const {k} = {json.dumps(v)};\n" for k, v in given.items())
    return run_node([], _prelude(pairs) + consts + snippet,
                    globals_js=tab_storage() + served(app_at), storage=True)


# ---------------------------------------------------------------------------
# § 9.3 — a read cannot be used to write
# ---------------------------------------------------------------------------

def test_changing_what_a_read_returned_leaves_the_viewer_untouched(app_at, pairs):
    """§ 9.3: "Every read of data returns a COPY, so changing what you were given
    can never change the viewer."

    The road's H2, its first atom held: the label a read hands out is one the
    atom really carries.
    """
    out = _run(app_at, pairs, """
        const m = await loaded("H2");
        const s = m.getStructure();
        const periodicityBefore = JSON.stringify(s.periodicity);
        s.elements[0] = "XX";
        s.annotations[0].labels.push("smuggled");
        s.periodicity = { cell: [[9,0,0],[0,9,0],[0,0,9]] };

        const coords = m.getCoordinates();
        coords.frames[0][0][0] = 999;

        const frame = m.getFrameAllAtoms(0);
        frame[0][0] = -999;

        const after = m.getStructure();
        console.log(JSON.stringify({
            element: after.elements[0],
            labels: after.annotations[0].labels,
            periodicityKept: JSON.stringify(after.periodicity) === periodicityBefore,
            x: m.getFrameAllAtoms(0)[0][0],
        }));
        """)
    held = [name for name, atoms in H2["regions"].items() if 0 in atoms]
    assert out["element"] == H2["elements"][0], (
        "a read handed out the model's own array")
    assert out["labels"] == held, "a read handed out the model's own labels"
    assert out["periodicityKept"], "a read handed out a writable structure"
    assert out["x"] == H2["positions"][0][0], (
        "a read handed out the model's own coordinates")


def test_no_read_at_all_can_be_written_through(app_at, pairs):
    """The same rule as above, asked of EVERY read rather than of three.

    § 13.1 forbids a pinned list of names — "a transcription, not a contract" —
    so the reads are enumerated from the surface at run time and each one's
    result is mutated as destructively as its shape allows. What is asserted is
    the RULE: after all of that, the structure and the coordinates are what they
    were.

    The defect it guards: `getUnitCellInfo` — the MAIN way in for the cell
    (§ 9.3's table) — handed out live references into the master copy.
    """
    out = _run(app_at, pairs, """
        // A structure WITH A CELL AND AN ASSIGNED ORIGIN, because the cell
        // reads are the ones this was written for — and with nothing to hand
        // out, they hand out nothing and the test proves nothing.
        const m = await loaded("CO2 labelled");
        if (!m.getUnitCellInfo().cell) throw new Error("the fixture has no cell");
        if (!m.getUnitCellOrigin()) throw new Error("the fixture has no assigned origin");

        const before = JSON.stringify({
            structure: m.getStructure(), coordinates: m.getCoordinates(),
        });

        // Mutate anything reachable: arrays get an extra entry and a changed
        // first element, objects get every value overwritten, recursively.
        function vandalise(value, depth) {
            if (depth > 4 || value == null) return;
            if (Array.isArray(value)) {
                value.push("smuggled");
                if (value.length > 1) {
                    if (typeof value[0] === "number") value[0] = -999;
                    else if (typeof value[0] === "string") value[0] = "XX";
                    else vandalise(value[0], depth + 1);
                }
                return;
            }
            if (typeof value === "object") {
                for (const key of Object.keys(value)) {
                    const held = value[key];
                    if (held && typeof held === "object") vandalise(held, depth + 1);
                    else value[key] = (typeof held === "number") ? -999 : "XX";
                }
            }
        }

        const reads = Object.keys(m).filter((name) =>
            typeof m[name] === "function"
            && (name.startsWith("get") || name === "exportFile"));
        for (const name of reads) {
            let got;
            try { got = m[name](0); } catch (_) { continue; }
            vandalise(got, 0);
        }

        const after = JSON.stringify({
            structure: m.getStructure(), coordinates: m.getCoordinates(),
        });
        console.log(JSON.stringify({ reads, unchanged: before === after }));
        """)
    assert len(out["reads"]) >= 10, (
        f"the enumeration found almost nothing — it is not exercising the "
        f"surface: {out['reads']}"
    )
    assert out["unchanged"] is True, (
        "one of the model's reads handed out something that writes through to "
        f"the master copy; the reads tried were {out['reads']}"
    )


def test_with_nothing_loaded_a_read_returns_nothing_not_an_empty_structure(
        app_at, pairs):
    """§ 9.3: "there is nothing here" and "here is a structure with no atoms" are
    different answers, and a caller has to be able to tell them apart.

    An empty structure walks straight past every `if (!data) return` guard.
    """
    out = _run(app_at, pairs, """
        const m = createModel({});
        console.log(JSON.stringify({
            structure: m.getStructure(),
            atoms: m.getAtoms(),
            elements: m.getElements(),
            coordinates: m.getCoordinates(),
            frame: m.getFrameAllAtoms(0),
            regions: m.getRegions(),
            exported: m.exportFile(),
            count: m.frameCount(),
        }));
        """)
    for name in ("structure", "atoms", "elements", "coordinates", "frame",
                 "regions", "exported"):
        assert out[name] is None, (
            f"{name} answered with an empty structure instead of nothing"
        )
    assert out["count"] == 0


def test_a_narrower_cut_cannot_disagree_with_the_main_way_in(app_at, pairs):
    """§ 13.3: "a narrower cut returns exactly what the main way in holds for
    that field — the two cannot disagree."

    § 9.3: a cut may disappear, but it must never grow into a rival.

    The road's CO2, an ``anchor`` and its held carbon among its labels, in its
    box with an origin assigned; a run's frame delivered after it, so the
    coordinate cuts have more than one frame to agree about.
    """
    out = _run(app_at, pairs, """
        const m = await loaded("CO2 labelled");

        m.addFrames([shifted(m.getFrameAllAtoms(0), 0.1)]);

        const whole = m.getStructure();
        console.log(JSON.stringify({
            elementsAgree: JSON.stringify(m.getElements()) === JSON.stringify(whole.elements),
            atomsAgree: m.getAtoms().map(a => a.element).join() === whole.elements.join(),
            cellAgree: JSON.stringify(m.getUnitCell()) === JSON.stringify(whole.periodicity.cell),
            // An ASSIGNED origin is read as the corner the server put the box
            // at for it -- `box_corner`, never worked out here (§ 9.3).
            originAgree: JSON.stringify(m.getUnitCellOrigin()) === JSON.stringify(whole.periodicity.box_corner),
            // The coordinate cuts, against the same one read. `getCoordinates`
            // is listed as a cut of `getStructure` (§ 9.3), so the whole has to
            // hold what it returns — it did not, and that was the hole.
            framesAgree: JSON.stringify(m.getCoordinates().frames)
                         === JSON.stringify(whole.frames),
            forcesAgree: JSON.stringify(m.getCoordinates().forcesPerFrame)
                         === JSON.stringify(whole.forcesPerFrame),
            oneFrameAgrees: JSON.stringify(m.getFrameAllAtoms(1))
                            === JSON.stringify(whole.frames[1]),
            regions: m.getRegions(),
            frozen: m.getFrozen(),
            infoCell: m.getUnitCellInfo().cell,
        }));
        """)
    assert out["elementsAgree"] and out["atomsAgree"], (
        "a cut of the structure disagreed with the whole"
    )
    assert out["cellAgree"] and out["originAgree"]
    assert out["framesAgree"] and out["forcesAgree"], (
        "getCoordinates is a cut of getStructure (§ 9.3), so the whole must hold "
        "what it returns — a cut that answers something the main way in does not "
        "have is a rival, not a cut"
    )
    assert out["oneFrameAgrees"], (
        "getFrameAllAtoms(i) disagreed with the same frame in the whole read"
    )
    assert out["regions"]["anchor"] == LABELLED["regions"]["anchor"], (
        "labels group into name -> atoms")
    assert out["frozen"] == LABELLED["regions"]["frozen_atoms"], (
        "the frozen cut reads the reserved label off the same one mechanism"
    )
    assert out["infoCell"] == np.diag(LABELLED["cell"]).tolist(), (
        "the cell as it will be used must agree with the raw cell when the "
        "structure states one — the resolved value only fills in what was left "
        "unsaid (§ 9.3)"
    )


def test_the_cell_page_and_the_drawing_cannot_describe_different_structures(
        app_at, pairs):
    """§ 5.2 and § 9.3: "the cell as it will actually be used" is one fact, so
    the panel that reports it and the pipeline that draws it must read the same
    answer.

    They did not. `getUnitCellInfo` read the RESOLVED values and `sceneFor` read
    the RAW ones, so for a structure with no explicit cell — every plain `.xyz` —
    the Cell page listed a lattice while the drawing had none and "Show unit
    cell" drew nothing. Two readers of one fact, disagreeing, with no error
    anywhere.

    The structure is one nobody gave a cell or an origin to (Automatic): the
    road's water as a molecule comes, stating no box.  What the server worked
    out for it is read off the route's own answer (``__answered``): its raw
    fields empty, the box it resolved beside them.
    """
    out = _run(app_at, pairs, """
        const m = createModel({});
        await opened(m, "no box stated");
        const served = __answered.at(-1);

        const panel = m.getUnitCellInfo();          // what the Cell page shows
        const drawn = ENGINE.sceneFor(m.getStructure().periodicity);   // what is drawn

        console.log(JSON.stringify({
            meta: served.structure.metadata,
            box: served.resolved_box,
            panelCell:  panel.cell,
            drawnCell:  drawn.cellBox && drawn.cellBox.lattice,
            panelOrigin: panel.box_corner,
            drawnOrigin: drawn.cellBox && drawn.cellBox.origin,
            // The RAW reads still say what the structure itself states, which is
            // nothing — that is their job (§ 9.3) and it is not a disagreement.
            rawCell: m.getUnitCell(),
            rawOrigin: m.getUnitCellOrigin(),
        }));
        """)
    meta, box = out["meta"], out["box"]
    assert meta["cell"] is None and meta["engine_offset"] is None, \
        "the fixture states a box: it is not Automatic"
    assert box["resolved_cell"] and box["box_corner"], \
        "Automatic resolves a box for atoms (§ 9.3: it always has an answer)"
    assert out["panelCell"] == box["resolved_cell"], (
        "the Cell page lost the cell the structure actually uses"
    )
    assert out["drawnCell"] == out["panelCell"], (
        f"the drawing and the Cell page describe different cells: drawn "
        f"{out['drawnCell']} vs panel {out['panelCell']}"
    )
    assert out["drawnOrigin"] == out["panelOrigin"] == box["box_corner"], (
        f"the box is not anchored where the panel says it is: {out}"
    )
    assert out["rawCell"] is None, (
        "the raw read must still report what the structure itself states — "
        "nothing — or it has stopped being the narrower cut § 9.3 describes"
    )
    assert out["rawOrigin"] is None, (
        "Automatic states no origin: the box is drawn at the server's corner, "
        "but nobody assigned one (§ 9.3, `getUnitCellOrigin`)"
    )


def test_what_the_viewer_is_has_one_answer_and_a_replacement_can_be_enforced(
        app_at, pairs):
    """§ 5.2, applied to the viewer's own lifecycle.

    "Has this viewer got its core data?" is ONE question, answered by one
    state, checked and set in one place, so coming back to the install door
    never finds the initial state a second time.

    And no state is a dead end. A read-only viewer refuses to have its structure
    swapped out from under it, but the HOST can say it means to: deciding which
    structure a viewer shows is the host's business, and the host is the one that
    asked for read-only. What read-only protects is the core data from being
    EDITED (§ 9.4) — `applyOp`, the cell, the labels — and those stay shut
    whatever is passed here.

    The viewer does not judge what it was handed: a structure with no atoms is a
    structure with no atoms, and carrying it is the module's job (§ 6.2).
    """
    out = _run(app_at, pairs, """
        const m = createModel({ mode: "readonly" });
        await opened(m, "H2");                  // leaves EMPTY
        const held = m.getAtoms().length;

        // A second install is refused: not a dead end, a default.
        const casual = await opened(m, "water");
        const afterCasual = m.getAtoms().length;

        // Said deliberately, it lands.
        const enforced = await opened(m, "water", { enforce: true });
        const afterEnforced = m.getAtoms().map((a) => a.element);

        // And editing is still shut, enforce or not.
        __calls.length = 0;
        await m.applyOp("translate", { dx: 1 });
        m.selection.adopt([0]);
        await m.selection.writeLabel("smuggled", "replace");

        console.log(JSON.stringify({
            held, casual, afterCasual,
            enforced: enforced !== null, afterEnforced,
            labels: m.getAtoms()[0].labels,
            requests: __calls.map((c) => c.url),
        }));
        """)
    assert out["held"] == len(H2["elements"]), (
        "the first install is allowed in any mode")
    assert out["casual"] is None and out["afterCasual"] == len(H2["elements"]), (
        "a read-only viewer let its structure be swapped without being asked"
    )
    assert out["enforced"] is True and out["afterEnforced"] == WATER["elements"], (
        f"an enforced install did not land: {out}"
    )
    assert out["labels"] == [], (
        "editing reached the core data in a read-only viewer"
    )
    assert out["requests"] == [], (
        "an edit left for the network in a read-only viewer"
    )


def test_every_door_answers_what_the_read_only_table_says_and_none_of_them_throws(
        app_at, pairs):
    """§ 9.4's per-door table, pinned as values rather than as prose.

    "It returns without effect AND WITHOUT THROWING" is the rule, not a
    courtesy: a viewer that threw would make every caller wrap its writes, which
    is the list of special cases § 9.4 exists to avoid. So each swallowed door is
    checked for BOTH — that nothing happened, and that the caller got an answer
    it can test rather than an exception.

    The answers differ by door (`null`, `false`, `undefined`) because they are
    each that call's own "no", not one invented for the gate.

    The viewer is handed the real workspace, as the Results tab's is; each
    door is asked with what its control would send -- the Translate box, the
    Cell page's lattice, the Label box, a run's frames.
    """
    out = _run(app_at, pairs, """
        const m = createModel({ mode: "readonly", owner: "ro", workspace: ws });
        await opened(m, "H2");
        const f0 = m.getFrameAllAtoms(0);
        m.addFrames([shifted(f0, 0.1)]);         // delivery: runs
        m.selection.adopt([0]);

        const before = JSON.stringify({
            structure: m.getStructure(), frames: m.frameCount(),
        });
        const threw = [];
        const answered = {};
        async function door(name, fn) {
            // `undefined` does not survive JSON, and it is one of the answers
            // the table names — so it is spelled out rather than dropped.
            try {
                const got = await fn();
                answered[name] = (got === undefined) ? "undefined" : got;
            } catch (e) { threw.push(name + ": " + e.message); }
        }

        await door("installOver",  () => m.installMolecule({ path: PAIRS["water"] }));
        await door("applyOp",      () => m.applyOp("translate", { dx: 1 }));
        await door("cell",         () => m.commitPeriodicityOp("cell", [[9,0,0],[0,9,0],[0,0,9]]));
        await door("writeLabel",   () => m.selection.writeLabel("smuggled", "replace"));
        await door("reloadFrames", () => m.reloadFrames([shifted(f0, 0.5)]));
        await door("save",         () => m.save(1));
        await door("load",         () => m.load(0));
        await door("undo",         () => m.undo());
        // The count guard sits INSIDE the door, so a misshapen reload that the
        // gate already answered must not raise either.
        await door("reloadMisshapen", () => m.reloadFrames([f0.slice(0, 1)]));
        await __quiet();

        console.log(JSON.stringify({
            threw, answered, mode: m.mode,
            unchanged: before === JSON.stringify({
                structure: m.getStructure(), frames: m.frameCount() }),
        }));
        """)
    assert out["threw"] == [], (
        f"a read-only no-op threw instead of answering: {out['threw']}"
    )
    assert out["unchanged"] is True, "a swallowed door changed something anyway"
    assert out["mode"] == "readonly"
    # Each door's own "no" — the values § 9.4's table names.
    assert out["answered"] == {
        "installOver":     None,
        "applyOp":         None,
        "cell":            None,
        "writeLabel":      False,
        "reloadFrames":    "undefined",
        "save":            False,
        "load":            None,
        "undo":            None,
        "reloadMisshapen": "undefined",
    }, f"a door answered something other than § 9.4's table says: {out['answered']}"


def test_the_reads_answer_nothing_when_nothing_is_loaded_except_the_counts(
        app_at, pairs):
    """§ 9.3's surface table, at the one moment it is easiest to get wrong.

    "With nothing loaded, a read returns NOTHING rather than an empty structure"
    — but three entries are deliberately not `null`, and a caller has to be able
    to rely on that: the two counts are counts, and `mode` is what the viewer IS
    rather than what it holds. `getUnitCellInfo` is the fourth: it is "the cell
    as it will actually be used... so it ALWAYS has an answer", which means an
    object with empty fields, not nothing.
    """
    out = _run(app_at, pairs, """
        const m = createModel({});
        const reads = {};
        for (const name of ["getStructure", "getAtoms", "getElements",
                            "getCoordinates", "getRegions", "getFrozen",
                            "exportFile", "getUnitCell", "getUnitCellOrigin",
                            "getAxisKind", "getVacuum"]) {
            reads[name] = m[name]();
        }
        m.setCurrentFrame(5);      // not gated, and there is nothing to move to
        console.log(JSON.stringify({
            reads,
            frame: m.getFrameAllAtoms(0),
            cellInfo: m.getUnitCellInfo(),
            counts: [m.frameCount(), m.currentFrame()],
            history: [m.state_index, m.uncommitted],
            mode: m.mode,
        }));
        """)
    for name, value in out["reads"].items():
        assert value is None, (
            f"{name} answered with an empty structure instead of nothing: {value}"
        )
    assert out["frame"] is None
    assert out["cellInfo"] == {"cell": None, "box_corner": None,
                               "axis_kind": None, "vacuum": None}, (
        "the cell as it will be used must ALWAYS have an answer — an object with "
        f"empty fields, never nothing: {out['cellInfo']}"
    )
    assert out["counts"] == [0, 0], (
        f"the counts are counts, not nothing, and a seek finds nowhere to go: "
        f"{out['counts']}"
    )
    assert out["history"] == [0, False]
    assert out["mode"] == "editable", "mode is what the viewer IS, not what it holds"


def test_appending_is_the_same_in_both_modes_but_replacing_is_not(app_at, pairs):
    """The line is REWRITE versus APPEND, not "coordinates versus everything
    else".

    `addFrame`, `addFrames` and `setForces` only EXTEND: after them, frame 0 is
    still exactly what the run produced, and § 10.8 forbids an arriving frame
    from carrying different atoms. Nothing already in the master copy is altered,
    so they are the same in both modes — an editable viewer follows a running job
    exactly as a read-only one does.

    `reloadFrames` is the odd one out and belongs with the edits: it REPLACES
    every coordinate and can shrink the trajectory, so after it frame 0 need not
    be what the calculation produced at all. In a read-only viewer that is
    refused unless the caller says it means it — the same word, for the same
    reason, as replacing the structure outright (§ 9.4).

    None of them raises the unsaved badge. It means "there is work here that is
    not on the sequence yet" (§ 11.2), and a run's own output is not the user's
    work: it is reproducible from the run, and a poll arriving every few seconds
    would otherwise flicker the badge continuously. Only an edit raises it.

    THE BADGE IS WATCHED, NOT SAMPLED: an edit lays its own point (§ 11.2),
    and with the real workspace that point is sent within the edit, so the
    badge an edit raises is down again by the time the call returns.  A
    subscriber is told of every change of it (`onBadge`), so what each one
    saw is the measure -- a flicker included.
    """
    out = _run(app_at, pairs, """
        const answer = {};
        for (const mode of ["editable", "readonly"]) {
            const m = createModel(mode === "readonly"
                ? { mode: "readonly", owner: "ro", workspace: ws }
                : { owner: "ed", workspace: ws });
            await opened(m, "H2");
            const badge = [];
            m.subscribe(() => badge.push(m.uncommitted));

            // APPENDING — identical either way.
            const f0 = m.getFrameAllAtoms(0);
            m.addFrames([shifted(f0, 1), shifted(f0, 2)],
                        { forces: [pushed(f0, 1), null] });
            m.addFrame(shifted(f0, 3));
            m.setForces([null, null, null, pushed(f0, 4)]);

            answer[mode] = {
                frames: m.frameCount(),
                frameZero: m.getFrameAllAtoms(0),
                forces: m.getCoordinates().forcesPerFrame,
                badgeSeen: badge.slice(),
            };

            // REPLACING — an edit-shaped act, so read-only wants to be asked.
            m.reloadFrames([shifted(f0, 99)]);
            answer[mode].afterCasualReload = m.frameCount();
            m.reloadFrames([shifted(f0, 99)], { enforce: true });
            answer[mode].afterEnforcedReload = m.frameCount();
            await __quiet();
        }

        // An EDIT does raise the badge, so it is not simply dead.
        const e = createModel({ owner: "label", workspace: ws });
        await opened(e, "H2");
        const raised = [];
        e.subscribe(() => raised.push(e.uncommitted));
        e.selection.adopt([0]);
        e.selection.writeLabel("bridge", "replace");
        answer.badgeOnAnEdit = raised;
        await __quiet();

        console.log(JSON.stringify(answer));
        """)
    for mode in ("editable", "readonly"):
        got = out[mode]
        assert got["frames"] == 4, f"{mode}: appending did not extend: {got}"
        assert got["frameZero"] == H2["positions"], (
            f"{mode}: appending altered frame 0 — an append must leave what the "
            f"run produced exactly as it was: {got['frameZero']}"
        )
        assert got["forces"] == [None, None, None,
                                 [[0, 0, 4]] * len(H2["elements"])]
        assert got["badgeSeen"] and True not in got["badgeSeen"], (
            f"{mode}: arriving frames raised the unsaved badge: "
            f"{got['badgeSeen']}"
        )
        assert got["afterEnforcedReload"] == 1, (
            f"{mode}: an enforced replacement did not land: {got}"
        )
    assert out["editable"]["afterCasualReload"] == 1, (
        "an editable viewer must replace its frames when asked, plainly"
    )
    assert out["readonly"]["afterCasualReload"] == 4, (
        "a read-only viewer let every coordinate be replaced without being "
        "asked — `reloadFrames` rewrites where the append doors only extend, so "
        "it belongs with the edits, not with delivery"
    )
    assert True in out["badgeOnAnEdit"], (
        "an edit must still raise the badge, or the badge is simply dead: "
        f"{out['badgeOnAnEdit']}"
    )


def test_a_trajectory_arrives_whole_in_one_install(app_at, pairs):
    """§ 9.3: `installMolecule` is "the ONLY way a structure gets in", and it
    "replaces the whole model at once and resets the session history". § 6.4: the
    master copy is updated "first, and COMPLETELY... No one ever observes a
    half-updated state."

    The server answers with ONE geometry, because it parses a file and a file has
    one. The frames of a run come from the tab's own parsed run file (§ 6.3), so
    a caller opening a trajectory hands them over WITH the load.

    Doing it in a second call broke three rules at once, and the third loses
    data: § 9.3's one entrance was not one, because the frames came through
    another door; subscribers saw a one-frame structure that never existed; and
    § 11.2's point 0 was anchored on that one frame, so RETRACTING TO THE ANCHOR
    threw the trajectory away.

    The road's H2 opened with three frames of a run of it, the last carrying
    forces; the points go to the real workspace and come back from it.
    """
    n = len(H2["elements"])
    run = _moved(H2, 3)
    forces = [None, None, [[0.0, 0.0, 0.9]] * n]
    out = _run(app_at, pairs, """
        const m = createModel({ owner: "s", workspace: ws });
        const seen = [];
        m.subscribe(() => seen.push(m.frameCount()));

        await opened(m, "H2", { frames: RUN, forces: FORCES });
        const installed = { frames: m.frameCount(), at: m.currentFrame(),
                            forces: m.getCoordinates().forcesPerFrame };

        /* The anchor's write has landed (`opened` waits for the wire); MOVE
         * AWAY FROM IT, then go back.
         *
         * The step away is what makes the retract measurable.  Retracting from
         * the anchor onto the anchor leaves the structure where it already was
         * whatever point 0 holds -- `restoreState` returns early on a state it
         * cannot read -- so the assertion below passed even with nothing
         * stored at all.  A fourth frame appended first means point 0 and the
         * present differ, and coming back has to undo the difference. */
        // What subscribers saw is a fact about THE INSTALL, so it is taken
        // before anything else touches the frames.
        const seenAtInstall = seen.slice();
        m.addFrame(shifted(RUN[2], 0.01));
        const beforeRetract = m.frameCount();
        await m.load(0);

        // A frame that does not carry the loaded atoms is refused at the load,
        // exactly as it is at an append (§ 10.8).
        let refused = null;
        try {
            await m.installMolecule({ path: PAIRS["H2"],
                                      frames: [RUN[0].slice(0, 1)] });
        } catch (e) { refused = e.message; }
        // ...and a set whose frames differ names the frame by its number.
        let uneven = null;
        try {
            await m.installMolecule({ path: PAIRS["H2"],
                                      frames: [RUN[0], RUN[1].slice(0, 1)] });
        } catch (e) { uneven = e.message; }
        await __quiet();

        console.log(JSON.stringify({
            uneven,
            installed, seen: seenAtInstall, beforeRetract,
            afterRetract: m.frameCount(), refused,
        }));
        """, RUN=run, FORCES=forces)
    assert out["installed"]["frames"] == 3, (
        f"the trajectory did not arrive with the load: {out['installed']}"
    )
    assert out["installed"]["at"] == 0, "a full load resets to frame 0 (§ 10.8)"
    assert out["installed"]["forces"] == forces
    # § 6.4 is about WHAT a subscriber sees, not how often it is told: every
    # notification must show a settled structure. More than one is fine — the
    # anchor's badge is a second — but a `1` among them is the half-updated
    # state the rule forbids.
    assert out["seen"] and all(k == 3 for k in out["seen"]), (
        "a subscriber saw the structure part-way in — the frames landed in a "
        f"second settle after the atoms: {out['seen']}"
    )
    assert out["beforeRetract"] == 4, (
        "the appended frame never landed, so the retract below measures nothing"
    )
    assert out["afterRetract"] == 3, (
        "retracting to the anchor did not restore the 3-frame trajectory: "
        f"got {out['afterRetract']}.  1 means point 0 was laid down before the "
        "frames arrived (§ 11.2); 4 means it restored nothing at all"
    )
    assert f"frame 2 has 1 atoms, expected {n}" in out["uneven"], out["uneven"]
    assert out["refused"] and "cannot join" in out["refused"], (
        f"a frame that does not fit the loaded atoms was accepted: {out['refused']}"
    )


def test_an_edit_lays_down_a_point_so_retract_steps_back_one_edit(app_at, pairs):
    """§ 11.2 (user, 2026-09-07): "every edit lays down a point of its own", so
    Retract always steps back exactly one edit.

    Driven through the model rather than the history module, because the claim
    is about the EDIT GATE spending the operation's declaration, not about the
    write machine underneath it.  The points are the ones the real workspace
    holds, read back through its own door.
    """
    out = _run(app_at, pairs, """
        const m = createModel({ owner: "s", workspace: ws });
        await opened(m, "H2");

        /* THREE EDITS -- the Translate box, 1 Å along x each time -- NO SAVE
           STATE PRESSED.  The count never changes, so nothing else clears;
           what is being watched is the SEQUENCE. */
        const counts = [];
        for (let i = 0; i < 3; i += 1) {
            await m.applyOp("translate", { dx: 1 });
            await __quiet();
            counts.push((await pointsOf("s")).length);
        }

        console.log(JSON.stringify({
            points: await pointsOf("s"),
            afterEachEdit: counts,
            badge: m.uncommitted,
        }));
        """)
    # Point 0 is the install's anchor; each edit adds one above it.
    assert out["points"] == [0, 1, 2, 3], (
        "three edits did not lay down three points -- Retract can only reach "
        f"the last manual save: {out['points']}"
    )
    assert out["afterEachEdit"] == [2, 3, 4], (
        f"the sequence did not advance once per edit: {out['afterEachEdit']}"
    )
    assert out["badge"] is False, (
        "the badge still says there is work off the sequence, after every edit "
        "put itself on it"
    )


def test_a_load_fits_the_camera_and_an_operation_keeps_it(app_at, pairs):
    """§ 9.6: the camera is fitted on load and on Reset, and at no other
    moment -- and the model is the one that knows which write is a load.

    THE SILENT FAILURE: every write to the structure reaching the drawing as
    "here is the data", so an operation on the Build tab threw away the
    angle the user had turned to; nothing above the drawing keeps the angle
    (§ 9.6), so it cannot be put back.  The other way round, a state put
    back opens with the drawing at its defaults (§ 11.2), and kept under
    the old camera it is drawn at a distance chosen for other settings.

    A drawing that records what it is told is attached -- a stand-in, as the
    drawing library needs WebGL, which node cannot run -- and each road's
    word is read: an install, a reopened page (a fresh viewer over the same
    real workspace, in the same tab), an undo and a Retract are loads; an
    operation is an edit.
    """
    out = _run(app_at, pairs, """
        const told = [];
        const drawing = () => ({
            setDataSource() {}, switchesChanged() {}, showFrame() {},
            appendFrames() {}, forcesChanged() {}, cellChanged() {},
            dataChanged(how) { told.push(how && how.edit ? "edit" : "load"); },
        });

        const one = createModel({ owner: "s", workspace: ws });
        one._attachRenderer(drawing());
        await opened(one, "H2");
        const install = told.splice(0);
        await one.applyOp("translate", { dx: 1 });
        await __quiet();
        const edit = told.splice(0);
        await one.applyOp("translate", { dx: 1 });
        await __quiet();
        told.splice(0);
        await one.undo();
        const undo = told.splice(0);
        await one.load(0);
        const retract = told.splice(0);
        await __quiet();

        const two = createModel({ owner: "s", workspace: ws });
        two._attachRenderer(drawing());
        await two.load(0);
        const reopen = told.splice(0);
        console.log(JSON.stringify({ install, edit, undo, retract, reopen }));
        """)
    assert out == {"install": ["load"], "edit": ["edit"], "undo": ["load"],
                   "retract": ["load"], "reopen": ["load"]}, (
        f"the drawing was told the wrong kind of write: {out}")


def test_an_atom_carries_a_label_once_however_often_it_is_applied(app_at, pairs):
    """§ 6.2: an atom's facts are "the labels it carries" — a set of names.
    Carrying the same name twice is not a state the model may hold.

    **Applying cannot create one.** Assigning a label an atom already has, twice
    over, leaves one — the write strips every occurrence of the name before
    adding one back, so it is impossible by construction rather than by a check
    that could be forgotten at a fourth call site.

    The road's water, which carries no label of its own.
    """
    out = _run(app_at, pairs, """
        const clean = await loaded("water");
        clean.selection.adopt([0]);
        clean.selection.writeLabel("bridge", "replace");
        clean.selection.writeLabel("bridge", "add");      // it already has it
        clean.selection.writeLabel("bridge", "add");      // and again
        clean.selection.writeLabel("bridge", "replace");  // and replace with itself
        console.log(JSON.stringify({
            labels:  clean.getAtoms()[0].labels,
            regions: clean.getRegions(),
        }));
        """)
    assert out["labels"] == ["bridge"], (
        f"applying a label an atom already carries added it again: {out['labels']}"
    )
    assert out["regions"] == {"bridge": [0]}


# ---------------------------------------------------------------------------
# § 6.4 — master copy, then range, then frame, then notify
# ---------------------------------------------------------------------------

def test_no_subscriber_sees_a_new_range_beside_an_old_frame_number(app_at, pairs):
    """§ 13.3: "after a load that shortens a trajectory, no subscriber ever sees
    a range from the new structure beside a frame number from the old one."

    § 6.4: "No one ever observes a half-updated state."

    The load is the Results tab's: a read-only viewer at the last frame of a
    five-frame set, handed a two-frame set in its place (`enforce`, § 9.4).
    """
    out = _run(app_at, pairs, """
        const m = createModel({ mode: "readonly" });
        await opened(m, "H2, 5 frames");
        m.setCurrentFrame(4);

        // Watch what every notification shows.
        const seen = [];
        m.subscribe(() => seen.push({ at: m.currentFrame(), of: m.frameCount() }));
        m.onFrameChange(() => seen.push({ at: m.currentFrame(), of: m.frameCount() }));

        await opened(m, "H2, 2 frames", { enforce: true });   // it shortens
        console.log(JSON.stringify({
            seen,
            bad: seen.filter(s => s.at >= s.of),
            finalAt: m.currentFrame(),
            finalOf: m.frameCount(),
        }));
        """)
    assert out["seen"], "the load told no one, so nothing here was observed"
    assert out["bad"] == [], (
        f"a subscriber saw a frame outside its own range: {out['seen']}"
    )
    assert out["finalAt"] == 0 and out["finalOf"] == 2


def test_an_out_of_range_write_is_resolved_against_the_range(app_at, pairs):
    """§ 6.4: "A number outside the range is resolved against the range, never
    taken on trust." Not an error — a slider at the end of a trajectory that just
    got shorter is asking a reasonable question.
    """
    out = _run(app_at, pairs, """
        const m = createModel({ mode: "readonly" });
        await opened(m, "H2, 5 frames");
        const seen = [];
        m.setCurrentFrame(99);   seen.push(m.currentFrame());
        m.setCurrentFrame(-5);   seen.push(m.currentFrame());
        m.setCurrentFrame(1.7);  seen.push(m.currentFrame());
        console.log(JSON.stringify({ seen }));
        """)
    assert out["seen"] == [4, 0, 1], (
        f"an out-of-range seek must land inside the range: {out['seen']}"
    )


def test_one_write_reaches_every_subscriber_whatever_moved_it(app_at, pairs):
    """§ 13.3: "one write reaches EVERY subscriber, whatever moved it."

    § 6.4: a subscriber never has to know which; nothing anywhere needs its own
    "did it change?" check.
    """
    out = _run(app_at, pairs, """
        const m = createModel({ mode: "readonly" });
        await opened(m, "H2, 5 frames");
        let a = 0, b = 0;
        m.onFrameChange(() => a++);
        m.onFrameChange(() => b++);
        m.setCurrentFrame(2);
        const both = { a, b };
        m.setCurrentFrame(2);                      // no move, no notification
        console.log(JSON.stringify({ both, after: { a, b } }));
        """)
    assert out["both"] == {"a": 1, "b": 1}, (
        f"a frame write did not reach every subscriber: {out['both']}"
    )
    assert out["after"] == {"a": 1, "b": 1}, (
        "setting the frame to where it already is must notify nobody"
    )


def test_appending_frames_recomputes_the_range_from_the_master_copy(app_at,
                                                                    pairs):
    """§ 6.4 step 2: the range is recomputed FROM THE MASTER COPY — "not from the
    drawing copy, and not from what the caller said it was adding."
    """
    out = _run(app_at, pairs, """
        const m = await loaded("H2");
        const f0 = m.getFrameAllAtoms(0);
        const start = m.frameCount();
        m.addFrames([shifted(f0, 1), shifted(f0, 2)]);
        const grown = m.frameCount();
        m.addFrame(shifted(f0, 3));
        console.log(JSON.stringify({ start, grown, final: m.frameCount() }));
        """)
    assert out["start"] == 1 and out["grown"] == 3 and out["final"] == 4


def test_forces_arriving_after_a_frameless_load_land_on_their_own_frame(app_at,
                                                                       pairs):
    """§ 10.3: frame f's arrows come from frame f.

    A run caught at its first geometry carries no forces. Pushing the first ones
    that arrive onto an empty list would attach them to frame 0 for ever after,
    so a growing run would show frame 0's forces on every frame.

    The Results tab following a run: its first geometry opened read-only, a
    frame arriving with no forces, then one carrying them.
    """
    out = _run(app_at, pairs, """
        const m = await loaded("H2", { mode: "readonly" });
        const f0 = m.getFrameAllAtoms(0);
        m.addFrame(shifted(f0, 0.01));                              // no forces
        m.addFrame(shifted(f0, 0.02), { forces: pushed(f0, 7) });   // it has them
        const c = m.getCoordinates();
        console.log(JSON.stringify({
            frames: c.frames.length,
            forces: c.forcesPerFrame,
        }));
        """)
    assert out["frames"] == 3
    assert out["forces"] == [None, None, [[0, 0, 7]] * len(H2["elements"])], (
        "the first forces to arrive must land on the frame that carried them, "
        f"not on frame 0: {out['forces']}"
    )


# ---------------------------------------------------------------------------
# § 9.4 — read-only freezes the master copy and nothing else
# ---------------------------------------------------------------------------

def test_read_only_freezes_the_core_data_and_gates_nothing_else(app_at, pairs):
    """§ 9.4, and the line the gate is actually drawn on: it protects THE CORE
    DATA — the structure and the metadata that travels with it — from being
    CHANGED. One write is allowed into an empty viewer, because a viewer with
    nothing in it has no core data to freeze.

    What it does NOT gate is delivery. Frames and forces arriving for the
    structure already installed are a running job's own output, not a user
    editing the structure the calculation ran on — and § 10.8's guards make that
    a fact rather than a reading, since those doors cannot alter the atom count,
    the elements, the labels or the cell.

    Gating them was reading "does this change the master copy?" literally, and it
    cost a read-only viewer the two things it exists for: following a running
    optimization (§ 12.2), and scrubbing a finished one (§ 12.3) — the only frame
    it could ever hold was the one it was seeded with.

    "It returns without effect AND WITHOUT THROWING": a read-only viewer that
    threw would make every caller wrap its writes, which is the list of special
    cases this rule exists to avoid.

    The Results tab's viewer: the road's H2 seeded read-only, two frames of a
    run of it and their forces delivered, then each change a person could ask
    of it -- another file opened over it, Delete with an atom picked, the Cell
    page's lattice, the Label box.
    """
    out = _run(app_at, pairs, """
        const m = await loaded("H2", { mode: "readonly" });
        const f0 = m.getFrameAllAtoms(0);

        // DELIVERY is open: the run's own frames and forces arrive.
        m.addFrames([shifted(f0, 0.1), shifted(f0, 0.2)]);
        m.setForces([null, null, pushed(f0, 9)]);
        m.setCurrentFrame(2);
        const delivered = { frames: m.frameCount(), at: m.currentFrame(),
                            forces: m.getCoordinates().forcesPerFrame };
        const core = () => JSON.stringify({
            elements: m.getStructure().elements,
            annotations: m.getStructure().annotations,
            periodicity: m.getStructure().periodicity,
        });
        const coreAfterDelivery = core();

        // CHANGING the core data is not, and none of it throws.
        const threw = [];
        async function tryIt(name, fn) {
            try { await fn(); } catch (e) { threw.push(name + ": " + e.message); }
        }
        m.selection.adopt([0]);
        await tryIt("installMolecule",
                    () => m.installMolecule({ path: PAIRS["water"] }));
        await tryIt("applyOp", () => m.applyOp("delete", {}));
        await tryIt("commitPeriodicityOp", () => m.commitPeriodicityOp(
            "cell", [[9,0,0],[0,9,0],[0,0,9]]));
        await tryIt("writeLabel",
                    () => m.selection.writeLabel("smuggled", "replace"));
        await __quiet();

        console.log(JSON.stringify({
            threw, delivered,
            coreUnchanged: coreAfterDelivery === core(),
            requests: __calls.map((c) => c.url),
        }));
        """)
    assert out["threw"] == [], f"a read-only no-op threw: {out['threw']}"
    assert {k: out["delivered"][k] for k in ("frames", "at")} == {
        "frames": 3, "at": 2}, (
        "a read-only viewer could not receive the run's own frames, so it can "
        f"neither follow a running job nor scrub a finished one: {out['delivered']}"
    )
    assert out["delivered"]["forces"] == [
        None, None, [[0, 0, 9]] * len(H2["elements"])], (
        f"the run's forces did not arrive: {out['delivered']['forces']}")
    assert out["coreUnchanged"] is True, (
        "the core data — the atoms, their labels, the cell — changed in a "
        "read-only viewer"
    )
    assert out["requests"] == [], (
        "a read-only viewer sent a request — the gate must stop a change before "
        f"the network, not after: {out['requests']}"
    )


def test_scrubbing_and_exporting_are_not_gated(app_at, pairs):
    """§ 9.4: "Somebody studying a finished calculation can still select atoms,
    isolate them, measure them, scrub the trajectory, turn on force arrows, spin
    the camera, and export" — none of which touches the master copy.

    The assertion is about WHICH SIDE OF THE GATE these fall on, so it is made
    on a read-only viewer holding something to scrub: the road's H2 as a
    five-frame set, opened as the Results tab opens one.  Neither door is
    wrapped, so neither is swallowed: the seek lands, and the export hands
    over the frame it landed on.

    (How a read-only viewer receives its structure is settled: one write into an
    empty viewer, and delivery of that structure's frames stays open —
    ``test_read_only_freezes_the_core_data_and_gates_nothing_else``.)
    """
    out = _run(app_at, pairs, """
        const m = await loaded("H2, 5 frames", { mode: "readonly" });
        let threw = null, file = null;
        try { m.setCurrentFrame(3); file = m.exportFile(); }
        catch (e) { threw = e.message; }
        console.log(JSON.stringify({
            threw, scrubbed: m.currentFrame(),
            positions: file && file.structure.positions,
        }));
        """)
    assert out["threw"] is None, f"a read-only viewer threw: {out['threw']}"
    assert out["scrubbed"] == 3, "scrubbing must work — it is looking at the picture"
    assert out["positions"] is not None, "export must work — it is a read"
    assert np.allclose(out["positions"], _moved(H2, 5)[3]), (
        f"the export did not hand over the frame scrubbed to: {out['positions']}")


def test_a_read_only_viewer_is_seeded_once_and_then_frozen(app_at, pairs):
    """§ 9.4 freezes THE MASTER COPY — and a viewer with nothing loaded has no
    master copy to freeze.

    So the first `installMolecule` is how a host says which structure this viewer
    shows, and it is allowed in any mode; every one after it meets the gate. That
    keeps § 9.3's "the only way a structure gets in" intact — no second door —
    while making § 8's "a viewer mounts before it has a structure" and § 12.3's
    read-only Results viewer both possible.

    What a read-only viewer still cannot do is exactly what § 9.4 promises:
    change the structure the calculation ran on -- and the gate stops the
    replacement before the network.
    """
    out = _run(app_at, pairs, """
        const m = createModel({ mode: "readonly" });
        const seed = await opened(m, "H2");
        const afterSeed = m.getElements();

        // A second load would REPLACE the structure the calculation ran on.
        __calls.length = 0;
        const replaced = await m.installMolecule({ path: PAIRS["water"] });
        await __quiet();

        console.log(JSON.stringify({
            seeded: seed !== null, afterSeed, replaced,
            stillFirst: m.getElements(),
            secondRequests: __calls.map((c) => c.url),
        }));
        """)
    assert out["seeded"] is True, (
        "a read-only viewer could not be given a structure at all, so the "
        "Results tab would have nothing to show"
    )
    assert out["afterSeed"] == H2["elements"]
    assert out["replaced"] is None, "the second load must be a no-op"
    assert out["stillFirst"] == H2["elements"], (
        "a read-only viewer's structure was replaced after it was seeded"
    )
    assert out["secondRequests"] == [], (
        "the gate must stop the replacement before the network, not after"
    )


def test_a_read_only_seed_anchors_no_history(app_at, pairs):
    """§ 9.4: "A read-only viewer has no history … `save`, `load` and `undo` are
    no-ops here too, and the unsaved-changes badge never appears."

    Anchoring point 0 would also write it to the workspace, which is a persist a
    read-only viewer has no business doing (§ 11.2: storage is touched by opening
    a structure, an explicit save, and a load).

    Two viewers on one page, each handed the real workspace under an owner of
    its own, each opening the road's H2: the points each owner holds are read
    back through the workspace's own door.  Only a POINT counts as anchoring
    -- an edit's draft goes to its own file under another rule (workspace.md
    § 7).
    """
    out = _run(app_at, pairs, """
        const ro = createModel({ mode: "readonly", owner: "ro", workspace: ws });
        await opened(ro, "H2");
        const editable = createModel({ owner: "ed", workspace: ws });
        await opened(editable, "H2");

        console.log(JSON.stringify({
            readonly: await pointsOf("ro"), editable: await pointsOf("ed"),
            badge: ro.uncommitted, at: ro.state_index,
        }));
        """)
    assert out["editable"] == [0], (
        f"the editable viewer did not anchor point 0: {out['editable']}")
    assert out["readonly"] == [], (
        "a read-only seed wrote a history point to the workspace — only the "
        f"editable viewer should have anchored: {out['readonly']}"
    )
    assert out["badge"] is False
    assert out["at"] == 0


def test_a_restored_session_still_receives_frames(app_at, pairs):
    """§ 11.2a's state machine, at the restore door: adoption re-enters
    HOLDING exactly as an install does.

    Until 2026-08-19 `restoreState` set the structure inside `settle` and
    never touched the unit state, so a reopened session sat in EMPTY while
    plainly holding a structure -- `addFrames`/`setForces` refused with
    "nothing loaded -- there is no atom identity", live on the Modify tab
    the moment anything appended frames after a reload.

    Session one opens the road's H2 on the real workspace (point 0 and the
    draft); session two is a fresh viewer over the same workspace in the same
    browser tab, as a reopened page builds one, and `load(0)` adopts the
    draft; then a run's frame arrives.
    """
    out = _run(app_at, pairs, """
        const one = createModel({ owner: "s", workspace: ws });
        await opened(one, "H2");

        const two = createModel({ owner: "s", workspace: ws });
        const adopted = await two.load(0);
        let refused = null;
        try {
            two.addFrames([shifted(two.getFrameAllAtoms(0), 0.1)]);
        } catch (e) { refused = String((e && e.message) || e); }

        console.log(JSON.stringify({
            adopted: adopted !== null && adopted !== undefined,
            refused,
            frames: two.frameCount(),
        }));
        """)
    assert out["adopted"] is True, "load(0) found nothing to adopt"
    assert out["refused"] is None, (
        f"a restored session refused arriving frames: {out['refused']!r} — "
        f"the viewer holds a structure and must say so (HOLDING)"
    )
    assert out["frames"] == 2, out


# ---------------------------------------------------------------------------
# § 9.3 / § 11.3 — writing the structure out
# ---------------------------------------------------------------------------

def test_export_hands_over_the_displayed_frame(app_at, pairs):
    """§ 13.3: "exporting data yields THE DISPLAYED FRAME's coordinates … scrub
    to frame 40 and frame 40 is what the file holds."

    It hands over the STRUCTURE, not bytes — a coordinate document is a format
    the server owns (§ 11.7) — so the frame is checked where it now lives: in
    the positions that leave, and the facts are checked leaving with them.

    The road's labelled CO2 in its box, open for editing, two frames of a run
    of it delivered (an editable viewer follows a running job as a read-only
    one does), scrubbed to the last.
    """
    out = _run(app_at, pairs, """
        const m = await loaded("CO2 labelled");
        const f0 = m.getFrameAllAtoms(0);
        m.addFrames([shifted(f0, 0.1), shifted(f0, 0.2)]);
        m.setCurrentFrame(2);
        const file = m.exportFile();
        console.log(JSON.stringify({
            positions: file.structure.positions,
            regions: file.structure.metadata.regions,
            cell: file.structure.metadata.cell,
        }));
        """)
    shown = np.asarray(LABELLED["positions"]) + [0.0, 0.0, 0.2]
    assert np.allclose(out["positions"], shown), (
        f"export handed over a frame the user was not looking at: {out['positions']}"
    )
    assert out["regions"] == LABELLED["regions"], (
        f"the labels did not leave with the atoms: {out['regions']}")
    assert out["cell"] == np.diag(LABELLED["cell"]).tolist(), (
        f"the cell did not leave with the atoms: {out['cell']}")


def test_the_frame_doors_refuse_to_create_a_structure_that_disagrees_with_itself(
        app_at, pairs):
    """§ 10.8 rules 1–3, at the three doors that could break the same-atoms rule.

    "Something must already be loaded. Appending with nothing loaded is a hard
    error — there is no atom identity to append to." "Each new frame is checked
    against that identity before anything reaches the drawing. Same atom count."
    "A mismatch is a hard error. Never padded, never truncated, never guessed
    into fitting."

    None of it was enforced, and the master copy would take a structure of two
    elements holding a frame with one position — the shape everything downstream
    indexes against (§ 6.2). This is the belt § 9.3's export refusal was standing
    in for: the disagreement is now unreachable through the public surface rather
    than merely refused on the way out.

    The road's H2 (two atoms); each short frame is its frame with one atom.
    """
    out = _run(app_at, pairs, """
        function refused(fn) {
            try { fn(); return null; } catch (e) { return e.message; }
        }
        const empty = createModel({});
        const nothingLoaded = refused(() => empty.addFrames([[[0,0,0],[1,0,0]]]));

        const m = await loaded("H2");
        const f0 = m.getFrameAllAtoms(0);
        const short = f0.slice(0, 1);
        const shortAppend  = refused(() => m.addFrames([short]));
        const shortOne     = refused(() => m.addFrame(short));
        const shortReload  = refused(() => m.reloadFrames([short]));
        // A batch whose frames disagree with EACH OTHER, checked before any of
        // them lands so the first half is not left applied.
        const ragged = refused(() => m.addFrames([shifted(f0, 0.1), short]));

        console.log(JSON.stringify({
            nothingLoaded, shortAppend, shortOne, shortReload, ragged,
            // Nothing was applied by any of the refusals.
            frames: m.frameCount(),
            stillWhole: m.getFrameAllAtoms(0).length,
        }));
        """)
    assert out["nothingLoaded"] and "nothing loaded" in out["nothingLoaded"], (
        "appending with nothing loaded invented an atom identity instead of "
        f"refusing: {out['nothingLoaded']}"
    )
    for door in ("shortAppend", "shortOne", "shortReload", "ragged"):
        assert out[door], (
            f"{door} accepted a frame that does not carry the structure's atoms "
            "— § 10.8 calls that a hard error, never coerced"
        )
    assert out["frames"] == 1 and out["stillWhole"] == len(H2["elements"]), (
        "a refused frame left something behind in the master copy"
    )


def test_one_read_of_the_structure_holds_everything_a_request_needs(app_at,
                                                                    pairs):
    """§ 9.3: "A request ... carries the coordinates, the labels, which atoms are
    held still, and the cell. EVERY ONE OF THOSE IS PART OF THE STRUCTURE, so one
    read of the structure already holds them. There is nothing a request needs
    that ``getStructure()`` does not have."

    Asked of the READ ITSELF, not of the body that leaves. The outbound body was
    already correct — one producer, one read — while ``getStructure()`` returned
    three of the five fields and left THE COORDINATES OUT, so any caller building
    a request off this surface had to take the positions from a second call. Two
    calls are two moments, and a set assembled from two moments is exactly the
    failure the section tells the story of.

    § 6.3 fixes the extent: the master copy is "every atom, EVERY FRAME, in the
    original order" — so a trajectory's whole movie is in this one answer, not
    just the frame on screen.

    The road's CO2 labelled -- an ``anchor`` on an oxygen, its carbon held --
    in its box with the origin a person assigned, a run's frame delivered
    after it carrying forces.
    """
    out = _run(app_at, pairs, """
        const m = await loaded("CO2 labelled");
        const f0 = m.getFrameAllAtoms(0);
        m.addFrames([shifted(f0, 0.1)], { forces: [pushed(f0, -1)] });

        // ONE call. Nothing else on the surface is touched.
        const whole = m.getStructure();

        console.log(JSON.stringify({
            keys:      Object.keys(whole).sort(),
            elements:  whole.elements,
            frames:    whole.frames,
            forces:    whole.forcesPerFrame,
            labels:    whole.annotations.map((a) => a.labels),
            cell:      whole.periodicity && whole.periodicity.cell,
            offset:    whole.periodicity && whole.periodicity.engine_offset,
        }));
        """)
    # THE WHOLE STRUCTURE, § 6.2's fields.
    # EXACT rather than a superset, deliberately: a field silently DROPPED is
    # what this catches, and a superset check cannot see one go.  A field
    # legitimately added updates this line, on purpose.
    assert out["keys"] == ["annotations", "channelDefs", "customized",
                           "elements", "forcesPerFrame", "frames", "info",
                           "labelsOffered", "orderSaid", "otherMetadata",
                           "periodicity", "rowsPerFrame", "title"], (
        f"the one read does not carry § 6.2's fields: {out['keys']}"
    )
    # The four facts § 9.3 names, each read off that single answer.
    at = np.asarray(LABELLED["positions"])
    assert np.allclose(out["frames"], [at, at + [0.0, 0.0, 0.1]]), (
        "the coordinates are missing from the read, or it handed back only the "
        f"displayed frame instead of every frame (§ 6.3): {out['frames']}"
    )
    assert out["forces"] == [None, [[0, 0, -1]] * len(LABELLED["elements"])]
    held = [sorted(name for name, atoms in LABELLED["regions"].items()
                   if i in atoms) for i in range(len(LABELLED["elements"]))]
    assert [sorted(labels) for labels in out["labels"]] == held, (
        f"the labels are missing from the read: {out['labels']}"
    )
    for i in LABELLED["regions"]["frozen_atoms"]:
        assert "frozen_atoms" in out["labels"][i], (
            "which atoms are held still is a label like any other (§ 6.6), so "
            "it rides in the same one read"
        )
    assert out["cell"] == np.diag(LABELLED["cell"]).tolist(), "the cell is missing"
    # THE OFFSET THE STRUCTURE STATES (§ 6.2): an origin a person sets at a
    # corner is stated as the offset that takes it there, its negative
    # (`structure-periodicity.md` § 6.0).
    assert out["offset"] == [-c for c in CORNER]
    assert out["elements"] == LABELLED["elements"]


def test_the_facts_a_request_carries_all_came_from_one_read(app_at, pairs):
    """§ 9.3: "after an edit, a request built from the viewer carries that edit
    in EVERY part of what it sends — no piece can be older than another, because
    it all came from one read of the structure."

    This went wrong once: a tab read the labels and the cell fresh while the
    coordinates came from a copy taken at page load, so the server judged a
    structure that was not the one on screen.

    The road's labelled CO2, edited twice -- the Translate box, 1 Å along x,
    and the Label box writing ``bridge`` onto its second oxygen -- then
    translated again: the request that leaves carries both edits, read off
    the wire.
    """
    out = _run(app_at, pairs, """
        const m = await loaded("CO2 labelled");
        await m.applyOp("translate", { dx: 1, dy: 0, dz: 0 });
        m.selection.adopt([1]);
        m.selection.writeLabel("bridge", "replace");
        m.selection.clear();
        await __quiet();

        __calls.length = 0;
        await m.applyOp("translate", { dx: 1, dy: 0, dz: 0 });
        const sent = __calls.find(
            (c) => c.url.endsWith("/api/modify/translate")).body.structure;
        console.log(JSON.stringify({
            positions: sent.positions,
            regions: sent.metadata.regions,
            cell: sent.metadata.cell,
            elements: sent.elements,
        }));
        """)
    assert np.allclose(out["positions"],
                       np.asarray(LABELLED["positions"]) + [1.0, 0.0, 0.0]), (
        f"the request carried coordinates that were not the current ones: "
        f"{out['positions']}"
    )
    assert out["regions"] == dict(LABELLED["regions"], bridge=[1]), (
        "the request carried labels from a different read than the "
        f"coordinates: {out['regions']}"
    )
    assert out["cell"] == np.diag(LABELLED["cell"]).tolist()
    assert out["elements"] == LABELLED["elements"]


def test_what_the_caller_knew_about_the_atoms_travels_in_the_envelope(
        app_at, pairs):
    """§ 9.3: `installMolecule` is "the only way a structure gets in", so
    everything the caller knows about that structure has to fit through it.

    THE CASE THIS EXISTS FOR is a trajectory. There is no `.molstruct.json`
    beside a run: the region labels and frozen tags come out of the input
    script the Build tab wrote, and the lattice out of the run's output. Both
    have to arrive WITH the structure, in one call, exactly as the frames do —
    the alternative is a second door, and § 9.3 has one.

    THEY TRAVEL IN THE ENVELOPE. The trajectory tab hands over frame 0 as an
    envelope the server assembled (`watch.py::_frame0_structure`), so the
    labels and the cell are simply IN the structure.

    (`web-api.md` § 1: "the browser sends what it holds; it never sends a
    document it wrote".)

    The envelope here is the door's other caller's, the same one shape
    (`Structure.to_dict`): a tab putting back what its viewer exported
    (`exportFile`'s exact inverse) -- the road's labelled CO2, an ``anchor``
    and its held carbon among its labels, in its box.  A run's own envelope
    needs a run's output, which a basic test does not make.  The body is read
    off the wire, and the viewer that took it is asked what it holds -- the
    server accepted what was sent.
    """
    out = _run(app_at, pairs, """
        const shown = await loaded("CO2 labelled");
        const envelope = shown.exportFile().structure;

        const m = createModel({});
        __calls.length = 0;
        await m.installMolecule({ structure: envelope, filename: "co2.xyz" });
        const sent = __calls[0].body;
        console.log(JSON.stringify({
            route: __calls[0].url,
            keys:  Object.keys(sent).sort(),
            regions: sent.structure.metadata.regions,
            cell: sent.structure.metadata.cell,
            held: m.getRegions(),
            heldCell: m.getUnitCell(),
        }));
        """)
    assert out["route"].endswith("/api/build/load")
    assert out["regions"] == LABELLED["regions"], (
        "the labels were dropped or rewritten. Dropped: a trajectory opens "
        "with no region labels and no frozen tags, at HTTP 200 — nothing "
        f"refused them. Got: {out['regions']}"
    )
    assert out["cell"] == np.diag(LABELLED["cell"]).tolist(), (
        "the caller's cell never left the browser, so no trajectory can draw "
        "the box its run used"
    )
    # ONE KEY: a body that grew a side-block would mean a caller is writing
    # what the envelope already holds.
    assert out["keys"] == ["structure"], (
        f"the envelope did not travel alone: {out['keys']}")
    assert out["held"] == LABELLED["regions"], (
        f"the viewer that took the envelope lost its labels: {out['held']}")
    assert out["heldCell"] == np.diag(LABELLED["cell"]).tolist(), (
        f"the viewer that took the envelope lost its cell: {out['heldCell']}")


def test_a_caller_that_knew_nothing_extra_sends_nothing_extra(app_at, pairs):
    """The fields are absent, not null, when there is nothing to say.

    An ordinary file load has no such block and states no cell, and a key
    carrying `null` is a statement the caller did not make — on the server side
    the periodicity block is applied verbatim, so an empty one is not nothing.

    The ordinary file load is a path: the road's water opened as the Load
    button opens a file, the body read off the wire.
    """
    out = _run(app_at, pairs, """
        const m = createModel({});
        await opened(m, "water");
        const body = __calls.find(
            (c) => c.url.endsWith("/api/build/load")).body;
        console.log(JSON.stringify({
            keys: Object.keys(body).sort(),
            nulls: Object.keys(body).filter((k) => body[k] == null),
        }));
        """)
    assert "atom_metadata" not in out["keys"], (
        f"an ordinary load sent an empty metadata block: {out['keys']}"
    )
    assert "periodicity" not in out["keys"], (
        f"an ordinary load stated a cell it was never given: {out['keys']}"
    )
    assert out["nulls"] == [], (
        f"an ordinary load sent a key carrying null: {out['nulls']}")


# ---------------------------------------------------------------------------
# § 11.1 — geometry edits go to the server
# ---------------------------------------------------------------------------

#: THE SLAB PANEL'S APPLY, as a person leaves it: gold (111), 2 x 2 cells of
#: three layers, placed from the world origin -- the panel's own defaults
#: (`modify.html`, `modify/slab-panel.js`) -- with the lattice constant its box
#: fills from the experimental reference the app lists (`/api/modify/meta`).
SLAB = {"element": "Au", "plane": "111", "m": 2, "n": 2, "layers": 3,
        "start_registry": 0, "sequence": "ABC", "grow": "+z", "start_z": 0,
        "orthogonal": False, "dx": 0, "dy": 0}


def test_the_count_requirement_is_checked_before_the_request_goes_out(app_at,
                                                                     pairs):
    """§ 13.3: "`orient` with one atom and `delete` with none are refused
    locally, with no request sent."

    The road's water: one atom picked on the ruler -- the track Orient reads
    -- and the Orient box applied; nothing selected and Delete pressed.
    """
    out = _run(app_at, pairs, """
        const m = await loaded("water");
        m.measurement.setActive(true);
        m.measurement.toggle(0);                       // one atom; needs 2
        const orient = await m.applyOp("orient",
            { axis: "z", angle: 0, center: "midpoint" });
        const del = await m.applyOp("delete", {});     // nothing selected
        console.log(JSON.stringify({
            orient, del, requests: __calls.map((c) => c.url),
        }));
        """)
    assert out["orient"] is None and out["del"] is None
    assert out["requests"] == [], (
        f"a refused operation still reached the network: {out['requests']}"
    )


def test_an_empty_selection_means_what_the_table_says(app_at, pairs):
    """§ 13.3: with nothing selected, `translate` acts on every atom, `orient`
    refuses, `delete` refuses and `slab` acts anyway — each read from the table
    rather than hand-coded per operation.

    The road's water, nothing selected: the Translate box (1 Å along x), the
    Orient box, Delete, and the Slab panel's Apply, each as its control sends
    it.
    """
    out = _run(app_at, pairs, """
        const meta = await (await fetch("/api/modify/meta")).json();
        const m = await loaded("water");
        const before = m.getFrameAllAtoms(0);
        const translate = await m.applyOp("translate", { dx: 1, dy: 0, dz: 0 });
        const moved = m.getFrameAllAtoms(0);
        const orient = await m.applyOp("orient",
            { axis: "z", angle: 0, center: "midpoint" });
        const del = await m.applyOp("delete", {});
        const slab = await m.applyOp("slab", Object.assign({}, SLAB, {
            lattice_constant: meta.lattice_table[SLAB.element].a_experimental,
        }));
        console.log(JSON.stringify({
            translate: translate !== null,
            shift: moved.map((p, i) => p.map((v, k) => v - before[i][k])),
            orient: orient !== null,
            del: del !== null,
            slab: slab !== null,
            routes: __calls.map((c) => new URL(c.url, "http://x").pathname),
        }));
        """, SLAB=SLAB)
    assert out["translate"] is True, "translate with nothing selected acts"
    assert np.allclose(out["shift"], [[1.0, 0.0, 0.0]] * len(WATER["elements"])), (
        f"translate with nothing selected did not act on every atom: {out['shift']}")
    assert out["orient"] is False, "orient with nothing selected refuses"
    assert out["del"] is False, "delete with nothing selected refuses"
    assert out["slab"] is True, "slab reads no selection, so it acts anyway"
    assert out["routes"] == ["/api/modify/translate", "/api/modify/slab"], (
        f"the two that refuse must not reach the network: {out['routes']}"
    )


def test_the_operation_name_is_the_route_segment(app_at, pairs):
    """§ 11.1: "The operation name IS the server route segment. The delete
    operation is `delete`, not `deleteAtoms`; the add operation is `add_atom`."

    The road's water, the Translate box and the Rotate box applied: each
    request goes to the route its operation names, and the route answers it.
    """
    out = _run(app_at, pairs, """
        const m = await loaded("water");
        const translated = await m.applyOp("translate", { dx: 1, dy: 0, dz: 0 });
        const rotated = await m.applyOp("rotate",
            { axis: "z", angle: 90, center: "centroid" });
        console.log(JSON.stringify({
            answered: [translated !== null, rotated !== null],
            routes: __calls.map((c) => new URL(c.url, "http://x").pathname),
        }));
        """)
    assert out["routes"] == ["/api/modify/translate", "/api/modify/rotate"]
    assert out["answered"] == [True, True], (
        f"a route its operation names did not answer it: {out}")


def test_an_unknown_operation_is_refused_rather_than_posted(app_at, pairs):
    """§ 11.1: one small table declares each operation's shape. A name that is not
    in it has no shape, so there is nothing to send.
    """
    out = _run(app_at, pairs, """
        const m = await loaded("water");
        let threw = false;
        try { await m.applyOp("deleteAtoms"); } catch (e) { threw = true; }
        console.log(JSON.stringify({ threw, requests: __calls.length }));
        """)
    assert out["threw"] is True, (
        "an unknown operation must be refused loudly — it is a caller mistake, "
        "not a user action the gate should swallow"
    )
    assert out["requests"] == 0


# ---------------------------------------------------------------------------
# § 6.1 / § 6.3
# ---------------------------------------------------------------------------

def test_one_frame_reads_the_same_way_as_many(app_at, pairs):
    """§ 6.1: "no read, edit, export or save path treats a single frame
    differently from four hundred."

    The road's H2 opened twice: once as it is, once followed by a run of four
    hundred frames of it (399 delivered after the first).
    """
    out = _run(app_at, pairs, """
        const one = await loaded("H2");
        const many = await loaded("H2");
        const f0 = many.getFrameAllAtoms(0);
        many.addFrames(Array.from({ length: 399 },
                                  (_, k) => shifted(f0, 0.001 * (k + 1))));
        console.log(JSON.stringify({
            oneCount: one.frameCount(),
            manyCount: many.frameCount(),
            oneExports: !!one.exportFile(),
            manyExports: !!many.exportFile(),
            oneFrameShape: one.getFrameAllAtoms(0).length,
            manyFrameShape: many.getFrameAllAtoms(0).length,
        }));
        """)
    assert out["oneCount"] == 1 and out["manyCount"] == 400
    assert out["oneExports"] and out["manyExports"]
    assert out["oneFrameShape"] == out["manyFrameShape"] == len(H2["elements"])


# ---------------------------------------------------------------------------
# § 11.2 / § 9.4 — the badge, raised inside the gate
# ---------------------------------------------------------------------------

def test_an_edit_records_itself_and_a_failed_one_records_nothing(app_at, pairs):
    """§ 11.1: when the server refuses, "no history state is recorded".
    Recording inside the gate and AFTER the change lands is what makes that
    fall out rather than needing a case of its own: a refused edit never
    reaches the line.

    THE SEQUENCE IS WHAT IS MEASURED, not the badge: an edit lays down a point
    and the badge goes back down when that point lands -- so the badge is False
    either side of a successful edit, and asking it whether the edit was
    recorded gets the same answer as asking it whether the edit was refused.
    The position on the sequence tells them apart, and so do the points the
    real workspace holds, read back through its own door.

    The road's water: the Translate box, then the Add box with an element
    typed that is no element (`Xx`), which the route refuses.
    """
    out = _run(app_at, pairs, """
        const m = createModel({ owner: "s", workspace: ws });
        await opened(m, "water");
        const now = async () => ({ at: m.state_index, badge: m.uncommitted,
                                   points: await pointsOf("s") });
        const fresh = await now();

        await m.applyOp("translate", { dx: 1, dy: 0, dz: 0 });
        await __quiet();
        const afterEdit = await now();

        // The refusal leaves by throwing (§ 6.9).
        let threw = false;
        try {
            await m.applyOp("add_atom", { element: "Xx", offset: [0, 0, 1.5] });
        } catch (_) { threw = true; }
        await __quiet();
        console.log(JSON.stringify({
            fresh, afterEdit, threw, afterFailure: await now(),
        }));
        """)
    assert out["fresh"]["badge"] is False, (
        "a freshly opened structure has no unsaved work")
    assert out["afterEdit"]["at"] == out["fresh"]["at"] + 1, (
        f"an edit did not lay down a point (§ 11.2): {out}")
    assert out["afterEdit"]["points"] == [0, 1], (
        f"the edit's point is not in the workspace: {out}")
    assert out["afterEdit"]["badge"] is False, (
        "the badge is still up after an edit put itself on the sequence")
    assert out["threw"] is True, "a refused edit must throw (§ 6.9)"
    assert out["afterFailure"] == out["afterEdit"], (
        "a failed edit changed the sequence or the badge — it recorded "
        f"nothing and changed nothing: {out}"
    )


def test_a_read_only_viewer_has_no_history_and_no_badge(app_at, pairs):
    """§ 9.4: "`save`, `load` and `undo` are no-ops here too, and the
    unsaved-changes badge (§ 11.2) never appears."

    Saving does not itself change the master copy — it records it — but a history
    exists to get back to a state you left, and in a read-only viewer nothing can
    leave one.

    A read-only viewer handed the real workspace, as the Results tab's is,
    holding the road's H2: Save state, Retract and Undo pressed, an edit and a
    label asked of it.  THE BADGE IS WATCHED, NOT SAMPLED -- every change of
    it reaches a subscriber -- and the points its owner holds are read back.
    """
    out = _run(app_at, pairs, """
        const m = createModel({ mode: "readonly", owner: "ro", workspace: ws });
        await opened(m, "H2");
        const badge = [];
        m.subscribe(() => badge.push(m.uncommitted));
        const threw = [];
        for (const [name, fn] of [
            ["save", () => m.save(1)], ["load", () => m.load(-1)],
            ["undo", () => m.undo()],
            ["applyOp", () => m.applyOp("translate", { dx: 1, dy: 0, dz: 0 })],
            ["writeLabel", () => { m.selection.adopt([0]);
                                   return m.selection.writeLabel("anchor", "replace"); }],
        ]) {
            try { await fn(); } catch (e) { threw.push(name + ": " + e.message); }
        }
        await __quiet();
        console.log(JSON.stringify({
            threw, badgeSeen: badge, badge: m.uncommitted, at: m.state_index,
            points: await pointsOf("ro"),
        }));
        """)
    assert out["threw"] == [], f"a read-only history door threw: {out['threw']}"
    assert out["badge"] is False and True not in out["badgeSeen"], (
        f"a read-only viewer raised the unsaved-changes badge: {out['badgeSeen']}"
    )
    assert out["at"] == 0
    assert out["points"] == [], (
        f"a read-only viewer laid a point in the workspace: {out['points']}")


def test_writing_a_label_records_a_point_and_is_frozen_read_only(app_at, pairs):
    """§ 9.4: "Tagging is an edit. A label becomes part of what an atom IS and
    travels to the calculation, so writing one is frozen along with the rest."

    § 9.5: applying a label REPLACES that label's previous set of atoms.

    The road's water, which carries no label of its own, open for editing on
    the real workspace; and the same water in a read-only viewer.
    """
    out = _run(app_at, pairs, """
        const m = createModel({ owner: "s", workspace: ws });
        await opened(m, "water");
        m.selection.add([0]);
        const before = m.state_index;
        m.selection.writeLabel("anchor");
        await __quiet();
        const first = m.getRegions();
        const recorded = m.state_index - before;
        const points = await pointsOf("s");

        m.selection.clear();
        m.selection.add([1]);
        m.selection.writeLabel("anchor");     // replaces the previous set
        await __quiet();

        const ro = await loaded("water", { mode: "readonly" });
        ro.selection.add([0]);
        const roWrote = ro.selection.writeLabel("anchor");

        console.log(JSON.stringify({
            first, recorded, points, replaced: m.getRegions(), roWrote,
            roRegions: ro.getRegions(),
        }));
        """)
    assert out["first"] == {"anchor": [0]}
    # Tagging is an EDIT, so it lays down a point like any other (§ 11.2, and
    # the gate says so itself: `recordEdit(true)`).
    assert out["recorded"] == 1 and out["points"] == [0, 1], (
        f"tagging did not lay down a timeline point: {out}")
    assert out["replaced"] == {"anchor": [1]}, (
        f"applying a label must replace that label's previous set: {out['replaced']}"
    )
    assert out["roWrote"] is False, "a read-only viewer must not write a label"
    assert out["roRegions"] == {}, (
        f"a label reached a read-only viewer's atoms: {out['roRegions']}")


def test_a_count_changing_edit_clears_the_selection(app_at, pairs):
    """A kept selection could point at an atom that is no longer the one it
    meant. A count-preserving transform leaves it alone — the two cases are
    genuinely different, which is why the operation table carries the effect on
    atom count (§ 11.1).

    The road's water, two of its atoms picked: the Translate box moves them;
    Delete removes them.
    """
    out = _run(app_at, pairs, """
        const m = await loaded("water");
        m.selection.add([0, 1]);

        // Same count back: the selection survives.
        await m.applyOp("translate", { dx: 1, dy: 0, dz: 0 });
        const afterTransform = m.selection.get();

        // One atom back where there were three: the count changed.
        m.selection.add([0, 1]);
        await m.applyOp("delete", {});
        console.log(JSON.stringify({
            afterTransform, afterShrink: m.selection.get(),
            atoms: m.getElements(),
        }));
        """)
    assert out["afterTransform"] == [0, 1], (
        "a count-preserving transform must leave the selection alone"
    )
    assert out["atoms"] == WATER["elements"][2:], (
        f"the delete did not remove the two picked atoms: {out['atoms']}")
    assert out["afterShrink"] == [], (
        f"an edit that changed the atom count must clear the selection: "
        f"{out['afterShrink']}"
    )


# ---------------------------------------------------------------------------
# § 6.2 / § 11.3 — the two doors carry the WHOLE structure (2026-08-20)
# ---------------------------------------------------------------------------

def test_identity_columns_and_channels_ride_the_two_doors_whole(app_at, pairs):
    """§ 11.3: an export of the truth carries everything the structure says —
    identity columns and per-atom channels included.  The way in is the fold
    of the server's answer onto each atom (``structureFromServer``); the way
    out the unfold back into the envelope (``structureForServer``).  The
    silent failure: a fold that reads fewer fields than the unfold rebuilds,
    so channels die at adoption and the first edit round-trip replaces every
    real residue name with the server's re-synthesized placeholder.  A
    metadata key the module does not shape -- the order record of a
    structure in an engine's order (`model/overview.md` § 2.2a) -- goes back
    as it came; a key the doors listed by hand would be dropped on every
    save.

    The peptide the build door makes, carrying :data:`CHANNELS`, opened for
    editing and translated: what the viewer holds is the fold, the request
    that leaves is the unfold, each against the pair on disk read by the
    codec.  The road's CO2 as an engine's decks hold it, opened as the
    Results tab opens a file -- as it is, its order record attached -- and
    exported.
    """
    from support.road import road_structure
    pep = road_structure(Path(pairs["peptide"]))
    out = _run(app_at, pairs, """
        const m = await loaded("peptide");
        const held = m.getStructure();
        await m.applyOp("translate", { dx: 1, dy: 0, dz: 0 });
        const sent = __calls.find(
            (c) => c.url.endsWith("/api/modify/translate")).body.structure;

        const o = await loaded("CO2, engine order", { mode: "readonly" });
        console.log(JSON.stringify({
            facts: held.annotations, title: held.title, defs: held.channelDefs,
            env: sent,
            order: o.exportFile().structure.metadata.atom_order,
        }));
        """)
    # THE FOLD: every column lands on its own atom, channels included.
    tag, value = CHANNELS["spin_up"], CHANNELS["charge"]
    for i, facts in enumerate(out["facts"]):
        channels = {**({"spin_up": True} if i in tag[1] else {}),
                    **({"charge": value[1][i]} if i in value[1] else {})}
        assert facts == {"labels": [], "residue": pep.residue_names[i],
                         "name": pep.atom_names[i],
                         "resid": pep.residue_ids[i], "chain": pep.chain_ids[i],
                         **({"channels": channels} if channels else {})}, (
            f"atom {i} lost a fact at the fold: {facts}")
    assert out["title"] == pep.title
    assert out["defs"]["spin_up"] == {"kind": "tag", **tag[2]}

    # THE UNFOLD: the envelope carries every column back, verbatim.
    env = out["env"]
    assert env["title"] == pep.title
    assert env["atom_names"] == pep.atom_names
    assert env["residue_ids"] == pep.residue_ids
    assert env["residue_names"] == pep.residue_names
    assert env["chain_ids"] == pep.chain_ids
    ch = env["metadata"]["annotations"]
    assert ch["spin_up"] == {"kind": "tag", "data": tag[1], **tag[2]}
    assert ch["charge"]["data"] == {str(k): v for k, v in value[1].items()}
    assert ch["charge"]["fdf"] == value[2]["fdf"]
    assert out["order"] == CO2_HELD_FIRST, (
        f"the order record did not go back as it came: {out['order']}")


def test_a_load_clears_the_selection(app_at, pairs):
    """A LOAD replaces the atoms wholesale, so a kept selection names
    nothing -- index 7 of the molecule just replaced is not index 7 of the
    one now open.

    The edit door clears only when the count changed, and that is right for
    an edit: a count-preserving transform moves the SAME atoms, so the
    selection still means what it meant
    (``test_a_count_changing_edit_clears_the_selection``).  A load has no
    such claim.

    Found in the browser 2026-08-24: after a 312-atom structure was
    replaced by ethanol, the atom list read **"75 of 9 selected"** -- the
    count from the new structure, the selection from the old.  The display
    was the visible half; the dangerous half is that "Delete selected" and
    "Assign" would have run against indices that no longer exist, or that
    now name different atoms.

    The road's H2, both atoms picked, then its water opened over it.
    """
    out = _run(app_at, pairs, """
        const m = await loaded("H2");
        m.selection.add([0, 1]);
        const before = m.selection.get().length;
        await opened(m, "water");          // a DIFFERENT molecule
        console.log(JSON.stringify({
            before: before,
            after: m.selection.get().length,
            atoms: m.getAtoms().length,
        }));
    """)
    assert out["before"] == 2, out
    assert out["after"] == 0, f"a load must clear the selection: {out}"
    assert out["atoms"] == len(WATER["elements"]), out


# --------------------------------------------------------------------- #
#  § 6.7a — empty is a state, and it is drawn                           #
# --------------------------------------------------------------------- #

def test_an_empty_structure_raises_the_axis_triad(app_at, pairs):
    """`molview.md` § 6.7a.  An axis in an empty window is how you tell a
    working viewer from a broken one -- the thing a blank card cannot say.

    The condition is *the window is empty*, not *which door emptied it*, so
    it settles in `settle` -- the one place every change to the structure
    passes through, and where the atom count becomes final.  The same shape
    as `stores.js`'s "isolate turns itself off when the selection empties".

    This is the FIRST of the three ways to arrive empty; the other two are
    `clear()` (below) and reopening a page whose saved state is empty.

    The road's water, every atom picked, Delete pressed.
    """
    out = _run(app_at, pairs, """
        const m = await loaded("water");
        const before = m.selection.switches().showAxis;
        m.selection.all(m.getElements().length);
        await m.applyOp("delete", {});
        console.log(JSON.stringify({
            before: before,
            atoms: m.getElements(),
            after:  m.selection.switches().showAxis,
        }));
    """)
    assert out["before"] is False, "the switch starts off (SWITCH_DEFAULTS)"
    assert out["atoms"] == [], f"the delete left atoms: {out['atoms']}"
    assert out["after"] is True, "an empty window did not raise the triad"


def test_deleting_every_atom_is_an_ORDINARY_EDIT(app_at, pairs):
    """**It keeps the metadata and the cell** *(user, 2026-09-02: "clear is
    just the init status or clear operation, but atom deletion would just
    update list properly")*.

    Wiping `info` here would make an ordinary edit quietly destructive -- a
    second thing the button does that its label does not say.  `clear()` is the door that means START EMPTY, and § 6.7a's table is
    the difference between them.

    The road's water in its box, an entry added on the Metadata page (the
    store edited as JSON, `info.edit`), then every atom picked and Delete
    pressed: the whole structure goes to the route, and what it answers is
    what the viewer holds.
    """
    out = _run(app_at, pairs, """
        const m = await loaded("water");
        m.info.edit("note", "mine");
        m.selection.all(m.getElements().length);
        await m.applyOp("delete", {});
        console.log(JSON.stringify({
            atoms: m.getElements(),
            info: m.info.get(),
            cell: m.getUnitCell(),
            mode: m.mode,
        }));
    """)
    assert out["atoms"] == [], f"the delete left atoms: {out['atoms']}"
    assert out["info"] == {"note": "mine"}, (
        "deleting atoms destroyed the metadata -- that is `clear`'s job: "
        f"{out['info']}")
    assert out["cell"] == np.diag(WATER["cell"]).tolist(), (
        f"deleting atoms destroyed the cell -- that is `clear`'s job: "
        f"{out['cell']}")


def test_a_persons_metadata_edit_is_an_edit_and_a_hosts_write_is_not(
        app_at, pairs, recorded):
    """`web/molview.md` § 8.4a *(user, 2026-10-03: "we need to allow user to
    edit it too"; "a: yes, b: yes")*: the person's doors -- `info.edit`,
    `info.drop`, the Metadata page's -- are gated and recorded: a point on
    the sequence each, taken back by undo; refused on a read-only viewer,
    whose store is rebuilt at every load.  The host's `set` stays silent.
    A block that names its `source` is stamped `edited_by_hand` when a
    person changes it; a value with no source is not; an edit that changes
    nothing records nothing.

    The road's H2 carrying the record its run leaves (:func:`recorded`),
    open for editing on the real workspace: the run's mesh cutoff changed
    on the Metadata page, an entry of the person's own added, retyped as
    it was, removed, and the removal undone; and the same pair in a
    read-only viewer.

    MUTATIONS THIS MUST FAIL AGAINST: the person's doors ungated; an edit
    not recorded; no stamp on a sourced block; a host's write recorded; a
    stamp on a block that names no source."""
    out = _run(app_at, dict(pairs, recorded=recorded), """
        const m = createModel({ owner: "s", workspace: ws });
        await opened(m, "recorded");
        const run = m.info.get().calculation;
        const beforeSet = m.state_index;
        m.info.set("note", "as loaded");                // the host's write
        await __quiet();
        const fresh = m.state_index;

        const edited = m.info.edit("calculation", Object.assign({}, run, {
            contract: Object.assign({}, run.contract, { mesh_cutoff: 400 }) }));
        const noted = m.info.edit("note", "mine");
        // A BLOCK OF THE PERSON'S OWN, naming no source: not stamped.
        const own = m.info.edit("mine", { value: 1 });
        await __quiet();
        const afterEdits = m.state_index;
        const same = m.info.edit("note", "mine");        // changes nothing
        await __quiet();
        const afterSame = m.state_index;
        const dropped = m.info.drop("note");
        await __quiet();
        const afterDrop = { at: m.state_index, info: m.info.get() };
        await m.undo();
        await __quiet();
        const afterUndo = m.info.get();

        const ro = await loaded("recorded", { mode: "readonly" });
        ro.info.set("note", "the run's");               // a host still writes
        const roEdit = ro.info.edit("note", "mine");
        const roDrop = ro.info.drop("note");
        console.log(JSON.stringify({
            run, beforeSet, fresh, edited, noted, own, afterEdits, same,
            afterSame, dropped,
            afterDrop, afterUndo, info: afterDrop.info,
            ro: { edit: roEdit, drop: roDrop, info: ro.info.get() },
        }));
    """)
    assert out["run"]["source"], f"the record names no source: {out['run']}"
    assert out["fresh"] == out["beforeSet"], (
        "the host's write recorded a point -- only a person's edit is one")
    assert out["edited"] is True and out["noted"] is True and out["own"], out
    assert out["afterEdits"] == out["fresh"] + 3, (
        f"each person's edit lays down a point (§ 11.2): {out}")
    assert out["afterSame"] == out["afterEdits"], (
        "an edit that changes nothing recorded a point")
    calc = out["info"]["calculation"]
    assert calc["contract"]["mesh_cutoff"] == 400 and calc.get("edited_by_hand"), (
        f"a record that names its source was not stamped: {calc}")
    assert "edited_by_hand" not in out["info"]["mine"], (
        f"a block naming no source was stamped: {out['info']['mine']}")
    assert out["dropped"] is True
    assert out["afterDrop"]["at"] == out["afterEdits"] + 1, out
    assert "note" not in out["afterDrop"]["info"], out
    assert out["afterUndo"].get("note") == "mine", (
        f"undo did not bring the dropped entry back: {out['afterUndo']}")
    assert out["ro"]["edit"] is False and out["ro"]["drop"] is False, (
        f"a read-only viewer took a person's edit: {out['ro']}")
    assert out["ro"]["info"] == {"calculation": out["run"],
                                 "note": "the run's"}, (
        f"a read-only viewer's store is not the run's and the host's: "
        f"{out['ro']['info']}")


def test_clear_is_the_door_that_means_start_empty(app_at, pairs):
    """§ 6.7a's other row: `clear()` returns the viewer to EMPTY -- nothing
    loaded, no atom identity -- and takes the metadata with it.

    Not the same as deleting every atom, and the `unit` half is why: in
    HOLDING `addFrames` accepts frames for the structure being held; in EMPTY
    it refuses by name (§ 10.8).  A `clear` that left the viewer HOLDING
    would leave it accepting frames for a structure that is not there.

    The road's water, an entry added on the Metadata page, then Clear."""
    out = _run(app_at, pairs, """
        const m = await loaded("water");
        m.info.edit("note", "keep me");
        const f0 = m.getFrameAllAtoms(0);
        const ok = m.clear();
        /* The refusal must be EMPTY's, by name.  Asserting only "it threw"
         * passed with `unit` left at HOLDING -- a null structure throws for
         * its own reasons, so the weaker check could not tell the two
         * lifecycle states apart (mutation-tested 2026-09-02). */
        let refusal = "";
        try { m.addFrames([f0]); } catch (e) { refusal = String(e.message || e); }
        console.log(JSON.stringify({
            ok: ok,
            info: m.info.get(),
            atoms: m.getElements(),
            refusal: refusal,
            showAxis: m.selection.switches().showAxis,
        }));
    """)
    assert out["ok"] is True
    assert out["info"] == {}, "`clear` left the metadata behind"
    assert out["atoms"] is None, "`clear` left a structure held"
    assert "nothing loaded" in out["refusal"], (
        "after `clear` the viewer did not refuse frames as EMPTY does "
        "(§ 10.8) -- it is still HOLDING: " + repr(out["refusal"]))
    assert out["showAxis"] is True


def test_clear_restarts_the_timeline_and_IS_the_state_it_starts_from(app_at,
                                                                     pairs):
    """§ 6.7a, *"`Clear` is where the sequence starts again"*.

    `Clear` means START EMPTY, so the sequence that could bring the old
    structure back is not the sequence of the thing now on the canvas: the
    position and the high-water mark go back to **#0**, the unsaved badge
    goes down, and the empty canvas is written as point 0 **and** as the
    draft -- so a reload finds the empty canvas rather than the structure
    that was discarded *(user, 2026-09-02: "timeline state should be #0,
    persistent data should be cleared" ... "when `clear()` is called, that
    becomes the persistent state and start of state")*.

    Both files are checked, because they are two writes with two rules
    (§ 11.3) and one of them was enough to pass while the other still held
    the discarded molecule.  The high-water mark is checked through
    behaviour -- a step forward off #0 finds nothing -- rather than by
    reading a number the module does not publish.

    The road's water on the real workspace: Save state (point 1), an atom
    picked and Delete pressed (point 2), then Clear; both files read back
    through the workspace's own door, and a fresh viewer over it in the
    same browser tab, as a reopened page builds one.
    """
    out = _run(app_at, pairs, """
        // A structure, a saved point and an edit -- so there is a real
        // sequence above 0 for `clear` to have to end.
        const m = createModel({ owner: "s", workspace: ws });
        await opened(m, "water");
        await m.save();                       // point 1
        m.selection.adopt([0]);
        await m.applyOp("delete", {});        // point 2 -- an edit records one
        await __quiet();
        const wasAt = m.state_index, wasDirty = m.uncommitted;
        const wasAbove0 = await pointsOf("s");

        m.clear();
        await __quiet();

        /* WHAT IS ON DISK, both files.  `state.structure` is null for an
         * empty canvas; anything else means the discarded molecule is still
         * the thing a reload would find. */
        const point0 = await ws.readState("s", 0);
        const draft  = await ws.readDraft("s");
        const held = (f) => !f ? "missing"
            : (f.state && f.state.structure ? "a structure" : "empty");

        const two = createModel({ owner: "s", workspace: ws });
        await two.load(0);

        console.log(JSON.stringify({
            wasAt, wasDirty, wasAbove0,
            at: m.state_index, dirty: m.uncommitted,
            point0: held(point0), draft: held(draft),
            draftAt: draft ? draft.at : null,
            reopenedAtoms: two.getElements(),
            points: await pointsOf("s"),
        }));
        """)
    # THE PRECONDITION IS A SEQUENCE WITH SOMETHING ABOVE #0, which is what
    # `clear` has to end.
    assert out["wasAt"] == 2, (
        "the setup did not build a sequence to end -- this test would pass "
        f"on a viewer that was already at #0: {out}")
    assert out["wasAbove0"] == [0, 1, 2], (
        f"the setup wrote no points above #0 for `clear` to prune: {out}")
    assert out["wasDirty"] is False, (
        "an edit left work off the sequence, so it did not checkpoint "
        f"(§ 11.2): {out}")
    assert out["at"] == 0, (
        f"`clear` left the timeline standing at #{out['at']}")
    assert out["dirty"] is False, "`clear` left the unsaved badge up"
    assert out["point0"] == "empty", (
        f"point 0 still holds {out['point0']} -- a Retract would bring the "
        "discarded structure back")
    assert out["draft"] == "empty", (
        f"the draft still holds {out['draft']} -- reopening the page would "
        "restore the structure that was cleared")
    assert out["draftAt"] == 0, (
        f"the draft says the user is standing at #{out['draftAt']}")
    assert out["reopenedAtoms"] is None, (
        f"a reopened page came back holding atoms: {out['reopenedAtoms']}")
    assert out["points"] == [0], (
        "the points above 0 were left on the server, so a later session could "
        f"still step forward into the discarded structure: {out['points']}")


def test_reopening_a_cleared_page_still_shows_the_triad(app_at, pairs):
    """§ 6.7a's *"Why not at the doors"*: the THIRD way to arrive empty.

    Neither the edit path nor `clear()` runs when a page is reopened -- the
    restore path installs the saved state through `settle` and touches
    neither door, so a rule at the doors would bring a cleared page back a
    grey rectangle indistinguishable from a broken viewer.

    A fresh viewer is the point.  The switch starts OFF on one
    (`SWITCH_DEFAULTS`), so nothing here can pass by inheriting the switch
    the first viewer raised.  The road's water opened and cleared on the
    real workspace, then the page reopened in the same browser tab.
    """
    out = _run(app_at, pairs, """
        const one = createModel({ owner: "s", workspace: ws });
        await opened(one, "water");
        one.clear();
        await __quiet();

        const two = createModel({ owner: "s", workspace: ws });
        const fresh = two.selection.switches().showAxis;
        await two.load(0);
        console.log(JSON.stringify({
            fresh,
            atoms: two.getElements(),
            showAxis: two.selection.switches().showAxis,
        }));
        """)
    assert out["fresh"] is False, (
        "the switch did not start off on a fresh viewer, so this test cannot "
        "tell the rule from an inherited switch")
    assert out["atoms"] is None, "the reopened page did not come back empty"
    assert out["showAxis"] is True, (
        "a reopened empty page drew nothing at all -- the triad rule did not "
        "reach the restore path")


# ---------------------------------------------------------------------------
# § 8.4a — two flags, because they invalidate different things
# ---------------------------------------------------------------------------

def _with_a_recorded_contract(extra=""):
    """The pair that carries the record its run leaves (:func:`recorded`),
    opened for editing -- the record arrives the way it really does, in the
    structure the load door answers (`structure.info`) -- then ``extra``,
    then the two flags as the store holds them."""
    return """
        const m = await loaded("recorded");
        const had = m.info.get().calculation;
    """ + extra + """
        const c = (m.info.get() || {}).calculation || {};
        console.log(JSON.stringify({had: !!had,
                                    structure: !!c.structure_modified,
                                    labels: !!c.labels_modified}));
    """


def test_a_label_write_flags_labels_and_not_the_structure(app_at, pairs,
                                                         recorded):
    """A rename moves the regions, not the convergence.

    `molview.md` § 8.4a. The settings a later calculation inherits are the
    mesh cutoff and the transverse k-mesh, and **no setting is a function of
    a name** -- so a label write must not raise the flag that says they were
    converged for a cell that is gone. With one flag for both, warning here
    would mean telling a person their mesh cutoff is suspect because they renamed an
    electrode, and on a junction renaming an electrode is routine.

    The Label box writing ``device`` onto the first atom.
    """
    out = _run(app_at, dict(pairs, recorded=recorded), _with_a_recorded_contract(
        'm.selection.writeLabel("device", "replace", [0]);'))
    assert out["had"] is True, "the pair opened without its record"
    assert out["labels"] is True, "a label write set no flag at all"
    assert out["structure"] is False, (
        "a label write claimed the geometry changed -- a reader acting on "
        "that tells people their convergence is suspect because they renamed "
        "a region")


def test_a_cell_edit_flags_the_structure_and_not_the_labels(app_at, pairs,
                                                           recorded):
    """The other half, and it has to be checked separately.

    Mesh cutoff is a grid density over the CELL and the transverse k-mesh
    samples the reciprocal cell, so a cell edit is exactly what invalidates
    the inherited settings.

    WHO DECIDES IT MOVED (2026-09-21).  `periodicity_gate.apply_edit` sets
    the flag and the answer carries it; the browser adopts the answer, and
    what the store ends up holding is the browser's half.  On the real route
    both halves run here together: the Cell page's Apply -- its one
    ``block``, a 9 Å periodic box typed -- sent to
    `/api/structure/periodicity`, and the store read after.  The decision on
    its own is pinned where it is made, over all five ops of the door --
    `test_an_edit_on_a_recorded_structure_e2e.py`
    ::test_a_box_edit_marks_the_record_outdated.
    """
    out = _run(app_at, dict(pairs, recorded=recorded), _with_a_recorded_contract("""
        await m.commitPeriodicityOp("block", {
            axis_kind: ["periodic", "periodic", "periodic"], vacuum: null,
            cell: [[9,0,0],[0,9,0],[0,0,9]], box_corner: null });
    """))
    assert out["had"] is True, "the pair opened without its record"
    assert out["structure"] is True, "the browser did not adopt the server's mark"
    assert out["labels"] is False, "a cell edit claimed the labels moved"


def test_a_structure_with_no_recorded_contract_is_flagged_at_all(app_at, pairs):
    """Nothing to outdate, so nothing is written.

    The flags live INSIDE `info.calculation`. A structure that never came
    from a run has no such block, and the viewer must not invent one -- a
    bare `{structure_modified: true}` would reach `recorded_contract_of`,
    which answers None without a contract, and would be dead weight in every
    pair the Modify tab ever saves.

    The road's water, which no run left, and the Label box writing
    ``device`` onto its first atom.
    """
    out = _run(app_at, pairs, """
        const m = await loaded("water");
        m.selection.writeLabel("device", "replace", [0]);
        const info = m.info.get() || {};
        console.log(JSON.stringify({none: !info.calculation,
                                    labelled: m.getRegions()}));
    """)
    assert out["labelled"] == {"device": [0]}, "the label write did not land"
    assert out["none"] is True, (
        "the viewer invented a calculation block on a structure that never "
        "came from a run")
