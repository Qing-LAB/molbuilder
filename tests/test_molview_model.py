"""The model — every test derived from ``docs/web/molview.md``, never from the
source it checks (§ 13).

Step D of the rebuild (``docs/web/molview-rework-plan.md``). The rows of § 13.3
guarded here:

    § 6.4  nothing keeps its own copy / master copy, then range, then frame, then notify
    § 9.3  a read cannot be used to write
    § 9.3  one need, one main way in
    § 9.3  the facts a request carries are read together
    § 9.3  a structure that cannot be written out is not written out
    § 9.4  read-only freezes the master copy and nothing else
    § 6.3  each question goes to the copy that can answer it
    § 6.1  one frame is not a special case
    § 11.1 the count requirement is checked first
    § 11.1 an empty selection means what the table says
    § 11.1 a failed edit changes nothing
    § 6.6  MolView interprets no reserved label

The server is a stand-in that obeys the document's account of it (§ 13.1): it
answers the three routes of § 11.1 and nothing else.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests._node_esm import run_node

REPO = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO / "molbuilder" / "web" / "static" / "lib" / "molview"
MODEL = MODULE_DIR / "model.js"
# The pipeline's scene derivation, so a test can ask the DRAWING what cell it
# would draw and compare it with what the Cell page reports (§ 5.2).
ENGINE = MODULE_DIR / "render-engine.js"

# A stand-in server. It answers the three routes of § 11.1 and records what it
# was sent, so a test can assert what actually left the browser.
SERVER = """
globalThis.__requests = [];
globalThis.__serverFails = false;
globalThis.__nextPayload = null;

function atomRow(i, element, x, opts) {
    /* The labels the atom carries, and nothing beside them: a reserved label
     * is IN `regions` (§ 6.6). The row used to carry an `is_frozen` flag too,
     * which is what the server sent while it kept a second store. */
    return Object.assign({ index: i, element, x, y: 0, z: 0, regions: [] },
                         opts || {});
}
globalThis.__atomRow = atomRow;

globalThis.__payload = function (atoms, extra) {
    return Object.assign({ atoms }, extra || {});
};

/* THE STAND-IN SPEAKS THE SERVER'S NAMES, not the module's (§ 13.1: a stand-in
 * "must obey THAT LEVEL's rules"). Its periodicity block is `{cell, cell_origin,
 * axis_kind, vacuum}` — what /api/build/load actually sends. It used to carry
 * `{lattice, origin}`, which is what the MODULE calls them, so the module's
 * inbound translation was tested against its own output and the cell was silently
 * null against the real server for every structure ever loaded. */

globalThis.fetch = async function (route, init) {
    const body = JSON.parse(init.body);
    globalThis.__requests.push({ route, body });
    if (globalThis.__serverFails) {
        return { ok: false, status: 500, json: async () => ({}) };
    }
    const payload = globalThis.__nextPayload || globalThis.__payload([
        atomRow(0, "C", 0), atomRow(1, "O", 1),
    ]);
    return { ok: true, status: 200, json: async () => payload };
};
"""

PRELUDE = f"""
const {{ createModel }} = await import({json.dumps(MODEL.resolve().as_uri())});
const ENGINE = await import({json.dumps(ENGINE.resolve().as_uri())});

async function loaded(opts) {{
    globalThis.__requests = [];
    globalThis.__serverFails = false;
    globalThis.__nextPayload = null;
    const m = createModel(opts || {{}});
    await m.installMolecule({{ text: "2\\n\\nC 0 0 0\\nO 1 0 0\\n", filename: "x.xyz" }});
    return m;
}}
"""


def _run(snippet: str):
    return run_node([], PRELUDE + snippet, globals_js=SERVER)


# ---------------------------------------------------------------------------
# § 9.3 — a read cannot be used to write
# ---------------------------------------------------------------------------

def test_changing_what_a_read_returned_leaves_the_viewer_untouched():
    """§ 9.3: "Every read of data returns a COPY, so changing what you were given
    can never change the viewer."
    """
    out = _run(
        """
        const m = await loaded();
        const s = m.getStructure();
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
            cell: after.periodicity,
            x: m.getFrameAllAtoms(0)[0][0],
        }));
        """
    )
    assert out["element"] == "C", "a read handed out the model's own array"
    assert out["labels"] == [], "a read handed out the model's own labels"
    assert out["cell"] is None, "a read handed out a writable structure"
    assert out["x"] == 0, "a read handed out the model's own coordinates"


def test_no_read_at_all_can_be_written_through():
    """The same rule as above, asked of EVERY read rather than of three.

    § 13.1 forbids a pinned list of names — "a transcription, not a contract" —
    so the reads are enumerated from the surface at run time and each one's
    result is mutated as destructively as its shape allows. What is asserted is
    the RULE: after all of that, the structure and the coordinates are what they
    were.

    Written because the three-read version passed while `getUnitCellInfo` — the
    MAIN way in for the cell (§ 9.3's table) — handed out live references into
    the master copy. The four narrower cuts beside it all copied, so the one
    read a caller is told to use was the one that could be written through.
    """
    out = _run(
        """
        // A structure WITH A CELL, because the cell reads are the ones this was
        // written for — and with nothing to hand out, they hand out nothing and
        // the test proves nothing.
        globalThis.__nextPayload = globalThis.__payload(
            [atomRow(0, "C", 0), atomRow(1, "O", 1)],
            { periodicity: { cell: [[8,0,0],[0,8,0],[0,0,8]], cell_origin: [1,1,1],
                             axis_kind: ["periodic","periodic","isolated"],
                             vacuum: [0,0,12] } });
        const m = createModel({});
        await m.installMolecule({ text: "x", filename: "x.xyz" });
        if (!m.getUnitCellInfo().cell) throw new Error("the fixture has no cell");

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
        """
    )
    assert len(out["reads"]) >= 10, (
        f"the enumeration found almost nothing — it is not exercising the "
        f"surface: {out['reads']}"
    )
    assert out["unchanged"] is True, (
        "one of the model's reads handed out something that writes through to "
        f"the master copy; the reads tried were {out['reads']}"
    )


def test_with_nothing_loaded_a_read_returns_nothing_not_an_empty_structure():
    """§ 9.3: "there is nothing here" and "here is a structure with no atoms" are
    different answers, and a caller has to be able to tell them apart.

    An empty structure walks straight past every `if (!data) return` guard.
    """
    out = _run(
        """
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
        """
    )
    for name in ("structure", "atoms", "elements", "coordinates", "frame",
                 "regions", "exported"):
        assert out[name] is None, (
            f"{name} answered with an empty structure instead of nothing"
        )
    assert out["count"] == 0


def test_a_narrower_cut_cannot_disagree_with_the_main_way_in():
    """§ 13.3: "a narrower cut returns exactly what the main way in holds for
    that field — the two cannot disagree."

    § 9.3: a cut may disappear, but it must never grow into a rival.
    """
    out = _run(
        """
        globalThis.__nextPayload = globalThis.__payload([
            globalThis.__atomRow(0, "C", 0, { regions: ["anchor"] }),
            globalThis.__atomRow(1, "O", 1, { regions: ["frozen_atoms"], residue_name: "ALA" }),
        ], { periodicity: { cell: [[4,0,0],[0,4,0],[0,0,4]], cell_origin: [1,1,1] } });
        const m = createModel({});
        await m.installMolecule({ text: "x", filename: "x.xyz" });

        m.addFrames([[[0,0,1],[1,0,1]]]);

        const whole = m.getStructure();
        console.log(JSON.stringify({
            elementsAgree: JSON.stringify(m.getElements()) === JSON.stringify(whole.elements),
            atomsAgree: m.getAtoms().map(a => a.element).join() === whole.elements.join(),
            cellAgree: JSON.stringify(m.getUnitCell()) === JSON.stringify(whole.periodicity.cell),
            originAgree: JSON.stringify(m.getUnitCellOrigin()) === JSON.stringify(whole.periodicity.cell_origin),
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
        """
    )
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
    assert out["regions"]["anchor"] == [0], "labels group into name -> atoms"
    assert out["frozen"] == [1], (
        "the frozen cut reads the reserved label off the same one mechanism"
    )
    assert out["infoCell"] == [[4, 0, 0], [0, 4, 0], [0, 0, 4]], (
        "the cell as it will be used must agree with the raw cell when the "
        "structure states one — the resolved value only fills in what was left "
        "unsaid (§ 9.3)"
    )


def test_the_cell_page_and_the_drawing_cannot_describe_different_structures():
    """§ 5.2 and § 9.3: "the cell as it will actually be used" is one fact, so
    the panel that reports it and the pipeline that draws it must read the same
    answer.

    They did not. `getUnitCellInfo` read the RESOLVED values and `sceneFor` read
    the RAW ones, so for a structure with no explicit cell — every plain `.xyz` —
    the Cell page listed a lattice while the drawing had none and "Show unit
    cell" drew nothing. Two readers of one fact, disagreeing, with no error
    anywhere.
    """
    out = _run(
        """
        // The server's own block for a structure nobody gave a cell to: the raw
        // field is empty and the box it worked out sits beside it.
        globalThis.__nextPayload = globalThis.__payload(
            [globalThis.__atomRow(0, "O", 0), globalThis.__atomRow(1, "H", 1)],
            { periodicity: {
                axis_kind: ["isolated","isolated","isolated"],
                cell: null, cell_origin: null, vacuum: [0,0,0],
                resolved_cell: [[7,0,0],[0,7,0],[0,0,6]],
                resolved_cell_origin: [-3,-3,-3],
                resolved_vacuum: [3,3,3] } });
        const m = createModel({});
        await m.installMolecule({ text: "x", filename: "x.xyz" });

        const panel = m.getUnitCellInfo();          // what the Cell page shows
        const drawn = ENGINE.sceneFor(m.getStructure().periodicity);   // what is drawn

        console.log(JSON.stringify({
            panelCell:  panel.cell,
            drawnCell:  drawn.cellBox && drawn.cellBox.lattice,
            panelOrigin: panel.cell_origin,
            drawnOrigin: drawn.cellBox && drawn.cellBox.origin,
            // The RAW reads still say what the structure itself states, which is
            // nothing — that is their job (§ 9.3) and it is not a disagreement.
            rawCell: m.getUnitCell(),
        }));
        """
    )
    assert out["panelCell"] == [[7, 0, 0], [0, 7, 0], [0, 0, 6]], (
        "the Cell page lost the cell the structure actually uses"
    )
    assert out["drawnCell"] == out["panelCell"], (
        f"the drawing and the Cell page describe different cells: drawn "
        f"{out['drawnCell']} vs panel {out['panelCell']}"
    )
    assert out["drawnOrigin"] == out["panelOrigin"] == [-3, -3, -3], (
        f"the box is not anchored where the panel says it is: {out}"
    )
    assert out["rawCell"] is None, (
        "the raw read must still report what the structure itself states — "
        "nothing — or it has stopped being the narrower cut § 9.3 describes"
    )


def test_what_the_viewer_is_has_one_answer_and_a_replacement_can_be_enforced():
    """§ 5.2, applied to the viewer's own lifecycle.

    "Has this viewer got its core data?" is ONE question and was answered two
    ways — a `seeded` flag at the install door, and "is the atom count zero?" at
    the frame doors — which could disagree. It is one state now, checked and set
    in one place, so coming back to the install door never finds the initial
    state a second time.

    And no state is a dead end. A read-only viewer refuses to have its structure
    swapped out from under it, but the HOST can say it means to: deciding which
    structure a viewer shows is the host's business, and the host is the one that
    asked for read-only. What read-only protects is the core data from being
    EDITED (§ 9.4) — `applyOp`, the cell, the labels — and those stay shut
    whatever is passed here.

    The viewer does not judge what it was handed: a structure with no atoms is a
    structure with no atoms, and carrying it is the module's job (§ 6.2).
    """
    out = _run(
        """
        const m = createModel({ mode: "readonly" });
        await m.installMolecule({ text: "x", filename: "x.xyz" });   // leaves EMPTY
        const held = m.getAtoms().length;

        // A second install is refused: not a dead end, a default.
        globalThis.__nextPayload = globalThis.__payload([
            globalThis.__atomRow(0, "N", 5)]);
        const casual = await m.installMolecule({ text: "y", filename: "y.xyz" });
        const afterCasual = m.getAtoms().length;

        // Said deliberately, it lands.
        const enforced = await m.installMolecule({
            text: "y", filename: "y.xyz", enforce: true });
        const afterEnforced = m.getAtoms().map((a) => a.element);

        // And editing is still shut, enforce or not.
        globalThis.__requests = [];
        await m.applyOp("translate", { dx: 1 });
        m.selection.adopt([0]);
        await m.selection.writeLabel("smuggled", "replace");

        console.log(JSON.stringify({
            held, casual, afterCasual,
            enforced: enforced !== null, afterEnforced,
            labels: m.getAtoms()[0].labels,
            requests: globalThis.__requests.length,
        }));
        """
    )
    assert out["held"] == 2, "the first install is allowed in any mode"
    assert out["casual"] is None and out["afterCasual"] == 2, (
        "a read-only viewer let its structure be swapped without being asked"
    )
    assert out["enforced"] is True and out["afterEnforced"] == ["N"], (
        f"an enforced install did not land: {out}"
    )
    assert out["labels"] == [], (
        "editing reached the core data in a read-only viewer"
    )
    assert out["requests"] == 0, (
        "an edit left for the network in a read-only viewer"
    )


def test_every_door_answers_what_the_read_only_table_says_and_none_of_them_throws():
    """§ 9.4's per-door table, pinned as values rather than as prose.

    "It returns without effect AND WITHOUT THROWING" is the rule, not a
    courtesy: a viewer that threw would make every caller wrap its writes, which
    is the list of special cases § 9.4 exists to avoid. So each swallowed door is
    checked for BOTH — that nothing happened, and that the caller got an answer
    it can test rather than an exception.

    The answers differ by door (`null`, `false`, `undefined`) because they are
    each that call's own "no", not one invented for the gate.
    """
    out = _run(
        """
        const store = { read: async () => null, write: async () => {} };
        const m = createModel({ mode: "readonly", workspace: store });
        await m.installMolecule({ text: "x", filename: "x.xyz" });   // leaves EMPTY
        await new Promise((r) => setTimeout(r, 0));
        m.addFrames([[[0,0,1],[1,0,1]]]);        // delivery: runs
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

        await door("installOver",  () => m.installMolecule({ text: "y", filename: "y.xyz" }));
        await door("applyOp",      () => m.applyOp("translate", { dx: 1 }));
        await door("cell",         () => m.commitPeriodicityOp("cell", [[9,0,0],[0,9,0],[0,0,9]]));
        await door("writeLabel",   () => m.selection.writeLabel("smuggled", "replace"));
        await door("reloadFrames", () => m.reloadFrames([[[5,0,0],[6,0,0]]]));
        await door("save",         () => m.save(1));
        await door("load",         () => m.load(0));
        await door("undo",         () => m.undo());
        // The count guard sits INSIDE the door, so a misshapen reload that the
        // gate already answered must not raise either.
        await door("reloadMisshapen", () => m.reloadFrames([[[9,9,9]]]));

        console.log(JSON.stringify({
            threw, answered, mode: m.mode,
            unchanged: before === JSON.stringify({
                structure: m.getStructure(), frames: m.frameCount() }),
        }));
        """
    )
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


def test_the_reads_answer_nothing_when_nothing_is_loaded_except_the_counts():
    """§ 9.3's surface table, at the one moment it is easiest to get wrong.

    "With nothing loaded, a read returns NOTHING rather than an empty structure"
    — but three entries are deliberately not `null`, and a caller has to be able
    to rely on that: the two counts are counts, and `mode` is what the viewer IS
    rather than what it holds. `getUnitCellInfo` is the fourth: it is "the cell
    as it will actually be used... so it ALWAYS has an answer", which means an
    object with empty fields, not nothing.
    """
    out = _run(
        """
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
        """
    )
    for name, value in out["reads"].items():
        assert value is None, (
            f"{name} answered with an empty structure instead of nothing: {value}"
        )
    assert out["frame"] is None
    assert out["cellInfo"] == {"cell": None, "cell_origin": None,
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


def test_appending_is_the_same_in_both_modes_but_replacing_is_not():
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
    """
    out = _run(
        """
        const store = { read: async () => null, write: async () => {} };
        const answer = {};
        for (const mode of ["editable", "readonly"]) {
            const m = createModel(mode === "readonly"
                ? { mode: "readonly", workspace: store } : { workspace: store });
            await m.installMolecule({ text: "x", filename: "x.xyz" });
            await new Promise((r) => setTimeout(r, 0));

            // APPENDING — identical either way.
            m.addFrames([[[0,0,1],[1,0,1]], [[0,0,2],[1,0,2]]],
                        { forces: [[[0,0,1],[0,0,1]], null] });
            m.addFrame([[0,0,3],[1,0,3]]);
            m.setForces([null, null, null, [[0,0,4],[0,0,4]]]);

            answer[mode] = {
                frames: m.frameCount(),
                frameZero: m.getFrameAllAtoms(0),
                forces: m.getCoordinates().forcesPerFrame,
                badgeAfterDelivery: m.uncommitted,
            };

            // REPLACING — an edit-shaped act, so read-only wants to be asked.
            m.reloadFrames([[[99,0,0],[99,0,0]]]);
            answer[mode].afterCasualReload = m.frameCount();
            m.reloadFrames([[[99,0,0],[99,0,0]]], { enforce: true });
            answer[mode].afterEnforcedReload = m.frameCount();
        }

        // An EDIT does raise the badge, so it is not simply dead.
        const e = createModel({ workspace: store });
        await e.installMolecule({ text: "x", filename: "x.xyz" });
        await new Promise((r) => setTimeout(r, 0));
        e.selection.adopt([0]);
        e.selection.writeLabel("bridge", "replace");
        answer.badgeAfterAnEdit = e.uncommitted;

        console.log(JSON.stringify(answer));
        """
    )
    for mode in ("editable", "readonly"):
        got = out[mode]
        assert got["frames"] == 4, f"{mode}: appending did not extend: {got}"
        assert got["frameZero"] == [[0, 0, 0], [1, 0, 0]], (
            f"{mode}: appending altered frame 0 — an append must leave what the "
            f"run produced exactly as it was: {got['frameZero']}"
        )
        assert got["forces"] == [None, None, None, [[0, 0, 4], [0, 0, 4]]]
        assert got["badgeAfterDelivery"] is False, (
            f"{mode}: arriving frames raised the unsaved badge"
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
    assert out["badgeAfterAnEdit"] is True, (
        "an edit must still raise the badge, or the badge is simply dead"
    )


def test_a_trajectory_arrives_whole_in_one_install():
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
    """
    out = _run(
        """
        const slots = new Map();
        // The real front door (workspace.md § 5), answered from a Map.
        const m = createModel({ workspace: {
            workspaceId: () => "id",
            persist: (tag, session, point, identity) => {
                if (point) slots.set(identity.state_index, point);
                return true;
            },
            readState: async (identity) => (slots.has(identity.state_index)
                ? slots.get(identity.state_index) : null),
            pruneStatesAbove: () => {},
        } });
        const seen = [];
        m.subscribe(() => seen.push(m.frameCount()));

        await m.installMolecule({
            text: "x", filename: "run.xyz",
            frames: [[[0,0,0],[1,0,0]], [[0,0,1],[1,0,1]], [[0,0,2],[1,0,2]]],
            forces: [null, null, [[0,0,9],[0,0,9]]],
        });
        const installed = { frames: m.frameCount(), at: m.currentFrame(),
                            forces: m.getCoordinates().forcesPerFrame };

        // Let the anchor's write land, then go back to it.
        await new Promise((r) => setTimeout(r, 0));
        await m.load(0);

        // A frame that does not carry the loaded atoms is refused at the load,
        // exactly as it is at an append (§ 10.8).
        let refused = null;
        try {
            await m.installMolecule({ text: "x", filename: "b.xyz",
                                      frames: [[[0,0,0]]] });
        } catch (e) { refused = e.message; }

        console.log(JSON.stringify({
            installed, seen, afterRetract: m.frameCount(), refused,
        }));
        """
    )
    assert out["installed"]["frames"] == 3, (
        f"the trajectory did not arrive with the load: {out['installed']}"
    )
    assert out["installed"]["at"] == 0, "a full load resets to frame 0 (§ 10.8)"
    assert out["installed"]["forces"] == [None, None, [[0, 0, 9], [0, 0, 9]]]
    # § 6.4 is about WHAT a subscriber sees, not how often it is told: every
    # notification must show a settled structure. More than one is fine — the
    # anchor's badge is a second — but a `1` among them is the half-updated
    # state the rule forbids, and that is exactly what the two-call load gave.
    assert out["seen"] and all(n == 3 for n in out["seen"]), (
        "a subscriber saw the structure part-way in — the frames landed in a "
        f"second settle after the atoms: {out['seen']}"
    )
    assert out["afterRetract"] == 3, (
        "retracting to the anchor lost the trajectory, because point 0 was laid "
        "down before the frames arrived (§ 11.2)"
    )
    assert out["refused"] and "§ 10.8" in out["refused"], (
        f"a frame that does not fit the loaded atoms was accepted: {out['refused']}"
    )


def test_an_atom_carries_a_label_once_however_often_it_is_applied():
    """§ 6.2: an atom's facts are "the labels it carries" — a set of names.
    Carrying the same name twice is not a state the model may hold.

    Two halves, because they fail differently:

    **Applying cannot create one.** Assigning a label an atom already has, twice
    over, leaves one — the write strips every occurrence of the name before
    adding one back, so it is impossible by construction rather than by a check
    that could be forgotten at a fourth call site.

    **An arriving one is dropped.** The count TRAVELS: `groupByLabel` turns a
    name an atom carries twice into that atom's index listed twice, and that goes
    out in `regions`, into the sidecar, and into the generated input — where
    `frozen_atoms: [0, 0]` is the same atom held still twice. The writing side
    cannot produce it, so the only way in is a payload that already had it.
    """
    out = _run(
        """
        const clean = await loaded();
        clean.selection.adopt([0]);
        clean.selection.writeLabel("bridge", "replace");
        clean.selection.writeLabel("bridge", "add");      // it already has it
        clean.selection.writeLabel("bridge", "add");      // and again
        clean.selection.writeLabel("bridge", "replace");  // and replace with itself
        const afterRepeats = {
            labels:  clean.getAtoms()[0].labels,
            regions: clean.getRegions(),
        };

        // A payload that arrives already carrying the name twice.
        globalThis.__nextPayload = globalThis.__payload([
            globalThis.__atomRow(0, "C", 0, { regions: ["bridge", "bridge"] }),
            globalThis.__atomRow(1, "O", 1, { regions: [] }),
        ]);
        const m = createModel({});
        await m.installMolecule({ text: "x", filename: "x.xyz" });

        console.log(JSON.stringify({
            afterRepeats,
            arrived: m.getAtoms()[0].labels,
            grouped: m.getRegions(),
            leaving: m.exportFile().structure.metadata.regions,
        }));
        """
    )
    assert out["afterRepeats"]["labels"] == ["bridge"], (
        "applying a label an atom already carries added it again: "
        f"{out['afterRepeats']['labels']}"
    )
    assert out["afterRepeats"]["regions"] == {"bridge": [0]}
    assert out["arrived"] == ["bridge"], (
        f"a repeated name survived the way in: {out['arrived']}"
    )
    assert out["grouped"] == {"bridge": [0]}, (
        f"one atom was listed twice under one label: {out['grouped']}"
    )
    assert out["leaving"] == {"bridge": [0]}, (
        "the duplicate reached the wire, so it would reach the sidecar and the "
        f"generated input: {out['leaving']}"
    )


def test_the_reserved_label_needs_no_boundary_translation():
    """§ 6.6: MolView's end is "one mechanism, no special case".

    This used to assert a FOLD — the server sent an `is_frozen` flag beside the
    labels and the module turned it into a label on the way in. The server keeps
    one store now, so there is nothing to fold: the label arrives as a label and
    the translator that existed for it is gone.

    (The reserved label's full contract — one store, one designated read, both
    boundaries — is `test_molview_reserved_label_js.py`. This keeps the § 6.6 row
    of § 13.3 anchored in the model's own suite.)
    """
    out = _run(
        """
        globalThis.__nextPayload = globalThis.__payload([
            globalThis.__atomRow(0, "C", 0, { regions: ["mine", "frozen_atoms"] }),
        ]);
        const m = createModel({});
        await m.installMolecule({ text: "x", filename: "x.xyz" });
        console.log(JSON.stringify({
            labels: m.getAtoms()[0].labels,
            regions: Object.keys(m.getRegions()).sort(),
            frozen: m.getFrozen(),
        }));
        """
    )
    assert out["labels"] == ["mine", "frozen_atoms"], (
        f"the reserved label is not carried like any other: {out['labels']}"
    )
    assert out["regions"] == ["frozen_atoms", "mine"]
    assert out["frozen"] == [0], "the designated read did not find it in the labels"



# ---------------------------------------------------------------------------
# § 6.4 — master copy, then range, then frame, then notify
# ---------------------------------------------------------------------------

def test_no_subscriber_sees_a_new_range_beside_an_old_frame_number():
    """§ 13.3: "after a load that shortens a trajectory, no subscriber ever sees
    a range from the new structure beside a frame number from the old one."

    § 6.4: "No one ever observes a half-updated state."
    """
    out = _run(
        """
        const m = await loaded();
        // Two atoms per frame, because the structure was opened with two and
        // § 10.8 fixes that identity for every frame after it.
        m.reloadFrames([[[0,0,0],[1,0,0]], [[1,0,0],[2,0,0]], [[2,0,0],[3,0,0]],
                        [[3,0,0],[4,0,0]], [[4,0,0],[5,0,0]]]);
        m.setCurrentFrame(4);

        // Watch what every notification shows.
        const seen = [];
        m.subscribe(() => seen.push({ at: m.currentFrame(), of: m.frameCount() }));
        m.onFrameChange(() => seen.push({ at: m.currentFrame(), of: m.frameCount() }));

        m.reloadFrames([[[0,0,0],[1,0,0]], [[1,0,0],[2,0,0]]]);   // it shortens
        console.log(JSON.stringify({
            seen,
            bad: seen.filter(s => s.at >= s.of),
            finalAt: m.currentFrame(),
            finalOf: m.frameCount(),
        }));
        """
    )
    assert out["bad"] == [], (
        f"a subscriber saw a frame outside its own range: {out['seen']}"
    )
    assert out["finalAt"] == 0 and out["finalOf"] == 2


def test_an_out_of_range_write_is_resolved_against_the_range():
    """§ 6.4: "A number outside the range is resolved against the range, never
    taken on trust." Not an error — a slider at the end of a trajectory that just
    got shorter is asking a reasonable question.
    """
    out = _run(
        """
        const m = await loaded();
        m.reloadFrames([[[0,0,0],[1,0,0]], [[1,0,0],[2,0,0]], [[2,0,0],[3,0,0]]]);
        const seen = [];
        m.setCurrentFrame(99);   seen.push(m.currentFrame());
        m.setCurrentFrame(-5);   seen.push(m.currentFrame());
        m.setCurrentFrame(1.7);  seen.push(m.currentFrame());
        console.log(JSON.stringify({ seen }));
        """
    )
    assert out["seen"] == [2, 0, 1], (
        f"an out-of-range seek must land inside the range: {out['seen']}"
    )


def test_one_write_reaches_every_subscriber_whatever_moved_it():
    """§ 13.3: "one write reaches EVERY subscriber, whatever moved it."

    § 6.4: a subscriber never has to know which; nothing anywhere needs its own
    "did it change?" check.
    """
    out = _run(
        """
        const m = await loaded();
        m.reloadFrames([[[0,0,0],[1,0,0]], [[1,0,0],[2,0,0]], [[2,0,0],[3,0,0]]]);
        let a = 0, b = 0;
        m.onFrameChange(() => a++);
        m.onFrameChange(() => b++);
        m.setCurrentFrame(2);
        const both = { a, b };
        m.setCurrentFrame(2);                      // no move, no notification
        console.log(JSON.stringify({ both, after: { a, b } }));
        """
    )
    assert out["both"] == {"a": 1, "b": 1}, (
        f"a frame write did not reach every subscriber: {out['both']}"
    )
    assert out["after"] == {"a": 1, "b": 1}, (
        "setting the frame to where it already is must notify nobody"
    )


def test_appending_frames_recomputes_the_range_from_the_master_copy():
    """§ 6.4 step 2: the range is recomputed FROM THE MASTER COPY — "not from the
    drawing copy, and not from what the caller said it was adding."
    """
    out = _run(
        """
        const m = await loaded();
        m.reloadFrames([[[0,0,0],[1,0,0]]]);
        const start = m.frameCount();
        m.addFrames([[[1,0,0],[2,0,0]], [[2,0,0],[3,0,0]]]);
        const grown = m.frameCount();
        m.addFrame([[3,0,0],[4,0,0]]);
        console.log(JSON.stringify({ start, grown, final: m.frameCount() }));
        """
    )
    assert out["start"] == 1 and out["grown"] == 3 and out["final"] == 4


def test_forces_arriving_after_a_frameless_load_land_on_their_own_frame():
    """§ 10.3: frame f's arrows come from frame f.

    A run caught at its first geometry carries no forces. Pushing the first ones
    that arrive onto an empty list would attach them to frame 0 for ever after,
    so a growing run would show frame 0's forces on every frame.
    """
    out = _run(
        """
        const m = await loaded();
        m.reloadFrames([[[0,0,0],[1,0,0]], [[1,0,0],[2,0,0]]]);  // no forces
        m.addFrame([[2,0,0],[3,0,0]], { forces: [[7,0,0],[7,0,0]] });  // it has them
        const c = m.getCoordinates();
        console.log(JSON.stringify({
            frames: c.frames.length,
            forces: c.forcesPerFrame,
        }));
        """
    )
    assert out["frames"] == 3
    assert out["forces"] == [None, None, [[7, 0, 0], [7, 0, 0]]], (
        "the first forces to arrive must land on the frame that carried them, "
        f"not on frame 0: {out['forces']}"
    )


# ---------------------------------------------------------------------------
# § 9.4 — read-only freezes the master copy and nothing else
# ---------------------------------------------------------------------------

def test_read_only_freezes_the_core_data_and_gates_nothing_else():
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
    """
    out = _run(
        """
        const m = createModel({ mode: "readonly" });
        // The one write into an empty viewer.
        await m.installMolecule({ text: "x", filename: "x.xyz" });
        const seeded = JSON.stringify(m.getStructure());

        // DELIVERY is open: the run's own frames and forces arrive.
        m.addFrames([[[0,0,1],[1,0,1]], [[0,0,2],[1,0,2]]]);
        m.setForces([null, null, [[0,0,9],[0,0,9]]]);
        m.setCurrentFrame(2);
        const delivered = { frames: m.frameCount(), at: m.currentFrame() };
        const coreAfterDelivery = JSON.stringify({
            elements: m.getStructure().elements,
            annotations: m.getStructure().annotations,
            periodicity: m.getStructure().periodicity,
        });

        // CHANGING the core data is not, and none of it throws.
        globalThis.__requests = [];
        globalThis.__nextPayload = globalThis.__payload([
            globalThis.__atomRow(0, "XX", 9, { regions: ["smuggled"] }),
        ]);
        const threw = [];
        async function tryIt(name, fn) {
            try { await fn(); } catch (e) { threw.push(name); }
        }
        await tryIt("installMolecule", () => m.installMolecule({ text: "y", filename: "y.xyz" }));
        await tryIt("applyOp",         () => m.applyOp("delete"));
        await tryIt("commitPeriodicityOp", () => m.commitPeriodicityOp("cell", [[9,0,0],[0,9,0],[0,0,9]]));
        m.selection.adopt([0]);
        await tryIt("writeLabel",      () => m.selection.writeLabel("smuggled", "replace"));

        console.log(JSON.stringify({
            threw, delivered, seeded: seeded !== null,
            coreUnchanged: coreAfterDelivery === JSON.stringify({
                elements: m.getStructure().elements,
                annotations: m.getStructure().annotations,
                periodicity: m.getStructure().periodicity,
            }),
            requests: globalThis.__requests.length,
        }));
        """
    )
    assert out["threw"] == [], f"a read-only no-op threw: {out['threw']}"
    assert out["delivered"] == {"frames": 3, "at": 2}, (
        "a read-only viewer could not receive the run's own frames, so it can "
        f"neither follow a running job nor scrub a finished one: {out['delivered']}"
    )
    assert out["coreUnchanged"] is True, (
        "the core data — the atoms, their labels, the cell — changed in a "
        "read-only viewer"
    )
    assert out["requests"] == 0, (
        "a read-only viewer sent a request — the gate must stop a change before "
        f"the network, not after: {out['requests']}"
    )


def test_scrubbing_and_exporting_are_not_gated():
    """§ 9.4: "Somebody studying a finished calculation can still select atoms,
    isolate them, measure them, scrub the trajectory, turn on force arrows, spin
    the camera, and export" — none of which touches the master copy.

    The assertion is about WHICH SIDE OF THE GATE these fall on, so it is made
    against the gate itself: neither door is wrapped, so neither can be swallowed
    in a read-only viewer.

    (How a read-only viewer receives its structure is settled: one write into an
    empty viewer, and delivery of that structure's frames stays open —
    ``test_read_only_freezes_the_core_data_and_gates_nothing_else``.)
    """
    out = _run(
        """
        const m = await loaded();
        m.reloadFrames([[[0,0,0],[1,0,0]], [[9,0,0],[8,0,0]]]);
        m.setCurrentFrame(1);
        const file = m.exportFile();

        // Same two doors on a read-only model: they must behave identically,
        // because the gate does not stand in front of either.
        const ro = createModel({ mode: "readonly" });
        let threw = false;
        try { ro.setCurrentFrame(1); ro.exportFile(); } catch (e) { threw = true; }

        console.log(JSON.stringify({
            exported: !!file,
            wroteDisplayedFrame: file.structure.positions[0][0] === 9,
            scrubbed: m.currentFrame(),
            roThrew: threw,
        }));
        """
    )
    assert out["exported"] is True, "export must work — it is a read"
    assert out["wroteDisplayedFrame"] is True
    assert out["scrubbed"] == 1, "scrubbing must work — it is looking at the picture"
    assert out["roThrew"] is False


def test_a_read_only_viewer_is_seeded_once_and_then_frozen():
    """§ 9.4 freezes THE MASTER COPY — and a viewer with nothing loaded has no
    master copy to freeze.

    So the first `installMolecule` is how a host says which structure this viewer
    shows, and it is allowed in any mode; every one after it meets the gate. That
    keeps § 9.3's "the only way a structure gets in" intact — no second door —
    while making § 8's "a viewer mounts before it has a structure" and § 12.3's
    read-only Results viewer both possible.

    What a read-only viewer still cannot do is exactly what § 9.4 promises:
    change the structure the calculation ran on.
    """
    out = _run(
        """
        const m = createModel({ mode: "readonly" });
        const seed = await m.installMolecule({ text: "x", filename: "first.xyz" });
        const afterSeed = m.getAtoms().map(a => a.element);

        // A second load would REPLACE the structure the calculation ran on.
        globalThis.__nextPayload = globalThis.__payload([
            globalThis.__atomRow(0, "XX", 9),
        ]);
        globalThis.__requests = [];
        const replaced = await m.installMolecule({ text: "y", filename: "second.xyz" });

        console.log(JSON.stringify({
            seeded: seed !== null,
            afterSeed,
            replaced,
            stillFirst: m.getAtoms().map(a => a.element),
            secondRequests: globalThis.__requests.length,
        }));
        """
    )
    assert out["seeded"] is True, (
        "a read-only viewer could not be given a structure at all, so the "
        "Results tab would have nothing to show"
    )
    assert out["afterSeed"] == ["C", "O"]
    assert out["replaced"] is None, "the second load must be a no-op"
    assert out["stillFirst"] == ["C", "O"], (
        "a read-only viewer's structure was replaced after it was seeded"
    )
    assert out["secondRequests"] == 0, (
        "the gate must stop the replacement before the network, not after"
    )


def test_a_read_only_seed_anchors_no_history():
    """§ 9.4: "A read-only viewer has no history … `save`, `load` and `undo` are
    no-ops here too, and the unsaved-changes badge never appears."

    Anchoring point 0 would also write it to the workspace, which is a persist a
    read-only viewer has no business doing (§ 11.2: storage is touched by opening
    a structure, an explicit save, and a load).
    """
    out = _run(
        """
        // A stand-in workspace speaking the real front door (workspace.md § 5).
        // Only a POINT counts as anchoring: `persist` also carries the session
        // copy, which is a different write with a different rule (§ 11.3).
        // A stand-in workspace speaking the real front door. Only a MILESTONE
        // counts as anchoring — an edit's draft goes to its own file and is a
        // different rule (workspace.md § 7).
        const writes = [];
        function spy(label) {
            return {
                workspaceId: (tag) => "id-" + tag,
                persist: (tag, bytes, identity) => {
                    if (!/-draft$/.test(identity.workspace_id)) {
                        writes.push(label + ":" + identity.state_index);
                    }
                    return true;
                },
                readState: async () => null,
                pruneStatesAbove: () => {},
            };
        }
        const ro = createModel({ mode: "readonly", owner: "ro",
                                 workspace: spy("readonly") });
        await ro.installMolecule({ text: "x", filename: "x.xyz" });

        const editable = createModel({ owner: "ed", workspace: spy("editable") });
        await editable.installMolecule({ text: "x", filename: "x.xyz" });

        console.log(JSON.stringify({
            writes, badge: ro.uncommitted, at: ro.state_index,
        }));
        """
    )
    assert out["writes"] == ["editable:0"], (
        "a read-only seed wrote a history point to the workspace — only the "
        f"editable viewer should have anchored: {out['writes']}"
    )
    assert out["badge"] is False
    assert out["at"] == 0


# ---------------------------------------------------------------------------
# § 9.3 / § 11.3 — writing the structure out
# ---------------------------------------------------------------------------

def test_export_hands_over_the_displayed_frame():
    """§ 13.3: "exporting data yields THE DISPLAYED FRAME's coordinates … scrub
    to frame 40 and frame 40 is what the file holds."

    It hands over the STRUCTURE, not bytes — a coordinate document is a format
    the server owns (§ 11.7) — so the frame is checked where it now lives: in
    the positions that leave.
    """
    out = _run(
        """
        const m = await loaded();
        m.reloadFrames([[[0,0,0],[1,0,0]], [[40,0,0],[41,0,0]]]);
        m.setCurrentFrame(1);
        const file = m.exportFile();
        console.log(JSON.stringify({
            positions: file.structure.positions,
            keys: Object.keys(file.structure).sort(),
        }));
        """
    )
    assert out["positions"][0] == [40, 0, 0], (
        f"export handed over a frame the user was not looking at: {out['positions']}"
    )
    assert "metadata" in out["keys"], (
        f"the facts did not leave with the atoms: {out['keys']}"
    )


def test_the_frame_doors_refuse_to_create_a_structure_that_disagrees_with_itself():
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
    """
    out = _run(
        """
        function refused(fn) {
            try { fn(); return null; } catch (e) { return e.message; }
        }
        const empty = createModel({});
        const nothingLoaded = refused(() => empty.addFrames([[[0,0,0],[1,0,0]]]));

        const m = await loaded();          // two atoms — C and O
        const shortAppend  = refused(() => m.addFrames([[[0,0,0]]]));
        const shortOne     = refused(() => m.addFrame([[0,0,0]]));
        const shortReload  = refused(() => m.reloadFrames([[[0,0,0]]]));
        // A batch whose frames disagree with EACH OTHER, checked before any of
        // them lands so the first half is not left applied.
        const ragged = refused(() => m.addFrames([[[0,0,0],[1,0,0]], [[2,0,0]]]));

        console.log(JSON.stringify({
            nothingLoaded, shortAppend, shortOne, shortReload, ragged,
            // Nothing was applied by any of the refusals.
            frames: m.frameCount(),
            stillWhole: m.getFrameAllAtoms(0).length,
        }));
        """
    )
    assert out["nothingLoaded"] and "nothing loaded" in out["nothingLoaded"], (
        "appending with nothing loaded invented an atom identity instead of "
        f"refusing: {out['nothingLoaded']}"
    )
    for door in ("shortAppend", "shortOne", "shortReload", "ragged"):
        assert out[door], (
            f"{door} accepted a frame that does not carry the structure's atoms "
            "— § 10.8 calls that a hard error, never coerced"
        )
    assert out["frames"] == 1 and out["stillWhole"] == 2, (
        "a refused frame left something behind in the master copy"
    )


def test_one_read_of_the_structure_holds_everything_a_request_needs():
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
    """
    out = _run(
        """
        globalThis.__nextPayload = globalThis.__payload([
            globalThis.__atomRow(0, "C", 0, { regions: ["anchor", "frozen_atoms"] }),
            globalThis.__atomRow(1, "O", 1, { regions: [] }),
        ], { periodicity: { cell: [[6,0,0],[0,6,0],[0,0,6]], cell_origin: [1,1,1],
                            axis_kind: ["periodic","periodic","isolated"],
                            vacuum: [0,0,10] } });
        const m = createModel({});
        await m.installMolecule({ text: "x", filename: "x.xyz" });
        m.addFrames([[[0,0,1],[1,0,1]]], { forces: [[[0,0,-1],[0,0,-2]]] });

        // ONE call. Nothing else on the surface is touched.
        const whole = m.getStructure();

        console.log(JSON.stringify({
            keys:      Object.keys(whole).sort(),
            elements:  whole.elements,
            frames:    whole.frames,
            forces:    whole.forcesPerFrame,
            labels:    whole.annotations.map((a) => a.labels),
            cell:      whole.periodicity && whole.periodicity.cell,
            origin:    whole.periodicity && whole.periodicity.cell_origin,
        }));
        """
    )
    assert out["keys"] == ["annotations", "elements", "forcesPerFrame",
                           "frames", "periodicity"], (
        f"the one read does not carry § 6.2's five fields: {out['keys']}"
    )
    # The four facts § 9.3 names, each read off that single answer.
    assert out["frames"] == [[[0, 0, 0], [1, 0, 0]], [[0, 0, 1], [1, 0, 1]]], (
        "the coordinates are missing from the read, or it handed back only the "
        "displayed frame instead of every frame (§ 6.3)"
    )
    assert out["forces"] == [None, [[0, 0, -1], [0, 0, -2]]]
    assert out["labels"] == [["anchor", "frozen_atoms"], []], (
        "the labels are missing from the read"
    )
    assert "frozen_atoms" in out["labels"][0], (
        "which atoms are held still is a label like any other (§ 6.6), so it "
        "rides in the same one read"
    )
    assert out["cell"] == [[6, 0, 0], [0, 6, 0], [0, 0, 6]], "the cell is missing"
    assert out["origin"] == [1, 1, 1]
    assert out["elements"] == ["C", "O"]


def test_the_facts_a_request_carries_all_came_from_one_read():
    """§ 9.3: "after an edit, a request built from the viewer carries that edit
    in EVERY part of what it sends — no piece can be older than another, because
    it all came from one read of the structure."

    This went wrong once: a tab read the labels and the cell fresh while the
    coordinates came from a copy taken at page load, so the server judged a
    structure that was not the one on screen.
    """
    out = _run(
        """
        globalThis.__nextPayload = globalThis.__payload([
            globalThis.__atomRow(0, "C", 5, { regions: ["edited"] }),
        ], { periodicity: { cell: [[7,0,0],[0,7,0],[0,0,7]] } });
        const m = createModel({});
        await m.installMolecule({ text: "x", filename: "x.xyz" });

        globalThis.__requests = [];
        await m.applyOp("translate");
        const sent = globalThis.__requests[0].body.structure;
        console.log(JSON.stringify({
            positions: sent.positions,
            regions: sent.metadata.regions,
            cell: sent.metadata.cell,
            elements: sent.elements,
        }));
        """
    )
    assert out["positions"] == [[5, 0, 0]], (
        "the request carried coordinates that were not the current ones"
    )
    assert out["regions"] == {"edited": [0]}, (
        "the request carried labels from a different read than the coordinates"
    )
    assert out["cell"] == [[7, 0, 0], [0, 7, 0], [0, 0, 7]]
    assert out["elements"] == ["C"]


def test_what_the_caller_knew_about_the_atoms_reaches_the_server():
    """§ 9.3: `installMolecule` is "the only way a structure gets in", so
    everything the caller knows about that structure has to fit through it.

    THE CASE THIS EXISTS FOR is a trajectory. There is no `.molstruct.json`
    beside a run: the region labels and frozen tags come out of the input
    script the Build tab wrote, and the lattice out of the run's output. Both
    are handed over with the structure, in one call, exactly as the frames are
    — the alternative is a second door, and § 9.3 has one.

    THEY ARE TWO FIELDS BECAUSE THEY ARE TWO FACTS from two places, and one of
    them is checked. The label block is a document the server wrote, carried as
    BYTES and never opened here — it holds its own atom-count guard, and a
    caller that parsed it and put a key back would be writing a format it does
    not own. The cell is a plain value in the block every other structure door
    already takes, so the server applies it through the seam that also refuses
    a bad one.

    Both were silently dropped: the request builder forwarded neither, so a
    trajectory opened with no labels and no cell at HTTP 200.
    """
    out = _run(
        """
        const m = createModel({});
        await m.installMolecule({
            text: "x", filename: "run.xyz",
            atomMetadata: '{"n_atoms_total":2,"regions":{"frozen_atoms":[1]}}',
            periodicity: { cell: [[9,0,0],[0,9,0],[0,0,9]] },
        });
        const sent = globalThis.__requests[0].body;
        console.log(JSON.stringify({
            route: globalThis.__requests[0].route,
            carried: sent.atom_metadata || null,
            cell: (sent.periodicity || {}).cell || null,
            keys: Object.keys(sent).sort(),
        }));
        """
    )
    assert out["route"] == "/api/build/load"
    assert out["carried"] == (
        '{"n_atoms_total":2,"regions":{"frozen_atoms":[1]}}'
    ), (
        "the label block was dropped or rewritten. Dropped: a trajectory opens "
        "with no region labels and no frozen tags, at HTTP 200 — nothing "
        "refused them. Rewritten: whatever re-stated `n_atoms_total` also "
        "disabled the guard that stops a label set landing on the wrong atoms"
    )
    assert out["cell"] == [[9, 0, 0], [0, 9, 0], [0, 0, 9]], (
        "the caller's cell never left the browser, so no trajectory can draw "
        "the box its run used"
    )
    assert "atom_metadata" in out["keys"] and "periodicity" in out["keys"]


def test_a_caller_that_knew_nothing_extra_sends_nothing_extra():
    """The fields are absent, not null, when there is nothing to say.

    An ordinary file load has no such block and states no cell, and a key
    carrying `null` is a statement the caller did not make — on the server side
    the periodicity block is applied verbatim, so an empty one is not nothing.
    """
    out = _run(
        """
        const m = createModel({});
        await m.installMolecule({ text: "x", filename: "x.xyz" });
        console.log(JSON.stringify({
            keys: Object.keys(globalThis.__requests[0].body).sort(),
        }));
        """
    )
    assert "atom_metadata" not in out["keys"], (
        f"an ordinary load sent an empty metadata block: {out['keys']}"
    )
    assert "periodicity" not in out["keys"], (
        f"an ordinary load stated a cell it was never given: {out['keys']}"
    )


# ---------------------------------------------------------------------------
# § 11.1 — geometry edits go to the server
# ---------------------------------------------------------------------------

def test_the_count_requirement_is_checked_before_the_request_goes_out():
    """§ 13.3: "`orient` with one atom and `delete` with none are refused
    locally, with no request sent."
    """
    out = _run(
        """
        const m = await loaded();
        globalThis.__requests = [];
        const orient = await m.applyOp("orient");     // nothing selected, needs 2
        const del = await m.applyOp("delete");        // nothing selected, refuses
        console.log(JSON.stringify({
            orient, del, requests: globalThis.__requests.map(r => r.route),
        }));
        """
    )
    assert out["orient"] is None and out["del"] is None
    assert out["requests"] == [], (
        f"a refused operation still reached the network: {out['requests']}"
    )


def test_an_empty_selection_means_what_the_table_says():
    """§ 13.3: "with nothing selected, `translate` acts on every atom, `orient`
    refuses and `electrode` centres on the origin — three different answers, each
    read from the table rather than hand-coded per operation."
    """
    out = _run(
        """
        const m = await loaded();
        globalThis.__requests = [];
        const translate = await m.applyOp("translate");
        const orient = await m.applyOp("orient");
        const electrode = await m.applyOp("electrode");
        console.log(JSON.stringify({
            translate: translate !== null,
            orient: orient !== null,
            electrode: electrode !== null,
            routes: globalThis.__requests.map(r => r.route),
        }));
        """
    )
    assert out["translate"] is True, "translate with nothing selected acts on all atoms"
    assert out["orient"] is False, "orient with nothing selected refuses"
    assert out["electrode"] is True, "electrode falls back to centring on the origin"
    assert out["routes"] == ["/api/modify/translate", "/api/modify/electrode"], (
        f"three different answers, and only two go out: {out['routes']}"
    )


def test_the_operation_name_is_the_route_segment():
    """§ 11.1: "The operation name IS the server route segment. The delete
    operation is `delete`, not `deleteAtoms`; the add operation is `add_atom`."
    """
    out = _run(
        """
        const m = await loaded();
        globalThis.__requests = [];
        await m.applyOp("translate");
        await m.applyOp("calibrate");
        console.log(JSON.stringify({ routes: globalThis.__requests.map(r => r.route) }));
        """
    )
    assert out["routes"] == ["/api/modify/translate", "/api/modify/calibrate"]


def test_a_failed_edit_changes_nothing():
    """§ 13.3: "when the server refuses, the structure is exactly as it was and
    no history state is recorded."

    § 11.1: that is the other half of "all at once" — it is what lets a failed
    edit be a state the viewer can sit in without being wrong.

    A refusal REACHES THE CALLER BY THROWING (§ 6.9), which is what this asserts
    now; it used to expect `null`, back when a refused edit and an empty viewer
    gave the same answer. What the failure route is does not change what this
    test is about: nothing moved.
    """
    out = _run(
        """
        const m = await loaded();
        const before = JSON.stringify(m.getStructure());
        const beforeFrame = m.getFrameAllAtoms(0);
        globalThis.__serverFails = true;
        let told = false;
        try { await m.applyOp("translate"); } catch (_) { told = true; }
        console.log(JSON.stringify({
            told,
            unchanged: JSON.stringify(m.getStructure()) === before,
            coordsUnchanged: JSON.stringify(m.getFrameAllAtoms(0)) === JSON.stringify(beforeFrame),
        }));
        """
    )
    assert out["told"] is True, "a failed edit must tell the caller"
    assert out["unchanged"] is True, "a failed edit changed the structure"
    assert out["coordsUnchanged"] is True


def test_an_unknown_operation_is_refused_rather_than_posted():
    """§ 11.1: one small table declares each operation's shape. A name that is not
    in it has no shape, so there is nothing to send.
    """
    out = _run(
        """
        const m = await loaded();
        globalThis.__requests = [];
        let threw = false;
        try { await m.applyOp("deleteAtoms"); } catch (e) { threw = true; }
        console.log(JSON.stringify({ threw, requests: globalThis.__requests.length }));
        """
    )
    assert out["threw"] is True, (
        "an unknown operation must be refused loudly — it is a caller mistake, "
        "not a user action the gate should swallow"
    )
    assert out["requests"] == 0


# ---------------------------------------------------------------------------
# § 6.1 / § 6.3
# ---------------------------------------------------------------------------

def test_one_frame_reads_the_same_way_as_many():
    """§ 6.1: "no read, edit, export or save path treats a single frame
    differently from four hundred."
    """
    out = _run(
        """
        const one = await loaded();
        one.reloadFrames([[[0,0,0],[1,0,0]]]);
        const many = await loaded();
        many.reloadFrames(Array.from({length: 400}, (_, f) => [[f,0,0],[f+1,0,0]]));
        console.log(JSON.stringify({
            oneCount: one.frameCount(),
            manyCount: many.frameCount(),
            oneExports: !!one.exportFile(),
            manyExports: !!many.exportFile(),
            oneFrameShape: one.getFrameAllAtoms(0).length,
            manyFrameShape: many.getFrameAllAtoms(0).length,
        }));
        """
    )
    assert out["oneCount"] == 1 and out["manyCount"] == 400
    assert out["oneExports"] and out["manyExports"]
    assert out["oneFrameShape"] == out["manyFrameShape"] == 2


def test_the_model_never_names_the_drawing_library():
    """§ 7 level 3: the model NEVER touches the drawing library.

    § 5.3: everything above the sealed layer can be read end to end without
    learning which library draws the molecule.
    """
    for name in ("model.js", "model-jobs.js"):
        text = (MODULE_DIR / name).read_text()
        assert "3Dmol" not in text, f"{name} names the drawing library"


# ---------------------------------------------------------------------------
# § 11.2 / § 9.4 — the badge, raised inside the gate
# ---------------------------------------------------------------------------

def test_an_edit_raises_the_badge_and_a_failed_one_does_not():
    """§ 11.2: the unsaved-changes flag is "not bookkeeping" — it says there is
    work here that is not on the sequence yet.

    § 11.1: when the server refuses, "no history state is recorded". Raising the
    badge inside the gate and AFTER the change lands is what makes that fall out
    rather than needing a case of its own: a refused edit never reaches the line.
    """
    out = _run(
        """
        const m = await loaded();
        const fresh = m.uncommitted;

        await m.applyOp("translate");
        const afterEdit = m.uncommitted;

        globalThis.__serverFails = true;
        const before = m.uncommitted;
        // The refusal leaves by throwing (§ 6.9); the badge is what is on trial.
        try { await m.applyOp("translate"); } catch (_) {}
        console.log(JSON.stringify({
            fresh, afterEdit, afterFailure: m.uncommitted, before,
        }));
        """
    )
    assert out["fresh"] is False, "a freshly opened structure has no unsaved work"
    assert out["afterEdit"] is True, "an edit must raise the badge"
    assert out["afterFailure"] is True, (
        "a failed edit must leave the badge exactly as it was — it recorded "
        "nothing and changed nothing"
    )


def test_a_read_only_viewer_has_no_history_and_no_badge():
    """§ 9.4: "`save`, `load` and `undo` are no-ops here too, and the
    unsaved-changes badge (§ 11.2) never appears."

    Saving does not itself change the master copy — it records it — but a history
    exists to get back to a state you left, and in a read-only viewer nothing can
    leave one.
    """
    out = _run(
        """
        const m = createModel({ mode: "readonly" });
        const threw = [];
        for (const [name, fn] of [
            ["save", () => m.save(1)], ["load", () => m.load(-1)], ["undo", () => m.undo()],
        ]) {
            try { await fn(); } catch (e) { threw.push(name); }
        }
        await m.applyOp("translate");
        console.log(JSON.stringify({
            threw, badge: m.uncommitted, at: m.state_index,
        }));
        """
    )
    assert out["threw"] == [], f"a read-only history door threw: {out['threw']}"
    assert out["badge"] is False, (
        "a read-only viewer raised the unsaved-changes badge"
    )
    assert out["at"] == 0


def test_writing_a_label_raises_the_badge_and_is_frozen_read_only():
    """§ 9.4: "Tagging is an edit. A label becomes part of what an atom IS and
    travels to the calculation, so writing one is frozen along with the rest."

    § 9.5: applying a label REPLACES that label's previous set of atoms.
    """
    out = _run(
        """
        const m = await loaded();
        m.selection.add([0]);
        m.selection.writeLabel("anchor");
        const first = m.getRegions();
        const badge = m.uncommitted;

        m.selection.clear();
        m.selection.add([1]);
        m.selection.writeLabel("anchor");     // replaces the previous set

        const ro = createModel({ mode: "readonly" });
        const roWrote = ro.selection.writeLabel("anchor");

        console.log(JSON.stringify({
            first, badge, replaced: m.getRegions(), roWrote,
        }));
        """
    )
    assert out["first"] == {"anchor": [0]}
    assert out["badge"] is True, "tagging is an edit, so it raises the badge"
    assert out["replaced"] == {"anchor": [1]}, (
        f"applying a label must replace that label's previous set: {out['replaced']}"
    )
    assert out["roWrote"] is False, "a read-only viewer must not write a label"


def test_a_count_changing_edit_clears_the_selection():
    """A kept selection could point at an atom that is no longer the one it
    meant. A count-preserving transform leaves it alone — the two cases are
    genuinely different, which is why the operation table carries the effect on
    atom count (§ 11.1).
    """
    out = _run(
        """
        const m = await loaded();
        m.selection.add([0, 1]);

        // Same count back: the selection survives.
        await m.applyOp("translate");
        const afterTransform = m.selection.get();

        // One atom back where there were two: the count changed.
        m.selection.add([0, 1]);
        globalThis.__nextPayload = globalThis.__payload([globalThis.__atomRow(0, "C", 0)]);
        await m.applyOp("delete");
        console.log(JSON.stringify({ afterTransform, afterShrink: m.selection.get() }));
        """
    )
    assert out["afterTransform"] == [0, 1], (
        "a count-preserving transform must leave the selection alone"
    )
    assert out["afterShrink"] == [], (
        f"an edit that changed the atom count must clear the selection: "
        f"{out['afterShrink']}"
    )
