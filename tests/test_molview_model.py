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

# A stand-in server. It answers the three routes of § 11.1 and records what it
# was sent, so a test can assert what actually left the browser.
SERVER = """
globalThis.__requests = [];
globalThis.__serverFails = false;
globalThis.__nextPayload = null;

function atomRow(i, element, x, opts) {
    return Object.assign({ index: i, element, x, y: 0, z: 0, regions: [],
                           is_frozen: false }, opts || {});
}
globalThis.__atomRow = atomRow;

globalThis.__payload = function (atoms, extra) {
    return Object.assign({ atoms }, extra || {});
};

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
        s.cell = { lattice: [[9,0,0],[0,9,0],[0,0,9]] };

        const coords = m.getCoordinates();
        coords.frames[0][0][0] = 999;

        const frame = m.getFrameAllAtoms(0);
        frame[0][0] = -999;

        const after = m.getStructure();
        console.log(JSON.stringify({
            element: after.elements[0],
            labels: after.annotations[0].labels,
            cell: after.cell,
            x: m.getFrameAllAtoms(0)[0][0],
        }));
        """
    )
    assert out["element"] == "C", "a read handed out the model's own array"
    assert out["labels"] == [], "a read handed out the model's own labels"
    assert out["cell"] is None, "a read handed out a writable structure"
    assert out["x"] == 0, "a read handed out the model's own coordinates"


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
            globalThis.__atomRow(1, "O", 1, { is_frozen: true, residue_name: "ALA" }),
        ], { periodicity: { lattice: [[4,0,0],[0,4,0],[0,0,4]], origin: [1,1,1] } });
        const m = createModel({});
        await m.installMolecule({ text: "x", filename: "x.xyz" });

        const whole = m.getStructure();
        console.log(JSON.stringify({
            elementsAgree: JSON.stringify(m.getElements()) === JSON.stringify(whole.elements),
            atomsAgree: m.getAtoms().map(a => a.element).join() === whole.elements.join(),
            cellAgree: JSON.stringify(m.getUnitCell()) === JSON.stringify(whole.cell.lattice),
            originAgree: JSON.stringify(m.getUnitCellOrigin()) === JSON.stringify(whole.cell.origin),
            regions: m.getRegions(),
            frozen: m.getFrozen(),
            infoLattice: m.getUnitCellInfo().lattice,
        }));
        """
    )
    assert out["elementsAgree"] and out["atomsAgree"], (
        "a cut of the structure disagreed with the whole"
    )
    assert out["cellAgree"] and out["originAgree"]
    assert out["regions"]["anchor"] == [0], "labels group into name -> atoms"
    assert out["frozen"] == [1], (
        "the frozen cut reads the reserved label off the same one mechanism"
    )
    assert out["infoLattice"] == [[4, 0, 0], [0, 4, 0], [0, 0, 4]], (
        "the cell as it will be used must agree with the raw cell"
    )


def test_the_frozen_flag_becomes_an_ordinary_label_at_the_boundary():
    """§ 6.6: MolView's end is "one mechanism, no special case".

    The server keeps `is_frozen` apart from `regions`, so the fold happens once
    at the inbound translation and downstream there is one mechanism — a frozen
    atom's label sits in the same list as any other.
    """
    out = _run(
        """
        globalThis.__nextPayload = globalThis.__payload([
            globalThis.__atomRow(0, "C", 0, { is_frozen: true, regions: ["mine"] }),
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
    assert len(out["labels"]) == 2, (
        f"a frozen atom must carry its frozen label like any other: {out['labels']}"
    )
    assert "mine" in out["labels"]
    assert out["frozen"] == [0]
    assert len(out["regions"]) == 2, (
        "the frozen label must appear in the label grouping like any other — "
        f"a separate field would keep it out: {out['regions']}"
    )


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
        m.reloadFrames([[[0,0,0]], [[1,0,0]], [[2,0,0]], [[3,0,0]], [[4,0,0]]]);
        m.setCurrentFrame(4);

        // Watch what every notification shows.
        const seen = [];
        m.subscribe(() => seen.push({ at: m.currentFrame(), of: m.frameCount() }));
        m.onFrameChange(() => seen.push({ at: m.currentFrame(), of: m.frameCount() }));

        m.reloadFrames([[[0,0,0]], [[1,0,0]]]);       // the trajectory shortens
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
        m.reloadFrames([[[0,0,0]], [[1,0,0]], [[2,0,0]]]);
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
        m.reloadFrames([[[0,0,0]], [[1,0,0]], [[2,0,0]]]);
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
        m.reloadFrames([[[0,0,0]]]);
        const start = m.frameCount();
        m.addFrames([[[1,0,0]], [[2,0,0]]]);
        const grown = m.frameCount();
        m.addFrame([[3,0,0]]);
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
        m.reloadFrames([[[0,0,0]], [[1,0,0]]]);      // two frames, no forces
        m.addFrame([[2,0,0]], [[7,0,0]]);            // the third carries forces
        const c = m.getCoordinates();
        console.log(JSON.stringify({
            frames: c.frames.length,
            forces: c.forcesPerFrame,
        }));
        """
    )
    assert out["frames"] == 3
    assert out["forces"] == [None, None, [[7, 0, 0]]], (
        "the first forces to arrive must land on the frame that carried them, "
        f"not on frame 0: {out['forces']}"
    )


# ---------------------------------------------------------------------------
# § 9.4 — read-only freezes the master copy and nothing else
# ---------------------------------------------------------------------------

def test_every_truth_change_is_a_no_op_and_does_not_throw():
    """§ 9.4: "it returns without effect AND WITHOUT THROWING."

    A read-only viewer that threw would make every caller wrap its writes, which
    is the list of special cases this rule exists to avoid.
    """
    out = _run(
        """
        globalThis.__nextPayload = globalThis.__payload([
            globalThis.__atomRow(0, "C", 0), globalThis.__atomRow(1, "O", 1),
        ]);
        const m = createModel({ mode: "readonly" });
        const threw = [];
        async function tryIt(name, fn) {
            try { await fn(); } catch (e) { threw.push(name); }
        }
        await tryIt("installMolecule", () => m.installMolecule({ text: "x", filename: "x.xyz" }));
        await tryIt("applyOp",         () => m.applyOp("delete"));
        await tryIt("commitPeriodicityOp", () => m.commitPeriodicityOp("set_cell", {}));
        await tryIt("reloadFrames",    () => m.reloadFrames([[[0,0,0]]]));
        await tryIt("addFrame",        () => m.addFrame([[1,0,0]]));
        await tryIt("addFrames",       () => m.addFrames([[[1,0,0]]]));
        await tryIt("setForces",       () => m.setForces([[[1,0,0]]]));
        console.log(JSON.stringify({
            threw,
            structure: m.getStructure(),
            requests: globalThis.__requests.length,
        }));
        """
    )
    assert out["threw"] == [], f"a read-only no-op threw: {out['threw']}"
    assert out["structure"] is None, (
        "a read-only viewer's master copy changed"
    )
    assert out["requests"] == 0, (
        "a read-only viewer sent a request — the gate must stop it before the "
        f"network, not after: {out['requests']}"
    )


def test_scrubbing_and_exporting_are_not_gated():
    """§ 9.4: "Somebody studying a finished calculation can still select atoms,
    isolate them, measure them, scrub the trajectory, turn on force arrows, spin
    the camera, and export" — none of which touches the master copy.

    The assertion is about WHICH SIDE OF THE GATE these fall on, so it is made
    against the gate itself: neither door is wrapped, so neither can be swallowed
    in a read-only viewer.

    (How a read-only viewer RECEIVES its structure is an open question — see
    ``test_a_read_only_viewer_cannot_be_given_a_structure_today``.)
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
            wroteDisplayedFrame: file.text.indexOf("9 0 0") >= 0,
            scrubbed: m.currentFrame(),
            roThrew: threw,
        }));
        """
    )
    assert out["exported"] is True, "export must work — it is a read"
    assert out["wroteDisplayedFrame"] is True
    assert out["scrubbed"] == 1, "scrubbing must work — it is looking at the picture"
    assert out["roThrew"] is False


def test_a_read_only_viewer_cannot_be_given_a_structure_today():
    """OPEN — a contradiction in the contract, pinned here so it cannot be lost.

    § 9.3's table marks `installMolecule` as changing the master copy, so § 9.4's
    one question answers "yes" and the gate swallows it. But § 8 says "a viewer
    mounts before it has a structure … the bar appears once a structure with more
    than one frame is LOADED INTO IT", and § 12.3 describes a read-only Results
    viewer showing a finished calculation. § 9.3 also says installMolecule is
    "the only way a structure gets in".

    So today a read-only viewer can never receive one. This test asserts the
    contract AS WRITTEN; when the question is decided, it changes with it.

    The two readings, for whoever decides:
      - installing is how a HOST says which structure this viewer shows, and the
        gate is about a USER editing it — § 9.4's "the structure the calculation
        ran on" reads as already present. Then installMolecule is not gated, and
        § 9.3's table row is what changes.
      - installing really is a truth change, and a read-only viewer is seeded by
        some other means at mount. Then § 8 or § 9.3's "only way in" is what
        changes.
    """
    out = _run(
        """
        const m = createModel({ mode: "readonly" });
        const result = await m.installMolecule({ text: "x", filename: "x.xyz" });
        console.log(JSON.stringify({
            result,
            structure: m.getStructure(),
            requests: globalThis.__requests.length,
        }));
        """
    )
    assert out["result"] is None
    assert out["structure"] is None, (
        "the gate swallowed the only door a structure can arrive through — this "
        "is § 9.3's table applied as written, and is the open question above"
    )
    assert out["requests"] == 0


# ---------------------------------------------------------------------------
# § 9.3 / § 11.3 — writing the structure out
# ---------------------------------------------------------------------------

def test_export_writes_the_displayed_frame():
    """§ 13.3: "exporting data yields THE DISPLAYED FRAME's coordinates … scrub
    to frame 40 and frame 40 is what the file holds."
    """
    out = _run(
        """
        const m = await loaded();
        m.reloadFrames([[[0,0,0],[1,0,0]], [[40,0,0],[41,0,0]]]);
        m.setCurrentFrame(1);
        const file = m.exportFile();
        console.log(JSON.stringify({ text: file.text }));
        """
    )
    assert "40 0 0" in out["text"], (
        f"export wrote a frame the user was not looking at: {out['text']!r}"
    )


def test_a_structure_that_cannot_be_written_out_is_not_written_out():
    """§ 9.3: "when the geometry and the per-atom labels disagree about how many
    atoms there are, the export door returns nothing rather than a corrupt
    structure."
    """
    out = _run(
        """
        const m = await loaded();
        // A frame with the wrong number of atoms — the disagreement § 9.3 names.
        m.reloadFrames([[[0,0,0]]]);
        console.log(JSON.stringify({ exported: m.exportFile() }));
        """
    )
    assert out["exported"] is None, (
        "a structure whose geometry and labels disagree about the atom count "
        "must produce nothing, not a corrupt file"
    )


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
        ], { periodicity: { lattice: [[7,0,0],[0,7,0],[0,0,7]] } });
        const m = createModel({});
        await m.installMolecule({ text: "x", filename: "x.xyz" });

        globalThis.__requests = [];
        await m.applyOp("translate");
        const sent = globalThis.__requests[0].body.structure;
        console.log(JSON.stringify({
            positions: sent.positions,
            regions: sent.regions,
            cell: sent.periodicity.lattice,
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
    """
    out = _run(
        """
        const m = await loaded();
        const before = JSON.stringify(m.getStructure());
        const beforeFrame = m.getFrameAllAtoms(0);
        globalThis.__serverFails = true;
        const result = await m.applyOp("translate");
        console.log(JSON.stringify({
            result,
            unchanged: JSON.stringify(m.getStructure()) === before,
            coordsUnchanged: JSON.stringify(m.getFrameAllAtoms(0)) === JSON.stringify(beforeFrame),
        }));
        """
    )
    assert out["result"] is None, "a failed edit must tell the caller"
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
        await m.applyOp("translate");
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
        m.selection.applyLabel("anchor");
        const first = m.getRegions();
        const badge = m.uncommitted;

        m.selection.clear();
        m.selection.add([1]);
        m.selection.applyLabel("anchor");     // replaces the previous set

        const ro = createModel({ mode: "readonly" });
        const roWrote = ro.selection.applyLabel("anchor");

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
