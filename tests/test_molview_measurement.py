"""The measurement track — its own list, and the wall around it.

User, 2026-08-30, twice and unprompted the second time:

    *"make sure your code does not make the measurement selection
    conflict/overlap with atom selection data in the molview that is used
    elsewhere"*

    *"Make sure the measurement selection is not interfering with any atom
    selection that is used for labeling etc. they should be independent and
    measurement selection is not part of the structure meta data but rather
    molview internal status"*

Both halves are asserted here, because they fail differently.  **Behaviour**:
the track caps at three, the fourth pick drops the oldest, and the order is the
click order because the vertex of an angle is the atom picked second
(``docs/web/molview.md`` § 11.6).  **The wall**: the track never enters the
selection's snapshot, never reaches a label, never becomes structure metadata,
never leaves the browser, and is persisted in exactly one place — MolView's own
``<owner>:ui`` view-context lane (§ 11.2b), which is *how you were looking* and
not part of the work.

The wall is the half worth having tests for.  A leak would not throw: it would
quietly widen what the next Delete removes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from tests._node_esm import run_node
from tests._molview_sources import module_files

REPO = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO / "molbuilder" / "web" / "static" / "lib" / "molview"
STORES = MODULE_DIR / "stores.js"
MODEL = MODULE_DIR / "model.js"

STORE_PRELUDE = f"""
const S = await import({json.dumps(STORES.resolve().as_uri())});
"""

SERVER = """
globalThis.fetch = async function (route, init) {
    const body = JSON.parse(init.body);
    return { ok: true, status: 200, json: async () => ({
        ok: true,
        atoms: [0, 1, 2, 3].map((i) => ({
            index: i, element: "C", x: i, y: 0, z: 0, regions: [] })),
        n_atoms: 4,
        periodicity: { cell: null, cell_origin: null,
                       axis_kind: ["free", "free", "free"], vacuum: null },
    }) };
};
"""

MODEL_PRELUDE = f"""
const {{ createModel }} = await import({json.dumps(MODEL.resolve().as_uri())});
async function loaded() {{
    const m = createModel({{}});
    await m.installMolecule({{
        text: "4\\n\\nC 0 0 0\\nC 1 0 0\\nC 2 0 0\\nC 3 0 0\\n",
        filename: "x.xyz",
    }});
    return m;
}}
"""


def _store(snippet: str):
    return run_node([], STORE_PRELUDE + snippet)


def _model(snippet: str):
    return run_node([], MODEL_PRELUDE + snippet, globals_js=SERVER)


def _sources() -> dict:
    """Every layer of the module, comments stripped.

    Stripped because this file's own prose quotes the names it forbids, and a
    guard that reads its neighbours' explanations is a guard that fires on the
    commit documenting it.
    """
    out = {}
    for name, path in module_files().items():
        text = path.read_text()
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        text = re.sub(r"^\s*//.*$", "", text, flags=re.M)
        out[name] = text
    return out


# ---------------------------------------------------------------------------
# What the user asked for
# ---------------------------------------------------------------------------

def test_the_track_caps_at_three_and_the_fourth_drops_the_oldest():
    """User: max three, and (agreed in the same exchange) a fourth pick drops
    the oldest so measuring along a chain stays one click per step.

    Refusing the fourth would make the user clear and re-pick two atoms they
    had already chosen — the opposite of fluid.
    """
    out = _store("""
        const m = S.createMeasurementStore();
        m.toggle(5); m.toggle(6); m.toggle(7);
        const three = m.get();
        m.toggle(8);
        const four = m.get();
        m.toggle(9);
        console.log(JSON.stringify({ three, four, five: m.get() }));
    """)
    assert out["three"] == [5, 6, 7]
    assert out["four"] == [6, 7, 8], "the fourth pick must drop the OLDEST"
    assert out["five"] == [7, 8, 9]


def test_the_order_is_the_click_order_so_the_second_pick_is_the_vertex():
    """§ 11.6: the vertex of an angle is the atom picked SECOND.  Only the
    click order can carry that, so the track is a list and not a set — picking
    7, then 5, then 9 must not come back sorted."""
    out = _store("""
        const m = S.createMeasurementStore();
        m.toggle(7); m.toggle(5); m.toggle(9);
        console.log(JSON.stringify({ picks: m.get() }));
    """)
    assert out["picks"] == [7, 5, 9], \
        "a sorted track would make the vertex the middle atom BY NUMBER"


def test_clicking_a_picked_atom_takes_it_back_out():
    out = _store("""
        const m = S.createMeasurementStore();
        m.toggle(1); m.toggle(2);
        m.toggle(1);
        console.log(JSON.stringify({ picks: m.get() }));
    """)
    assert out["picks"] == [2]


def test_turning_measuring_off_ends_the_session_and_clear_only_empties_it():
    """Two controls, two jobs — and OFF means nothing is being measured.

    The picks survived the toggle until 2026-08-31, on the reasoning that
    coming back to a half-finished measurement costs nothing.  It was not
    free: the Cell page's pick buttons read the COUNT, so they stayed enabled
    with the ruler off, titled "turn measuring on", and staged a row from picks
    nothing on screen was showing.  Off-with-picks and off-without behaved
    differently and looked identical.

    Clear is still its own control: it empties the track WITHOUT leaving the
    mode, which is what a user wants mid-measurement.
    """
    out = _store("""
        const m = S.createMeasurementStore();
        m.setActive(true); m.toggle(1); m.toggle(2);
        m.setActive(false);
        const afterOff = m.get();
        m.setActive(true);
        const backOn = m.get();
        m.toggle(1); m.toggle(2);
        m.clear();
        console.log(JSON.stringify({ afterOff, backOn, afterClear: m.get(),
                                     stillActive: m.getState().active }));
    """)
    assert out["afterOff"] == [], (
        "turning measuring off ends the session — leaving picks behind is the "
        "state that made the Cell page's buttons act on marks nobody could see"
    )
    assert out["backOn"] == [], "coming back starts a new measurement, not the old one"
    assert out["afterClear"] == []
    assert out["stillActive"] is True, "Clear empties the track, not the toggle"


# ---------------------------------------------------------------------------
# The wall
# ---------------------------------------------------------------------------

def test_the_track_never_appears_in_the_selections_snapshot():
    """§ 8.4: the panel, the halo, the count and isolate are all readers of ONE
    settled snapshot.  If the track were a field on it, every one of them would
    see the measurement — the panel would tick rows nobody selected, and isolate
    would hide the structure around three atoms picked to read an angle."""
    out = _store("""
        const sel = S.createSelectionStore({});
        const m = S.createMeasurementStore();
        m.setActive(true); m.toggle(1); m.toggle(2); m.toggle(3);
        const state = sel.getState();
        console.log(JSON.stringify({
            keys: Object.keys(state).sort(),
            values: JSON.stringify(state),
            selection: sel.get(),
        }));
    """)
    assert out["selection"] == [], "measuring must not select anything"
    for forbidden in ("picks", "measurement", "measuring", "active"):
        assert forbidden not in out["keys"], \
            f"`{forbidden}` in the selection's snapshot is the wall breaking"


def test_a_click_lands_in_exactly_one_track():
    """The router (§ 11.6).  With measuring ON the selection must not move; with
    it OFF the track must not.  Asserting only the first half would pass on a
    router that wrote to BOTH."""
    out = _model("""
        const m = await loaded();
        m.pickAtom(0);
        const offTrack = m.measurement.getState().picks, offSel = m.selection.get();
        m.measurement.setActive(true);
        m.pickAtom(2);
        const onTrack = m.measurement.getState().picks, onSel = m.selection.get();
        console.log(JSON.stringify({ offTrack, offSel, onTrack, onSel }));
    """)
    assert out["offSel"] == [0] and out["offTrack"] == [], \
        "with measuring off a click is a selection and nothing else"
    assert out["onTrack"] == [2], "with measuring on a click feeds the ruler"
    assert out["onSel"] == [0], \
        "measuring moved the selection — this is the leak the user named"


def test_clearing_a_measurement_leaves_the_selection_alone():
    """The failure the user is guarding against, stated as its consequence: if
    the two shared a list, clearing a measurement would silently change what the
    next Delete removes."""
    out = _model("""
        const m = await loaded();
        m.selection.add([0, 1, 2]);
        m.measurement.setActive(true);
        m.pickAtom(3);
        m.measurement.clear();
        console.log(JSON.stringify({ selection: m.selection.get() }));
    """)
    assert out["selection"] == [0, 1, 2]


def test_the_track_is_not_structure_metadata():
    """User: *"not part of the structure meta data but rather molview internal
    status"*.  A label IS metadata and travels with the file; the track must not
    reach one, and must not appear in what a state records."""
    out = _model("""
        const m = await loaded();
        m.measurement.setActive(true);
        m.pickAtom(1); m.pickAtom(2);
        const atoms = m.getAtoms();
        const struct = m.getStructure();
        console.log(JSON.stringify({
            labelsOnPicked: atoms.filter((a) => [1, 2].includes(a.index))
                                 .map((a) => a.labels),
            structureKeys: Object.keys(struct).sort(),
            serialised: JSON.stringify(struct),
        }));
    """)
    assert out["labelsOnPicked"] == [[], []], "measuring wrote a label"
    for key in ("measurement", "measuring", "picks"):
        assert key not in out["structureKeys"], \
            f"`{key}` became structure metadata"
        assert key not in out["serialised"], \
            f"`{key}` is inside the structure and would travel with the file"


# ---------------------------------------------------------------------------
# The mark on the atom
# ---------------------------------------------------------------------------

ENGINE = MODULE_DIR / "render-engine.js"
ENGINE_PRELUDE = f"""
const E = await import({json.dumps(ENGINE.resolve().as_uri())});
const FRAME = {{
    elements: ["C", "O", "N", "H"],
    positions: [[0,0,0], [1,0,0], [2,0,0], [3,0,0]],
}};
const at = (over) => E.processFrame(Object.assign({{}}, FRAME, over));
"""


def _engine(snippet: str):
    return run_node([], ENGINE_PRELUDE + snippet)


def test_the_picked_atoms_are_marked_in_the_drawing():
    """User, 2026-08-30: *"the measurement selection need some indicator at the
    atom?"*  Without one, picking in the 3D window is blind — the chip names the
    atoms, but nothing on the molecule says which three they are.

    The marks are content, never styling (§ 6.5): this says WHICH atoms, and
    what a mark looks like is the sealed layer's constant.
    """
    out = _engine("""
        const off = at({ measurement: { active: false, picks: [1, 2] } });
        const on  = at({ measurement: { active: true,  picks: [1, 2] } });
        const none = at({ measurement: { active: true, picks: [] } });
        console.log(JSON.stringify({
            off: off.measured, on: on.measured, none: none.measured,
        }));
    """)
    assert out["on"] == [1, 2], "the picked atoms are not marked"
    assert out["off"] is None, (
        "the marks belong to the ruler: with it off there are none"
    )
    assert out["none"] is None


def test_the_marks_survive_isolate_where_the_highlight_does_not():
    """The one place the two glows differ, and it is not an oversight.

    Under isolate the highlight is null because the drawn set already IS the
    selection, so marking all of it says nothing.  The measured atoms are a
    handful WITHIN that set, so which of them they are is exactly the thing
    still worth saying — and the marks are renumbered through the same map, so
    they land on the right atoms in the cut-down list.
    """
    out = _engine("""
        const iso = at({
            selection: [1, 2, 3],
            switches: { isolate: true },
            measurement: { active: true, picks: [3, 1] },
        });
        console.log(JSON.stringify({
            highlight: iso.selection, measured: iso.measured,
            sourceIndex: iso.sourceIndex,
        }));
    """)
    assert out["highlight"] is None, "the highlight's isolate rule changed"
    # The drawn list is [1, 2, 3] renumbered to [0, 1, 2]; atoms 3 and 1 are
    # drawn seats 2 and 0.  Asserting the ORIGINAL numbers here would pass on a
    # marker that lands on the wrong atom under isolate.
    assert out["sourceIndex"] == [1, 2, 3]
    assert out["measured"] == [2, 0], (
        f"the marks were not renumbered into the drawn list: {out['measured']}"
    )


def test_a_pick_that_is_no_longer_in_the_structure_is_dropped():
    """A track restored from the lane, or held across an edit, can name an atom
    the drawn list does not have.  A mark with no atom must vanish, not throw."""
    out = _engine("""
        const gone = at({ measurement: { active: true, picks: [1, 99] } });
        console.log(JSON.stringify({ measured: gone.measured }));
    """)
    assert out["measured"] == [1]


# ---------------------------------------------------------------------------
# Source pins — where the track may and may not be named
# ---------------------------------------------------------------------------

def test_only_the_view_context_lane_persists_the_track():
    """§ 11.2b: *looking is not changing*.  The track is written in one file and
    no other — not `history.js` (a state, a draft, the badge), not
    `model-jobs.js` (a request body), not the export paths."""
    src = _sources()

    # READING is not the wall.  The track has readers by design — the readout
    # draws it, the drawing marks it, the router picks into it — and each is
    # listed with what it does, so a NEW name here is a decision somebody made
    # rather than a line that slipped in.
    readers = {
        "stores.js":      "owns it",
        "model.js":       "assembles it and routes clicks into it",
        "ui.js":          "the rail toggle and the readout",
        "render-engine.js": "derives the marks on the picked atoms (§ 11.6)",
        "ui-context.js":  "the one lane that persists it (§ 11.2b)",
        # ADDED 2026-08-31, and it reverses the note that stood here.  The
        # window's click still goes through `model.pickAtom` -- WHICH track it
        # lands in is still not this file's opinion.  What it now asks is
        # whether the ruler is on, and that is a different question: under
        # isolate the drawn numbering is not the real one, so the entry
        # translates the index (§ 6.5's map) and lets a MEASURING click
        # through while still refusing a SELECTING one -- because isolate
        # draws only the selected atoms, and clicking one to toggle it would
        # make it vanish under the cursor.  The guard is about this entry, not
        # about the tracks, which is why it can live here without the routing
        # following it.
        "mount.js":       "asks whether the ruler is on, to let a pick "
                          "through under isolate (§ 11.6)",
        # ADDED 2026-08-31.  The marks became ARROWS, so the sealed layer now
        # names the thing it is drawing.  It draws only -- it is handed which
        # atoms, in order, and owns what a mark LOOKS like (§ 6.5); it reads no
        # track and decides nothing about one.
        "3dmol-embed.js": "draws the marks, one arrow per step (§ 11.6)",
    }
    named = sorted(n for n, t in src.items() if "measurement" in t.lower())
    assert named == sorted(readers), (
        f"the track is named in {named}; the readers it is allowed are "
        f"{sorted(readers)} — a new one is a decision, not an oversight")

    # WHAT IT MAY NEVER REACH, each for its own reason.
    assert "measurement" not in src["history.js"], \
        "a measurement is not an edit: no state, no draft, no badge"
    assert "measurement" not in src["model-jobs.js"], \
        "the track must never leave the browser in a request body"
    assert "measurement" in src["ui-context.js"], \
        "the <owner>:ui lane is where it IS kept (§ 11.2b)"


def test_the_readout_reads_the_track_and_not_the_selection():
    """§ 11.6's input changed; this is the line that says so.  A readout still
    reading `selection` would look identical until the user selected an atom
    for an edit and got a measurement they never asked for."""
    ui = _sources()["ui.js"]
    body = ui.split("function mountReadout", 1)[1].split("\nfunction ", 1)[0]
    assert "model.measurement" in body
    assert "model.selection" not in body, \
        "the readout must take its atoms from the track alone"


def test_the_geometric_vertex_guess_is_gone():
    """It existed only because a SELECTION can arrive with no pick order — from
    All, Invert, a filter, a restored session.  A track is only ever built by
    clicks, so the case is unreachable and the guess is deleted rather than left
    to be maintained (`archive/2026-09-01-modify-redesign-plan.md` § 1.3)."""
    ui = _sources()["ui.js"]
    assert "byGeometry" not in ui and "orderedForMeasurement" not in ui


def test_the_ops_group_still_comes_from_the_selection():
    """The hazard items 1 and 2 create together: item 2 makes Center act on the
    selection, and every op resolves its group through one door.  If that door
    ever read the track, clearing a measurement would change what an edit
    operates on."""
    viewer = re.sub(r"/\*.*?\*/", "", (REPO / "molbuilder" / "web" / "static"
                    / "modify" / "viewer.js").read_text(), flags=re.S)
    viewer = re.sub(r"^\s*//.*$", "", viewer, flags=re.M)
    # The functions here are INDENTED (module-closure style), so splitting on a
    # column-0 `function` swallowed the rest of the file -- 1199 lines, one of
    # which says "measurement" in a banner comment.  A pin that reads the whole
    # file is not reading the door.
    def door(name):
        after = viewer.split("function " + name, 1)[1]
        return after.split("\n    function ", 1)[0]

    assert "measurement" not in door("selectedIndices"), \
        "an op's group must come from the selection, never from the ruler"
    # ...and the door it goes through resolves the SELECTION store by name.
    resolver = door("_selStore")
    assert "d.selection" in resolver and "measurement" not in resolver

    jobs = _sources()["model-jobs.js"]
    assert "readSelection" in jobs and "measurement" not in jobs


# ---------------------------------------------------------------------- #
#  THE CLEARING RULE HAS ONE HOME (§ 11.6, user 2026-08-31)               #
#                                                                        #
#  It lived at three call sites and the fourth door -- the cell commit -- #
#  had already forgotten it.  It is now stated once in `settle`, which    #
#  every structure change passes through, with a single exemption for the #
#  doors that PROVE the atoms are unchanged (`requireSameAtoms` runs      #
#  before they land): a running job's frames arriving.  Those must not    #
#  clear, because § 12.4 is measuring an angle WHILE a trajectory plays.  #
# ---------------------------------------------------------------------- #

def test_a_structure_change_clears_the_track_without_the_door_asking():
    """The rule, from the outside: pick atoms, change the structure, and the
    picks are gone -- with no door having decided that for itself."""
    out = _model("""
        const m = await loaded();
        m.measurement.setActive(true);
        m.pickAtom(2); m.pickAtom(0);
        const before = m.measurement.getState().picks;
        await m.installMolecule({
            text: "4\\n\\nO 0 0 0\\nO 1 0 0\\nO 2 0 0\\nO 3 0 0\\n",
            filename: "y.xyz",
        });
        console.log(JSON.stringify({ before, after: m.measurement.getState().picks }));
    """)
    assert out["before"] == [2, 0], "the picks were not taken in click order"
    assert out["after"] == [], (
        "a structure change must clear the ordered track -- the picks name "
        "atoms in the molecule that just went away (molview.md § 11.6)"
    )


def test_a_frame_arriving_from_a_job_does_NOT_clear_the_track():
    """The one exemption, and the reason it exists: § 12.4 is measuring while
    a trajectory plays.  The picks are indices and the readout re-reads the
    current frame, so a growing movie must leave them standing."""
    out = _model("""
        const m = await loaded();
        m.measurement.setActive(true);
        m.pickAtom(1); m.pickAtom(3);
        m.addFrame([[0,0,0],[1.1,0,0],[2,0,0],[3,0,0]]);
        m.addFrames([[[0,0,0],[1.2,0,0],[2,0,0],[3,0,0]]]);
        m.setForces(null);
        console.log(JSON.stringify({ picks: m.measurement.getState().picks }));
    """)
    assert out["picks"] == [1, 3], (
        "frames arriving from a running job prove the atoms are unchanged, so "
        "they must not clear the track -- clearing here deletes the very "
        "measurement § 12.4 exists to show"
    )


# ---------------------------------------------------------------------- #
#  AN ORDERED OP READS THE ORDERED TRACK (§ 11.6, user 2026-08-31)        #
#                                                                        #
#  `orient`'s answer is WHICH ATOM WAS FIRST -- first -> second is the    #
#  tilt direction.  It read `selection`, which SORTS, so the same two     #
#  atoms picked by clicking and by shift-range oriented in opposite       #
#  directions with nothing said.  The op table now declares the track.    #
# ---------------------------------------------------------------------- #

_RECORDING_SERVER = """
globalThis.__sent = [];
globalThis.fetch = async function (route, init) {
    globalThis.__sent.push({ route, body: JSON.parse(init.body) });
    return { ok: true, status: 200, json: async () => ({
        ok: true,
        atoms: [0, 1, 2, 3].map((i) => ({
            index: i, element: "C", x: i, y: 0, z: 0, regions: [] })),
        n_atoms: 4,
        periodicity: { cell: null, cell_origin: null,
                       axis_kind: ["free", "free", "free"], vacuum: null },
    }) };
};
"""


def _model_recording(snippet: str):
    return run_node([], MODEL_PRELUDE + snippet, globals_js=_RECORDING_SERVER)


def test_orient_sends_the_anchors_in_CLICK_order_not_sorted_order():
    """The bug, driven: pick 2 then 0 with the ruler while the selection holds
    the same two atoms sorted.  What reaches the server must be the click
    order, because reversing the pair reverses the tilt."""
    out = _model_recording("""
        const m = await loaded();
        // the SET says {0, 2}: `add` SORTS, and `add` is what shift-range and
        // the drag box call -- which is exactly how the same two atoms came to
        // orient in two directions depending on how they were picked.
        m.selection.add([2, 0]);
        // the TRACK says 2 then 0 -- the order they were clicked
        m.measurement.setActive(true);
        m.pickAtom(2); m.pickAtom(0);
        await m.applyOp("orient", { axis: "z", angle: 0, center: "midpoint" });
        const sent = globalThis.__sent.filter((s) => /orient/.test(s.route));
        console.log(JSON.stringify({
            selection: m.selection.get(),
            anchors: sent.length ? sent[0].body.anchors : null,
        }));
    """)
    assert out["selection"] == [0, 2], \
        "the selection store is a SET and sorts -- that is the premise"
    assert out["anchors"] == [2, 0], (
        "orient must send the CLICK order from the ruler's track, not the "
        "sorted selection -- reversing the pair reverses the tilt "
        "(molview.md § 11.6, `ordered` column in § 11.1's table)"
    )


def test_a_set_op_still_reads_the_selection_and_ignores_the_ruler():
    """The other half, or the column would be a rule with no edge: `delete`
    declares no `ordered`, so the picks must not reach it."""
    out = _model_recording("""
        const m = await loaded();
        m.selection.adopt([1, 3]);
        m.measurement.setActive(true);
        m.pickAtom(0);
        await m.applyOp("delete", {});
        const sent = globalThis.__sent.filter((s) => /delete/.test(s.route));
        console.log(JSON.stringify({
            indices: sent.length ? sent[0].body.indices : null }));
    """)
    assert out["indices"] == [1, 3], (
        "a set op reads the selection; the ruler's picks are a different "
        "track and must not reach it"
    )


# ---------------------------------------------------------------------- #
#  THE SURFACE IS NARROWER THAN THE STORE (user, 2026-08-31)              #
#                                                                        #
#  "The internal selection states should not be accessible from outside   #
#  the module ... this is the only way to guarantee that you're not       #
#  misusing any of those states."  The model used to hand out the STORE,  #
#  so every internal write door came with it and a consumer could add a   #
#  pick without going through `pickAtom` -- the one place that decides    #
#  measuring-vs-selecting and the one place isolate has been translated.  #
# ---------------------------------------------------------------------- #

def test_the_router_is_what_a_pick_goes_through():
    """The other half: `pickAtom` must actually reach the track, or the door
    above would be closed with nothing open in its place."""
    out = _model("""
        const m = await loaded();
        m.measurement.setActive(true);
        m.pickAtom(2); m.pickAtom(0);
        const measuring = m.measurement.getState().picks;
        m.measurement.setActive(false);      // ...which also ends the session
        m.pickAtom(3);                       // now it means SELECT
        console.log(JSON.stringify({
            measuring, after: m.measurement.getState().picks, selection: m.selection.get(),
        }));
    """)
    assert out["measuring"] == [2, 0], "picks land in click order while measuring"
    assert out["after"] == [], (
        "turning the ruler off clears the track, and a click with it off must "
        "not put anything back into it"
    )
    assert out["selection"] == [3], "...it selects instead"


def test_asking_for_the_ruler_is_the_modules_decision_not_the_pages():
    """`requestPicking` answers whether it turned the ruler ON, so a caller
    never has to read `active` and set it itself — which is the two-step both
    panels were copying, and which lived in a page-side file until the module
    took it back."""
    out = _model("""
        const m = await loaded();
        const first  = m.requestPicking();   // was off -> turns on
        const second = m.requestPicking();   // already on
        const active = m.measurement.getState().active;
        console.log(JSON.stringify({ first, second, active }));
    """)
    assert out["first"] is True, "asking with the ruler off turns it on and says so"
    assert out["second"] is False, (
        "already on is not an event — announcing it would greet every tab "
        "opening with a notice about a mode nobody touched"
    )
    assert out["active"] is True


# ---------------------------------------------------------------------- #
#  NOTHING ESCAPES THE MODULE (user, 2026-08-31)                          #
#                                                                        #
#  "All the measurement API has to be within MolView, and all the users   #
#  have to call through it.  Nothing escapes."                            #
#                                                                        #
#  A design invariant, so it is checked statically: it is violated by     #
#  NEW code reaching past the surface, which no behavioural test can see  #
#  until someone writes that code.                                       #
# ---------------------------------------------------------------------- #

#: What the model hands a consumer (model.js).  Anything else named on a
#: `.measurement.` is a reach past the surface into the store.
_SURFACE = {"getState", "positions", "subscribe", "setActive",
            "requestPicking", "clear"}

_STATIC = REPO / "molbuilder" / "web" / "static"


def _outside_module_js():
    for path in sorted(_STATIC.rglob("*.js")):
        rel = path.relative_to(_STATIC).as_posix()
        if rel.startswith("lib/molview/") or rel.startswith("vendor/"):
            continue
        yield rel, path


def test_no_consumer_reaches_past_the_measurement_surface():
    offenders = {}
    for rel, path in _outside_module_js():
        src = path.read_text(encoding="utf-8", errors="ignore")
        src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
        src = re.sub(r"^\s*//.*$", "", src, flags=re.M)
        for m in re.finditer(r"\.measurement\s*\.\s*(\w+)", src):
            if m.group(1) not in _SURFACE:
                offenders.setdefault(rel, set()).add(m.group(1))
    assert not offenders, (
        f"these reach past the measurement surface: "
        f"{ {k: sorted(v) for k, v in offenders.items()} }.  The model hands "
        f"out {sorted(_SURFACE)} and nothing else; a pick is written through "
        f"`pickAtom` alone (molview.md § 11.6)."
    )
    # ...and the scan is not vacuous: the two real consumers must be seen.
    seen = {rel for rel, p in _outside_module_js()
            if ".measurement." in p.read_text(encoding="utf-8", errors="ignore")}
    assert {"modify/periodicity.js", "modify/viewer.js"} <= seen, (
        f"the corpus lost its known consumers, so a pass means nothing: {seen}"
    )


def test_no_consumer_imports_a_molview_internal():
    """The module is `index.js` and its two exports.  A consumer that imports
    `model.js` or `stores.js` directly has the store itself, and every rule
    above is moot."""
    offenders = {}
    for rel, path in _outside_module_js():
        src = path.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"""from\s+["'][^"']*molview/([\w.-]+)["']""", src):
            if m.group(1) != "index.js":
                offenders.setdefault(rel, set()).add(m.group(1))
    assert not offenders, (
        f"these import a MolView internal instead of index.js: "
        f"{ {k: sorted(v) for k, v in offenders.items()} } — "
        f"'a consumer that imports any of them directly has broken the "
        f"module, not found a shortcut' (molview.md § 4)"
    )
