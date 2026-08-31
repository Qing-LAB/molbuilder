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


def test_turning_measuring_off_keeps_the_picks_and_clear_empties_them():
    """Two controls, two jobs.  Coming back to a half-finished measurement
    costs nothing; Clear is how the user says they are done, and it says so in
    one place rather than being a side effect of the toggle."""
    out = _store("""
        const m = S.createMeasurementStore();
        m.setActive(true); m.toggle(1); m.toggle(2);
        m.setActive(false);
        const afterOff = m.get();
        m.setActive(true);
        const backOn = m.get();
        m.clear();
        console.log(JSON.stringify({ afterOff, backOn, afterClear: m.get(),
                                     stillActive: m.getState().active }));
    """)
    assert out["afterOff"] == [1, 2], "turning it off must not discard the work"
    assert out["backOn"] == [1, 2]
    assert out["afterClear"] == []
    assert out["stillActive"] is True, "Clear empties the track, not the toggle"


def test_a_restored_track_is_trimmed_and_cleaned():
    """The lane is bytes that were on disk (§ 11.2b), so what comes back is
    checked rather than trusted — a saved four, a negative, a string."""
    out = _store("""
        const m = S.createMeasurementStore();
        m.adopt([1, 2, 3, 4]);
        const tooMany = m.get();
        m.adopt([-1, "2", null, 3]);
        console.log(JSON.stringify({ tooMany, junk: m.get() }));
    """)
    assert out["tooMany"] == [1, 2, 3], "a saved four must not raise the cap"
    assert out["junk"] == [3]


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
        const offTrack = m.measurement.get(), offSel = m.selection.get();
        m.measurement.setActive(true);
        m.pickAtom(2);
        const onTrack = m.measurement.get(), onSel = m.selection.get();
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
# Source pins — where the track may and may not be named
# ---------------------------------------------------------------------------

def test_only_the_view_context_lane_persists_the_track():
    """§ 11.2b: *looking is not changing*.  The track is written in one file and
    no other — not `history.js` (a state, a draft, the badge), not
    `model-jobs.js` (a request body), not the export paths."""
    src = _sources()
    writes = sorted(n for n, t in src.items()
                    if "measurement" in t and n not in
                    ("stores.js", "model.js", "ui.js", "ui-context.js", "mount.js"))
    assert writes == [], f"the track is named in {writes}; it belongs to the lane"
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
    to be maintained (`plans/modify-redesign-plan.md` § 1.3)."""
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
