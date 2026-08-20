"""The view context (molview.md § 11.2b) — run, not described.

The REAL model and the REAL lane execute together against the stand-in
server; only the sealed layer's pose pair is a stub (its real half is
vendor 3Dmol, pinned in test_molview_3dmol_embed.py).  What these pin:

  1. looking is never changing — a lane write raises no badge and rewrites
     no draft (the user's rule, verbatim: "not a change of data");
  2. the round trip — switches, style, frame and camera come back;
  3. the match guard — a different structure gets the preferences but
     never the pose, the frame, or the selection;
  4. selection ownership — the truth lane owns it where one exists, the
     ui lane carries it only for read-only viewers;
  5. ordering — the context applies AFTER a draft restore's view.reset(),
     not under it.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests._node_esm import run_node

REPO = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO / "molbuilder" / "web" / "static" / "lib" / "molview"
MODEL = MODULE_DIR / "model.js"
UICTX = MODULE_DIR / "ui-context.js"

# The same stand-in server test_molview_model.py boots the model against.
from tests.test_molview_model import SERVER  # noqa: E402

PRELUDE = f"""
const {{ createModel }} = await import({json.dumps(MODEL.resolve().as_uri())});
const {{ attachUiContext }} =
    await import({json.dumps(UICTX.resolve().as_uri())});

// A workspace that STORES (the real front door's shape, workspace.md § 5),
// with every write visible to the test.
function makeStore() {{
    const files = new Map();
    const writes = [];
    return {{
        files, writes,
        workspaceId: (tag) => "id-" + tag,
        persist(tag, bytes, identity) {{
            const key = identity.workspace_id + ":" + identity.state_index;
            files.set(key, JSON.parse(JSON.stringify(bytes)));
            writes.push(key);
            return true;
        }},
        readState: async (identity) => {{
            const key = identity.workspace_id + ":" + identity.state_index;
            return files.has(key) ? files.get(key) : null;
        }},
        pruneStatesAbove: () => {{}},
    }};
}}

// The pose pair, stubbed at the engine seam (§ 9.7's two bounded askings).
function makeEngine(pose) {{
    return {{
        pose: pose || null,
        pointed: [],
        getCamera() {{ return this.pose ? this.pose.slice() : null; }},
        setCamera(p) {{ this.pointed.push(p.slice()); return true; }},
    }};
}}

const WAIT_FLUSH = () => new Promise((r) => setTimeout(r, 500));
const WAIT_RESTORE = () => new Promise((r) => setTimeout(r, 50));

async function session(store, opts) {{
    globalThis.__requests = [];
    globalThis.__serverFails = false;
    const model = createModel(Object.assign({{ owner: "s", workspace: store }},
                                            (opts && opts.model) || {{}}));
    const engine = makeEngine((opts && opts.pose) || null);
    const off = attachUiContext({{
        model, engine, workspace: store, owner: "s",
        canvasEl: null,
        hasTruthLane: !(opts && opts.model && opts.model.mode === "readonly"),
    }});
    return {{ model, engine, off }};
}}

const TWO_ATOMS = "2\\n\\nC 0 0 0\\nO 1 0 0\\n";
"""


def _run(snippet: str):
    return run_node([], PRELUDE + snippet, globals_js=SERVER)


def test_looking_is_never_changing():
    """A lane write raises no badge and rewrites no draft — the timeline
    cannot even see it (the module holds no reference to the history)."""
    out = _run(
        """
        const store = makeStore();
        const one = await session(store);
        await one.model.installMolecule({ text: TWO_ATOMS, filename: "x.xyz" });
        await WAIT_RESTORE();
        const writesAfterInstall = store.writes.slice();

        one.model.selection.setSwitch("showCell", true);
        one.model.view.set("style", "sphere");
        await WAIT_FLUSH();

        const newWrites = store.writes.slice(writesAfterInstall.length);
        console.log(JSON.stringify({
            newWrites,
            badge: one.model.uncommitted,
            at: one.model.state_index,
        }));
        """
    )
    assert out["newWrites"] == ["id-s:ui:0"], (
        f"a look change wrote outside the lane: {out['newWrites']} — the "
        f"draft and the points are the truth's, and looking is not changing"
    )
    assert out["badge"] is False
    assert out["at"] == 0


def test_the_round_trip_restores_how_you_were_looking():
    out = _run(
        """
        const store = makeStore();
        const one = await session(store, { pose: [1, 2, 3, 4] });
        await one.model.installMolecule({ text: TWO_ATOMS, filename: "x.xyz" });
        await WAIT_RESTORE();
        one.model.view.set("style", "sphere");
        one.model.selection.setSwitch("showAxis", true);
        one.model.selection.setSwitch("forceScale", 2);
        await WAIT_FLUSH();
        one.off();

        const two = await session(store);
        await two.model.installMolecule({ text: TWO_ATOMS, filename: "x.xyz" });
        await WAIT_RESTORE();
        console.log(JSON.stringify({
            style: two.model.view.get().style,
            axis: two.model.selection.switches().showAxis,
            scale: two.model.selection.switches().forceScale,
            pointed: two.engine.pointed,
            badge: two.model.uncommitted,
        }));
        """
    )
    assert out["style"] == "sphere"
    assert out["axis"] is True and out["scale"] == 2
    assert out["pointed"] == [[1, 2, 3, 4]], (
        "the pose was not put back on a matching structure"
    )
    assert out["badge"] is False, "restoring a look raised the badge"


def test_a_different_structure_gets_the_preferences_never_the_pose():
    """§ 11.2b's guard: style and switches are the user's standing
    preferences; the camera, the frame and the selection belonged to a
    structure, and a stale pose can leave a molecule off-screen."""
    out = _run(
        """
        const store = makeStore();
        const one = await session(store, { pose: [9, 9, 9] });
        await one.model.installMolecule({ text: TWO_ATOMS, filename: "x.xyz" });
        await WAIT_RESTORE();
        one.model.view.set("style", "sphere");
        await WAIT_FLUSH();
        one.off();

        const two = await session(store);
        globalThis.__nextPayload = __payload([
            __atomRow(0, "C", 0), __atomRow(1, "O", 1), __atomRow(2, "H", 2),
        ]);
        await two.model.installMolecule({ text: "3\\n\\nC\\nO\\nH\\n",
                                          filename: "y.xyz" });
        await WAIT_RESTORE();
        console.log(JSON.stringify({
            style: two.model.view.get().style,
            pointed: two.engine.pointed,
            frame: two.model.currentFrame(),
        }));
        """
    )
    assert out["style"] == "sphere", "the standing preference must carry over"
    assert out["pointed"] == [], "a pose was applied onto a different structure"
    assert out["frame"] == 0


def test_selection_rides_the_lane_only_where_no_truth_lane_exists():
    out = _run(
        """
        // READ-ONLY: the lane carries the selection home.
        const roStore = makeStore();
        const ro1 = await session(roStore, { model: { mode: "readonly" } });
        await ro1.model.installMolecule({ text: TWO_ATOMS, filename: "x.xyz" });
        await WAIT_RESTORE();
        ro1.model.selection.add([1]);
        await WAIT_FLUSH();
        ro1.off();
        const ro2 = await session(roStore, { model: { mode: "readonly" } });
        await ro2.model.installMolecule({ text: TWO_ATOMS, filename: "x.xyz" });
        await WAIT_RESTORE();

        // EDITABLE: the draft owns the selection; the lane must not fight it.
        const edStore = makeStore();
        const ed1 = await session(edStore);
        await ed1.model.installMolecule({ text: TWO_ATOMS, filename: "x.xyz" });
        await WAIT_RESTORE();
        ed1.model.selection.add([0, 1]);
        await WAIT_FLUSH();
        ed1.off();
        const uiSlot = edStore.files.get("id-s:ui:0");

        console.log(JSON.stringify({
            roSelection: ro2.model.selection.get(),
            editableLaneCarries: "selection" in (uiSlot || {}),
        }));
        """
    )
    assert out["roSelection"] == [1], (
        "a read-only viewer's selection did not survive — it has no other "
        "way home"
    )
    assert out["editableLaneCarries"] is False, (
        "the lane carries a selection beside a truth lane that owns it — "
        "two lanes restoring one fact is the drift § 5.2 forbids"
    )


def test_the_context_applies_after_a_draft_restore_not_under_it():
    """§ 11.2b's ordering sentence: a draft's own restore calls
    `view.reset()` after its settle announces; the context must land on
    top of that, or the style comes back only to be reset."""
    out = _run(
        """
        const store = makeStore();
        const one = await session(store);
        await one.model.installMolecule({ text: TWO_ATOMS, filename: "x.xyz" });
        await WAIT_RESTORE();
        one.model.view.set("style", "sphere");
        await WAIT_FLUSH();
        one.off();

        // A reopened page: a FRESH viewer adopts the draft (load(0) calls
        // view.reset()), and the context must still win the ordering.
        const two = await session(store);
        const adopted = await two.model.load(0);
        await WAIT_RESTORE();
        console.log(JSON.stringify({
            adopted: adopted !== null && adopted !== undefined,
            style: two.model.view.get().style,
        }));
        """
    )
    assert out["adopted"] is True
    assert out["style"] == "sphere", (
        "the draft's view.reset() landed AFTER the context — the ordering "
        "§ 11.2b states is broken"
    )
