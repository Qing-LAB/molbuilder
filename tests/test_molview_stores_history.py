"""The stores and the session history — every test derived from
``docs/web/molview.md``, never from the source it checks (§ 13).

Step E of the rebuild (``docs/web/molview-rework-plan.md``). The rows of § 13.3
guarded here:

    § 9.5  the selection survives an editor switch
    § 9.5  a half-typed row constrains nothing
    § 9.5  by atom index crosses the numbering boundary once
    § 9.5  a label is a change to the truth
    § 9.6  the camera is not kept, saved or read back
    § 11.2 state is the truth, not the view of it
    § 11.2 there is no automatic write
    § 11.2 a bracketed change writes once, at the end
    § 11.2 saving a state is the user's act, and undo returns to it
    § 11.2 a Retract spends unsaved work first
    § 11.2 Save state drops what was above it
    § 11.2 reopening returns to the point you were on
    § 11.2 a new structure invalidates the old one's pending writes
    § 11.2 the history is offered as calls, not as a control
"""
from __future__ import annotations

import json
from pathlib import Path

from tests._node_esm import run_node

REPO = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO / "molbuilder" / "web" / "static" / "lib" / "molview"
STORES = MODULE_DIR / "stores.js"
HISTORY = MODULE_DIR / "history.js"

PRELUDE = f"""
const S = await import({json.dumps(STORES.resolve().as_uri())});
const H = await import({json.dumps(HISTORY.resolve().as_uri())});

// A store that obeys what § 11.2 says sits underneath: it holds bytes at a step
// and hands them back. It knows nothing about what is in them.
function makeStore() {{
    const slots = {{}};
    const log = [];
    let failNext = false;
    return {{
        slots, log,
        failWrite() {{ failNext = true; }},
        async write(step, bytes) {{
            if (failNext) {{ failNext = false; throw new Error("storage refused"); }}
            log.push({{ step, bytes }});
            slots[step] = bytes;
        }},
        async read(step) {{ return step in slots ? slots[step] : null; }},
    }};
}}

// The model's end: a way to make a state and a way to put one back. The history
// never looks inside either.
function makeHistory(store) {{
    const restored = [];
    const badges = [];
    let truth = "opened";
    const h = H.createHistory({{
        recordState: () => truth,
        restoreState: (s) => restored.push(s),
        store: store,
        onBadge: (b) => badges.push(b),
    }});
    return {{ h, restored, badges, edit: (t) => {{ truth = t; h.edited(); }} }};
}}
"""


def _run(snippet: str):
    return run_node([], PRELUDE + snippet)


# ---------------------------------------------------------------------------
# § 9.5 — the selection
# ---------------------------------------------------------------------------

def test_the_selection_survives_an_editor_switch():
    """§ 13.3: "moving between click and filter mode leaves the selection exactly
    as it was."

    § 9.5: the selection is the truth; click and filter are two EDITORS of it.
    """
    out = _run(
        """
        const sel = S.createSelectionStore({});
        sel.add([3, 1, 7]);
        const before = sel.get();
        sel.setEditor("filter");
        const inFilter = sel.get();
        sel.setEditor("click");
        console.log(JSON.stringify({ before, inFilter, back: sel.get() }));
        """
    )
    assert out["before"] == [1, 3, 7]
    assert out["inFilter"] == [1, 3, 7], "switching editors moved the selection"
    assert out["back"] == [1, 3, 7]


def test_a_half_typed_row_constrains_nothing():
    """§ 13.3: "a blank row combined under AND leaves the other rows' result
    intact rather than emptying it."

    § 9.5: "You have not told me anything to intersect with yet" is the correct
    reading of a blank row; treating it as "match nothing" would make the panel
    feel broken mid-typing.
    """
    out = _run(
        """
        const one = S.buildRule([{ kind: "element", value: "Au" }], "and");
        const withBlank = S.buildRule([
            { kind: "element", value: "Au" },
            { kind: "residue", value: "   " },
        ], "and");
        const allBlank = S.buildRule([{ kind: "element", value: "" }], "and");
        const none = S.buildRule([], "and");
        console.log(JSON.stringify({
            one, withBlank, allBlank, none,
            same: JSON.stringify(one) === JSON.stringify(withBlank),
        }));
        """
    )
    assert out["same"] is True, (
        f"a blank row changed the rule: {out['one']} vs {out['withBlank']}"
    )
    assert out["allBlank"] is None and out["none"] is None, (
        "no rows means no filter at all — not a filter that matches nothing"
    )


def test_by_atom_index_crosses_the_numbering_boundary_exactly_once():
    """§ 13.3: "a typed range like `1-4, 6` selects the atoms a user would count
    off on screen, at any structure size, without drifting by one — and the shift
    happens at one point, not at each row."
    """
    out = _run(
        """
        const rule = S.buildRule([{ kind: "index", value: "1-4, 6" }], "and");
        const two = S.buildRule([
            { kind: "index", value: "1-4" },
            { kind: "index", value: "10-11" },
        ], "or");
        const named = S.buildRule([{ kind: "element", value: "Au" }], "and");
        console.log(JSON.stringify({ rule, two, named }));
        """
    )
    assert out["rule"] == {"op": "by_index_range", "expression": "0-3, 5"}, (
        f"the typed range must shift once, into the server's rule: {out['rule']}"
    )
    assert [r["expression"] for r in out["two"]["operands"]] == ["0-3", "9-10"], (
        "each index row shifts, and shifts once"
    )
    assert out["named"] == {"op": "by_element", "elements": ["Au"]}, (
        f"a row that compares names to names must never touch a number: {out['named']}"
    )


def test_the_rule_vocabulary_is_the_servers():
    """§ 11.1: the field-level JSON of these payloads belongs to web-api.md, and
    the rule names are ``molbuilder/selection.py``'s own.

    § 9.5 describes four kinds of row in prose. Inventing a shape for them here
    would give a panel that builds rules nothing can evaluate — which no unit
    test of the panel alone would ever notice.
    """
    out = _run(
        """
        console.log(JSON.stringify({
            element: S.buildRule([{ kind: "element", value: "Au, C" }], "and"),
            residue: S.buildRule([{ kind: "residue", value: "ALA,DA" }], "and"),
            label:   S.buildRule([{ kind: "label", value: "L-electrode" }], "and"),
            index:   S.buildRule([{ kind: "index", value: "1-4" }], "and"),
            both:    S.buildRule([
                { kind: "element", value: "Au" },
                { kind: "label", value: "bridge" },
            ], "or"),
            unknown: S.buildRule([{ kind: "charge", value: "0.3" }], "and"),
        }));
        """
    )
    assert out["element"] == {"op": "by_element", "elements": ["Au", "C"]}
    assert out["residue"] == {"op": "by_residue_name", "names": ["ALA", "DA"]}
    assert out["label"] == {"op": "by_region", "name": "L-electrode"}
    assert out["index"] == {"op": "by_index_range", "expression": "0-3"}
    assert out["both"] == {
        "op": "or",
        "operands": [{"op": "by_element", "elements": ["Au"]},
                     {"op": "by_region", "name": "bridge"}],
    }, f"rows compose under one combinator over operands: {out['both']}"
    assert out["unknown"] is None, (
        "a row kind the server has no rule for must build nothing, rather than "
        "a rule that will be rejected at the far end"
    )


def test_a_label_is_written_through_the_model_not_the_store():
    """§ 9.5: writing a label "is a change to the structure … so it behaves like
    every other truth change and not like its neighbours."

    § 9.4: it is the one truth change reached through the selection door, which
    is precisely why it is listed in § 9.3's table where the gate can see it. A
    change the gate cannot see is a change the gate does not stop.
    """
    out = _run(
        """
        const wrote = [];
        const sel = S.createSelectionStore({
            writeLabel: (name, atoms) => { wrote.push({ name, atoms }); return true; },
        });
        sel.add([2, 5]);
        sel.applyLabel("anchor");
        console.log(JSON.stringify({ wrote }));
        """
    )
    assert out["wrote"] == [{"name": "anchor", "atoms": [2, 5]}], (
        "a label must leave the store through the door the gate stands on"
    )


def test_the_pick_order_survives_separately_from_the_selection():
    """§ 11.6: "the vertex of an angle is the atom picked SECOND, not the middle
    one by number." So the order has to be kept, and kept apart from the set.
    """
    out = _run(
        """
        const sel = S.createSelectionStore({});
        sel.toggle(7); sel.toggle(2); sel.toggle(5);
        console.log(JSON.stringify({ set: sel.get(), order: sel.order() }));
        """
    )
    assert out["order"] == [7, 2, 5], (
        f"the order atoms were picked in was lost: {out['order']}"
    )


def test_filtering_asks_the_server_and_holds_no_matching_logic():
    """§ 9.5: "Filtering is a question asked of the server, not a scan done here.
    MolView holds no matching logic" — the same boundary as § 2's: one place
    decides what a structure means.
    """
    out = _run(
        """
        const asked = [];
        const sel = S.createSelectionStore({
            resolveFilter: async (rule) => { asked.push(rule); return [4, 9]; },
        });
        sel.add([1]);
        sel.setRows([{ kind: "element", value: "Au" }], "and");
        const result = await sel.applyFilter();
        console.log(JSON.stringify({ asked, result, selection: sel.get() }));
        """
    )
    assert out["asked"] == [{"op": "by_element", "elements": ["Au"]}]
    assert out["selection"] == [4, 9], (
        "applying a filter REPLACES the selection (§ 9.5)"
    )


# ---------------------------------------------------------------------------
# § 9.6 — the camera is not kept, saved or read back
# ---------------------------------------------------------------------------

def test_no_store_holds_a_camera():
    """§ 13.3: "nothing above the drawing reports where the camera is pointing."

    § 9.6: MolView never records where it ended up, never reads it back and never
    saves it.
    """
    out = _run(
        """
        const view = S.createViewStore();
        const before = view.get();
        view.set("camera", { x: 1 });           // not a setting this store has
        view.set("orientation", [1,2,3]);
        console.log(JSON.stringify({
            keys: Object.keys(view.get()).sort(),
            unchanged: JSON.stringify(view.get()) === JSON.stringify(before),
        }));
        """
    )
    assert out["unchanged"] is True, "a camera was accepted into the view store"
    assert out["keys"] == ["background", "orthographic", "radius", "style"], (
        f"`view` must be exactly § 9.6's four settings: {out['keys']}"
    )
    for name in ("stores.js", "history.js", "model.js"):
        text = (MODULE_DIR / name).read_text()
        code = "\n".join(
            line for line in text.splitlines()
            if not line.lstrip().startswith(("*", "//", "/*"))
        )
        assert "camera" not in code.lower(), (
            f"{name} names the camera in code — it is held nowhere above the drawing"
        )


def test_a_drawing_setting_is_not_a_switch():
    """§ 9.6's test, which is checkable rather than a convention: "does working
    out what a frame contains require reading it?"

    A switch the frame calculation has to read belongs to `selection`; a setting
    the sealed layer applies without that calculation ever seeing it belongs to
    `view`. Neither store may hold the other's.
    """
    out = _run(
        """
        const sel = S.createSelectionStore({});
        const view = S.createViewStore();
        sel.setSwitch("style", "sphere");        // a drawing setting, not a switch
        view.set("isolate", true);               // a switch, not a drawing setting
        console.log(JSON.stringify({
            switches: Object.keys(sel.switches()).sort(),
            view: Object.keys(view.get()).sort(),
            styleLeaked: "style" in sel.switches(),
            isolateLeaked: "isolate" in view.get(),
        }));
        """
    )
    assert out["styleLeaked"] is False, "a drawing setting got into the switches"
    assert out["isolateLeaked"] is False, "a switch got into the drawing settings"
    assert out["switches"] == ["forceScale", "isolate", "showAxis", "showCell",
                               "showForces", "showIndex"], (
        f"the switches must be exactly § 6.2's: {out['switches']}"
    )


# ---------------------------------------------------------------------------
# § 11.2 — the history
# ---------------------------------------------------------------------------

def test_there_is_no_automatic_write():
    """§ 13.3: "nothing persists except through installing, saving or loading."

    § 11.2: nothing is written on a timer, and nothing is written because
    something changed.
    """
    out = _run(
        """
        const store = makeStore();
        const { h, edit } = makeHistory(store);
        await h.anchor();                        // opening a structure
        const afterOpen = store.log.length;
        edit("one"); edit("two"); edit("three"); // three edits record nothing
        console.log(JSON.stringify({
            afterOpen, afterEdits: store.log.length, badge: h.uncommitted,
        }));
        """
    )
    assert out["afterOpen"] == 1, "opening a structure lays down point 0"
    assert out["afterEdits"] == 1, (
        f"an edit wrote to storage on its own: {out['afterEdits']} writes"
    )
    assert out["badge"] is True, (
        "an edit must raise the badge — without it an explicit-save history "
        "silently loses work the user assumed was being kept"
    )


def test_saving_is_the_users_act_and_undo_returns_to_it():
    """§ 13.3: "an edit records nothing and raises the badge; after three edits
    with no save between them, one undo restores the state before all three."

    § 11.2: three edits after a save "were never three points — they were one
    stretch of work between two of them."
    """
    out = _run(
        """
        const store = makeStore();
        const { h, edit, restored } = makeHistory(store);
        await h.anchor();                        // point 0 == "opened"
        edit("one"); edit("two"); edit("three");
        await h.undo();
        console.log(JSON.stringify({
            restored, at: h.state_index, badge: h.uncommitted,
        }));
        """
    )
    assert out["restored"] == ["opened"], (
        f"undo must land on the state before all three edits: {out['restored']}"
    )
    assert out["at"] == 0
    assert out["badge"] is False


def test_a_retract_spends_unsaved_work_first():
    """§ 13.3: "from a saved point with edits on top, one Retract lands ON that
    point with the edits discarded; a second lands on the point before it."

    § 11.2: the first press undoes what you just did, not what you had already
    decided to keep.
    """
    out = _run(
        """
        const store = makeStore();
        const { h, edit, restored } = makeHistory(store);
        await h.anchor();                        // point 0 == "opened"
        edit("first edit");
        await h.save(1);                         // point 1 == "first edit"
        edit("unsaved work");                    // sits on top of point 1

        await h.undo();
        const afterFirst = { at: h.state_index, restored: restored.slice() };
        await h.undo();
        const afterSecond = { at: h.state_index, restored: restored.slice() };
        console.log(JSON.stringify({ afterFirst, afterSecond }));
        """
    )
    assert out["afterFirst"]["at"] == 1, (
        "the first Retract must land ON the saved point, discarding the edits — "
        f"it went to {out['afterFirst']['at']}"
    )
    assert out["afterFirst"]["restored"] == ["first edit"]
    assert out["afterSecond"]["at"] == 0, (
        "only the second Retract steps to the point before it"
    )
    assert out["afterSecond"]["restored"] == ["first edit", "opened"]


def test_save_state_drops_what_was_above_it():
    """§ 13.3: "after retracting past two points and saving, stepping forward is
    no longer possible — the abandoned points are gone."

    § 11.2: the moment a user commits to a different path, the abandoned one
    stops existing.
    """
    out = _run(
        """
        const store = makeStore();
        const { h, edit } = makeHistory(store);
        await h.anchor();
        edit("a"); await h.save(1);              // point 1
        edit("b"); await h.save(1);              // point 2
        const top = h.state_index;

        await h.load(-1);                        // back to 1
        await h.load(-1);                        // back to 0
        edit("different path");
        await h.save(1);                         // a new point 1, dropping 2
        const afterSave = h.state_index;
        const forward = await h.load(+1);        // nothing above to step into
        console.log(JSON.stringify({ top, afterSave, forward }));
        """
    )
    assert out["top"] == 2
    assert out["afterSave"] == 1
    assert out["forward"] is None, (
        "stepping forward must be impossible after a save dropped the abandoned "
        f"points: {out['forward']}"
    )


def test_stepping_forward_lasts_until_you_save():
    """§ 11.2: "After a Retract you can step forward again into the points you
    moved away from. Saving ends that."
    """
    out = _run(
        """
        const store = makeStore();
        const { h, edit, restored } = makeHistory(store);
        await h.anchor();
        edit("a"); await h.save(1);
        edit("b"); await h.save(1);
        await h.load(-1);
        const back = h.state_index;
        const forward = await h.load(+1);
        console.log(JSON.stringify({ back, forward, restored }));
        """
    )
    assert out["back"] == 1
    assert out["forward"] == 2, "a Retract must be reversible until a save"


def test_reopening_returns_to_the_point_you_were_on():
    """§ 13.3: "a reload comes back to the current point rather than to the
    anchor, and does not move the position."

    § 11.2: `load(0)` is not "move by nothing" — it is a different verb.
    """
    out = _run(
        """
        const store = makeStore();
        const { h, edit, restored } = makeHistory(store);
        await h.anchor();
        edit("a"); await h.save(1);
        edit("b"); await h.save(1);              // sitting on point 2

        const at = await h.load(0);
        console.log(JSON.stringify({
            at, position: h.state_index, restored,
        }));
        """
    )
    assert out["at"] == 2 and out["position"] == 2, (
        f"load(0) must put back the point you were on, and not move: {out}"
    )
    assert out["restored"] == ["b"], (
        "a reopened page comes back to the current point, not the anchor"
    )


def test_a_bracketed_change_writes_once_at_the_end():
    """§ 13.3: "a write requested mid-bracket does not land until the bracket
    closes, and what lands is the settled state."

    § 11.2: between the coordinates arriving and the labels arriving, the viewer
    holds the new positions with the previous file's labels — a structure that
    never existed.
    """
    out = _run(
        """
        const store = makeStore();
        const { h, edit } = makeHistory(store);
        await h.anchor();
        store.log.length = 0;

        h.beginChange();
        const during = h.writeState;
        edit("halfway"); await h.save(1);
        edit("also halfway"); await h.save(1);
        const midBracket = store.log.length;
        edit("settled");
        await h.endChange();
        console.log(JSON.stringify({
            during, midBracket, after: store.log.map(w => w.bytes),
        }));
        """
    )
    assert out["during"] == "changing"
    assert out["midBracket"] == 0, (
        f"a write landed while the structure was halfway between two files: "
        f"{out['midBracket']}"
    )
    assert len(out["after"]) == 1, (
        f"a bracket must write once, not once per request: {out['after']}"
    )
    assert out["after"] == ["settled"], (
        f"what lands must be the SETTLED state, not the halfway one: {out['after']}"
    )


def test_a_saved_state_wins_over_a_routine_write_held_beside_it():
    """§ 11.2: "At most one is remembered; if a SAVED STATE is among them, that
    is the one sent, and a routine write arriving after it does not replace it."

    Two rules of § 11.2 meet here and it is worth separating them, because they
    answer different questions:

      - WHICH BYTES land is settled by "what lands is the settled state" — so
        they are taken when the write goes out, not when it was asked for.
      - WHICH REQUEST wins is what this rule decides, and the difference between
        a save and a routine write is that A SAVE MOVES YOUR POSITION. So the
        assertion is about the position, not the payload.

    Getting that backwards would mean holding stale bytes in order to honour a
    rule that was never about bytes.
    """
    out = _run(
        """
        const store = makeStore();
        const { h, edit } = makeHistory(store);
        await h.anchor();
        store.log.length = 0;

        h.beginChange();
        edit("halfway"); await h.save(1);         // a SAVE, held
        edit("also halfway"); await h.save(0);    // a routine write, held after it
        edit("settled");                          // the change finishes
        await h.endChange();
        console.log(JSON.stringify({
            wrote: store.log.map(w => ({ step: w.step, bytes: w.bytes })),
            at: h.state_index,
            badge: h.uncommitted,
        }));
        """
    )
    assert len(out["wrote"]) == 1, f"a bracket must write once: {out['wrote']}"
    assert out["at"] == 1, (
        "the held SAVE is the one that went out, so the position moved — a "
        f"routine write would have left it at 0: {out['at']}"
    )
    assert out["wrote"][0]["step"] == 1
    assert out["wrote"][0]["bytes"] == "settled", (
        f"the bytes must be the settled state, not the halfway one: {out['wrote']}"
    )
    assert out["badge"] is False, "a landed save clears the badge"


def test_a_new_structure_drops_what_was_held_for_the_old_one():
    """§ 13.3: "a save still in flight when a new structure is opened does not
    apply its state over the new one."

    § 11.2: applying it would put an old state over a freshly opened structure.
    The more authoritative statement about what the structure is beats whatever
    is in flight.
    """
    out = _run(
        """
        const store = makeStore();
        const { h, edit } = makeHistory(store);
        await h.anchor();

        h.beginChange();
        edit("belongs to the OLD structure");
        await h.save(1);                          // held, never sent
        store.log.length = 0;

        // A new structure arrives and anchors over it.
        edit("the NEW structure");
        await h.anchor();
        await h.endChange();

        console.log(JSON.stringify({
            wrote: store.log.map(w => ({ step: w.step, bytes: w.bytes })),
            at: h.state_index,
        }));
        """
    )
    assert all(w["bytes"] != "belongs to the OLD structure" for w in out["wrote"]), (
        f"a held write for the replaced structure was applied over the new one: "
        f"{out['wrote']}"
    )
    assert out["at"] == 0, "a new structure anchors at point 0"


def test_a_failed_write_does_not_move_the_position():
    """§ 11.2's machine: "it landed: the position moves" / "it failed: the
    position does not move."

    A position describing a state that was never written is how a reload comes
    back to something that does not exist.
    """
    out = _run(
        """
        const store = makeStore();
        const { h, edit } = makeHistory(store);
        await h.anchor();
        edit("a");
        store.failWrite();
        await h.save(1);
        console.log(JSON.stringify({ at: h.state_index, badge: h.uncommitted }));
        """
    )
    assert out["at"] == 0, (
        f"the position moved for a write that never landed: {out['at']}"
    )
    assert out["badge"] is True, (
        "the badge must stay up — the work is still not on the sequence"
    )


def test_the_mechanism_never_looks_inside_a_state():
    """§ 11.2: "The mechanism does not know or care what is in it … So NOTHING
    ABOUT SAVING CONSTRAINS WHAT MAY BE SAVED."

    A trajectory needs no new mechanism to become restorable, and neither does
    anything added to the truth later.
    """
    out = _run(
        """
        const store = makeStore();
        // States that are not text at all, and are never the same shape twice.
        const shapes = [
            { frames: [[[0,0,0]]], labels: {} },
            "a string",
            [1, 2, 3],
            null,
        ];
        let which = 0;
        const restored = [];
        const h = H.createHistory({
            recordState: () => shapes[which],
            restoreState: (s) => restored.push(s),
            store: store,
        });
        await h.anchor();
        for (which = 1; which < shapes.length; which++) { h.edited(); await h.save(1); }
        await h.load(0);
        console.log(JSON.stringify({
            wrote: store.log.map(w => w.bytes),
            restored,
        }));
        """
    )
    assert out["wrote"][0] == {"frames": [[[0, 0, 0]]], "labels": {}}, (
        "a structure with frames must pass through untouched — frames are the "
        "truth and need no new mechanism"
    )
    assert out["wrote"][1] == "a string" and out["wrote"][2] == [1, 2, 3]


def test_the_history_draws_no_control_of_its_own():
    """§ 13.3: "a mounted viewer draws no save-state or retract button of its
    own, and the calls work all the same when a host wires its own."

    § 11.2: saving a state carries no decision, and a host knows better than a
    viewer when it is worth offering one.
    """
    code = (MODULE_DIR / "history.js").read_text()
    for token in ("document.", "createElement", "addEventListener", "innerHTML"):
        assert token not in code, (
            f"the history reaches the DOM ({token}) — it is offered as calls, "
            "not as a control"
        )
