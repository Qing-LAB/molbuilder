"""The stores and the session history — every test derived from
``docs/web/molview.md``, never from the source it checks (§ 13).

Step E of the rebuild (``docs/web/molview.md``). The rows of § 13.3
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

/* A STAND-IN WORKSPACE — the calls docs/web/workspace.md § 5 lists, answered from
 * memory, keyed per tag as the real one is.
 *
 * IT IMPLEMENTS THE REAL DOOR ON PURPOSE. What was here before was a two-call
 * store, `read(step)` / `write(step, bytes)`, which is what the module asked for
 * and what no workspace has ever had — so every test in this file passed while
 * nothing on any page could save. A stand-in has to obey the level it stands in
 * for (§ 13.1); one shaped like the caller's wish only ever confirms the wish.
 *
 * `slots` is the numbered points, `sessions` the browser copy — the two things
 * § 11.3 keeps apart, kept apart here so a test can ask about either.
 */
function makeStore() {{
    const slots  = {{}};       // step -> a milestone, as it was written
    const drafts = {{}};       // id   -> the latest draft, replaced each time
    const log = [];           // every write, in order
    const pruned = [];        // every tail-drop asked for
    let failNext = false;
    const isDraft = (identity) => /-draft$/.test(identity.workspace_id);
    return {{
        slots, drafts, log, pruned,
        failWrite() {{ failNext = true; }},
        workspaceId(tag) {{ return "id-" + tag; }},
        persist(tag, bytes, identity) {{
            if (failNext) {{ failNext = false; return false; }}
            const draft = isDraft(identity);
            log.push({{ tag, bytes, draft, step: identity.state_index,
                       // what the caller actually saved, unwrapped
                       state: bytes && bytes.state }});
            if (draft) drafts[identity.workspace_id] = bytes;
            else slots[identity.state_index] = bytes;
            return true;
        }},
        async readState(identity) {{
            if (isDraft(identity)) {{
                return identity.workspace_id in drafts
                    ? drafts[identity.workspace_id] : null;
            }}
            const step = identity.state_index;
            return step in slots ? slots[step] : null;
        }},
        pruneStatesAbove(id, index) {{
            pruned.push({{ id, index }});
            Object.keys(slots).forEach((step) => {{
                if (Number(step) > index) delete slots[step];
            }});
        }},
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
        workspace: store,
        tag: "test-viewer",
        onBadge: (b) => badges.push(b),
    }});
    return {{
        h, restored, badges,
        edit: (t) => {{ truth = t; return h.edited(); }},
    }};
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


def test_invert_takes_the_complement_and_reports_no_pick_trail():
    """`invert` was the one door on the selection store with NO test at all.

    Two things about it are easy to get wrong, and both matter downstream.
    It is the COMPLEMENT against the atom count it is handed -- not against
    whatever happens to be drawn -- and, like All and an applied filter, it is
    NOT a click, and after 2026-08-31 there is no pick order for it to carry.

    **The trail half of this test is retired**, not weakened.  It asserted that
    a bulk operation reports an EMPTY `pickOrder`, because a fabricated trail
    was indistinguishable from a real one to the measurement readout.  That
    readout no longer reads this store at all -- measuring has its own track --
    and the field itself is gone: order and count limits belong to the ruler,
    and this store is a SET (`molview.md` § 9.5).  Asserting a rule no document
    states is how a test outlives its contract, so what is left is the half
    that is still true.
    """
    out = _run(
        """
        const sel = S.createSelectionStore({});
        sel.add([1, 3]);
        sel.invert(5);
        const state = sel.getState();
        console.log(JSON.stringify({
            selection: state.selection,
            keys: Object.keys(state).sort(),
        }));
        """
    )
    assert out["selection"] == [0, 2, 4], (
        f"invert is the complement against the count it was handed: {out['selection']}")
    assert "pickOrder" not in out["keys"], (
        "the snapshot must not carry an order field: this store is a set, and "
        "a second field for 'in what order' is what made the two tracks "
        "overlap (molview.md § 9.5)")


def test_isolate_turns_itself_off_when_invert_empties_the_selection():
    """§ 1.1: isolate turns itself off when the selection empties, "since there
    would be nothing left to show". It is a SELECTION-STATE rule, so it holds
    whichever operation did the emptying -- including this one, which no test
    reached before."""
    out = _run(
        """
        const sel = S.createSelectionStore({});
        sel.add([0, 1, 2]);
        sel.setIsolate(true);
        const on = sel.getState().isolate;
        sel.invert(3);                      // everything was selected -> none is
        const state = sel.getState();
        console.log(JSON.stringify({
            on, after: state.isolate, selection: state.selection,
        }));
        """
    )
    assert out["on"] is True
    assert out["selection"] == []
    assert out["after"] is False, (
        "isolate stayed on with nothing selected -- the window would be empty")


def test_a_half_typed_row_constrains_nothing():
    """§ 13.3: "a blank row combined under AND leaves the other rows' result
    intact rather than emptying it."

    § 9.5: "You have not told me anything to intersect with yet" is the correct
    reading of a blank row; treating it as "match nothing" would make the panel
    feel broken mid-typing.
    """
    out = _run(
        """
        const one = S.buildRule([{ kind: "by_element", value: "Au" }], "and");
        const withBlank = S.buildRule([
            { kind: "by_element", value: "Au" },
            { kind: "by_residue", value: "   " },
        ], "and");
        const allBlank = S.buildRule([{ kind: "by_element", value: "" }], "and");
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
        const rule = S.buildRule([{ kind: "by_index", value: "1-4, 6" }], "and");
        const two = S.buildRule([
            { kind: "by_index", value: "1-4" },
            { kind: "by_index", value: "10-11" },
        ], "or");
        const named = S.buildRule([{ kind: "by_element", value: "Au" }], "and");
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
            element: S.buildRule([{ kind: "by_element", value: "Au, C" }], "and"),
            residue: S.buildRule([{ kind: "by_residue", value: "ALA,DA" }], "and"),
            label:   S.buildRule([{ kind: "by_label", value: "L-electrode" }], "and"),
            index:   S.buildRule([{ kind: "by_index", value: "1-4" }], "and"),
            both:    S.buildRule([
                { kind: "by_element", value: "Au" },
                { kind: "by_label", value: "bridge" },
            ], "or"),
            unknown: S.buildRule([{ kind: "by_charge", value: "0.3" }], "and"),
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
        sel.writeLabel("anchor");
        console.log(JSON.stringify({ wrote }));
        """
    )
    assert out["wrote"] == [{"name": "anchor", "atoms": [2, 5]}], (
        "a label must leave the store through the door the gate stands on"
    )


def test_the_panel_is_handed_one_settled_state_whole():
    """§ 8.4: the panel is given ONE snapshot — what is selected and in what
    order, which editor is showing, the rows and how they combine, and every
    switch — handed over on subscribing and again after every change.

    This is the fix for a real failure, not a style preference: the pick order
    was maintained correctly in the store for months and simply LEFT OUT of the
    snapshot, so the panel read nothing, fell back to guessing an angle's vertex
    from geometry, and § 11.6's chemist's-pick rule was dead end to end while
    looking implemented.

    A FACT THE STORE KEEPS BUT DOES NOT HAND OVER DOES NOT EXIST. So the check is
    not "does the store track the pick order" — it is "does the snapshot carry
    it", which is the thing that was actually missing.
    """
    out = _run(
        """
        const sel = S.createSelectionStore({});
        // Handed one on subscribing, so the first paint needs no separate fetch.
        const seen = [];
        sel.subscribe((state) => seen.push(state));
        const onSubscribe = seen.length;

        sel.toggle(7); sel.toggle(2); sel.toggle(5);
        sel.setIsolate(true);
        sel.addFilter({ kind: "by_element", value: "Au" });
        sel.setCombinator("or");
        const latest = seen[seen.length - 1];

        console.log(JSON.stringify({
            onSubscribe,
            keys: Object.keys(latest).sort(),
            selection: latest.selection,
            isolate: latest.isolate,
            filters: latest.filters,
            combinator: latest.combinator,
            everyChangeDelivered: seen.length,
        }));
        """
    )
    assert out["onSubscribe"] == 1, (
        "subscribing must hand over a state immediately, or the first paint "
        "needs a separate fetch"
    )
    assert out["selection"] == [7, 2, 5], (
        "clicks APPEND -- `toggle` does not sort, so what a click-built "
        "selection holds is what was clicked.  That is a fact about this "
        "store, not an order promise: `add`, `all`, `invert` and a filter all "
        "sort, and nothing may depend on the difference (molview.md § 9.5)")
    assert out["isolate"] is True
    assert out["filters"] == [{"kind": "by_element", "value": "Au"}]
    assert out["combinator"] == "or"
    # Everything the panel draws, in one object.
    # `filterOutcome` joined 2026-08-04: what the last apply matched, so the
    # panel can say "that rule found nothing" — which an empty selection alone
    # cannot distinguish from never having filtered.
    assert out["keys"] == ["combinator", "filterOutcome", "filters", "forceScale",
                           "isolate", "mode", "selection",
                           "showAxis", "showCell", "showForces", "showIndex"], (
        f"the snapshot must be everything the panel draws: {out['keys']}"
    )


def test_a_filter_row_is_edited_one_at_a_time():
    """§ 8.4: "a user adds a row, types in it, changes its kind, removes it, and
    chooses how the rows combine — each of those is its own small change, because
    that is what the controls are."

    A surface that only accepted the whole set at once would make the panel
    rebuild and re-send state it was in the middle of editing.
    """
    out = _run(
        """
        const sel = S.createSelectionStore({});
        const states = [];
        sel.subscribe((s) => states.push(s.filters));

        sel.addFilter();                                  // "+ Add filter"
        const blankRow = states[states.length - 1];
        sel.updateFilter(0, { value: "Au" });              // typing
        sel.addFilter({ kind: "by_label", value: "" });
        sel.updateFilter(1, { kind: "by_residue" });       // re-kinding
        const two = states[states.length - 1];
        sel.removeFilter(0);                              // the row's ×
        console.log(JSON.stringify({
            blankRow, two, afterRemove: states[states.length - 1],
        }));
        """
    )
    assert out["blankRow"] == [{"kind": "by_element", "value": ""}], (
        "adding a row must give a blank row of a sensible default kind, not "
        f"require the caller to supply one: {out['blankRow']}"
    )
    assert out["two"] == [{"kind": "by_element", "value": "Au"},
                          {"kind": "by_residue", "value": ""}], (
        f"each edit must touch only its own row: {out['two']}"
    )
    assert out["afterRemove"] == [{"kind": "by_residue", "value": ""}]


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
        sel.addFilter({ kind: "by_element", value: "Au" });
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


def test_isolate_turns_itself_off_when_the_selection_empties():
    """§ 1.1: "Isolate turns itself off when the selection becomes empty, since
    there would be nothing left to show."

    It belongs to the STORE and not to the control that emptied the selection:
    Clear, Remove, an inverted selection that lands on nothing, a filter that
    matches nothing and a restored empty session all reach the same line. Put in
    a button's handler instead, it would be right for that button and wrong for
    the other four.

    It settles before the snapshot goes out, so no reader ever sees isolate on
    with nothing selected (§ 8.4 — one settled state).
    """
    out = _run(
        """
        const sel = S.createSelectionStore({});
        sel.add([0, 1]);
        sel.setSwitch("isolate", true);
        const isolating = sel.getState().isolate;

        const seen = [];
        sel.subscribe((s) => seen.push({ isolate: s.isolate, selection: s.selection }));
        sel.clear();
        const atClear = seen[seen.length - 1];   // the snapshot the clear sent

        // And it does not switch itself back on when atoms are picked again:
        // it is a switch the user sets.
        sel.add([2]);
        console.log(JSON.stringify({
            isolating,
            afterClear:   sel.getState().isolate,
            snapshotSaw:  atClear,
            afterPicking: sel.getState().isolate,
        }));
        """
    )
    assert out["isolating"] is True
    assert out["afterClear"] is False, (
        "isolate stayed on with nothing selected — the viewer is hiding every "
        "atom it has to show"
    )
    assert out["snapshotSaw"] == {"isolate": False, "selection": []}, (
        f"a reader saw isolate on beside an empty selection: {out['snapshotSaw']}"
    )
    assert out["afterPicking"] is False, "isolate switched itself back on"


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

def test_an_edit_refreshes_the_session_copy_and_lays_down_no_point():
    """§ 11.2 and § 11.3 — the two rules that must not collapse into each other.

    An edit changes what would be lost if the tab closed, so it **refreshes the
    session copy**. An edit is not a milestone, so it **lays down no point**: the
    user decides what is worth coming back to and says so.

    Persistence is about the accident; the timeline is about the decision. Make
    persistence explicit and an unsaved edit dies on a reload; make the timeline
    automatic and every keystroke becomes a milestone, so there is nothing left
    to come back *to*.

    This replaced a test asserting that an edit wrote NOTHING, which was the rule
    when the two were one thing.
    """
    out = _run(
        """
        const store = makeStore();
        const { h, edit } = makeHistory(store);
        await h.anchor();                        // opening a structure
        const pointsAfterOpen = Object.keys(store.slots).length;
        await edit("one"); await edit("two"); await edit("three");
        console.log(JSON.stringify({
            pointsAfterOpen,
            points:  Object.keys(store.slots).sort(),
            draft:   store.drafts["id-test-viewer-draft"],
            badge:   h.uncommitted,
            at:      h.state_index,
        }));
        """
    )
    assert out["pointsAfterOpen"] == 1, "opening a structure lays down point 0"
    assert out["points"] == ["0"], (
        f"an edit laid down a point of its own — the user never asked for one: "
        f"{out['points']}"
    )
    assert out["at"] == 0, "an edit moved the position on the sequence"
    assert out["draft"]["state"] == "three", (
        f"the draft does not hold the latest edit, so a reload would lose it: "
        f"{out['draft']!r}"
    )
    # AND WHERE THAT LEFT YOU (molview.md § 11.2a).  A reopened page builds a
    # fresh viewer that starts at position 0 with no sequence, so these three are
    # the only way it can come back standing where the user left off.  Asserted
    # as a whole envelope rather than key-by-key: this is the shape the reader
    # in `adopt()` expects, and the two drifting apart is the failure that would
    # restore the right structure into the wrong place on the sequence.
    assert out["draft"] == {
        "v": 1, "state": "three", "at": 0, "highest": 0, "dirty": True,
    }, f"the draft does not describe where the session was: {out['draft']!r}"
    # The version stamp is not decoration: the reader parses the bytes, looks for
    # it, and hands back nothing without it. Every save went in unstamped until
    # 2026-08-02 and no read ever came back out.
    assert out["draft"]["v"] == 1, "the draft carries no version stamp"
    assert out["badge"] is True, (
        "an edit must raise the badge — the work is kept against an accident, "
        "but it is not on the sequence and Retract cannot reach it"
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


def test_load_zero_puts_back_the_point_you_are_on_without_moving():
    """§ 11.2: `load(0)` is not "move by nothing" — it is a different verb. The
    three things this surface does are *step back*, *step forward*, and *restore
    where I was*, and this is the third.

    WHAT THIS DOES NOT SHOW, named rather than left as a silent hole: that a
    REOPENED PAGE comes back to where it was. That needs a second history over
    the same store — a fresh viewer, as a reload builds — and such a viewer has
    no sequence to load from until something anchors one (§ 11.2a). This test
    carried that promise in its name while exercising a single instance, so the
    § 13.3 row above it was guarded by a test of something else.
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
            during, midBracket, after: store.log.filter(w => !w.draft).map(w => w.state),
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
            wrote: store.log.filter(w => !w.draft).map(w => ({ step: w.step, bytes: w.state })),
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
            wrote: store.log.filter(w => !w.draft).map(w => ({ step: w.step, bytes: w.state })),
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
            workspace: store,
            tag: "test-viewer",
        });
        await h.anchor();
        for (which = 1; which < shapes.length; which++) { h.edited(); await h.save(1); }
        await h.load(0);
        console.log(JSON.stringify({
            // Only the writes that carried a POINT: an edit refreshes the session
            // copy and sends no point (§ 11.3), so those rows carry null and are
            // not what this rule is about.
            wrote: store.log.filter(w => !w.draft).map(w => w.state),
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


# ---------------------------------------------------------------------------
# § 11.2 — a read waits for a write that has not finished
# ---------------------------------------------------------------------------

def test_a_read_cannot_answer_from_before_a_write_that_is_under_way():
    """§ 11.2: "a read waits for a write that has not finished."

    THE ORDERING NOW FALLS OUT OF THE WRITE BEING SYNCHRONOUS. `persist` writes
    the session copy straight away and sends the numbered point without waiting,
    so by the time `save` has returned to its caller the position has already
    moved — there is no instant in between for a read to land in.

    It was not always so. When the history awaited a store's own `write`, a
    Retract or a reopen arriving during that await chose its point from a
    position the save had not yet moved, and then read bytes the save had not yet
    written. The guard that waits for an unfinished write is still in the code,
    because "the session copy is written synchronously" is the workspace's
    promise rather than this module's, and a store that broke it would put the
    window straight back.

    What is asserted here is the property, not the mechanism: interleaved as
    tightly as this module allows, the read answers about the point the save
    created.
    """
    out = _run(
        """
        const store = makeStore();
        const { h, restored, edit } = makeHistory(store);
        await h.anchor();
        await edit("first edit");
        await h.save(1);
        await edit("second edit");

        // Started, not awaited — then a read immediately behind it.
        const saving = h.save(1);
        const reading = h.load(0);
        await saving;
        const target = await reading;

        console.log(JSON.stringify({
            target, restored, points: Object.keys(store.slots).sort(),
        }));
        """
    )
    assert out["points"] == ["0", "1", "2"], (
        f"the save did not land: {out['points']}"
    )
    assert out["target"] == 2, (
        f"the read answered about point {out['target']}, which the save had "
        f"already moved past"
    )
    assert out["restored"] == ["second edit"], (
        f"the read returned bytes from before the save: {out['restored']}"
    )


def test_a_failed_write_does_not_wedge_the_next_read():
    """The wait must end when the write ENDS, not when it succeeds. A write that
    failed has settled too — it left the position where it was — so a read after
    it answers from the point that is actually current, and does not hang.
    """
    out = _run(
        """
        const store = makeStore();
        const { h, restored, edit } = makeHistory(store);
        await h.anchor();
        edit("first edit");
        await h.save(1);

        edit("second edit");
        store.failWrite();
        const failed = await h.save(1);      // refused by the storage

        const target = await h.load(0);      // put back where I am
        console.log(JSON.stringify({ failed, target, restored, index: h.state_index }));
        """
    )
    assert out["failed"] is False, "a refused write reported as landed"
    assert out["index"] == 1, (
        f"a refused write moved the position: {out['index']}"
    )
    assert out["target"] == 1, f"the read after a failed write answered {out['target']}"
    assert out["restored"] == ["first edit"], (
        f"the read did not return the point that is actually current: "
        f"{out['restored']}"
    )


def test_a_sequence_outlives_its_page_and_a_fresh_viewer_takes_it_up():
    """PINS molview.md § 11.2a — coming back to a session without re-opening the file.

    § 11.2 says the sequence is persistent: "it outlives the page". A reopened
    page builds a FRESH viewer, so `load(0)` is the one call that has to work
    before anything has been installed. It used to refuse — `if (!anchored)
    return null` — which made that sentence false: the bytes sat on disk and
    nothing could reach them.

    WHAT COMES BACK IS THE DRAFT, NOT THE POINT. The point is where the user last
    chose to be able to return to; the draft is what was actually on screen.
    Coming back to the point throws away every edit made after it, silently,
    because the structure that appears looks perfectly reasonable.

    The second session shares the STORE and nothing else — a new history, as a
    new page would build.
    """
    out = _run(
        """
        const store = makeStore();

        // Session one: open, edit, save a point, then edit again.
        const one = makeHistory(store);
        await one.h.anchor();
        await one.edit("after-open");
        await one.h.save(1);              // the point the user chose
        await one.edit("work-since-the-save");

        // Session two: a NEW history over the same store, as a reopened page
        // builds a new viewer.
        const two = makeHistory(store);
        const beforeAdopt = { at: two.h.state_index, badge: two.h.uncommitted };
        const adopted = await two.h.load(0);
        // READ THE MOMENT IT LANDS. The Retract below correctly lowers the
        // badge, so asking afterwards measures the retract, not the restore.
        const afterAdopt = {
            at: two.h.state_index, badge: two.h.uncommitted,
            restored: two.restored.slice(),
        };

        // And the sequence is usable: Retract spends the unsaved work first,
        // landing on the point that was saved (§ 11.2).
        const retracted = await two.h.load(-1);

        console.log(JSON.stringify({
            beforeAdopt,
            adopted,
            restored: afterAdopt.restored,
            at: afterAdopt.at,
            badge: afterAdopt.badge,
            retracted,
            atAfterRetract: two.h.state_index,
            restoredAfterRetract: two.restored[two.restored.length - 1],
        }));
        """
    )
    assert out["beforeAdopt"] == {"at": 0, "badge": False}, (
        "a fresh viewer must start knowing nothing — otherwise this test is "
        "measuring leftover state, not a restore"
    )
    assert out["adopted"] == 1, (
        f"load(0) on a fresh viewer did not take up the sequence: {out['adopted']!r}. "
        f"§ 11.2 says it outlives the page"
    )
    assert out["restored"][0] == "work-since-the-save", (
        f"the reopened page came back to the POINT, losing the work done after "
        f"it — the exact loss persistence exists to prevent: {out['restored']!r}"
    )
    assert out["at"] == 1, (
        f"the work came back but the position did not: standing on {out['at']}, "
        f"not on the point the session was on. Retract would go somewhere the "
        f"user never was"
    )
    assert out["badge"] is True, (
        "the badge came back down over work that is not on the sequence — an "
        "explicit-save history that lies about this loses work silently (§ 11.2)"
    )
    # And the adopted sequence actually works.
    assert out["retracted"] == 1 and out["atAfterRetract"] == 1, (
        f"Retract on an adopted sequence did not spend the unsaved work first: "
        f"{out['retracted']!r} / {out['atAfterRetract']!r}"
    )
    assert out["restoredAfterRetract"] == "after-open", (
        f"Retract restored the wrong point: {out['restoredAfterRetract']!r}"
    )


def test_a_first_visit_finds_nothing_and_stays_empty():
    """The other half of § 11.2a: no draft is a FIRST VISIT, not a failure.

    `load(0)` answers null, the viewer stays EMPTY and ready for an install, and
    nothing is restored. Distinguishing this from a broken read matters because
    the two want opposite responses — one waits for the user to open something,
    the other must not pretend it did.
    """
    out = _run(
        """
        const store = makeStore();
        const fresh = makeHistory(store);
        const answer = await fresh.h.load(0);
        console.log(JSON.stringify({
            answer, restored: fresh.restored, at: fresh.h.state_index,
            wrote: store.log.length,
        }));
        """
    )
    assert out["answer"] is None, "a first visit must answer nothing"
    assert out["restored"] == [], "nothing was stored, so nothing may be restored"
    assert out["wrote"] == 0, (
        "adopting wrote something — it must not: the sequence being taken up is "
        "the one that was already there (§ 11.2a, adopting is not anchoring)"
    )


def test_a_draft_this_build_cannot_read_is_not_guessed_at():
    """A version stamp this code does not know reads as nothing (§ 11.2a).

    These files outlive the code that wrote them: someone upgrades molbuilder and
    opens a tab whose draft was written by the old one. Bytes from a layout this
    build has never seen are not something to guess at — and guessing would put a
    structure on screen assembled from fields that meant something else.
    """
    out = _run(
        """
        const store = makeStore();
        store.drafts["id-test-viewer-draft"] =
            { v: 99, state: "from-a-later-molbuilder", at: 3, highest: 3 };
        const fresh = makeHistory(store);
        const answer = await fresh.h.load(0);
        console.log(JSON.stringify({
            answer, restored: fresh.restored, at: fresh.h.state_index,
        }));
        """
    )
    assert out["answer"] is None, "an unreadable draft must answer nothing"
    assert out["restored"] == [], (
        "bytes from an unknown layout were restored anyway — a structure "
        "assembled from fields that meant something else"
    )
    assert out["at"] == 0, "an unreadable draft moved the position"


def test_match_none_is_the_complement_of_match_any():
    """§ 1.1's third combine option, and the reading that makes it useful.

    "Match none" wraps the rows in the server's own `not` (selection.py::Not) —
    MolView still holds no matching logic (§ 9.5). It complements ANY, not ALL:
    with rows `Au` and `S` a user wants every atom that is neither, and
    NOT(Au AND S) would be almost the whole structure, since no atom is both.
    """
    out = _run(
        """
        const rows = [{ kind: "by_element", value: "Au" },
                      { kind: "by_element", value: "S" }];
        console.log(JSON.stringify({
            and:  S.buildRule(rows, "and"),
            or:   S.buildRule(rows, "or"),
            nor:  S.buildRule(rows, "nor"),
            one:  S.buildRule([rows[0]], "nor"),
        }));
        """
    )
    assert out["nor"]["op"] == "not", out["nor"]
    # The complement is taken over OR, so every atom matching NO row is picked.
    assert out["nor"]["rule"] == out["or"], (
        f"Match none must complement Match any, not Match all: {out['nor']}")
    # One row negates too — "everything except the gold" is a real request.
    assert out["one"] == {"op": "not", "rule": out["and"]["operands"][0]}, out["one"]


def test_an_applied_filter_says_how_many_it_matched():
    """A filter that matches nothing empties the selection, and an empty
    selection is indistinguishable from never having filtered — so the panel
    could not tell the user their rule found nothing, which is the one thing
    they need to hear (and why `◉` then appears to do nothing: isolate requires
    a non-empty selection).

    The outcome rides in the snapshot like every other fact the panel draws, and
    is cleared the moment the rule changes so a stale answer cannot show.
    """
    out = _run(
        """
        const store = S.createSelectionStore({
            resolveFilter: async () => [],        // the rule matches nothing
            writeLabel:    async () => null,
        });
        const before = store.getState().filterOutcome;
        store.addFilter({ kind: "by_element", value: "Xx" });
        await store.applyFilter();
        const after = store.getState().filterOutcome;
        store.updateFilter(0, { value: "Au" });   // the question changed
        const edited = store.getState().filterOutcome;
        console.log(JSON.stringify({ before, after, edited,
                                     selected: store.getState().selection }));
        """
    )
    assert out["before"] is None, "nothing applied yet is not an outcome"
    assert out["after"] == {"matched": 0, "isolateTurnedOff": False}, out["after"]
    assert out["selected"] == [], "a filter matching nothing selects nothing"
    assert out["edited"] is None, (
        "editing a row must clear the recorded outcome — a stale answer to a "
        "question the user has since changed is worse than none")


def test_a_filter_that_empties_the_selection_reports_isolate_switching_off():
    """The store already turns isolate off when the selection empties ("there
    would be nothing left to show", § 1.1) — so after a filter matches nothing,
    `isolate` is ALREADY false and "it was on and switched off" cannot be told
    from "it was off all along".

    The panel needs that difference: a control changing state without the user
    touching it is worth a sentence, and without it the message said isolate
    "needs a selection, so the structure is still shown" — implying the switch
    was still on. It is not. Every test passed while that sentence was wrong;
    a browser caught it.
    """
    out = _run(
        """
        const store = S.createSelectionStore({
            resolveFilter: async () => [],
            writeLabel:    async () => null,
        });
        store.add([1, 2]);                    // something to isolate
        store.setIsolate(true);
        const litBefore = store.getState().isolate;
        store.addFilter({ kind: "by_element", value: "Xx" });
        await store.applyFilter();
        const st = store.getState();
        console.log(JSON.stringify({
            litBefore, isolateAfter: st.isolate, outcome: st.filterOutcome,
        }));
        """
    )
    assert out["litBefore"] is True
    assert out["isolateAfter"] is False, (
        "isolate must switch off when the selection empties")
    assert out["outcome"] == {"matched": 0, "isolateTurnedOff": True}, (
        f"the outcome must record that the switch moved: {out['outcome']}")
