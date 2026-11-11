"""Unit tests for the Molbuilder-tab page orchestrator.

Pins the public API of ``molbuilder/web/static/modify/structure/
page.js`` — the gate every Sources panel routes Load / Generate
calls through.  The orchestrator's load-bearing job is the load gate:

  * empty canvas → install directly
  * something open → the incoming structure is appended to it, no modal

These tests use a fake canvas-state + warning-modal API so the
state machine can be exercised without the real DOM / sessionStorage
machinery (which has its own dedicated test files).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/modify/structure/page.js"


# Fake canvas-state + warning-modal — minimum surface the page
# orchestrator pokes.  Kept inline in the bootstrap so each test
# starts from a fresh state.
_FAKES = r"""
// page.js binds against MolView's unified DATA model
// (window.molbuilder.molview.data), NOT the workspace.  The gate
// reads isEmpty/isDirty and routes through the ONE atomic load
// door: molview.data.load({text, filename, source, periodicity,
// annotations}).  Helper preserved as _mkFakeCanvas for
// test-name continuity.
/* A stand-in VIEWER, speaking the surface molview.md § 9.3 actually lists.
 *
 * It used to speak isEmpty / isDirty / getSource / getLastSavedTo / markDirty /
 * markSaved — six names MolView has never had, so this file tested page.js
 * against an invention while every one of those calls was `undefined` on the
 * real thing. What replaced them:
 *
 *   isEmpty()        -> getStructure() === null   (nothing loaded reads as nothing)
 *   isDirty()        -> the PAGE's note: the view against what was written (tabs.md § 6, 2.68.2)
 *   markDirty()      -> nothing; the edit raised it
 *   markSaved(path)  -> nothing; the PAGE keeps where it saved (§ 6.7)
 *   getSource()      -> nothing; the viewer tracks contents, not files
 */
function _mkFakeCanvas(initial) {
    initial = initial || {};
    let state = {
        structure:   initial.empty === false
            ? (initial.structure || { title: "seeded", elements: ["C"] })
            : (initial.structure || null),
        uncommitted: !!initial.dirty,
    };
    const calls = [];
    /* The viewer announces each structure change to its subscribers, with
     * nothing in hand (`model.js` `announceStructure`) -- an install, an
     * edit, an append alike.  Subscribing is listening, not touching, so it
     * is not a call. */
    const subs = [];
    const announce = () => subs.slice().forEach((f) => f());
    return {
        getStructure: () => state.structure,
        // What a save writes: the structure as the viewer exports it.
        exportFile: () => (state.structure
            ? { structure: JSON.parse(JSON.stringify(state.structure)) }
            : null),
        get uncommitted() { return state.uncommitted; },
        installMolecule: (arg) => {
            state.structure   = arg.structure;
            state.uncommitted = false;
            calls.push({fn: "installMolecule", arg: arg,
                        structure: arg.structure});
            announce();
            return Promise.resolve();
        },
        /* The edit door (§ 11.1).  The gate reaches it for `append`, which is
         * what a load into a NON-EMPTY canvas is since 2026-09-07.  The
         * stand-in answers with a structure, because `applyOp` answers with
         * the structure the server sent and the gate reads that to tell an
         * applied edit from a refused one. */
        applyOp: (name, params) => {
            calls.push({fn: "applyOp", op: name, arg: params,
                        addition: params && params.addition});
            const merged = { title: state.structure && state.structure.title,
                             elements: (state.structure
                                        ? state.structure.elements : []).concat(
                                 (params && params.addition
                                  ? params.addition.elements : []) || []) };
            state.structure = merged;
            announce();
            return Promise.resolve(merged);
        },
        subscribe: (cb) => {
            subs.push(cb);
            return () => {
                const at = subs.indexOf(cb);
                if (at >= 0) subs.splice(at, 1);
            };
        },
        /* An edit as the real viewer makes it: the change is announced, and
         * the edit lays down its own timeline point, which clears the
         * viewer's flag again (`history.js`) -- "✓ saved #1". */
        _edit:  () => {
            state.structure = Object.assign({}, state.structure || {}, {
                elements: ((state.structure && state.structure.elements) || [])
                    .concat(["H"]) });
            state.uncommitted = true; announce();
            state.uncommitted = false; },
        // A timeline save: the badge moves, the structure does not, and the
        // viewer announces all the same (`model.js` `onBadge`).
        _badge: () => { announce(); },
        // A Retract to a structure held before.
        _set:   (s) => { state.structure = s; announce(); },
        _state: () => ({ empty: state.structure === null,
                         dirty: state.uncommitted,
                         structure: state.structure }),
        _calls: () => calls.slice(),
    };
}
function _mkFakeModal(verdict) {
    // ``verdict`` is the boolean the modal will resolve to.
    const calls = [];
    return {
        confirmDiscardUnsaved: () => {
            calls.push("confirmDiscardUnsaved");
            return Promise.resolve(verdict);
        },
        _calls: () => calls.slice(),
    };
}
"""


def _run_node(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    module_path = MODULE.resolve()
    bootstrap = f"""
        {_FAKES}
        const page = require({json.dumps(str(module_path))});
        try {{
            (async () => {{
                {snippet}
            }})().catch(err => {{
                console.log(JSON.stringify({{
                    __test_unexpected_throw: true,
                    message: err && err.message ? err.message : String(err),
                    stack:   err && err.stack ? err.stack : null,
                }}));
            }});
        }} catch (err) {{
            console.log(JSON.stringify({{
                __test_unexpected_throw: true,
                message: err && err.message ? err.message : String(err),
                stack:   err && err.stack ? err.stack : null,
            }}));
        }}
    """
    proc = subprocess.run(
        [node, "--input-type=commonjs", "-e", bootstrap],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"node exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\n"
            f"stdout:\n{proc.stdout}"
        )
    last_line = proc.stdout.strip().splitlines()[-1]
    out = json.loads(last_line)
    if isinstance(out, dict):
        assert "__test_unexpected_throw" not in out, (
            "module threw: " + str(out)
        )
    return out


# ----- loadIntoCanvas gate --------------------------------------- #


class TestLoadGateEmptyCanvas:

    def test_empty_canvas_sets_directly_no_modal(self):
        out = _run_node('''
            const canvas = _mkFakeCanvas({ empty: true });
            const modal  = _mkFakeModal(false);  // would say Cancel
            page._bind(canvas, modal);
            const r = await page.loadIntoCanvas(
                { structure: { title: "eth", elements: ["C"],
                               positions: [[0, 0, 0]] } },
                { kind: "smiles", generator_input: { smiles: "CCO" } }
            );
            console.log(JSON.stringify({
                envelope:   r,
                canvasCalls: canvas._calls(),
                modalCalls:  modal._calls(),
            }));
        ''')
        assert out["envelope"] == {"ok": True}
        # Routed through the ONE atomic load door, carrying the envelope
        # itself.  The gate forwards what it was handed and adds nothing:
        # a document written in the browser is what web-api.md § 1 forbids.
        loads = [c for c in out["canvasCalls"] if c["fn"] == "installMolecule"]
        assert len(loads) == 1
        assert loads[0]["arg"]["structure"] == {
            "title": "eth", "elements": ["C"], "positions": [[0, 0, 0]]}
        assert "text" not in loads[0]["arg"]
        # Modal NOT consulted — the empty canvas gate didn't ask.
        assert out["modalCalls"] == []
        # AND NOT APPENDED INTO (2026-09-07).  A load into a canvas that holds
        # something adds to it; with NOTHING open there is nothing to add to,
        # and an install is also what anchors the timeline at point 0 -- which
        # an appended-into canvas would have no floor to retract to.
        assert not [c for c in out["canvasCalls"] if c["fn"] == "applyOp"], (
            "an empty canvas was appended into rather than installed")


class TestLoadGateNonEmptyCanvas:
    """SOMETHING OPEN -> the incoming structure is ADDED to it.

    *"the load from project or other generators should by default ADD their
    results into the molview structure instead of clear the existing one"*
    (user, 2026-09-07).

    Nothing is discarded, so there is nothing to confirm and no cancel to
    make -- the modal is not consulted in either case below.  Replacing rather than adding is a separate
    gesture: Clear structure, then load.
    """

    def test_a_loaded_canvas_is_appended_to(self):
        out = _run_node('''
            const canvas = _mkFakeCanvas({
                empty: false, dirty: false,
                structure: { title: "old", elements: ["C"] },
            });
            const modal = _mkFakeModal(false);
            page._bind(canvas, modal);
            const r = await page.loadIntoCanvas(
                { structure: { title: "new", elements: ["N", "N"] } },
                { kind: "smiles" }
            );
            console.log(JSON.stringify({
                envelope:   r,
                modalCalls: modal._calls(),
                ops:        canvas._calls().map(c => c.fn),
                append:     canvas._calls().filter(
                                c => c.fn === "applyOp").pop(),
                elements:   canvas.getStructure().elements,
            }));
        ''')
        assert out["envelope"] == {"ok": True}
        assert out["modalCalls"] == [], (
            "the discard confirm was asked for a load that discards nothing")
        assert "installMolecule" not in out["ops"], (
            f"a loaded canvas was replaced instead of added to: {out['ops']}")
        assert out["append"]["op"] == "append"
        assert out["append"]["addition"]["title"] == "new"
        assert out["elements"] == ["C", "N", "N"], (
            f"the two structures did not both survive: {out['elements']}")

    def test_a_dirty_canvas_is_appended_to_without_asking(self):
        """Unsaved work is not a reason to ask: an append does not
        overwrite it.  It also cannot be lost -- every edit records a timeline
        point (§ 11.2, 2026-09-07), so there is nothing sitting off the
        sequence for a dialog to protect.
        """
        out = _run_node('''
            const canvas = _mkFakeCanvas({
                empty: false, dirty: true,
                structure: { title: "edited", elements: ["C"] },
            });
            const modal = _mkFakeModal(false);   // would say Cancel
            page._bind(canvas, modal);
            const r = await page.loadIntoCanvas(
                { structure: { title: "new", elements: ["N"] } },
                { kind: "file", file: "/p/b.xyz" }
            );
            console.log(JSON.stringify({
                envelope:   r,
                modalCalls: modal._calls(),
                ops:        canvas._calls().map(c => c.fn),
                elements:   canvas.getStructure().elements,
            }));
        ''')
        assert out["envelope"] == {"ok": True}
        assert out["modalCalls"] == [], (
            "a dirty canvas still triggered the discard confirm, which would "
            "let a Cancel refuse a load that discards nothing")
        assert out["elements"] == ["C", "N"]


# ----- modifier helpers ------------------------------------------ #


class TestModifierHelpers:

    def test_an_edit_marks_the_pages_own_note_and_touches_nothing(self):
        """Edited since written is the PAGE's note (tabs.md § 6, 2.68.2): its
        own listener marks it at a structure change; the page writes nothing
        to the viewer, and `markDirtyAfterModification` adds nothing.
        """
        out = _run_node('''
            const canvas = _mkFakeCanvas({ empty: false, dirty: false });
            page._bind(canvas, _mkFakeModal(false));
            (async () => {
                await page.restorePanelNote();    // the note is read back first
                canvas._edit();                   // the viewer announces it
                page.markDirtyAfterModification();
                console.log(JSON.stringify({
                    isDirty: page.getCanvasSnapshot().isDirty,
                    touched: canvas._calls().map(c => c.fn),
                }));
            })();
        ''')
        assert out["isDirty"] is True
        assert out["touched"] == [], (
            f"the page wrote to the viewer to say something the viewer already "
            f"knew: {out['touched']}"
        )

    def test_where_it_was_saved_is_the_pages_own_note(self):
        """The page chose the destination, so the page remembers it.

        The viewer tracks contents, not files (molview.md § 6.7): it was never
        told where anything was written.
        """
        out = _run_node('''
            const canvas = _mkFakeCanvas({ empty: false, dirty: true });
            page._bind(canvas, _mkFakeModal(false));
            page.markSavedTo("/p/saved.xyz");
            console.log(JSON.stringify({
                lastSavedTo: page.getCanvasSnapshot().lastSaveTo,
                touched:     canvas._calls().map(c => c.fn),
            }));
        ''')
        assert out["lastSavedTo"] == "/p/saved.xyz"
        assert out["touched"] == [], (
            "recording where the page saved must not write to the viewer"
        )


class TestALoadRetiresTheSaveTarget:
    """`lastSaveTo` means "where the thing on the canvas was written".

    After a load, the thing on the canvas is something else -- so the note
    is about a structure that is no longer there.  It was left standing, and
    the Save readout went on naming the PREVIOUS structure's file: building
    ethanol over a restored BDT-Au junction left the panel saying
    "Target: BDT-Au-junction.xyz" (browser walk, 2026-08-24).

    `_saveDataset` does confirm an overwrite, so this misled rather than
    silently destroyed -- but a confirmation you answer while reading the
    wrong filename is not much of a guard.
    """

    def test_loading_a_file_clears_where_the_last_one_was_saved(self):
        out = _run_node('''
            const canvas = _mkFakeCanvas({ empty: false, dirty: false });
            page._bind(canvas, _mkFakeModal(false));
            page.markSavedTo("/p/first.xyz");
            const before = page.getCanvasSnapshot().lastSaveTo;
            page.markLoadedFrom("/p/second.xyz");
            console.log(JSON.stringify({
                before: before,
                after:  page.getCanvasSnapshot().lastSaveTo,
                loaded: page.getCanvasSnapshot().loadedFrom,
            }));
        ''')
        assert out["before"] == "/p/first.xyz"
        assert out["after"] is None, (
            "a load must retire the previous structure's save target")
        assert out["loaded"] == "/p/second.xyz"

    def test_a_generator_build_clears_it_too(self):
        """A SMILES / DNA / peptide build passes no filename, so the page
        records `null` -- and must equally stop claiming the old target."""
        out = _run_node('''
            const canvas = _mkFakeCanvas({ empty: false, dirty: false });
            page._bind(canvas, _mkFakeModal(false));
            page.markSavedTo("/p/first.xyz");
            page.markLoadedFrom(undefined);
            console.log(JSON.stringify({
                after:  page.getCanvasSnapshot().lastSaveTo,
                loaded: page.getCanvasSnapshot().loadedFrom,
            }));
        ''')
        assert out["after"] is None
        assert out["loaded"] is None


# ----- snapshot -------------------------------------------------- #


class TestSnapshot:

    def test_getCanvasSnapshot_returns_full_state(self):
        out = _run_node('''
            const canvas = _mkFakeCanvas({
                empty: false, dirty: true,
                structure: { title: "lig", elements: ["C", "O"] },
                source: { kind: "smiles",
                          generator_input: { smiles: "CCO" } },
                lastSaveTo: null,
            });
            page._bind(canvas, _mkFakeModal(false));
            console.log(JSON.stringify(page.getCanvasSnapshot()));
        ''')
        assert out["isEmpty"] is False
        # The viewer's timeline flag is not the page's note: nothing was
        # edited since the page started following (tabs.md § 6).
        assert out["isDirty"] is False
        assert out["structure"] == {
            "title": "lig", "elements": ["C", "O"]}
        # No `source`: the viewer tracks contents, not where they came from
        # (molview.md § 6.7), so the snapshot cannot carry one.
        assert "source" not in out
        assert out["lastSaveTo"] is None



#: EDITED SINCE WRITTEN, the page's own note (tabs.md § 6, 2.68.2), through
#: the page's API with a viewer that announces its changes -- one row per
#: road, each step a call or the viewer's announcement, then what the
#: snapshot says.  The viewer's timeline flag clears at every edit (each edit
#: lays down its own point), so it never answered "Unsaved -- Target".
_NOTE_ROWS = [
    ("an edit after a save reads unsaved",
     "restore; save /p/x.xyz; edit", True),
    ("...and the next save clears it",
     "restore; save /p/x.xyz; edit; save /p/x.xyz", False),
    ("the viewer's own restore, before the note is read back, is no edit",
     "edit; restore", False),
    ("a file loaded over the view is what the file holds",
     "restore; edit; load a.xyz", False),
    ("a generator's build is held by no file",
     "restore; load -", True),
    ("a timeline save changes nothing and is no edit",
     "restore; save /p/x.xyz; badge", False),
    ("a Retract back to the written structure reads saved",
     "restore; save /p/x.xyz; edit; back", False),
    ("an edit landing while the save's dialog is open stays unsaved",
     "restore; edit; save-before /p/x.xyz", True),
]


@pytest.mark.parametrize("road, steps, edited", _NOTE_ROWS,
                         ids=[r[0] for r in _NOTE_ROWS])
def test_the_page_knows_whether_the_view_was_edited_since_written(
        road, steps, edited):
    """MUTATION THIS MUST FAIL AGAINST: the snapshot reading the viewer's
    timeline flag again (each edit clears it); the note not marked by the
    page's own listener; a save or a load not clearing it."""
    out = _run_node('''
        const canvas = _mkFakeCanvas({ empty: false, dirty: false });
        page._bind(canvas, _mkFakeModal(false));
        const steps = STEPS.split("; ");
        let written = null;
        const before = canvas.exportFile().structure;
        (async () => {
            for (const s of steps) {
                if (s === "restore") await page.restorePanelNote();
                else if (s === "edit") canvas._edit();
                else if (s === "badge") canvas._badge();
                else if (s === "back") canvas._set(written);
                else if (s.startsWith("save-before ")) {
                    // the payload captured BEFORE the last edit landed
                    page.markSavedTo(s.slice(12), before);
                }
                else if (s.startsWith("save ")) {
                    written = canvas.exportFile().structure;
                    page.markSavedTo(s.slice(5), written);
                }
                else if (s.startsWith("load ")) {
                    const file = s.slice(5) === "-" ? null : s.slice(5);
                    await page.loadIntoCanvas(
                        { structure: { title: "t", elements: ["C"] } },
                        file ? { file: file } : {}, { replace: true });
                }
            }
            console.log(JSON.stringify({
                isDirty: page.getCanvasSnapshot().isDirty }));
        })();
    '''.replace("STEPS", json.dumps(steps)))
    assert out["isDirty"] is edited, (road, out)


def test_a_note_read_back_after_a_reload_reaches_its_readers():
    """After a reload the page reads its note back (where the structure was
    saved, whether it was edited since), and the readout painted before that
    must hear it -- else it says "Not saved yet" over a structure saved before
    the reload, as it did through 2.68.1 (tabs.md § 6).

    MUTATION THIS MUST FAIL AGAINST: `restorePanelNote` setting the note
    without telling the page's own channel."""
    out = _run_node('''
        const canvas = _mkFakeCanvas({ empty: false, dirty: false });
        page._bind(canvas, _mkFakeModal(false));
        const root = (typeof window !== "undefined") ? window : globalThis;
        root.molbuilder = root.molbuilder || {};
        root.molbuilder.workspace = {
            persistDraft: () => true,
            readDraft: async () => ({ v: 1, loadedFrom: null,
                                      lastSavedTo: "/p/x.xyz", edited: true }),
        };
        const heard = [];
        page.onPanelChange((note) => heard.push(note));
        await page.restorePanelNote();
        console.log(JSON.stringify({ heard: heard,
            snap: page.getCanvasSnapshot().lastSaveTo }));
    ''')
    assert out["snap"] == "/p/x.xyz", out
    assert out["heard"] == [{"loadedFrom": None, "lastSavedTo": "/p/x.xyz",
                             "edited": True}], out



def test_the_note_survives_a_reload_through_what_was_stored():
    """What the page wrote is what it reads back: a save, an edit, then a
    fresh page over the same workspace says Unsaved with its target -- the
    round trip through a store that returns what was stored (tabs.md § 6).

    MUTATION THIS MUST FAIL AGAINST: the note persisted without its
    edited fact."""
    out = _run_node('''
        const root = (typeof window !== "undefined") ? window : globalThis;
        const store = new Map();
        root.molbuilder = root.molbuilder || {};
        root.molbuilder.workspace = {
            persistDraft: (tag, bytes) => {
                store.set(tag, JSON.stringify(bytes));
                return true;
            },
            readDraft: async (tag) => {
                const b = store.get(tag);
                return b ? JSON.parse(b) : null;
            },
        };
        const canvas = _mkFakeCanvas({ empty: false, dirty: false });
        page._bind(canvas, _mkFakeModal(false));
        await page.restorePanelNote();
        page.markSavedTo("/p/x.xyz", canvas.exportFile().structure);
        canvas._edit();
        // a fresh page over the same store, the view restored as it was
        const MOD = require.resolve(MODULE_PATH_FOR_RELOAD);
        delete require.cache[MOD];
        const again = require(MOD);
        again._bind(canvas, _mkFakeModal(false));
        await again.restorePanelNote();
        const s = again.getCanvasSnapshot();
        console.log(JSON.stringify({ isDirty: s.isDirty,
                                     lastSaveTo: s.lastSaveTo }));
    '''.replace("MODULE_PATH_FOR_RELOAD", json.dumps(str(MODULE.resolve()))))
    assert out == {"isDirty": True, "lastSaveTo": "/p/x.xyz"}, out


def test_the_viewers_own_export_to_the_project_is_a_save():
    """The door the page hands its viewer records a structure written to the
    project, as Save to project does; an image, or a download, records
    nothing (tabs.md § 6, one door out).

    MUTATION THIS MUST FAIL AGAINST: the viewer handed the bare door."""
    out = _run_node('''
        const canvas = _mkFakeCanvas({ empty: false, dirty: false });
        page._bind(canvas, _mkFakeModal(false));
        await page.restorePanelNote();
        canvas._edit();
        const door = page.savingThrough({
            save: async (dest, stem, payload) => ({ ok: true,
                                                    path: "/p/" + stem }),
        });
        await door.save("project", "img.png", { image: "..." });
        const afterImage = page.getCanvasSnapshot();
        await door.save("download", "x.xyz", canvas.exportFile());
        const afterDownload = page.getCanvasSnapshot();
        await door.save("project", "x.xyz", canvas.exportFile());
        const afterSave = page.getCanvasSnapshot();
        console.log(JSON.stringify({
            image: [afterImage.isDirty, afterImage.lastSaveTo],
            download: [afterDownload.isDirty, afterDownload.lastSaveTo],
            save: [afterSave.isDirty, afterSave.lastSaveTo] }));
    ''')
    assert out == {"image": [True, None], "download": [True, None],
                   "save": [False, "/p/x.xyz"]}, out
