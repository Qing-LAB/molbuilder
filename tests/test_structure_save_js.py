"""The Save panel — `modify/structure/save.js`.

**It is a panel, not a save.**  `tabs.md` § 6: the writing is
`projects.molviewFiles.save("project", stem, viewer.exportFile())` — the one
door, which asks *where*, posts `/api/structure/save`, runs the overwrite
confirmation and refreshes the sidebar.  What this file owns is the readout,
the button's enabled state, and recording where the bytes went.

**What was retired here on 2026-09-02, and why.**  This file was 496 lines
against a 456-line module, and its harness faked `projects.parser.saveMolecule`
so it could exercise a save PIPELINE this panel no longer has: name
normalisation, the overwrite retry, the sidebar refresh, fetch-level failures.
All of that moved to `molviewFiles`, which had been mounted into this very tab's
viewer since 2026-08 — two roads to one route, and this file tested the older
one.

Its coverage was not lost with it.  The door's own behaviour is
`test_molview_files_door_js.py`; the server's 409 overwrite contract is
`test_web.py`; the menu-to-door handoff is `test_molview_mount.py`.  Those are
where the rules live, so a copy here would have been a second statement of them
free to drift — which is exactly what it had become.

What remains is what nothing else can say: that the PANEL delegates, and that it
tells the page where the bytes went.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/modify/structure/save.js"


def _run_node(snippet: str) -> object:
    """Load save.js as a classic script and run `snippet` against its API.

    It is an IIFE, not a module — the tab's scripts predate ESM and are
    handed their collaborators rather than importing them — so it is loaded
    through `module.exports`, which is the seam it already provides.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    bootstrap = f"""
        globalThis.window = globalThis;
        globalThis.molbuilder = {{}};

        /** A stand-in viewer DATA model: the two reads the panel makes, plus
         *  the export it hands to the door. */
        function fakeModel(opts) {{
            opts = opts || {{}};
            return {{
                getStructure: () => (opts.empty ? null : {{ elements: ["C"] }}),
                get uncommitted() {{ return !!opts.dirty; }},
                exportFile: () => (opts.exportNull ? null : {{
                    name: "x.xyz",
                    structure: {{ elements: ["C"], positions: [[0, 0, 0]] }},
                }}),
                subscribe: (fn) => {{ fn(); return () => {{}}; }},
            }};
        }}

        /** The door, recording what it was asked for and answering a queue. */
        function fakeDoor(replies) {{
            const calls = [];
            return {{
                calls,
                save: (destination, stem, payload) => {{
                    calls.push({{ destination, stem,
                                  hasStructure: !!(payload && payload.structure) }});
                    const q = replies || [];
                    return Promise.resolve(
                        q.length ? q.shift() : {{ ok: true, path: "/p/x.xyz" }});
                }},
            }};
        }}

        function fakePage(lastSaveTo, loadedFrom) {{
            let saved = lastSaveTo || null;
            const marks = [];
            return {{
                marks,
                markSavedTo: (p) => {{ saved = p; marks.push(p); }},
                getCanvasSnapshot: () => ({{ lastSaveTo: saved }}),
                getLoadedFrom: () => (loadedFrom || null),
            }};
        }}

        const MODULE_PATH = {json.dumps(str(MODULE.resolve()))};
        const mod = require(MODULE_PATH);
    """
    proc = subprocess.run(
        ["node", "-e", bootstrap + "\n" + snippet],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n"
                    f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------- #
#  The surface                                                          #
# --------------------------------------------------------------------- #

def test_the_panel_publishes_the_calls_the_page_wires():
    """`useViewer` is how the page hands over the viewer it mounted — this
    module cannot `import` it, which is the whole reason the bind door
    exists (`selection-bootstrap.js`)."""
    out = _run_node("""
        console.log(JSON.stringify(
            ["configure", "useViewer", "save", "targetPath", "wirePanel"]
                .map((k) => typeof mod[k])));
    """)
    assert out == ["function"] * 5


# --------------------------------------------------------------------- #
#  Where it last saved — the page's note, never the viewer's            #
# --------------------------------------------------------------------- #

def test_the_target_is_the_pages_note_and_is_null_before_a_save():
    """The viewer tracks contents, not files (`molview.md` § 6.7), so where
    a structure went is `structurePage`'s to remember — and `targetPath` is
    what the READOUT shows, not a destination.  The destination is the
    question the door's dialog asks."""
    out = _run_node("""
        mod.configure({ structurePage: fakePage(null) });
        const before = mod.targetPath();
        mod.configure({ structurePage: fakePage("/p/done.xyz") });
        console.log(JSON.stringify({ before, after: mod.targetPath() }));
    """)
    assert out["before"] is None
    assert out["after"] == "/p/done.xyz"


# --------------------------------------------------------------------- #
#  Saving = delegating                                                  #
# --------------------------------------------------------------------- #

def test_save_hands_the_viewers_export_to_the_project_door():
    """One door out (`tabs.md` § 6).  The panel supplies WHAT and a suggested
    stem; the door owns WHERE, the overwrite flow and the refresh."""
    out = _run_node("""
        const door = fakeDoor();
        molbuilder.projects = { molviewFiles: door };
        mod.configure({ projects: molbuilder.projects,
                        structurePage: fakePage(null, "/p/bdt.xyz"),
                        workspace: fakeModel() });
        mod.save().then((r) => {
            console.log(JSON.stringify({ r, calls: door.calls }));
        });
    """)
    assert out["r"]["ok"] is True
    assert len(out["calls"]) == 1
    call = out["calls"][0]
    assert call["destination"] == "project", call
    assert call["hasStructure"] is True, (
        "the door was handed no structure -- the model serialises itself and "
        "the panel passes that through")
    assert call["stem"] == "bdt", (
        f"the loaded file's stem should be SUGGESTED as the name: {call}")


def test_the_page_records_the_path_the_SERVER_answered():
    """Not the one we asked for: `auto_rename` may have written
    `<stem>-2.xyz`, and confirming a name we merely proposed would name a
    file that does not exist."""
    out = _run_node("""
        molbuilder.projects = { molviewFiles:
            fakeDoor([{ ok: true, path: "/p/bdt-2.xyz" }]) };
        const page = fakePage(null, "/p/bdt.xyz");
        mod.configure({ projects: molbuilder.projects, structurePage: page,
                        workspace: fakeModel() });
        mod.save().then(() => {
            console.log(JSON.stringify({ marks: page.marks,
                                         target: mod.targetPath() }));
        });
    """)
    assert out["marks"] == ["/p/bdt-2.xyz"]
    assert out["target"] == "/p/bdt-2.xyz"


def test_a_cancelled_save_records_nothing():
    """Cancelling either half of the door's dialog writes no file, so the
    page must not claim a target it does not have."""
    out = _run_node("""
        molbuilder.projects = { molviewFiles:
            fakeDoor([{ ok: false, cancelled: true }]) };
        const page = fakePage(null, "/p/bdt.xyz");
        mod.configure({ projects: molbuilder.projects, structurePage: page,
                        workspace: fakeModel() });
        mod.save().then((r) => {
            console.log(JSON.stringify({ cancelled: !!r.cancelled,
                                         marks: page.marks }));
        });
    """)
    assert out["cancelled"] is True
    assert out["marks"] == []


# --------------------------------------------------------------------- #
#  Refusals — said, never thrown                                        #
# --------------------------------------------------------------------- #

def test_nothing_loaded_is_refused_by_name():
    """Nothing loaded reads as nothing (`molview.md` § 9.3)."""
    out = _run_node("""
        molbuilder.projects = { molviewFiles: fakeDoor() };
        mod.configure({ projects: molbuilder.projects,
                        structurePage: fakePage(null),
                        workspace: fakeModel({ empty: true }) });
        mod.save().then((r) => console.log(JSON.stringify(r)));
    """)
    assert out["ok"] is False
    assert "No structure to save" in out["error"]


def test_a_missing_collaborator_is_an_ENVELOPE_not_a_throw():
    """The click handler shows `error` beside the button; a rejection would
    leave a hung "Saving…" and tell the user nothing.  Both halves the panel
    depends on are checked: the viewer the page hands over, and the door."""
    out = _run_node("""
        // no viewer
        mod.configure({ projects: { molviewFiles: fakeDoor() },
                        structurePage: fakePage(null) });
        mod.save().then((noViewer) => {
            // a viewer, but no door on this page
            molbuilder.projects = {};
            mod.configure({ projects: {}, structurePage: fakePage(null),
                            workspace: fakeModel() });
            return mod.save().then((noDoor) => {
                console.log(JSON.stringify({ noViewer, noDoor }));
            });
        });
    """)
    assert out["noViewer"]["ok"] is False and "viewer" in out["noViewer"]["error"]
    assert out["noDoor"]["ok"] is False and "door" in out["noDoor"]["error"]


def test_an_export_that_cannot_be_written_out_is_refused_not_sent():
    """`exportFile()` answers null when the geometry and the per-atom facts
    disagree (`molview.md` § 9.3).  Sending that would ask the server to
    write a structure the viewer itself could not state."""
    out = _run_node("""
        const door = fakeDoor();
        mod.configure({ projects: { molviewFiles: door },
                        structurePage: fakePage(null),
                        workspace: fakeModel({ exportNull: true }) });
        mod.save().then((r) => console.log(JSON.stringify(
            { ok: r.ok, error: r.error, calls: door.calls.length })));
    """)
    assert out["ok"] is False
    assert out["calls"] == 0, "a refused export still reached the door"


def test_the_readout_follows_the_model_whichever_arrives_first():
    """**A race the page does not control, and neither order may lose.**

    `wirePanel` runs from `DOMContentLoaded`; the viewer arrives later, after
    an `await MV.mount(...)` in `selection-bootstrap.js`.  Either can be
    second.  When the panel wired first there was no model to subscribe to,
    the `_wired` guard stopped a second attempt, and **the readout never moved
    again** — you could edit the structure and it went on saying
    "Target: x.xyz" with no "Unsaved", for the rest of the session.

    Nothing about that is visible in a reading of either file: each half is
    correct on its own and the defect is in their order.  Pinned in both
    directions because fixing one order and not the other is the shape the
    bug already had.
    """
    out = _run_node("""
        function panelDoc() {
            const el = () => ({ textContent: "", disabled: false,
                                addEventListener() {} });
            const nodes = { "save-to-source-btn": el(),
                            "save-readout": el(), "save-status": el() };
            return { getElementById: (id) => nodes[id] || null, _nodes: nodes };
        }
        molbuilder.status = { set() {} };

        /** A model that can be edited, and tells its subscribers. */
        function liveModel() {
            let dirty = false; const subs = [];
            return {
                getStructure: () => ({ elements: ["C"] }),
                get uncommitted() { return dirty; },
                exportFile: () => ({ structure: {} }),
                subscribe: (fn) => { subs.push(fn); fn(); return () => {}; },
                edit() { dirty = true; subs.forEach((f) => f()); },
            };
        }

        function readoutAfterEdit(viewerFirst) {
            // a fresh module instance per order -- `_wired` is module state
            delete require.cache[require.resolve(MODULE_PATH)];
            const m = require(MODULE_PATH);
            const model = liveModel();
            const doc = panelDoc();
            m.configure({ structurePage: fakePage("/p/x.xyz") });
            if (viewerFirst) {
                m.useViewer({ ok: true, data: model });
                m.wirePanel({ doc });
            } else {
                m.wirePanel({ doc });
                m.useViewer({ ok: true, data: model });
            }
            model.edit();
            return doc._nodes["save-readout"].textContent;
        }

        console.log(JSON.stringify({
            viewerFirst: readoutAfterEdit(true),
            panelFirst:  readoutAfterEdit(false),
        }));
    """)
    for order, said in out.items():
        assert "Unsaved" in said, (
            f"with the {order} order the readout did not follow the model "
            f"after an edit: {said!r} -- the panel is wired to nothing")
