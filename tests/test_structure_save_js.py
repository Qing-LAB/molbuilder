"""The Save panel — `modify/structure/save.js`.

**It is a panel, not a save.**  `tabs.md` § 6: the writing is
`projects.molviewFiles.save("project", stem, viewer.exportFile())` — the one
door, which asks *where*, posts `/api/structure/save`, runs the overwrite
confirmation and refreshes the sidebar.  What this file owns is the readout,
the button's enabled state, and recording where the bytes went.

The door's own behaviour is `test_molview_files_door_js.py`; the server's 409
overwrite contract is `test_structure_save_endpoint.py`; the menu-to-door handoff is
`test_molview_mount.py`.  Those are where the rules live, so a copy here would
be a second statement of them free to drift.

What this file pins is what nothing else can say: that the PANEL delegates, and
that it tells the page where the bytes went.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/modify/structure/save.js"
#: What the page header loads before any click: the one path door.
PATH_DOOR = ROOT / "molbuilder/web/static/lib/path-utils.js"


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
        // WHAT THE PAGE HEADER LOADS before any click: the one path door
        // (`lib/path-utils.js`), which save.js reads when called.
        require({json.dumps(str(PATH_DOOR.resolve()))});

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
            let edited = false;
            const marks = [];
            const noteReaders = [];
            const said = () => noteReaders.slice().forEach((f) => f());
            return {{
                marks,
                markSavedTo: (p) => {{ saved = p; edited = false; marks.push(p);
                                      said(); }},
                getCanvasSnapshot: () => ({{ lastSaveTo: saved,
                                            isDirty: edited }}),
                getLoadedFrom: () => (loadedFrom || null),
                // THE PAGE'S NOTE and its own channel (`page.js`,
                // tabs.md § 6): edited since written.
                onPanelChange: (fn) => {{ noteReaders.push(fn); return () => {{}}; }},
                _edit: () => {{ edited = true; said(); }},
                // marked without a word on its own channel: only the
                // model's announcement repaints the readout then
                _mark: () => {{ edited = true; }},
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
#  Saving = delegating                                                  #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("loaded_from", ["/p/bdt.xyz", "/p/bdt.pdb"],
                         ids=["an xyz", "a pdb"])
def test_save_hands_the_viewers_export_to_the_project_door(loaded_from):
    """One door out (`tabs.md` § 6).  The panel supplies WHAT and a suggested
    stem -- the loaded file's name without its extension, for a save writes
    `<name>.xyz` with its sidecar whatever was loaded (a `.pdb` is read,
    never written: `model/structure.md` § 2.4); the door owns WHERE, the
    overwrite flow and the refresh.  THE SILENT FAILURE (to 2.69.16): a
    `.pdb` loaded was offered as `bdt.pdb`, and saved as `bdt.pdb.xyz`."""
    out = _run_node("""
        const door = fakeDoor();
        molbuilder.projects = { molviewFiles: door };
        mod.configure({ projects: molbuilder.projects,
                        structurePage: fakePage(null, %s),
                        workspace: fakeModel() });
        mod.save().then((r) => {
            console.log(JSON.stringify({ r, calls: door.calls }));
        });
    """ % json.dumps(loaded_from))
    assert out["r"]["ok"] is True
    assert len(out["calls"]) == 1
    call = out["calls"][0]
    assert call["destination"] == "project", call
    assert call["hasStructure"] is True, (
        "the door was handed no structure -- the model serialises itself and "
        "the panel passes that through")
    assert call["stem"] == "bdt", (
        f"the loaded file's stem should be SUGGESTED as the name: {call}")


def test_the_page_records_the_path_the_door_answered():
    """The page records where the door says the bytes went, never the name
    the panel suggested.  The door answers the path the person chose in its
    save-where dialog (`molview-doors.js`, `projects.md` § 3), which may be
    any name, not the suggested stem: recording the suggestion would name a
    file nobody saved.

    The door here is a stand-in answering with a path other than the
    suggestion -- the person typed another name.  The panel is handed its
    collaborators (`configure`), and the real door's dialog is a person's
    answer, run in the browser by `test_results_export_e2e.py`; what this
    pins is only that the panel takes the door's answer."""
    out = _run_node("""
        molbuilder.projects = { molviewFiles:
            fakeDoor([{ ok: true, path: "/p/bdt-relaxed.xyz" }]) };
        const page = fakePage(null, "/p/bdt.xyz");
        mod.configure({ projects: molbuilder.projects, structurePage: page,
                        workspace: fakeModel() });
        mod.save().then(() => {
            console.log(JSON.stringify({ marks: page.marks,
                                         target: mod.targetPath() }));
        });
    """)
    assert out["marks"] == ["/p/bdt-relaxed.xyz"]
    assert out["target"] == "/p/bdt-relaxed.xyz"


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

        /** A model that holds a structure and tells its subscribers of each
         *  change.  Whether it was edited since it was written is the PAGE's
         *  note (tabs.md § 6), not the model's. */
        function liveModel() {
            const subs = [];
            return {
                getStructure: () => ({ elements: ["C"] }),
                exportFile: () => ({ structure: {} }),
                subscribe: (fn) => { subs.push(fn); return () => {}; },
                announce() { subs.forEach((f) => f()); },
            };
        }

        function readoutAfterEdit(viewerFirst, byModel) {
            // a fresh module instance per order -- `_wired` is module state
            delete require.cache[require.resolve(MODULE_PATH)];
            const m = require(MODULE_PATH);
            const model = liveModel();
            const doc = panelDoc();
            const pg = fakePage("/p/x.xyz");
            m.configure({ structurePage: pg });
            if (viewerFirst) {
                m.useViewer({ ok: true, data: model });
                m.wirePanel({ doc });
            } else {
                m.wirePanel({ doc });
                m.useViewer({ ok: true, data: model });
            }
            if (byModel) { pg._mark(); model.announce(); }
            else pg._edit();             // the page tells its own channel
            return doc._nodes["save-readout"].textContent;
        }

        console.log(JSON.stringify({
            viewerFirst: readoutAfterEdit(true, true),
            panelFirst:  readoutAfterEdit(false, true),
            byTheNote:   readoutAfterEdit(false, false),
        }));
    """)
    for order, said in out.items():
        assert "Unsaved" in said, (
            f"with the {order} order the readout did not follow the model "
            f"after an edit: {said!r} -- the panel is wired to nothing")


def test_a_save_says_what_the_server_said_of_it():
    """`tabs.md` § 6 step 4: on success the periodicity gate's `notices` come
    back verbatim -- and the panel's line says them after the outcome, in the
    tone that says *read this*.  The notices are the gate's own, for a water
    molecule with no box (`periodicity_gate.validate_periodicity`, which the
    save route asks).

    THE SILENT FAILURE: through 2.69.8 the line said only *Saved water.xyz.*
    and dropped what the server said of the box -- the viewer's own Export
    said it, the tab's button did not."""
    import numpy as np
    from molbuilder.periodicity_gate import validate_periodicity
    from molbuilder.structure import Structure
    _s, notices = validate_periodicity(Structure(
        elements=["O", "H", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0],
                            [-0.757, 0.586, 0.0]])))
    assert notices, "this row must reach the branch where the gate speaks"
    out = _run_node(f"""
        const said = [];
        molbuilder.status = {{ set: (el, msg, kind) => said.push({{ msg, kind }}) }};
        let click = null;
        const el = () => ({{ textContent: "", disabled: false,
            addEventListener: (t, f) => {{ if (t === "click") click = f; }} }});
        const nodes = {{ "save-to-source-btn": el(), "save-readout": el(),
                         "save-status": el() }};
        mod.configure({{ projects: {{ molviewFiles: fakeDoor([
                             {{ ok: true, path: "/p/water.xyz",
                                notices: {json.dumps(notices)} }}]) }},
                         structurePage: fakePage() }});
        mod.useViewer({{ ok: true, data: fakeModel() }});
        mod.wirePanel({{ doc: {{ getElementById: (id) => nodes[id] || null }} }});
        click();
        setTimeout(() => console.log(JSON.stringify(said[said.length - 1])), 20);
    """)
    assert out["msg"].startswith("Saved water.xyz."), out
    assert all(n["message"] in out["msg"] for n in notices), out
    assert out["kind"] == "warn", out
