"""molview.mount -- the owner-facing handle (molview-module.md §18 / §D).

Node unit test pinning the POST-CARVE contract:

  * §18/§C -- mount reads DATA from ``molbuilder.molview.data`` and uses the ``workspace``
    for PERSISTENCE only.  With a real ``molview.data`` present the handle routes every data
    op through it and NEVER touches the persistence workspace's data methods.  (A workspace
    fallback exists for the transitional case where ``molview.data`` is absent; it is tested
    HERE explicitly and labelled as a fallback -- it is not the contract path.)
  * §18.1 / §D -- the handle exposes exactly the owner-facing surface + the frame axis, and
    NO internals (no store / viewer / DOM).
  * §19.3 -- the model install primitive is ``data.installMolecule({text, filename[, ...]})``
    (text install).  Project-file PATH loads/saves go through the projects package
    (``projects.parser.openMolecule`` / ``saveMolecule``), NOT the mounted handle.  The
    obsolete doors (loadFromFile / loadFromText / installStructure / getScratchBlob /
    applyPayload) are GONE and no handle method reaches one.
  * §19.4 -- ``handle.exportFile()`` returns the serialized bytes via ``data.exportFile``
    (installMolecule's inverse).

The handle is exercised through a STUBBED molview.data + panel; a persistence-only workspace
is passed alongside with its own data methods wired as TRAPS.
"""
from pathlib import Path

from _node_esm import run_node

ROOT = Path(__file__).resolve().parents[1]
# mount.js is now a native ES module (the MolView ESM migration): the ``export { mount }`` it
# carries is a SyntaxError under a sloppy CommonJS text-concat run, so it loads through the
# shared ES-module harness (tests/_node_esm.run_node), which dynamic-``import()``s it -- the
# transitional ``window.molbuilder.molview.mount`` shim publish still fires the same global.
MODULES = [
    ROOT / "molbuilder/web/static/lib/molview/mount.js",
]

# The complete §18.1 handle key set (sorted): the seven core owner-API calls (§D) + the
# frame axis (§14.5), and NOTHING else -- no store, no viewer, no DOM refs.
HANDLE_KEYS = ["currentFrame", "dispose", "exportFile", "frameCount", "getFrame",
               "getSelection", "getStructure", "installMolecule", "isPlaying",
               "ok",   # uniform mount contract: success handle -> ok:true (mount.js _failMount)
               "onChange", "pause", "play", "setFrame", "undo"]


def _run_node(snippet: str) -> object:
    return run_node(MODULES, snippet)


# ---- The CONTRACT harness: a real molview.data is present (§18 / §C) ------------------- #
#
# ``molbuilder.molview.data`` is THE data layer.  It records every op it receives; the
# obsolete doors it also carries are TRAPS -- if the handle ever reaches one, the contract
# is broken.  The ``workspace`` is persistence-only; its data methods are traps too.
_HARNESS = """
    const dataCalls = [];       // ops the DATA layer (molview.data) receives
    const obsoleteHit = [];     // an obsolete door reached == a contract breach
    const wsTrap = [];          // the persistence workspace must NEVER serve data
    const subs = [];            // onChange subscribers, via data.subscribe
    const structure = { text: 'XYZ', atoms: [{ index: 0, x: 1 }] };
    const store = { getState: () => ({ indices: [1, 2] }),
                    subscribe: (fn) => { subs.push(fn);
                        return () => { const i = subs.indexOf(fn); if (i>=0) subs.splice(i,1); }; } };

    // THE DATA MODEL -- molbuilder.molview.data.  (Modules already mounted molview.mount; we
    // merge .data onto the same namespace, mirroring data-model.js.)
    global.molbuilder = global.molbuilder || {};
    global.molbuilder.molview = global.molbuilder.molview || {};
    global.molbuilder.molview.data = {
        selection:    store,
        getStructure: () => { dataCalls.push('getStructure'); return structure; },
        subscribe:    (fn) => store.subscribe(fn),
        // §19.3 -- the model install-from-text primitive ({text, filename}); the handle passes
        // its input straight through.  (Project-file PATH doors live in projects.parser, not here.)
        installMolecule: (arg) => { dataCalls.push(['installMolecule', arg]); return Promise.resolve('loaded'); },
        // §19.4 -- exportFile serialises the whole model to bytes (installMolecule's inverse).
        exportFile:   () => { dataCalls.push('exportFile'); return { xyz: 'XYZ-BYTES', sidecar: {} }; },
        undo:         () => { dataCalls.push('undo'); return Promise.resolve('undone'); },
        setFrame:     (i) => { dataCalls.push(['setFrame', i]); return i; },
        getFrame:     (i) => [[0, 0, 0]], frameCount: () => 3, currentFrame: () => 0,
        // OBSOLETE doors (retired at the carve) -- TRAPS: no handle method may call these.
        loadFromFile:     () => { obsoleteHit.push('loadFromFile');    return Promise.resolve(); },
        loadFromText:     () => { obsoleteHit.push('loadFromText');    return Promise.resolve(); },
        installStructure: () => { obsoleteHit.push('installStructure'); },
        getScratchBlob:   () => { obsoleteHit.push('getScratchBlob'); },
        applyPayload:     () => { obsoleteHit.push('applyPayload'); },
    };
    global.molbuilder.molview.selection = { mountPanel: async () => ({ panel: {}, dispose: () => {} }) };

    // THE PERSISTENCE WORKSPACE -- no data role.  Every data method is a TRAP.
    const workspace = {
        owner: 'modify', persist: () => {}, workspaceId: () => 'ws-x',
        selection:    { getState: () => { wsTrap.push('selection'); return { indices: [] }; },
                        subscribe: () => (() => {}) },
        getStructure: () => { wsTrap.push('getStructure'); return null; },
        installMolecule: () => { wsTrap.push('installMolecule'); return Promise.resolve(); },
        exportFile:   () => { wsTrap.push('exportFile'); return Promise.resolve(); },
        undo:         () => { wsTrap.push('undo'); return Promise.resolve(); },
        setFrame:     () => { wsTrap.push('setFrame'); },
    };
    // A PRE-BUILT fused-card host (wire-only path); the handle surface is identical on the
    // empty-host build path -- exercised in test_molview_render_js / the demo e2e.
    const host = { classList: { contains: () => true },
                   querySelector: (sel) => (sel === '.molview-panel' ? {} : null) };
    const mount = global.molbuilder.molview.mount;
"""


def test_handle_surface_one_load_door_and_save_is_loads_inverse():
    """§18.1/§D: the handle exposes exactly the owner surface + frame axis, no internals.
    §19.3: ``handle.installMolecule(input)`` passes its input straight to the SINGLE
    ``data.installMolecule`` primitive (no loadFromFile/loadFromText split; project-file PATH
    doors live in projects.parser).  §19.4: ``save()`` returns the serialized bytes via
    ``data.exportFile``."""
    out = _run_node(_HARNESS + """
        mount(host, workspace, { mode: 'modify' }).then(async (h) => {
            const keys  = Object.keys(h).sort();
            const r1    = await h.installMolecule('/path/to.xyz');                  // passthrough
            const r2    = await h.installMolecule({ text: 'T', filename: 'm.xyz' }); // {text,...}
            const bytes = await h.exportFile();                                // installMolecule's inverse -> bytes
            console.log(JSON.stringify({ keys, r1, r2, bytes, dataCalls, obsoleteHit }));
        });
    """)
    assert out["keys"] == HANDLE_KEYS                 # §D surface + frame axis, no internals
    # Both inputs pass straight through to the ONE data.installMolecule primitive:
    assert out["dataCalls"][0] == ["installMolecule", "/path/to.xyz"]
    assert out["dataCalls"][1] == ["installMolecule", {"text": "T", "filename": "m.xyz"}]
    assert out["dataCalls"][2] == "exportFile"
    assert out["r1"] == "loaded" and out["r2"] == "loaded"       # returns data.installMolecule's promise
    assert out["bytes"] == {"xyz": "XYZ-BYTES", "sidecar": {}}   # save() = serialized bytes
    assert out["obsoleteHit"] == []                             # no handle method reached an obsolete door


def test_all_handle_data_ops_route_to_molview_data_never_the_workspace():
    """§18/§C drift-guard: with a real molview.data present, EVERY handle data op
    (getStructure / installMolecule / exportFile / undo / setFrame) hits molview.data; the
    persistence workspace's data traps stay empty."""
    out = _run_node(_HARNESS + """
        mount(host, workspace, { mode: 'modify' }).then(async (h) => {
            const text = h.getStructure().text;
            await h.installMolecule('/p.xyz');
            await h.installMolecule({ text: 'T', filename: 'm.xyz' });
            await h.exportFile();
            await h.undo();
            h.setFrame(1);
            console.log(JSON.stringify({ text, dataCalls, wsTrap }));
        });
    """)
    assert out["text"] == "XYZ"                       # read came from molview.data
    assert out["dataCalls"] == ["getStructure", ["installMolecule", "/p.xyz"],
                                ["installMolecule", {"text": "T", "filename": "m.xyz"}],
                                "exportFile", "undo", ["setFrame", 1]]
    assert out["wsTrap"] == []                        # workspace NEVER served data


def test_handle_getSelection_returns_a_copy():
    """§E rule 2: reads are copies -- mutating the returned array can't leak into the store."""
    out = _run_node(_HARNESS + """
        mount(host, workspace, { mode: 'modify' }).then((h) => {
            const a = h.getSelection();
            a.push(999);                    // mutate the returned array
            const b = h.getSelection();     // must be a fresh copy
            console.log(JSON.stringify({ a, b }));
        });
    """)
    assert out["a"] == [1, 2, 999]
    assert out["b"] == [1, 2]              # unaffected -> getSelection returned a copy


def test_onChange_is_the_one_change_channel_through_molview_data():
    """§E rule 4: onChange is the ONE change channel; the owner subscribes here, not to the
    store/workspace directly.  It subscribes THROUGH molview.data.subscribe."""
    out = _run_node(_HARNESS + """
        mount(host, workspace, { mode: 'modify' }).then((h) => {
            let n = 0;
            const off = h.onChange(() => { n++; });
            subs.forEach((fn) => fn());     // a data change -> owner notified
            const afterFire = n;
            off();
            subs.forEach((fn) => fn());     // no listeners now
            console.log(JSON.stringify({ afterFire, afterOff: n, subCount: subs.length }));
        });
    """)
    assert out["afterFire"] == 1     # onChange fired on the data change
    assert out["afterOff"] == 1      # off() stopped further notifications
    assert out["subCount"] == 0      # unsubscribed cleanly


def test_dispose_tears_down_onChange_subscriptions():
    out = _run_node(_HARNESS + """
        mount(host, workspace, { mode: 'modify' }).then((h) => {
            h.onChange(() => {});
            h.onChange(() => {});
            const before = subs.length;
            h.dispose();                    // must tear down onChange subs it handed out
            console.log(JSON.stringify({ before, after: subs.length }));
        });
    """)
    assert out["before"] == 2
    assert out["after"] == 0          # dispose() unsubscribed both onChange listeners


# ---- The FALLBACK path (NOT the contract): molview.data absent -> use the workspace ----- #
#
# §18/§C mandate molview.data as THE data layer.  mount.js still carries a transitional
# fallback -- ``const data = (mb.molview && mb.molview.data) || workspace`` -- for the case
# where no data model is mounted.  This is verified explicitly and labelled as a FALLBACK; it
# is NOT the contract path (that is pinned by the two tests above).
_FALLBACK_HARNESS = """
    const wsCalls = [];
    const store = { getState: () => ({ indices: [7] }), subscribe: () => (() => {}) };
    // NO molview.data mounted -> mount falls back to the workspace as the data source.
    global.molbuilder = global.molbuilder || {};
    global.molbuilder.molview = global.molbuilder.molview || {};   // keep .mount; leave .data UNSET
    global.molbuilder.molview.selection = { mountPanel: async () => ({ panel: {}, dispose: () => {} }) };
    const workspace = {
        selection:    store,
        getStructure: () => { wsCalls.push('getStructure'); return { text: 'WS' }; },
        subscribe:    () => (() => {}),
        installMolecule: (a) => { wsCalls.push(['installMolecule', a]); return Promise.resolve('ws-loaded'); },
        exportFile:   () => { wsCalls.push('exportFile'); return 'ws-bytes'; },
        undo:         () => { wsCalls.push('undo'); return Promise.resolve('ws-undone'); },
    };
    const host = { classList: { contains: () => true },
                   querySelector: (sel) => (sel === '.molview-panel' ? {} : null) };
    const mount = global.molbuilder.molview.mount;
"""


def test_workspace_fallback_when_molview_data_absent():
    """FALLBACK ONLY (not the contract): with no molview.data mounted, the handle routes its
    data ops to the workspace.  Guards that the fallback branch stays wired -- while the
    contract path (data via molview.data) is pinned separately above."""
    out = _run_node(_FALLBACK_HARNESS + """
        mount(host, workspace, { mode: 'modify' }).then(async (h) => {
            const text  = h.getStructure().text;
            const r     = await h.installMolecule('/f.xyz');
            const bytes = await h.exportFile();
            console.log(JSON.stringify({ text, r, bytes, wsCalls }));
        });
    """)
    assert out["text"] == "WS"                 # read fell back to the workspace
    assert out["r"] == "ws-loaded"
    assert out["bytes"] == "ws-bytes"
    assert out["wsCalls"] == ["getStructure", ["installMolecule", "/f.xyz"], "exportFile"]
