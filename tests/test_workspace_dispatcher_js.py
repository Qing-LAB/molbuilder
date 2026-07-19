"""Contract tests for the MolView in-memory data model + the workspace
persistence dispatcher (JS unit tests, run under Node).

Two layers, two docs, pinned here:

* **MolView data model** — ``window.molbuilder.molview.data``
  (``lib/molview/data-model.js``).  Owns the structure / selection /
  periodicity / frames in memory and serialises itself.  Contract:
  ``docs/protocols/molview-module.md`` §19 (§19.2 reads, §19.3 writes,
  §19.3.1 the LOAD/SAVE atomic + coherence invariant, §19.4
  serialisation).  The authoritative public surface is the ``api``
  object at the end of ``data-model.js``.

* **Workspace** — ``window.molbuilder.workspace``
  (``lib/workspace/dispatcher.js``).  PERSISTENCE ONLY (session mirror
  + on-disk draft), format-blind.  Contract:
  ``docs/protocols/workspace-contract.md`` §3.5 / §4.

The model install / serialise primitives are
``molview.data.installMolecule({text, filename[, sidecar, ...]})`` (parses
the text via /api/build/load, installs it in ONE store write, RESETS the
timeline) and ``molview.data.exportFile()`` (whole model -> ``{xyz,
sidecar}`` project-file bytes, installMolecule's inverse).  The FORMAT-AWARE
project-file DOORS (``openMolecule(path)`` / ``saveMolecule(path)``) live in
the projects package (``molbuilder.projects.parser``), NOT on molview.data,
and are tested there — this file does NOT exercise them.  The SESSION-STATE
timeline (§19.5) is ``save(delta=0)`` /
``load(delta=0)`` -- checkpoint / restore parameterized by an index delta
(``save(1)`` = a new checkpoint, ``load(-1)`` = Retract, ``load(0)`` =
reload from the mirror).  The pre-carve + superseded doors are GONE and
must never appear in a test as a call: ``loadFromText`` / ``loadFromFile``
/ ``installStructure`` / ``getScratchBlob`` / ``applyPayload`` /
``pushState`` / ``popState`` / ``restoreSnapshot`` / the old file-writer
``save(opts)``.

The stores the harness mounts (``structureCanvas`` = canvas-state,
``selection.store``) are MolView-INTERNAL (molview-module.md §19).  A
few tests drive them directly to SET UP fixture state cheaply; the
CONTRACT under test is always read/written through the public
``molview.data.*`` surface.  Where an install must be exercised end-to-end
the real ``installMolecule()`` primitive is used with a stubbed ``/api/build/load``.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DISPATCHER_PATH   = ROOT / "molbuilder/web/static/lib/workspace/dispatcher.js"
STORE_PATH        = ROOT / "molbuilder/web/static/lib/molview/_selection-store-impl.js"
CANVAS_PATH       = ROOT / "molbuilder/web/static/lib/molview/_canvas-state-impl.js"
SNAPSHOT_IO_PATH  = ROOT / "molbuilder/web/static/lib/workspace/snapshot-io.js"
DATA_MODEL_PATH   = ROOT / "molbuilder/web/static/lib/molview/data-model.js"


def _run_node(snippet: str) -> object:
    """Run a Node snippet with the MolView data model + workspace
    dispatcher loaded over a minimal browser stub.

    Load order mirrors production: shared snapshot IO -> canvas-state
    impl -> selection-store impl -> the selection-store singleton ->
    the MolView data model (``molview.data``) -> the workspace
    persistence dispatcher.  (Frame coords are owned by the embed movie,
    §14.5, so no frame-series is loaded; frame tests attach a fake
    embed handle.)  The
    canvas-state + selection-store impls are MolView-internal; the
    bootstrap mounts them where the data model's private escape
    hatches (``_canvas()`` / ``_store()``) look for them so a test can
    seed fixture state cheaply.

    The snippet drives ``window.molbuilder.molview.data.*`` (the data
    model) and ``window.molbuilder.workspace.*`` (persistence) and
    prints a JSON blob as its LAST stdout line.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")

    bootstrap = (
        "require(" + json.dumps(str(SNAPSHOT_IO_PATH)) + ");\n"
        "window.molbuilder.structureCanvas = require("
        + json.dumps(str(CANVAS_PATH)) + ");\n"
        "require(" + json.dumps(str(STORE_PATH)) + ");\n"
        "window.molbuilder.selection.store = "
        "  window.molbuilder.selection._createStore();\n"
        "require(" + json.dumps(str(DATA_MODEL_PATH)) + ");\n"
        "require(" + json.dumps(str(DISPATCHER_PATH)) + ");\n"
        + snippet
    )
    header = """
        const _events = {};
        global.window = global;
        global.window.addEventListener = (evt, cb) => {
            (_events[evt] = _events[evt] || []).push(cb);
        };
        global.window.__fireEvent = (evt) => {
            (_events[evt] || []).forEach((cb) => cb({}));
        };
        global.document = {
            readyState: "complete",
            addEventListener: () => {},
            getElementById:  () => null,
        };
        const _storage = {};
        global.sessionStorage = {
            getItem:    (k) => (_storage[k] == null ? null : _storage[k]),
            setItem:    (k, v) => { _storage[k] = String(v); },
            removeItem: (k) => { delete _storage[k]; },
        };
        global.molbuilder = global.molbuilder || {};
        global.window.molbuilder = global.molbuilder;
        const _registry = {};
        const _waiters  = {};
        global.molbuilder.runtime = {
            register: (name, value) => {
                _registry[name] = value;
                if (_waiters[name]) {
                    _waiters[name].forEach((res) => res(value));
                    delete _waiters[name];
                }
            },
            whenReady: (name) => {
                if (name in _registry) return Promise.resolve(_registry[name]);
                return new Promise((res) => {
                    (_waiters[name] = _waiters[name] || []).push(res);
                });
            },
        };
    """
    full = header + bootstrap
    proc = subprocess.run(
        [node, "--input-type=commonjs", "-e", full],
        capture_output=True, text=True, timeout=15,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"node exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
    last = proc.stdout.strip().splitlines()[-1]
    return json.loads(last)


# A load fixture: stub ``/api/build/load`` so ``molview.data.installMolecule()``
# can drive the WHOLE atomic-load pipeline (§19.3.1) without a server.  Returns
# a 3-atom water payload in the WorkspacePayload wire shape (§21).
_STUB_WATER_FETCH = """
global.window.fetch = function (url, opts) {
    return Promise.resolve({ ok: true, json: function () {
        return Promise.resolve({
            ok: true, source_format: "xyz", title: "h2o", n_atoms: 3,
            text: "3\\nh2o\\nO 0 0 0\\nH 0.957 0 0\\nH -0.24 0.927 0\\n",
            atoms: [
                {index:0, element:"O", x:0,      y:0,     z:0, regions:[], is_frozen:false},
                {index:1, element:"H", x:0.957,  y:0,     z:0, regions:[], is_frozen:false},
                {index:2, element:"H", x:-0.24,  y:0.927, z:0, regions:[], is_frozen:false},
            ],
        });
    }});
};
"""


# A fake embed handle: the frame COORDINATES are owned by the embed's native 3Dmol
# movie (task #33, §14.5), so the data model delegates every frame op to the handle.
# This in-memory stand-in implements just the frame slice of the handle surface
# (setAnimation / setAnimationFrame / appendFrames + the cheap read probes) so frame
# tests exercise the delegation without a real 3Dmol viewer.  Attach it AFTER the data
# model loads and BEFORE the frame ops.
_FAKE_EMBED = """
window.molbuilder.molview.data.attachViewHandle((function () {
    let _frames = null, _cur = 0, _kind = null;
    const cp = (f) => f.map((p) => p.slice());
    return {
        setAnimation: function (a) {
            if (a && a.kind === "trajectory") {
                _frames = a.frames.map(cp); _kind = "trajectory"; _cur = 0;
            }
            // arrowsPerFrame-only partial update: ignore (overlay, not frames).
        },
        appendFrames: function (list) {
            if (_frames) list.forEach((f) => _frames.push(f.slice().map((p) => p.slice())));
        },
        appendFrameArrows: function () { /* overlay-only; frames unaffected */ },
        setAnimationFrame: function (i) {
            if (_frames && i >= 0 && i < _frames.length) _cur = i;
        },
        getAnimationFrame: function () { return _cur; },
        getAnimationKind:  function () { return _kind; },
        getFrameCount:     function () { return _frames ? _frames.length : 0; },
        getFrameCoords:    function (i) { return (_frames && _frames[i]) ? cp([_frames[i]])[0] : null; },
    };
})());
"""


# --------------------------------------------------------------------- #
#  1. The public molview.data surface (molview-module.md §19)           #
# --------------------------------------------------------------------- #

# The authoritative surface — the sorted Object.keys of the ``api`` object
# at the end of data-model.js.  Adding/removing a method here means the doc
# (§19) AND this pin change together; drift breaks the test on purpose.
_DATA_SURFACE = sorted([
    "subscribe", "getState", "getStructure", "getSource", "getSourceFile",
    "getLastSavedTo", "getSelection", "getAtoms", "getElements",
    "getCoordinates", "getUnitCell", "getLattice", "getAxisKind", "getVacuum",
    "getUnitCellInfo", "getUnitCellOrigin", "getVacuumInfo",
    "getAxisKindInfo", "getAtomsByLabel", "getFrozen", "getRegions",
    "atomFor3Dmol", "toAddAtoms", "draftIdentity", "suspendPersist",
    "resumePersist", "commitPeriodicity", "setUnitCell", "setLattice",
    "setAxisKind", "setVacuum", "setLabel", "isDirty", "isEmpty",
    "markDirty", "markSaved", "installMolecule", "exportFile",
    "save", "load",
    "generate", "applyOp",
    "discard", "undo", "reloadFrames", "addFrame", "addFrames", "setFrame",
    # setFrameArrows: per-frame force overlays baked into the native animation;
    # onFrameChange: the frame-only notification channel the frame bar subscribes
    # to (separate from the selection store, so a frame swap doesn't re-render the
    # panel + steal input focus during playback).
    "setFrameArrows", "appendFrameArrows", "onFrameChange",
    # getForces/currentForces removed with the frame-series (task #33): forces are
    # the CONSUMER's data now, and coords are owned by the embed movie -- getFrame
    # reads a frame through the handle, not a data-model coords copy.
    "getFrame", "currentFrame", "frameCount",
    "selection", "view",
    # §20 view-state lifecycle: the active molview registers its embed handle
    # (attach/detach) so view.get/applyState reach it, and flushViewState mirrors
    # the live view on pagehide (persistence is otherwise push-only).
    "attachViewHandle", "detachViewHandle", "flushViewState",
    # §19.5 state timeline — save(delta)/load(delta) + the two live reads.
    "state_index", "uncommitted",
])

# The pre-carve + superseded doors that the unified surface collapsed.  A
# test that CALLS any of these is testing a surface that no longer exists.
_OBSOLETE_DOORS = ["loadFromText", "loadFromFile", "installStructure",
                   "getScratchBlob", "applyPayload",
                   # §19.5 superseded the `applyState` stopgap (it also name-clashed
                   # with view.applyState) -- it must not reappear on the surface.
                   "applyState",
                   # §19.5 collapsed the fixed-delta timeline doors into save/load.
                   "pushState", "popState", "restoreSnapshot"]


class TestDataModelSurface:
    """The concealed data model exposes EXACTLY one documented surface
    on ``molbuilder.molview.data`` — and none of the pre-carve doors."""

    def test_molview_data_is_an_object_on_window(self):
        out = _run_node(
            "console.log(JSON.stringify("
            "  typeof window.molbuilder.molview.data));")
        assert out == "object"

    def test_public_surface_is_exactly_the_documented_set(self):
        out = _run_node(
            "console.log(JSON.stringify("
            "  Object.keys(window.molbuilder.molview.data).sort()));")
        assert out == _DATA_SURFACE

    def test_obsolete_doors_are_absent(self):
        """The unified I/O is installMolecule()+exportFile()+save(delta)/
        load(delta); the pre-carve + superseded fixed-delta timeline
        doors must NOT be reachable on the public surface."""
        out = _run_node(
            "const d = window.molbuilder.molview.data;\n"
            "console.log(JSON.stringify("
            + json.dumps(_OBSOLETE_DOORS)
            + ".filter(k => k in d)));")
        assert out == [], f"obsolete door(s) still on molview.data: {out}"

    def test_unified_io_and_core_methods_are_functions(self):
        out = _run_node(
            "const d = window.molbuilder.molview.data;\n"
            "console.log(JSON.stringify({\n"
            "  installMolecule: typeof d.installMolecule,\n"
            "  exportFile:   typeof d.exportFile,\n"
            "  save:         typeof d.save,\n"
            "  load:         typeof d.load,\n"
            "  subscribe:    typeof d.subscribe,\n"
            "  getState:     typeof d.getState,\n"
            "  getStructure: typeof d.getStructure,\n"
            "  getSource:    typeof d.getSource,\n"
            "  getSelection: typeof d.getSelection,\n"
            "  isDirty:      typeof d.isDirty,\n"
            "  isEmpty:      typeof d.isEmpty,\n"
            "  generate:     typeof d.generate,\n"
            "  applyOp:      typeof d.applyOp,\n"
            "  discard:      typeof d.discard,\n"
            "  undo:         typeof d.undo,\n"
            "  draftIdentity:  typeof d.draftIdentity,\n"
            "  suspendPersist: typeof d.suspendPersist,\n"
            "  resumePersist:  typeof d.resumePersist,\n"
            "}));")
        for k, v in out.items():
            assert v == "function", f"{k} is {v!r}; expected function"

    def test_sub_namespaces_present(self):
        out = _run_node(
            "const d = window.molbuilder.molview.data;\n"
            "console.log(JSON.stringify({\n"
            "  selection: typeof d.selection,\n"
            "  view:      typeof d.view,\n"
            "  sel_toggle: typeof d.selection.toggle,\n"
            "  sel_set:    typeof d.selection.set,\n"
            "  view_apply: typeof d.view.applyState,\n"
            "  view_get:   typeof d.view.getState,\n"
            "}));")
        assert out == {
            "selection": "object", "view": "object",
            "sel_toggle": "function", "sel_set": "function",
            "view_apply": "function", "view_get": "function",
        }


# --------------------------------------------------------------------- #
#  2. The workspace is PERSISTENCE-ONLY (workspace-contract.md §3.5)     #
# --------------------------------------------------------------------- #


class TestWorkspaceIsPersistenceOnly:
    _PERSIST_SURFACE = ["STORAGE_KEY", "mountRestoreTarget", "onPersistError",
                        "persist", "pruneStatesAbove", "readPersistedSnapshot",
                        "readState", "useNamespace", "workspaceId"]

    def test_workspace_public_surface_is_exactly_persistence(self):
        """§3.5: the workspace exposes EXACTLY the persistence surface
        (private ``_``-slots the impls mount are allowed; no data
        method is)."""
        out = _run_node(
            "console.log(JSON.stringify("
            "  Object.keys(window.molbuilder.workspace)"
            "    .filter(k => k[0] !== '_').sort()));")
        assert out == self._PERSIST_SURFACE

    def test_no_data_accessors_leaked_onto_the_workspace(self):
        """Explicit negative drift-guard: the data surface (incl. the
        pre-carve doors) is ABSENT on the workspace — it lives on
        ``molbuilder.molview.data``."""
        probe = ["getStructure", "getAtoms", "getSelection", "getCoordinates",
                 "installStructure", "loadFromText", "load", "save", "applyOp",
                 "applyPayload", "setFrame", "addFrame", "reloadFrames",
                 "discard", "undo", "selection", "view", "getScratchBlob",
                 "setUnitCell"]
        out = _run_node(
            "const ws = window.molbuilder.workspace;\n"
            "console.log(JSON.stringify("
            + json.dumps(probe) + ".filter(k => k in ws)));")
        assert out == [], f"data method(s) leaked onto the workspace: {out}"

    def test_persist_is_format_blind(self):
        """§3.5: ``persist`` writes the session bytes VERBATIM — it
        never parses or interprets them (they need not be a structure
        at all)."""
        out = _run_node(
            "const ws = window.molbuilder.workspace;\n"
            "ws.persist({foo: 1, bar: 'x', not_a_structure: true}, null, {source: '/x'});\n"
            "const raw = sessionStorage.getItem(ws.STORAGE_KEY);\n"
            "console.log(JSON.stringify(JSON.parse(raw)));")
        assert out == {"foo": 1, "bar": "x", "not_a_structure": True}

    def test_storage_key_is_the_unified_v1_key(self):
        out = _run_node(
            "console.log(JSON.stringify(window.molbuilder.workspace.STORAGE_KEY));")
        assert out == "molbuilder.workspace.v1"

    def test_useNamespace_isolates_each_owners_session_mirror(self):
        """§18.4: ``useNamespace(owner)`` folds the owner into the mirror key so one
        consumer's session never overwrites another's.  Two owners persisting in turn
        each land under ``<base>::<owner>``; the base key stays untouched; switching back
        to an owner still reads ITS snapshot (no clobber)."""
        out = _run_node(
            "const ws = window.molbuilder.workspace;\n"
            "const io = window.molbuilder.workspaceSnapshot;\n"
            "ws.useNamespace('owner-a');\n"
            "ws.persist({v: 1, state: {tag: 'A'}}, null, {});\n"
            "ws.useNamespace('owner-b');\n"
            "ws.persist({v: 1, state: {tag: 'B'}}, null, {});\n"
            "ws.useNamespace('owner-a');\n"
            "console.log(JSON.stringify({\n"
            "  aKey:  sessionStorage.getItem('molbuilder.workspace.v1::owner-a') !== null,\n"
            "  bKey:  sessionStorage.getItem('molbuilder.workspace.v1::owner-b') !== null,\n"
            "  base:  sessionStorage.getItem('molbuilder.workspace.v1'),\n"
            "  aTag:  (io.read() || {}).state.tag,\n"
            "}));")
        assert out["aKey"] is True   # owner-a's mirror exists under its namespaced key
        assert out["bKey"] is True   # owner-b's mirror exists under ITS namespaced key
        assert out["base"] is None   # neither leaked into the shared base key
        assert out["aTag"] == "A"    # switching back to owner-a reads A, not B

    def test_same_index_state_writes_land_in_issue_order(self):
        """§4.7 write ordering: the on-disk state write is fire-and-forget, so two
        writes to the SAME ``<workspace_id>.<state_index>`` file (a rapid
        save(1) -> load(-1) -> save(1)) must not race -- a stale write landing
        after a newer one would leave an abandoned state that a later Retract could
        restore.  The dispatcher CHAINS write-state POSTs: the 2nd is not even SENT
        until the 1st has fully completed, so last-issued is always last-written --
        even when the 1st resolves SLOWER than the 2nd would have."""
        out = _run_node(
            "var log = [];\n"
            "global.window.fetch = function (url, opts) {\n"
            "  if (url === '/api/state-timeline/write') {\n"
            "    var tag = JSON.parse(opts.body).data.tag;\n"
            "    log.push('sent:' + tag);\n"
            "    var delay = (tag === 'A') ? 40 : 0;\n"   # 1st write is the SLOW one
            "    return new Promise(function (res) { setTimeout(function () {\n"
            "      log.push('done:' + tag);\n"
            "      res({ ok: true, status: 200 });\n"
            "    }, delay); });\n"
            "  }\n"
            "  return Promise.resolve({ ok: true, status: 200,\n"
            "    json: function () { return Promise.resolve({}); } });\n"
            "};\n"
            "var ws = window.molbuilder.workspace;\n"
            "var id = { workspace_id: 'w', state_index: 1 };\n"   # SAME index for both
            "ws.persist({s: 1}, { tag: 'A' }, id);\n"
            "ws.persist({s: 1}, { tag: 'B' }, id);\n"
            "setTimeout(function () { console.log(JSON.stringify({ log: log })); }, 200);")
        # Strict serialisation: A fully completes before B is even sent, despite A
        # being the slow write.  Without the chain this would be
        # ['sent:A','sent:B','done:B','done:A'] -- both in flight, slow A landing LAST.
        assert out["log"] == ["sent:A", "done:A", "sent:B", "done:B"]


# --------------------------------------------------------------------- #
#  3. getStructure() is "null iff empty" (molview-module.md §19.2)       #
# --------------------------------------------------------------------- #


class TestGetStructureNullIffEmpty:

    def test_null_when_empty(self):
        out = _run_node(
            "const d = window.molbuilder.molview.data;\n"
            "console.log(JSON.stringify({\n"
            "  isEmpty:   d.isEmpty(),\n"
            "  structure: d.getStructure(),\n"
            "  atoms:     d.getAtoms(),\n"
            "  isDirty:   d.isDirty(),\n"
            "}));")
        assert out == {"isEmpty": True, "structure": None,
                       "atoms": [], "isDirty": False}

    def test_nonnull_with_atoms_after_a_load(self):
        """§19.3.1 coherence invariant: after ONE installMolecule() the model is
        populated across atoms + structure + periodicity together —
        getStructure() is non-null AND carries the loaded atoms."""
        out = _run_node(
            _STUB_WATER_FETCH +
            "const d = window.molbuilder.molview.data;\n"
            "d.installMolecule({ text: 'unused', filename: '/p/water.xyz' }).then(() => {\n"
            "  const s = d.getStructure();\n"
            "  console.log(JSON.stringify({\n"
            "    isEmpty:  d.isEmpty(),\n"
            "    nonNull:  s !== null,\n"
            "    nAtoms:   s ? s.atoms.length : -1,\n"
            "    elements: d.getElements(),\n"
            "    head:     s ? s.text.split('\\n')[0] : null,\n"
            "  }));\n"
            "});")
        assert out["isEmpty"] is False
        assert out["nonNull"] is True
        assert out["nAtoms"] == 3
        assert out["elements"] == ["O", "H", "H"]
        assert out["head"] == "3"

    def test_installMolecule_rejects_a_bad_input(self):
        out = _run_node(
            "const d = window.molbuilder.molview.data;\n"
            "d.installMolecule(42).then(\n"
            "  () => console.log(JSON.stringify({ rejected: false })),\n"
            "  (e) => console.log(JSON.stringify({ rejected: true,\n"
            "     msg: /installMolecule/.test(String(e.message)) })));")
        assert out == {"rejected": True, "msg": True}


class TestReadsAreDefensiveCopies:
    """§1.2.1: every read accessor returns COPIES -- a consumer holding a
    returned value can NEVER mutate the store by writing into it."""

    def test_getAtoms_returns_defensive_per_atom_copies(self):
        out = _run_node(
            _STUB_WATER_FETCH +
            "const d = window.molbuilder.molview.data;\n"
            "d.installMolecule({ text:'x', filename:'/p/water.xyz' }).then(() => {\n"
            "  const a = d.getAtoms();\n"
            "  a[0].x = 999; a[0].element = 'Xx';\n"       # mutate the returned copy
            "  const b = d.getAtoms();\n"
            "  console.log(JSON.stringify({ x: b[0].x, el: b[0].element }));\n"
            "});")
        assert out["x"] != 999, "getAtoms() leaked a live atom object"
        assert out["el"] != "Xx", "getAtoms() leaked a live atom object"

    def test_getStructure_annotations_are_a_deep_copy(self):
        out = _run_node(
            "const d = window.molbuilder.molview.data;\n"
            "const cs = window.molbuilder.structureCanvas;\n"
            "cs.setStructure({ source_format:'xyz', text:'1\\nx\\nH 0 0 0\\n',\n"
            "  annotations:{ note:'orig', arr:[1, 2] } }, {kind:'file', file:'/p/x.xyz'});\n"
            "const s1 = d.getStructure();\n"
            "s1.annotations.note = 'HACKED'; s1.annotations.arr.push(99);\n"
            "const s2 = d.getStructure();\n"
            "console.log(JSON.stringify({ note: s2.annotations.note,\n"
            "                             arrLen: s2.annotations.arr.length }));")
        assert out["note"] == "orig", "getStructure().annotations leaked a live reference"
        assert out["arrLen"] == 2, "getStructure().annotations leaked a live reference"


# --------------------------------------------------------------------- #
#  4. exportFile() = the whole-model serialisation (molview-module.md §19.4) #
# --------------------------------------------------------------------- #


class TestExportFile:
    """exportFile() reads the ENTIRE model out to {xyz, sidecar}
    project-file bytes — the inverse of installMolecule (§19.3.1 symmetry).
    It is NOT a file writer (the old ``save(opts)`` door is gone) and it
    is NOT the session-state timeline save; persisting the bytes is the
    consumer's job (two-saves-never-mix)."""

    def test_export_returns_xyz_and_sidecar_from_the_whole_model(self):
        out = _run_node(
            _STUB_WATER_FETCH +
            "const d = window.molbuilder.molview.data;\n"
            "d.installMolecule({ text: 'unused', filename: '/p/water.xyz' }).then(() => {\n"
            "  const blob = d.exportFile();\n"
            "  console.log(JSON.stringify({\n"
            "    keys:       blob ? Object.keys(blob).sort() : null,\n"
            "    xyzHead:    blob ? blob.xyz.split('\\n')[0] : null,\n"
            "    sidecarKeys: blob ? Object.keys(blob.sidecar).sort() : null,\n"
            "    nAtoms:     blob ? blob.sidecar.n_atoms_total : null,\n"
            "  }));\n"
            "});")
        assert out["keys"] == ["sidecar", "xyz"]
        assert out["xyzHead"] == "3"           # the whole geometry, serialised
        assert out["nAtoms"] == 3
        # The sidecar carries the full non-geometry state (labels/frozen +
        # periodicity + annotations), built through the §19.2 accessors.
        for field in ("regions", "frozen_atoms", "cell", "axis_kind",
                      "vacuum", "annotations"):
            assert field in out["sidecarKeys"], f"sidecar missing {field}"
        assert "kgrid" not in out["sidecarKeys"]   # k-grid is not geometry

    def test_export_carries_regions_frozen_and_periodicity(self):
        """The sidecar is assembled from the model accessors (getRegions
        / getFrozen / periodicity) — not a re-read of any old file."""
        out = _run_node(
            "const d = window.molbuilder.molview.data;\n"
            "const cs = window.molbuilder.structureCanvas;\n"
            "cs.setStructure(\n"
            "  {source_format:'xyz', text:'2\\nx\\nAu 0 0 0\\nS 1 0 0\\n',\n"
            "   periodicity:{cell:[[10,0,0],[0,10,0],[0,0,10]],\n"
            "     axis_kind:['periodic','periodic','transport'],\n"
            "     vacuum:[0,0,15]}},\n"
            "  {kind:'file', file:'/p/x.xyz'});\n"
            "window.molbuilder.selection.store.adoptAtoms([\n"
            "  {index:0, element:'Au', x:0, y:0, z:0, regions:['L-electrode'], is_frozen:true},\n"
            "  {index:1, element:'S',  x:1, y:0, z:0, regions:['BDT'], is_frozen:false}]);\n"
            "const s = d.exportFile();\n"
            "console.log(JSON.stringify(s.sidecar));")
        assert out["regions"] == {"L-electrode": [0], "BDT": [1]}
        assert out["frozen_atoms"] == [0]
        assert out["cell"] == [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
        assert out["axis_kind"] == ["periodic", "periodic", "transport"]
        assert "kgrid" not in out   # k-grid is not geometry -> not in the sidecar

    def test_export_refuses_geometry_labels_desync_returns_null_and_logs(self):
        """§19.4: a geometry (canvas text) ↔ labels (selection store)
        atom-count desync must NEVER reach disk — exportFile() returns
        null and logs, instead of emitting a mismatched .xyz/.json pair."""
        out = _run_node(
            "let errs = 0;\n"
            "const _err = console.error; console.error = function(){ errs++; _err.apply(console, arguments); };\n"
            "const d = window.molbuilder.molview.data;\n"
            "const cs = window.molbuilder.structureCanvas;\n"
            "cs.setStructure({source_format:'xyz',\n"
            "  text:'3\\nx\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n'},\n"    # geometry: 3 atoms
            "  {kind:'file', file:'/p/x.xyz'});\n"
            "window.molbuilder.selection.store.adoptAtoms([\n"       # store: only 2 atoms
            "  {index:0, element:'O', x:0, y:0, z:0, regions:[], is_frozen:false},\n"
            "  {index:1, element:'H', x:1, y:0, z:0, regions:[], is_frozen:false}]);\n"
            "const blob = d.exportFile();\n"
            "console.log(JSON.stringify({ blob: blob, logged: errs >= 1 }));")
        assert out["blob"] is None, "a mismatched .xyz/.json pair must not serialise"
        assert out["logged"] is True, "the desync refusal must be logged"


# --------------------------------------------------------------------- #
#  5. draftIdentity() = {workspace_id} ONLY (molview-module.md §19.4)    #
# --------------------------------------------------------------------- #


class TestDraftIdentity:
    """The AUTOMATIC crash-safe draft is keyed by the tab's workspace
    id and NOTHING else — no filename.  A filename belongs only to the
    explicit user "save to a project file" action (two-saves-never-mix,
    workspace-contract intro); leaking it into the draft key is the bug
    this pins against."""

    def test_draft_identity_is_workspace_id_only_even_after_a_file_load(self):
        out = _run_node(
            _STUB_WATER_FETCH +
            "const d = window.molbuilder.molview.data;\n"
            "d.installMolecule({ text: 'unused', filename: '/p/water.xyz' }).then(() => {\n"
            "  const id = d.draftIdentity();\n"
            "  console.log(JSON.stringify({\n"
            "    keys:      Object.keys(id).sort(),\n"
            "    hasWsId:   typeof id.workspace_id === 'string' && id.workspace_id.length > 0,\n"
            "    noSource:  !('source' in id),\n"
            "    noFile:    !('file' in id) && !('filename' in id),\n"
            "  }));\n"
            "});")
        assert out["keys"] == ["workspace_id"]
        assert out["hasWsId"] is True
        assert out["noSource"] is True
        assert out["noFile"] is True


# --------------------------------------------------------------------- #
#  6. Persistence is PUSH-ONLY — a bare edit writes NOTHING (§19.5)      #
# --------------------------------------------------------------------- #


class TestNoAutoPersist:
    """§19.5: persistence is EXPLICIT (push-only).  A data change (an
    edit, a periodicity/label edit, a frame append) updates the
    in-memory model but writes NOTHING to disk — only ``installMolecule``
    (the anchor), ``save``, and ``load`` touch ``ws.persist`` /
    ``ws.readState``.  suspendPersist/resumePersist survive as a
    coherence bracket but no longer release any write."""

    def test_a_bare_edit_never_calls_persist(self):
        """A canvas DATA change (no save) must not auto-persist —
        the old debounced auto-write is gone (§19.5)."""
        out = _run_node(
            "let n = 0;\n"
            "window.molbuilder.workspace.persist = function () { n++; };\n"
            "const cs = window.molbuilder.structureCanvas;\n"
            "cs.setStructure({source_format:'xyz', text:'1\\nx\\nC 0 0 0\\n'},\n"
            "  {kind:'file', file:'/p/x.xyz'});\n"
            "setTimeout(() => { console.log(JSON.stringify({ persists: n })); }, 200);")
        assert out["persists"] == 0, "a bare edit must not auto-persist (push-only)"

    def test_suspend_resume_release_no_write(self):
        """resumePersist() no longer releases a persist — under push-only
        there is nothing to flush; it is a pure coherence bracket."""
        out = _run_node(
            "let n = 0;\n"
            "window.molbuilder.workspace.persist = function () { n++; };\n"
            "const d = window.molbuilder.molview.data;\n"
            "d.suspendPersist();\n"
            "window.molbuilder.selection.store.adoptAtoms(\n"
            "  [{index:0, element:'C', x:0, y:0, z:0, regions:[], is_frozen:false}]);\n"
            "d.resumePersist();\n"
            "setTimeout(() => { console.log(JSON.stringify({ persists: n })); }, 200);")
        assert out["persists"] == 0, "resume must not release any write (push-only)"


# --------------------------------------------------------------------- #
#  Persistence round-trip + restore (workspace-contract.md §4.4)        #
# --------------------------------------------------------------------- #


class TestPersistRoundtrip:

    def test_persisted_snapshot_carries_the_documented_state_fields(self):
        """§4.4.1: the unified snapshot carries v=1 + structure +
        source + dirty + last_save_to + selection."""
        out = _run_node(
            _STUB_WATER_FETCH +
            "const d = window.molbuilder.molview.data;\n"
            "d.installMolecule({ text: 'unused', filename: '/p/water.xyz' }).then(() => {\n"
            "  setTimeout(() => {\n"
            "    const p = JSON.parse(sessionStorage.getItem(\n"
            "      window.molbuilder.workspace.STORAGE_KEY));\n"
            "    console.log(JSON.stringify({\n"
            "      v:             p.v,\n"
            "      hasStructure:  !!p.state.structure,\n"
            "      text:          p.state.structure.text.split('\\n')[0],\n"
            "      hasSource:     !!p.state.source,\n"
            "      hasDirty:      typeof p.state.dirty === 'boolean',\n"
            "      hasLastSaveTo: ('last_save_to' in p.state),\n"
            "      hasSelection:  !!p.state.selection,\n"
            "    }));\n"
            "  }, 200);\n"
            "});")
        assert out["v"] == 1
        assert out["hasStructure"] is True
        assert out["text"] == "3"
        assert out["hasSource"] is True
        assert out["hasDirty"] is True
        assert out["hasLastSaveTo"] is True
        assert out["hasSelection"] is True

    def test_readPersistedSnapshot_null_when_absent(self):
        out = _run_node(
            "console.log(JSON.stringify(\n"
            "  window.molbuilder.workspace.readPersistedSnapshot()));")
        assert out is None

    def test_readPersistedSnapshot_null_on_schema_mismatch(self):
        out = _run_node(
            "sessionStorage.setItem(window.molbuilder.workspace.STORAGE_KEY,\n"
            "  JSON.stringify({ v: 99, state: {} }));\n"
            "console.log(JSON.stringify(\n"
            "  window.molbuilder.workspace.readPersistedSnapshot()));")
        assert out is None

    def test_mountRestoreTarget_is_the_persisted_source_file(self):
        """§4.5: the single-authority restore target derives from the
        SAME persisted snapshot; a mount-time writer defers when it
        equals the file it was about to load."""
        out = _run_node(
            "sessionStorage.setItem(window.molbuilder.workspace.STORAGE_KEY,\n"
            "  JSON.stringify({ v: 1, state: {\n"
            "    structure: { text: '1\\nx\\nC 0 0 0\\n' },\n"
            "    source:    { kind: 'file', file: '/p/target.xyz' } } }));\n"
            "console.log(JSON.stringify(\n"
            "  window.molbuilder.workspace.mountRestoreTarget()));")
        assert out == "/p/target.xyz"

    def test_mountRestoreTarget_null_when_snapshot_has_no_structure(self):
        out = _run_node(
            "sessionStorage.setItem(window.molbuilder.workspace.STORAGE_KEY,\n"
            "  JSON.stringify({ v: 1, state: { source: { kind: 'blank', file: null } } }));\n"
            "console.log(JSON.stringify(\n"
            "  window.molbuilder.workspace.mountRestoreTarget()));")
        assert out is None

    def test_getState_snapshot_is_a_defensive_copy(self):
        out = _run_node(
            "const d = window.molbuilder.molview.data;\n"
            "window.molbuilder.selection.store.adoptAtoms([\n"
            "  {index:0, element:'O', regions:[], is_frozen:false},\n"
            "  {index:1, element:'H', regions:[], is_frozen:false}]);\n"
            "Promise.resolve().then(() => window.molbuilder.selection.store.setSelection([0,1]))\n"
            ".then(() => {\n"
            "  const snap = d.getState();\n"
            "  snap.selection.indices.push(999);\n"       # mutate the copy
            "  console.log(JSON.stringify(d.getState().selection.indices));\n"
            "});")
        assert out == [0, 1]


# --------------------------------------------------------------------- #
#  Subscriptions + selection passthrough (molview-module.md §12 / §19.2) #
# --------------------------------------------------------------------- #


class TestSubscribeAndSelection:

    def test_subscribe_fires_once_immediately_with_current_state(self):
        out = _run_node(
            "const d = window.molbuilder.molview.data;\n"
            "let calls = 0, last = null;\n"
            "d.subscribe((s) => { calls++; last = s; });\n"
            "console.log(JSON.stringify({ calls: calls, empty: last.structure === null }));")
        assert out == {"calls": 1, "empty": True}

    def test_subscribe_fans_in_selection_store_changes_and_unsubscribes(self):
        out = _run_node(
            "const d = window.molbuilder.molview.data;\n"
            "let calls = 0, afterChange = -1;\n"
            "const unsub = d.subscribe(() => { calls++; });\n"
            "Promise.resolve().then(() => {\n"
            "  calls = 0;\n"
            "  window.molbuilder.selection.store.adoptAtoms(\n"
            "    [{index:0, element:'C', regions:[], is_frozen:false}]);\n"
            "  return Promise.resolve();\n"
            "}).then(() => {\n"
            "  afterChange = calls;\n"
            "  unsub();\n"
            "  window.molbuilder.selection.store.adoptAtoms(\n"
            "    [{index:0, element:'C', regions:[], is_frozen:false}]);\n"
            "  return Promise.resolve();\n"
            "}).then(() => {\n"
            "  console.log(JSON.stringify({ afterChange: afterChange, afterUnsub: calls }));\n"
            "});")
        assert out == {"afterChange": 1, "afterUnsub": 1}

    def test_subscriber_error_does_not_wedge_the_others(self):
        out = _run_node(
            "const d = window.molbuilder.molview.data;\n"
            "let good = 0;\n"
            "d.subscribe(() => { throw new Error('boom'); });\n"
            "d.subscribe(() => { good++; });\n"
            "Promise.resolve().then(() => {\n"
            "  good = 0;\n"
            "  window.molbuilder.selection.store.adoptAtoms(\n"
            "    [{index:0, element:'C', regions:[], is_frozen:false}]);\n"
            "  return Promise.resolve();\n"
            "}).then(() => { console.log(JSON.stringify(good)); });")
        assert out == 1

    def test_selection_mutation_lands_on_the_store_in_contract_shape(self):
        """§12.2: ws.selection.set lands on the store; getState()
        returns the CONTRACT shape (raw ``selection`` renamed
        ``indices``)."""
        out = _run_node(
            "const d = window.molbuilder.molview.data;\n"
            "window.molbuilder.selection.store.adoptAtoms([\n"
            "  {index:0, element:'O', regions:[], is_frozen:false},\n"
            "  {index:1, element:'H', regions:[], is_frozen:false},\n"
            "  {index:2, element:'H', regions:[], is_frozen:false}]);\n"
            "Promise.resolve().then(() => d.selection.set([0, 2])).then(() => {\n"
            "  const s = d.selection.getState();\n"
            "  console.log(JSON.stringify({\n"
            "    onStore:  window.molbuilder.selection.store.getState().selection,\n"
            "    indices:  s.indices,\n"
            "    hasLegacy: 'selection' in s,\n"
            "  }));\n"
            "});")
        assert out["onStore"] == [0, 2]
        assert out["indices"] == [0, 2]
        assert out["hasLegacy"] is False   # legacy field name must not leak


# --------------------------------------------------------------------- #
#  Frames — the coordinate time axis (molview-module.md §14.5)           #
# --------------------------------------------------------------------- #


class TestFrames:

    _ATOMS2 = (
        "window.molbuilder.selection.store.adoptAtoms([\n"
        "  {index:0, element:'O', x:0, y:0, z:0},\n"
        "  {index:1, element:'H', x:1, y:0, z:0}]);\n"
    )

    def test_reload_then_setFrame_swaps_coords_and_keeps_selection(self):
        out = _run_node(
            _FAKE_EMBED + self._ATOMS2 +
            "const d = window.molbuilder.molview.data;\n"
            "window.molbuilder.selection.store.setSelection([1]);\n"
            "const nf = d.reloadFrames([[[0,0,0],[1,0,0]], [[0,0,0],[2,0,0]], [[0,0,0],[3,0,0]]]);\n"
            "const f0 = d.getAtoms().map(a => a.x);\n"
            "d.setFrame(2);\n"
            "const f2 = d.getAtoms().map(a => a.x);\n"
            "console.log(JSON.stringify({ nf, count: d.frameCount(),\n"
            "  current: d.currentFrame(), f0, f2,\n"
            "  sel: window.molbuilder.selection.store.getState().selection }));")
        assert out["nf"] == 3 and out["count"] == 3
        assert out["f0"] == [0, 1]          # reload lands on frame 0
        assert out["current"] == 2
        assert out["f2"] == [0, 3]          # setFrame(2) swapped the coords
        assert out["sel"] == [1]            # selection survives the frame swap

    def test_setFrame_makes_getStructure_text_and_export_reflect_visible_frame(self):
        """§14.5.4 coherence: after scrubbing to frame i, getStructure().text +
        exportFile() serialize the VISIBLE frame -- read on demand from the embed movie
        (the coord owner, task #33), not frame 0.  Text/atoms no longer diverge."""
        out = _run_node(
            _FAKE_EMBED + _STUB_WATER_FETCH +
            "const d = window.molbuilder.molview.data;\n"
            "d.installMolecule({ text:'x', filename:'/p/w.xyz' }).then(() => {\n"
            "  d.reloadFrames([\n"
            "    [[0,0,0],[0.957,0,0],[-0.24,0.927,0]],\n"
            "    [[0,0,0],[0.957,0,0],[9.9,0,0]]]);\n"      # frame 1: atom 2 x = 9.9
            "  d.setFrame(1);\n"
            "  const s = d.getStructure();\n"
            "  const blob = d.exportFile();\n"
            "  console.log(JSON.stringify({\n"
            "    fmt: s.source_format,\n"
            "    current: d.currentFrame(),\n"
            "    textHas99:   s.text.indexOf('9.900000') !== -1,\n"
            "    exportHas99: blob && blob.xyz.indexOf('9.900000') !== -1 }));\n"
            "});")
        assert out["current"] == 1
        assert out["fmt"] == "xyz"
        assert out["textHas99"] is True, "getStructure().text must serialize the visible frame"
        assert out["exportHas99"] is True, "exportFile() must save the visible frame"

    def test_wrong_atom_count_frame_is_a_hard_error(self):
        """§14.5 same-atoms invariant: a frame whose atom count differs
        from the loaded structure is rejected, never coerced."""
        out = _run_node(
            _FAKE_EMBED + self._ATOMS2 +
            "const d = window.molbuilder.molview.data;\n"
            "d.reloadFrames([[[0,0,0],[1,0,0]]]);\n"
            "const ok = d.addFrame([[0,1,0],[1,1,0]]);\n"
            "let threw = false;\n"
            "try { d.addFrame([[0,0,0],[1,0,0],[2,0,0]]); } catch(_) { threw = true; }\n"
            "console.log(JSON.stringify({ ok, threw, after: d.frameCount() }));")
        assert out["ok"] == 2
        assert out["threw"] is True
        assert out["after"] == 2            # the bad frame was not appended


# --------------------------------------------------------------------- #
#  §18.3 VIEW ops must NOT persist — the code violates this today.       #
#  Pinned as xfail (NOT weakened, NOT skipped) per the contract.        #
# --------------------------------------------------------------------- #


class TestViewOpsMustNotPersist:
    """molview-module.md §18.3 + §19.5: a VIEW change (frame-select,
    selection, isolate, k-grid, style, background) changes only what is
    DRAWN and must persist NOTHING.  Under push-only (§19.5) NOTHING
    auto-persists — not even a DATA change (a frame append flips the
    ``uncommitted`` flag instead of writing); persistence happens only on
    an explicit save / load (or the installMolecule anchor).  So a view op
    is doubly safe: it is neither data nor a checkpoint."""

    def test_setFrame_does_not_persist(self):
        out = _run_node(
            self._boot_one_frame() +
            "  let n = 0;\n"
            "  window.molbuilder.workspace.persist = function () { n++; };\n"
            "  d.reloadFrames([[[0,0,0]], [[1,0,0]]]);\n"        # DATA change -> uncommitted, NO write
            "  setTimeout(() => {\n"
            "    const afterReload = n;\n"
            "    d.setFrame(1);\n"                                # VIEW change -> must NOT persist
            "    setTimeout(() => {\n"
            "      console.log(JSON.stringify({\n"
            "        afterReload: afterReload,\n"
            "        setFramePersisted: n > afterReload,\n"
            "        uncommitted: d.uncommitted }));\n"
            "    }, 200);\n"
            "  }, 200);\n"
            "}, 200);")
        assert out["afterReload"] == 0              # push-only: a frame DATA change writes NOTHING
        assert out["setFramePersisted"] is False     # setFrame = VIEW -> zero writes (§18.3)
        assert out["uncommitted"] is True            # but the frame DATA change IS uncommitted (§19.5)

    def test_selection_change_does_not_persist(self):
        out = _run_node(
            self._boot_one_frame() +
            "  window.molbuilder.selection.store.adoptAtoms([\n"
            "    {index:0, element:'H', x:0, y:0, z:0, regions:[], is_frozen:false},\n"
            "    {index:1, element:'H', x:0, y:0, z:0.7, regions:[], is_frozen:false}]);\n"
            "  setTimeout(() => {\n"
            "    let n = 0;\n"
            "    window.molbuilder.workspace.persist = function () { n++; };\n"
            "    window.molbuilder.selection.store.setSelection([1]);\n"   # VIEW op
            "    setTimeout(() => {\n"
            "      console.log(JSON.stringify({ selectionPersisted: n > 0 }));\n"
            "    }, 200);\n"
            "  }, 200);\n"
            "}, 200);")
        assert out["selectionPersisted"] is False   # selection = VIEW -> zero writes (§18.3)

    @staticmethod
    def _boot_one_frame() -> str:
        """Load a 1-atom structure and open the async block the xfail
        snippets continue (they close the trailing ``}, 200);``)."""
        return (
            _FAKE_EMBED +
            "const d = window.molbuilder.molview.data;\n"
            "window.molbuilder.selection.store.adoptAtoms(\n"
            "  [{index:0, element:'H', x:0, y:0, z:0, regions:[], is_frozen:false}]);\n"
            "window.molbuilder.structureCanvas.setStructure(\n"
            "  {source_format:'xyz', text:'1\\n\\nH 0 0 0\\n'}, {kind:'x', file:null});\n"
            "setTimeout(() => {\n"
        )


# --------------------------------------------------------------------- #
#  §19.5 The state timeline — one save/load, index-delta parameterized   #
# --------------------------------------------------------------------- #

# An in-memory stand-in for the workspace's on-disk state timeline + the
# sessionStorage mirror.  The real ws.persist / readState /
# readPersistedSnapshot / pruneStatesAbove are fetch/storage-backed (inert
# under Node); these stubs capture the identities the data model sends and
# serve them back, emulating the tail-delete so the timeline round-trips
# like production.  ``persist(sessionBytes, snapshotBlob, id)`` writes
# ``sessionBytes`` to the MIRROR always AND ``snapshotBlob`` to the disk
# file for ``id.state_index`` — UNLESS ``snapshotBlob`` is null, which
# writes the MIRROR ONLY (the ``load(delta!=0)`` re-mirror).  At a
# ``save`` both args are the SAME snapshot, so the stub stores the blob
# and hands it straight back.
_STUB_TIMELINE_WS = """
const _disk = {};
let   _mirror = null;
const _persistIdx = [];
const _readIdx = [];
const _pruneIdx = [];
const _wsStub = window.molbuilder.workspace;
_wsStub.workspaceId = function () { return 'ws-test'; };
_wsStub.persist = function (sb, blob, id) {
    _persistIdx.push(id.state_index);
    _mirror = sb;                                    // the sessionStorage MIRROR (always)
    if (blob != null) _disk[id.state_index] = blob;  // null => MIRROR ONLY (skip disk)
    return Promise.resolve();
};
_wsStub.readState = function (id) {
    _readIdx.push(id.state_index);
    return Promise.resolve(_disk[id.state_index] != null ? _disk[id.state_index] : null);
};
_wsStub.readPersistedSnapshot = function () { return _mirror; };
_wsStub.pruneStatesAbove = function (wid, above) {
    _pruneIdx.push(above);
    Object.keys(_disk).forEach(function (k) { if (Number(k) > above) delete _disk[k]; });
    return Promise.resolve();
};
"""


class TestStateTimeline:
    """molview-module.md §19.5: the model owns a ``state_index`` (0 = the
    opened anchor) and each index is a full ``getState()`` session
    snapshot on disk.  ``installMolecule`` anchors a fresh timeline;
    ``save(1)`` commits an undoable checkpoint (advance + persist +
    tail-delete); ``load(-1)`` retracts (read index-1, apply, decrement,
    floor at 0); ``load(0)`` reloads from the mirror.  Persistence is
    push-only; save/load are serialized.  Built against the workspace
    interface (persist / readState / readPersistedSnapshot /
    pruneStatesAbove / workspaceId), stubbed here."""

    def test_openMolecule_anchors_index_zero(self):
        """§19.5: after ``installMolecule`` the timeline is reset — index 0,
        the anchor snapshot persisted at index 0, the whole prior timeline
        pruned (above_index = -1)."""
        out = _run_node(
            _STUB_TIMELINE_WS + _STUB_WATER_FETCH +
            "const d = window.molbuilder.molview.data;\n"
            "d.installMolecule({ text:'unused', filename:'/p/water.xyz' }).then(() => {\n"
            "  console.log(JSON.stringify({\n"
            "    index:       d.state_index,\n"
            "    uncommitted: d.uncommitted,\n"
            "    persistIdx:  _persistIdx,\n"
            "    pruneIdx:    _pruneIdx,\n"
            "    diskHas0:    _disk[0] != null,\n"
            "    mirrorHas0:  _mirror != null && _mirror.state.state_index === 0,\n"
            "  }));\n"
            "});")
        assert out["index"] == 0
        assert out["uncommitted"] is False
        assert out["persistIdx"] == [0]      # the ONE automatic write, at index 0
        assert out["pruneIdx"] == [-1]       # clear the whole prior timeline
        assert out["diskHas0"] is True
        assert out["mirrorHas0"] is True     # the anchor is mirrored too (reload target)

    def test_save_checkpoint_advances_persists_and_prunes_tail(self):
        """§19.5: ``save(1)`` persists the snapshot at index+1 (mirror +
        disk), then on success advances the index and tail-deletes every
        index above."""
        out = _run_node(
            _STUB_TIMELINE_WS + _STUB_WATER_FETCH +
            "const d = window.molbuilder.molview.data;\n"
            "d.installMolecule({ text:'unused', filename:'/p/water.xyz' }).then(() => {\n"
            "  window.molbuilder.structureCanvas.markDirty();\n"   # an edit -> uncommitted
            "  const before = d.uncommitted;\n"
            "  return d.save(1).then(() => {\n"
            "    console.log(JSON.stringify({\n"
            "      before:      before,\n"
            "      index:       d.state_index,\n"
            "      uncommitted: d.uncommitted,\n"
            "      persistIdx:  _persistIdx,\n"
            "      pruneIdx:    _pruneIdx,\n"
            "      diskHas1:    _disk[1] != null,\n"
            "    }));\n"
            "  });\n"
            "});")
        assert out["before"] is True             # the edit marked the model uncommitted
        assert out["index"] == 1                 # advanced only after the persist resolved
        assert out["uncommitted"] is False       # the checkpoint cleared it
        assert out["persistIdx"] == [0, 1]       # anchor, then the checkpoint
        assert out["pruneIdx"] == [-1, 1]        # anchor-clear, then the divergent-tail delete
        assert out["diskHas1"] is True

    def test_save0_resaves_current_index_without_pruning(self):
        """§19.5: ``save(0)`` re-saves the current index in place — no
        advance, and delta==0 does NOT tail-delete."""
        out = _run_node(
            _STUB_TIMELINE_WS + _STUB_WATER_FETCH +
            "const d = window.molbuilder.molview.data;\n"
            "d.installMolecule({ text:'unused', filename:'/p/water.xyz' }).then(() => {\n"
            "  return d.save(0).then(() => {\n"
            "    console.log(JSON.stringify({\n"
            "      index:      d.state_index,\n"
            "      persistIdx: _persistIdx,\n"
            "      pruneIdx:   _pruneIdx,\n"
            "    }));\n"
            "  });\n"
            "});")
        assert out["index"] == 0                 # save(0) does not move the index
        assert out["persistIdx"] == [0, 0]       # anchor, then the in-place re-save
        assert out["pruneIdx"] == [-1]           # delta==0 -> NO extra prune

    def test_reload_restores_full_selection_not_just_indices(self):
        """§19.2/§19.5: getSelection() snapshots the WHOLE selection state and
        load(0) restores it -- isolate + k-grid (the VIEW toggles), the click
        ORDER (pickOrder = angle vertex), and mode/filters/combinator -- NOT just
        indices.  Regression: "Show selected only" / k-grid silently reset to
        default on every reload / Retract because only indices were restored."""
        out = _run_node(
            _STUB_TIMELINE_WS + _STUB_WATER_FETCH +
            "const d = window.molbuilder.molview.data;\n"
            "const store = window.molbuilder.selection.store;\n"
            "d.installMolecule({ text:'x', filename:'/p/water.xyz' }).then(() => {\n"
            "  store.setSelection([2, 1]);\n"                       # pickOrder [2,1] -> vertex 1
            "  store.setIsolate(true);\n"
            "  return d.save(0).then(() => {\n"                     # persist the snapshot
            "    store.setSelection([]);\n"                          # stomp the LIVE store
            "    store.setIsolate(false);\n"
            "    const stomped = d.getSelection();\n"
            "    return d.load(0).then(() => {\n"                    # reload-restore from mirror
            "      const after = d.getSelection();\n"
            "      console.log(JSON.stringify({\n"
            "        stomped_isolate:  stomped.isolate,\n"
            "        after_indices:    after.indices,\n"
            "        after_pickOrder:  after.pickOrder,\n"
            "        after_isolate:    after.isolate,\n"
            "      }));\n"
            "    });\n"
            "  });\n"
            "});")
        assert out["stomped_isolate"] is False       # sanity: the live store WAS stomped
        assert out["after_indices"] == [1, 2]        # selection restored (sorted set)
        assert out["after_pickOrder"] == [2, 1]      # click ORDER restored (angle vertex)
        assert out["after_isolate"] is True          # the bug: isolate survives the reload

    def test_save_does_not_advance_when_persist_rejects(self):
        """§19.5 atomic: a failed write never leaves state_index pointing
        at a missing file — the index stays put on a rejected persist."""
        out = _run_node(
            "const d = window.molbuilder.molview.data;\n"
            "const ws = window.molbuilder.workspace;\n"
            "ws.workspaceId = () => 'ws-test';\n"
            "ws.pruneStatesAbove = () => Promise.resolve();\n"
            "ws.persist = () => Promise.reject(new Error('disk full'));\n"
            "d.save(1).then(\n"
            "  () => console.log(JSON.stringify({ rejected:false, index:d.state_index })),\n"
            "  () => console.log(JSON.stringify({ rejected:true,  index:d.state_index })));")
        assert out["rejected"] is True
        assert out["index"] == 0                 # unchanged — no advance on failure

    def test_load_minus1_while_dirty_reverts_to_current_checkpoint(self):
        """§19.5 Retract semantics: with UNCOMMITTED edits on top of a
        checkpoint, ``load(-1)`` reverts those edits by re-applying the CURRENT
        checkpoint -- it does NOT step past a saved checkpoint.  The uncommitted
        edit consumes the first retract step; the index stays put, uncommitted
        clears.  (Only a subsequent Retract, now clean, steps back -- see
        ``test_load_minus1_while_clean_decrements`` below.)"""
        out = _run_node(
            _STUB_TIMELINE_WS + _STUB_WATER_FETCH +
            "const d = window.molbuilder.molview.data;\n"
            "d.installMolecule({ text:'unused', filename:'/p/water.xyz' }).then(() => {\n"
            "  return d.save(1);\n"                              # index 1 (disk[1]=water)
            "}).then(() => {\n"
            # An UNCOMMITTED edit: replace the live model with a 1-atom carbon.
            "  window.molbuilder.structureCanvas.setStructure(\n"
            "    {source_format:'xyz', text:'1\\nc\\nC 0 0 0\\n'}, {kind:'file', file:'/p/c.xyz'});\n"
            "  window.molbuilder.selection.store.adoptAtoms(\n"
            "    [{index:0, element:'C', x:0, y:0, z:0, regions:[], is_frozen:false}]);\n"
            "  const beforeEls = d.getElements();\n"
            "  return d.load(-1).then(() => {\n"
            "    console.log(JSON.stringify({\n"
            "      beforeEls: beforeEls,\n"
            "      afterEls:  d.getElements(),\n"
            "      head:      d.getStructure().text.split('\\n')[0],\n"
            "      index:     d.state_index,\n"
            "      readIdx:   _readIdx,\n"
            "      uncommitted: d.uncommitted,\n"
            "    }));\n"
            "  });\n"
            "});")
        assert out["beforeEls"] == ["C"]           # the uncommitted edit was live
        assert out["afterEls"] == ["O", "H", "H"]  # reverted to the SAVED index-1 water
        assert out["head"] == "3"                  # whole structure re-applied
        assert out["index"] == 1                   # NOT decremented -- reverted in place
        assert out["readIdx"] == [1]               # read the CURRENT checkpoint, not prev
        assert out["uncommitted"] is False

    def test_load_minus1_while_clean_decrements(self):
        """§19.5 Retract semantics: at a CLEAN checkpoint (no uncommitted
        edits), ``load(-1)`` steps back one checkpoint -- reads {index-1},
        applies it, decrements, and re-mirrors."""
        out = _run_node(
            _STUB_TIMELINE_WS + _STUB_WATER_FETCH +
            "const d = window.molbuilder.molview.data;\n"
            "d.installMolecule({ text:'unused', filename:'/p/water.xyz' }).then(() => {\n"
            "  return d.save(1);\n"                              # index 1 (disk[1]=water); clean
            "}).then(() => {\n"
            "  return d.load(-1).then(() => {\n"                 # clean -> step back to 0
            "    console.log(JSON.stringify({\n"
            "      afterEls:  d.getElements(),\n"
            "      index:     d.state_index,\n"
            "      readIdx:   _readIdx,\n"
            "      uncommitted: d.uncommitted,\n"
            "    }));\n"
            "  });\n"
            "});")
        assert out["afterEls"] == ["O", "H", "H"]  # index-0 anchor water
        assert out["index"] == 0                   # decremented (clean -> step back)
        assert out["readIdx"] == [0]               # read index (state_index - 1)
        assert out["uncommitted"] is False

    def test_load_minus1_is_a_noop_and_floors_at_zero(self):
        """§19.5: ``load(-1)`` at index 0 is a no-op — the target index is
        below the anchor, so no read; index stays 0."""
        out = _run_node(
            _STUB_TIMELINE_WS + _STUB_WATER_FETCH +
            "const d = window.molbuilder.molview.data;\n"
            "d.installMolecule({ text:'unused', filename:'/p/water.xyz' }).then(() => {\n"
            "  return d.load(-1).then(() => {\n"
            "    console.log(JSON.stringify({ index:d.state_index, readIdx:_readIdx }));\n"
            "  });\n"
            "});")
        assert out["index"] == 0
        assert out["readIdx"] == []              # floored at 0 -> the timeline was never read

    def test_load0_restores_from_the_mirror(self):
        """§19.5: ``load(0)`` reloads from the sessionStorage MIRROR (the
        current committed snapshot), discarding uncommitted changes —
        the reload / mount-restore primitive; no disk read."""
        out = _run_node(
            _STUB_TIMELINE_WS + _STUB_WATER_FETCH +
            "const d = window.molbuilder.molview.data;\n"
            "d.installMolecule({ text:'unused', filename:'/p/water.xyz' }).then(() => {\n"
            # An UNCOMMITTED edit: replace the live model with a 1-atom carbon.
            "  window.molbuilder.structureCanvas.setStructure(\n"
            "    {source_format:'xyz', text:'1\\nc\\nC 0 0 0\\n'}, {kind:'file', file:'/p/c.xyz'});\n"
            "  window.molbuilder.selection.store.adoptAtoms(\n"
            "    [{index:0, element:'C', x:0, y:0, z:0, regions:[], is_frozen:false}]);\n"
            "  const beforeEls = d.getElements();\n"
            "  return d.load(0).then(() => {\n"
            "    console.log(JSON.stringify({\n"
            "      beforeEls: beforeEls,\n"
            "      afterEls:  d.getElements(),\n"
            "      index:     d.state_index,\n"
            "      readIdx:   _readIdx,\n"           # load(0) reads the MIRROR, never the disk
            "      uncommitted: d.uncommitted,\n"
            "    }));\n"
            "  });\n"
            "});")
        assert out["beforeEls"] == ["C"]         # the uncommitted edit was live
        assert out["afterEls"] == ["O", "H", "H"]  # load(0) restored the mirrored water
        assert out["index"] == 0                 # adopts the mirror's own state_index
        assert out["readIdx"] == []              # mirror read, not a disk readState
        assert out["uncommitted"] is False

    def test_uncommitted_flips_on_edit_and_clears_after_save_and_load(self):
        """§19.5: ``uncommitted`` is true iff the model changed since the
        last ``save``; installMolecule / save / load all clear it."""
        out = _run_node(
            _STUB_TIMELINE_WS + _STUB_WATER_FETCH +
            "const d = window.molbuilder.molview.data;\n"
            "d.installMolecule({ text:'unused', filename:'/p/water.xyz' }).then(() => {\n"
            "  const afterOpen = d.uncommitted;\n"               # false (anchor)
            "  d.reloadFrames([[[0,0,0],[0.9,0,0],[-0.2,0.9,0]]]);\n"  # frame DATA edit
            "  const afterEdit = d.uncommitted;\n"                # true
            "  return d.save(1).then(() => {\n"
            "    const afterSave = d.uncommitted;\n"             # false
            "    d.reloadFrames([[[0,0,0],[0.8,0,0],[-0.3,0.8,0]]]);\n"  # edit again
            "    const afterEdit2 = d.uncommitted;\n"            # true
            "    return d.load(-1).then(() => {\n"
            "      console.log(JSON.stringify({\n"
            "        afterOpen, afterEdit, afterSave, afterEdit2,\n"
            "        afterLoad: d.uncommitted }));\n"
            "    });\n"
            "  });\n"
            "});")
        assert out["afterOpen"] is False
        assert out["afterEdit"] is True
        assert out["afterSave"] is False
        assert out["afterEdit2"] is True
        assert out["afterLoad"] is False         # load(-1) discards + clears uncommitted

    def test_save_and_load_are_serialized(self):
        """§19.5: save/load are serialized through a queue — the index
        advances one at a time even when an earlier write resolves LATER
        than a later one (each op computes its index only when it runs)."""
        out = _run_node(
            "const d = window.molbuilder.molview.data;\n"
            "const ws = window.molbuilder.workspace;\n"
            "ws.workspaceId = () => 'ws-test';\n"
            "ws.pruneStatesAbove = () => Promise.resolve();\n"
            "const order = [];\n"
            # Make the FIRST checkpoint's write resolve slower than the second's:
            # if the two ran concurrently, index 2 would land before index 1.
            "ws.persist = (sb, blob, id) => new Promise((res) => {\n"
            "  setTimeout(() => { order.push(id.state_index); res(); },\n"
            "            id.state_index === 1 ? 120 : 10);\n"
            "});\n"
            "d.save(1);\n"
            "d.save(1).then(() => {\n"
            "  console.log(JSON.stringify({ order:order, index:d.state_index }));\n"
            "});")
        assert out["order"] == [1, 2]            # first-in first-out, not by write speed
        assert out["index"] == 2                 # sequential advance, no interleave


class TestAnchorTimelineDurableWrites:
    """Regression (2026-07-14): the state-timeline anchor write must be
    RACE-FREE.

    ``installMolecule``'s index-0 anchor intermittently vanished (~20% of runs),
    so a later Retract to index 0 read a missing file and no-op'd -- the flaky
    "retract never returns to the opened state" hang.  Root cause: within
    ``_anchorTimeline`` the two server calls had NO ordering between them --
    ``pruneStatesAbove(-1)`` (delete the WHOLE timeline) and the index-0 write
    were issued concurrently, so on the threaded server a late-landing
    delete-all could wipe the just-written anchor.

    FIX (ordering only): await the prune-all, THEN issue the anchor write.
    ``persist`` stays best-effort/fire-and-forget (workspace-contract: the
    on-disk state file is crash-safety, not a blocking write) -- making it
    awaitable would force ``installMolecule`` to block on the durable write, which
    is neither needed for the race nor free (it stalls the generation flow).

    Pin: the anchor's prune-all completes STRICTLY BEFORE the index-0 write is
    issued."""

    def test_anchor_prunes_all_before_writing_index0(self):
        out = _run_node(
            # Instrument fetch: /build/load returns water; prune-states resolves
            # on a LATER macrotask (so a CONCURRENT write-0 would be issued while
            # prune is still pending); write-state records whether prune had
            # already resolved when the index-0 write was issued.
            "global.__o = { pruneResolved:false, writeAfterPrune:null,\n"
            "               above:null };\n"
            "global.window.fetch = function (url, opts) {\n"
            "  var b = {}; try { b = JSON.parse((opts&&opts.body)||'{}'); } catch(e){}\n"
            "  if (url.indexOf('/api/build/load') >= 0) {\n"
            "    return Promise.resolve({ ok:true, json: function(){ return Promise.resolve({\n"
            "      ok:true, source_format:'xyz', title:'h2o', n_atoms:3,\n"
            "      text:'3\\\\nh2o\\\\nO 0 0 0\\\\nH 0.957 0 0\\\\nH -0.24 0.927 0\\\\n',\n"
            "      atoms:[{index:0,element:'O',x:0,y:0,z:0,regions:[],is_frozen:false},\n"
            "             {index:1,element:'H',x:0.957,y:0,z:0,regions:[],is_frozen:false},\n"
            "             {index:2,element:'H',x:-0.24,y:0.927,z:0,regions:[],is_frozen:false}] }); } });\n"
            "  }\n"
            "  if (url.indexOf('/api/state-timeline/prune') >= 0) {\n"
            "    global.__o.above = b.above_index;\n"
            "    return new Promise(function (res) { setTimeout(function () {\n"
            "      global.__o.pruneResolved = true;\n"
            "      res({ ok:true, json:function(){ return Promise.resolve({ ok:true, removed:0 }); } });\n"
            "    }, 25); });\n"
            "  }\n"
            "  if (url.indexOf('/api/state-timeline/write') >= 0) {\n"
            "    if (b.state_index === 0 && global.__o.writeAfterPrune === null) {\n"
            "      global.__o.writeAfterPrune = global.__o.pruneResolved;\n"
            "    }\n"
            "    return Promise.resolve({ ok:true, json:function(){ return Promise.resolve({ ok:true }); } });\n"
            "  }\n"
            "  return Promise.resolve({ ok:true, json:function(){ return Promise.resolve({ ok:true }); } });\n"
            "};\n"
            "(async () => {\n"
            "  await window.molbuilder.molview.data.installMolecule({\n"
            "    text:'3\\\\nh2o\\\\nO 0 0 0\\\\nH 0.957 0 0\\\\nH -0.24 0.927 0\\\\n',\n"
            "    filename:'w.xyz' });\n"
            "  console.log(JSON.stringify(global.__o));\n"
            "})();")
        assert out["above"] == -1, (
            "_anchorTimeline must prune the WHOLE timeline (above_index=-1) "
            "before writing the anchor."
        )
        assert out["writeAfterPrune"] is True, (
            "the index-0 anchor write was issued BEFORE pruneStatesAbove(-1) "
            "resolved -- the delete-all races the anchor write and can wipe it. "
            "_anchorTimeline must await the prune, THEN persist the anchor."
        )


class TestPersistErrorIsExplicit:
    """The state write is NON-BLOCKING but ERROR-EXPLICIT (workspace-contract
    §4.7): persist() fires the write-state POST fire-and-forget, but a failure
    -- a rejected fetch (network) OR a non-2xx response (server refused) -- is
    NEVER swallowed.  It reaches ``ws.onPersistError(handler)`` subscribers.

    This replaces the old silent ``.catch(() => {})`` that let a failed anchor
    write masquerade as a mysterious downstream hang."""

    def _run_persist(self, fetch_body):
        return _run_node(
            "var got = [];\n"
            "window.molbuilder.workspace.onPersistError(function (d) { got.push(d); });\n"
            + fetch_body +
            "window.molbuilder.workspace.persist({s:1}, {blob:1},\n"
            "  { workspace_id:'w', state_index:3 });\n"
            # wait a macrotask for the write-state fetch .then/.catch to run.
            "setTimeout(function () { console.log(JSON.stringify({ got: got })); }, 40);")

    def test_non_2xx_write_surfaces_via_onPersistError(self):
        out = self._run_persist(
            "global.window.fetch = function (url) {\n"
            "  var isWrite = String(url).indexOf('/state-timeline/write') >= 0;\n"
            "  return Promise.resolve({ ok: !isWrite, status: isWrite ? 500 : 200,\n"
            "    json: function () { return Promise.resolve({ ok: !isWrite }); } });\n"
            "};\n")
        assert len(out["got"]) == 1, out
        assert out["got"][0]["op"] == "write-state"
        assert out["got"][0]["status"] == 500
        assert out["got"][0]["state_index"] == 3

    def test_network_reject_write_surfaces_via_onPersistError(self):
        out = self._run_persist(
            "global.window.fetch = function () {\n"
            "  return Promise.reject(new TypeError('Failed to fetch'));\n"
            "};\n")
        assert len(out["got"]) == 1, out
        assert out["got"][0]["op"] == "write-state"
        assert "Failed to fetch" in out["got"][0]["error"]

    def test_successful_write_does_not_fire(self):
        out = self._run_persist(
            "global.window.fetch = function () {\n"
            "  return Promise.resolve({ ok: true, status: 200,\n"
            "    json: function () { return Promise.resolve({ ok: true }); } });\n"
            "};\n")
        assert out["got"] == [], out


# --------------------------------------------------------------------- #
