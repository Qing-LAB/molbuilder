"""Unit tests for ``molbuilder/web/static/lib/workspace/dispatcher.js``.

Phase 4 of the workspace-state migration (docs/protocols/
workspace-state.md § 4.2 + § 7).  The dispatcher is the single
client-side entry point for the Molbuilder tab's workspace state.

This phase pins:

* TestPublicSurface — every method named in § 4.2 exists with
  the right kind (function vs sub-namespace object).
* TestSubscribe — subscribers fire on registration and on
  underlying-store changes (fan-in from canvas-state + selection
  store + future view-state notifications).
* TestPersistRoundtrip — the dispatcher's ``getState()`` snapshot
  is a defensive copy; mutating it does NOT leak into the
  underlying stores.  Phase 8 of the migration will add the
  unified sessionStorage key and extend this with a round-trip
  through it.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DISPATCHER_PATH = ROOT / "molbuilder/web/static/lib/workspace/dispatcher.js"
STORE_PATH      = ROOT / "molbuilder/web/static/lib/selection/store.js"
CANVAS_PATH     = ROOT / "molbuilder/web/static/lib/structure/canvas-state.js"


def _run_node(snippet: str) -> object:
    """Run a Node.js snippet under a minimal stub environment that
    mounts the legacy stores ahead of the dispatcher.

    Mocks ``window`` + ``document`` + ``sessionStorage`` so the
    canvas-state + selection store + dispatcher IIFEs can load
    without a real browser.  Tests then drive the public surface
    via ``window.molbuilder.workspace.*`` and assert on JSON
    snapshots emitted to stdout.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")

    bootstrap = """
        // Minimal browser-shim for the IIFE stores.
        const _events = {};
        global.window = global;
        global.document = {
            readyState: "complete",
            addEventListener: () => {},
            getElementById:  () => null,
        };
        const _storage = {};
        global.sessionStorage = {
            getItem:   (k) => (_storage[k] == null ? null : _storage[k]),
            setItem:   (k, v) => { _storage[k] = String(v); },
            removeItem: (k) => { delete _storage[k]; },
        };
        global.molbuilder = global.molbuilder || {};
        global.window.molbuilder = global.molbuilder;
        // Minimal runtime registry stub — the dispatcher uses it
        // for whenReady("modify.postOp") + similar.
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

        // Load the canvas-state + selection store IIFEs (the
        // legacy stores the dispatcher delegates to).
        require({DISPATCHER_DEPS_LOAD});

        // Tests append their snippet here.
        {SNIPPET}
    """.replace("{DISPATCHER_DEPS_LOAD}", json.dumps(str(CANVAS_PATH))) \
       .replace("{SNIPPET}", snippet)

    # canvas-state, selection store, AND dispatcher all load via
    # the same require chain — keep it explicit.
    # canvas-state ships a UMD branch: in CommonJS it exports via
    # ``module.exports`` and skips the ``window.molbuilder.*`` mount
    # the browser would do.  Manually mount it so the dispatcher's
    # ``_canvas()`` resolver finds it.  selection store mounts
    # unconditionally on the window — just require it.  Dispatcher
    # mounts BOTH module.exports AND window.molbuilder.workspace
    # (Phase 4 UMD-ish update).
    bootstrap = (
        "window.molbuilder.structureCanvas = require("
        + json.dumps(str(CANVAS_PATH)) + ");\n"
        "require(" + json.dumps(str(STORE_PATH))      + ");\n"
        "require(" + json.dumps(str(DISPATCHER_PATH)) + ");\n"
        + snippet
    )
    header = """
        const _events = {};
        global.window = global;
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


# --------------------------------------------------------------------- #
#  TestPublicSurface — every method exists with the right shape         #
# --------------------------------------------------------------------- #


class TestPublicSurface:
    """Pin the exact public surface from workspace-state.md § 4.2.

    Adding a new public method here updates the protocol doc AND
    this test class — drift between the two breaks the test.
    """

    def test_workspace_namespace_exists_on_window(self):
        out = _run_node(
            "console.log(JSON.stringify("
            "  typeof window.molbuilder.workspace));")
        assert out == "object"

    def test_top_level_methods_are_functions(self):
        out = _run_node(
            "const ws = window.molbuilder.workspace;\n"
            "console.log(JSON.stringify({\n"
            "  subscribe:    typeof ws.subscribe,\n"
            "  getState:     typeof ws.getState,\n"
            "  getStructure: typeof ws.getStructure,\n"
            "  getSource:    typeof ws.getSource,\n"
            "  getSelection: typeof ws.getSelection,\n"
            "  isDirty:      typeof ws.isDirty,\n"
            "  isEmpty:      typeof ws.isEmpty,\n"
            "  loadFromFile: typeof ws.loadFromFile,\n"
            "  generate:     typeof ws.generate,\n"
            "  applyOp:      typeof ws.applyOp,\n"
            "  save:         typeof ws.save,\n"
            "  discard:      typeof ws.discard,\n"
            "  undo:         typeof ws.undo,\n"
            "}));"
        )
        for k, v in out.items():
            assert v == "function", f"{k} is {v!r}; expected function"

    def test_selection_sub_namespace_methods_are_functions(self):
        out = _run_node(
            "const sel = window.molbuilder.workspace.selection;\n"
            "console.log(JSON.stringify({\n"
            "  toggle:        typeof sel.toggle,\n"
            "  set:           typeof sel.set,\n"
            "  add:           typeof sel.add,\n"
            "  remove:        typeof sel.remove,\n"
            "  all:           typeof sel.all,\n"
            "  invert:        typeof sel.invert,\n"
            "  clear:         typeof sel.clear,\n"
            "  setMode:       typeof sel.setMode,\n"
            "  setFilters:    typeof sel.setFilters,\n"
            "  setCombinator: typeof sel.setCombinator,\n"
            "  applyFilter:   typeof sel.applyFilter,\n"
            "  writeLabel:    typeof sel.writeLabel,\n"
            "}));"
        )
        for k, v in out.items():
            assert v == "function", f"selection.{k} is {v!r}"

    def test_view_sub_namespace_methods_are_functions(self):
        out = _run_node(
            "const view = window.molbuilder.workspace.view;\n"
            "console.log(JSON.stringify({\n"
            "  applyState: typeof view.applyState,\n"
            "  getState:   typeof view.getState,\n"
            "}));"
        )
        for k, v in out.items():
            assert v == "function", f"view.{k} is {v!r}"


# --------------------------------------------------------------------- #
#  TestSubscribe — fan-in from the legacy stores                        #
# --------------------------------------------------------------------- #


class TestSubscribe:
    """The dispatcher's subscribe fans in changes from the
    underlying canvas-state + selection store.  A subscriber
    fires once on registration (current state) AND on every
    subsequent mutation.
    """

    def test_subscribe_fires_once_on_registration_with_current_state(self):
        out = _run_node(
            "const ws = window.molbuilder.workspace;\n"
            "let calls = 0; let last = null;\n"
            "ws.subscribe((s) => { calls++; last = s; });\n"
            "console.log(JSON.stringify({\n"
            "  calls: calls,\n"
            "  empty: last.structure === null,\n"
            "}));"
        )
        assert out["calls"] == 1
        assert out["empty"] is True   # fresh dispatcher, no canvas

    def test_subscribe_fires_on_selection_store_mutation(self):
        out = _run_node(
            "const ws = window.molbuilder.workspace;\n"
            "let calls = 0;\n"
            "// Skip the on-subscribe fire.\n"
            "ws.subscribe(() => { calls++; });\n"
            "Promise.resolve().then(() => {\n"
            "  calls = 0;\n"
            "  // Drive the selection store directly — the dispatcher\n"
            "  // forwards its subscribe to the store.\n"
            "  window.molbuilder.selection.store.adoptAtoms([\n"
            "    {index: 0, element: 'C', regions: [], is_frozen: false},\n"
            "  ]);\n"
            "  return Promise.resolve();\n"
            "}).then(() => {\n"
            "  console.log(JSON.stringify(calls));\n"
            "});"
        )
        # Selection store batches via microtask + dispatcher forwards
        # 1:1 — exactly one notification per adoptAtoms call.
        assert out == 1

    def test_unsubscribe_stops_notifications(self):
        out = _run_node(
            "const ws = window.molbuilder.workspace;\n"
            "let calls = 0;\n"
            "const unsub = ws.subscribe(() => { calls++; });\n"
            "Promise.resolve().then(() => {\n"
            "  calls = 0;\n"
            "  unsub();\n"
            "  window.molbuilder.selection.store.adoptAtoms([\n"
            "    {index: 0, element: 'C', regions: [], is_frozen: false},\n"
            "  ]);\n"
            "  return Promise.resolve();\n"
            "}).then(() => {\n"
            "  console.log(JSON.stringify(calls));\n"
            "});"
        )
        assert out == 0

    def test_subscriber_errors_do_not_wedge_the_dispatcher(self):
        """A subscriber that throws must not stop other
        subscribers from receiving the notification."""
        out = _run_node(
            "const ws = window.molbuilder.workspace;\n"
            "let good = 0;\n"
            "ws.subscribe(() => { throw new Error('boom'); });\n"
            "ws.subscribe(() => { good++; });\n"
            "Promise.resolve().then(() => {\n"
            "  good = 0;\n"
            "  window.molbuilder.selection.store.adoptAtoms([\n"
            "    {index: 0, element: 'C', regions: [], is_frozen: false},\n"
            "  ]);\n"
            "  return Promise.resolve();\n"
            "}).then(() => {\n"
            "  console.log(JSON.stringify(good));\n"
            "});"
        )
        # The "good" subscriber still fires once.
        assert out == 1


# --------------------------------------------------------------------- #
#  TestReads — synthesised state from the legacy stores                 #
# --------------------------------------------------------------------- #


class TestReads:
    """The dispatcher's getState() / getStructure() / getSource() /
    isDirty() / isEmpty() / getSelection() synthesise from the
    legacy stores."""

    def test_initial_state_is_empty(self):
        out = _run_node(
            "const ws = window.molbuilder.workspace;\n"
            "console.log(JSON.stringify({\n"
            "  isEmpty:   ws.isEmpty(),\n"
            "  isDirty:   ws.isDirty(),\n"
            "  structure: ws.getStructure(),\n"
            "}));"
        )
        assert out["isEmpty"] is True
        assert out["isDirty"] is False
        assert out["structure"] is None

    def test_getStructure_reads_canvas_text_after_setStructure(self):
        out = _run_node(
            "const ws = window.molbuilder.workspace;\n"
            "const cs = window.molbuilder.structureCanvas;\n"
            "cs.setStructure(\n"
            "  {source_format: 'xyz', text: '3\\nh2o\\nO 0 0 0\\nH 0.957 0 0\\nH -0.24 0.927 0\\n'},\n"
            "  {kind: 'file', file: '/projects/x/water.xyz'},\n"
            ");\n"
            "const s = ws.getStructure();\n"
            "console.log(JSON.stringify({\n"
            "  fmt:  s.source_format,\n"
            "  head: s.text.split('\\n')[0],\n"
            "  isEmpty: ws.isEmpty(),\n"
            "}));"
        )
        assert out["fmt"] == "xyz"
        assert out["head"] == "3"
        assert out["isEmpty"] is False

    def test_getSource_returns_kind_file_generator_input(self):
        out = _run_node(
            "const ws = window.molbuilder.workspace;\n"
            "const cs = window.molbuilder.structureCanvas;\n"
            "cs.setStructure(\n"
            "  {source_format: 'xyz', text: '1\\nC\\nC 0 0 0\\n'},\n"
            "  {kind: 'smiles', generator_input: {smiles: 'C'}},\n"
            ");\n"
            "console.log(JSON.stringify(ws.getSource()));"
        )
        assert out["kind"] == "smiles"
        assert out["file"] is None
        assert out["generator_input"] == {"smiles": "C"}


# --------------------------------------------------------------------- #
#  TestPersistRoundtrip — defensive-copy contract                       #
# --------------------------------------------------------------------- #


class TestPersistRoundtrip:
    """``ws.getState()`` returns a defensive snapshot — mutating
    it must not leak into the underlying stores.  Phase 8 of the
    migration adds the unified sessionStorage key; this class
    extends to cover the round-trip then."""

    def test_getState_selection_indices_is_a_copy(self):
        out = _run_node(
            "const ws = window.molbuilder.workspace;\n"
            "window.molbuilder.selection.store.adoptAtoms([\n"
            "  {index: 0, element: 'O', regions: [], is_frozen: false},\n"
            "  {index: 1, element: 'H', regions: [], is_frozen: false},\n"
            "]);\n"
            "Promise.resolve().then(() => "
            "  window.molbuilder.selection.store.setSelection([0, 1])\n"
            ").then(() => {\n"
            "  const snap = ws.getState();\n"
            "  snap.selection.indices.push(999);   // mutate the copy\n"
            "  const refetched = ws.getState();\n"
            "  console.log(JSON.stringify(refetched.selection.indices));\n"
            "});"
        )
        assert out == [0, 1]

    def test_storage_key_is_v1(self):
        out = _run_node(
            "console.log(JSON.stringify(window.molbuilder.workspace.STORAGE_KEY));")
        assert out == "molbuilder.workspace.v1"

    def test_persist_writes_to_unified_sessionStorage_key(self):
        """Phase 8 — on every state change the dispatcher writes
        a debounced snapshot to ``molbuilder.workspace.v1``.  We
        force a synchronous write via the dispatcher's pagehide
        flush path by simulating a state change and waiting for
        the debounce timer."""
        out = _run_node(
            "const ws = window.molbuilder.workspace;\n"
            "window.molbuilder.selection.store.adoptAtoms([\n"
            "  {index: 0, element: 'O', regions: [], is_frozen: false},\n"
            "]);\n"
            "// adoptAtoms fires _notify() in the store, which fans\n"
            "// into the dispatcher's _notify → _schedulePersist.\n"
            "// Wait for the 100ms debounce.\n"
            "setTimeout(() => {\n"
            "  const raw = sessionStorage.getItem(ws.STORAGE_KEY);\n"
            "  const parsed = JSON.parse(raw);\n"
            "  console.log(JSON.stringify({\n"
            "    v:     parsed.v,\n"
            "    atoms: parsed.state.selection.indices.length === 0,\n"
            "    hasSelectionSlot: !!parsed.state.selection,\n"
            "    hasSourceSlot:    !!parsed.state.source,\n"
            "  }));\n"
            "}, 150);"
        )
        assert out == {
            "v": 1,
            "atoms": True,
            "hasSelectionSlot": True,
            "hasSourceSlot":    True,
        }

    def test_readPersistedSnapshot_returns_null_when_no_key(self):
        out = _run_node(
            "console.log(JSON.stringify(\n"
            "  window.molbuilder.workspace.readPersistedSnapshot()));"
        )
        assert out is None

    def test_readPersistedSnapshot_returns_parsed_snapshot(self):
        out = _run_node(
            "const ws = window.molbuilder.workspace;\n"
            "sessionStorage.setItem(ws.STORAGE_KEY, JSON.stringify({\n"
            "  v: 1, saved_at: '2026-06-07T00:00:00Z',\n"
            "  state: {source: {kind: 'smiles', file: null}},\n"
            "}));\n"
            "console.log(JSON.stringify(ws.readPersistedSnapshot()));"
        )
        assert out["v"] == 1
        assert out["state"]["source"]["kind"] == "smiles"

    def test_readPersistedSnapshot_returns_null_on_schema_mismatch(self):
        """Future schema bumps (v=2, v=3, …) invalidate older
        saves cleanly — same back-compat policy as modify-state's
        STATE_SCHEMA_VERSION check.  Pins that the dispatcher
        rejects v=99 rather than silently passing it through."""
        out = _run_node(
            "const ws = window.molbuilder.workspace;\n"
            "sessionStorage.setItem(ws.STORAGE_KEY, JSON.stringify({\n"
            "  v: 99, state: {}\n"
            "}));\n"
            "console.log(JSON.stringify(ws.readPersistedSnapshot()));"
        )
        assert out is None


# --------------------------------------------------------------------- #
#  TestSelectionPassthrough — selection.* mirrors the store             #
# --------------------------------------------------------------------- #


class TestSelectionPassthrough:
    """The dispatcher's selection sub-namespace is a passthrough
    to ``window.molbuilder.selection.store``.  A mutation through
    ``ws.selection.set([...])`` MUST land on the underlying store
    so the existing panel + viewer-adapter (which still subscribe
    to the store directly during the migration window) see the
    change."""

    def test_set_lands_on_underlying_store(self):
        out = _run_node(
            "const ws = window.molbuilder.workspace;\n"
            "window.molbuilder.selection.store.adoptAtoms([\n"
            "  {index: 0, element: 'O', regions: [], is_frozen: false},\n"
            "  {index: 1, element: 'H', regions: [], is_frozen: false},\n"
            "  {index: 2, element: 'H', regions: [], is_frozen: false},\n"
            "]);\n"
            "Promise.resolve().then(() => "
            "  ws.selection.set([0, 2])\n"
            ").then(() => {\n"
            "  console.log(JSON.stringify(\n"
            "    window.molbuilder.selection.store.getState().selection\n"
            "  ));\n"
            "});"
        )
        assert out == [0, 2]

    def test_toggle_via_dispatcher_lands_on_underlying_store(self):
        out = _run_node(
            "const ws = window.molbuilder.workspace;\n"
            "window.molbuilder.selection.store.adoptAtoms([\n"
            "  {index: 0, element: 'O', regions: [], is_frozen: false},\n"
            "  {index: 1, element: 'H', regions: [], is_frozen: false},\n"
            "]);\n"
            "Promise.resolve().then(() => ws.selection.toggle(1))\n"
            ".then(() => {\n"
            "  console.log(JSON.stringify(\n"
            "    window.molbuilder.selection.store.getState().selection\n"
            "  ));\n"
            "});"
        )
        assert out == [1]

    def test_getState_returns_contract_shape_with_indices_not_selection(self):
        """Regression for 2026-06-09 audit BLOCKER:
        ``ws.selection.getState()`` MUST return ``indices`` (contract
        shape, workspace-contract.md §5) — NOT ``selection`` (legacy
        store's internal field name).

        Without this contract, the panel's click handlers (which
        call ``store.getState().indices``) silently broke when the
        legacy store passes ``selection`` to its subscribers.
        """
        out = _run_node(
            "const ws = window.molbuilder.workspace;\n"
            "window.molbuilder.selection.store.adoptAtoms([\n"
            "  {index: 0, element: 'O', regions: [], is_frozen: false},\n"
            "  {index: 1, element: 'H', regions: [], is_frozen: false},\n"
            "]);\n"
            "Promise.resolve().then(() => ws.selection.set([0, 1]))\n"
            ".then(() => {\n"
            "  const s = ws.selection.getState();\n"
            "  console.log(JSON.stringify({\n"
            "    has_indices:      Array.isArray(s.indices),\n"
            "    indices_value:    s.indices,\n"
            "    selection_field:  s.selection,\n"
            "    has_mode:         typeof s.mode === 'string',\n"
            "    has_atoms:        Array.isArray(s.atoms),\n"
            "    has_pickOrder:    Array.isArray(s.pickOrder),\n"
            "    has_sourceFile:   s.sourceFile === null\n"
            "                          || typeof s.sourceFile === 'string',\n"
            "  }));\n"
            "});"
        )
        assert out["has_indices"] is True
        assert out["indices_value"] == [0, 1]
        # Legacy field name MUST NOT leak through.  ``undefined``
        # values are dropped by JSON.stringify, so the contract is
        # satisfied iff the key is absent OR null in the output.
        assert out.get("selection_field") is None
        # Contract shape includes mode/atoms/pickOrder/sourceFile.
        assert out["has_mode"] is True
        assert out["has_atoms"] is True
        assert out["has_pickOrder"] is True
        assert out["has_sourceFile"] is True

    def test_subscribe_callback_receives_contract_shape_not_legacy(self):
        """Regression for the same audit BLOCKER:
        ``ws.selection.subscribe(fn)`` MUST translate the legacy
        store's state shape to the contract shape before passing it
        to ``fn`` — without this wrapper, panel renderers (which
        subscribe + read ``state.indices``) saw ``undefined`` and the
        UI rendered "undefined / 3 atoms" in the count readout.
        """
        out = _run_node(
            "const ws = window.molbuilder.workspace;\n"
            "window.molbuilder.selection.store.adoptAtoms([\n"
            "  {index: 0, element: 'O', regions: [], is_frozen: false},\n"
            "  {index: 1, element: 'H', regions: [], is_frozen: false},\n"
            "]);\n"
            "let received = null;\n"
            "ws.selection.subscribe((s) => { received = s; });\n"
            "Promise.resolve().then(() => ws.selection.set([1]))\n"
            ".then(() => {\n"
            "  console.log(JSON.stringify({\n"
            "    has_indices:    Array.isArray(received && received.indices),\n"
            "    indices_value:  received && received.indices,\n"
            "    legacy_field:   received && received.selection,\n"
            "  }));\n"
            "});"
        )
        assert out["has_indices"] is True
        assert out["indices_value"] == [1]
        # Legacy ``state.selection`` field must NOT appear on the
        # subscriber-delivered shape — that's the whole point of the
        # contract normalisation.  ``undefined`` is dropped by
        # JSON.stringify so the key is absent or null.
        assert out.get("legacy_field") is None
