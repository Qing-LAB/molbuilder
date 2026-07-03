"""Unit tests for ``lib/workspace/_selection-store-impl.js`` driven via Node.

Task #146 — Phase B.1.15.  The selection store is the canonical
state holder for the /modify selection panel (see
``docs/protocols/atom-selection.md`` for the full spec).  Until
this commit it was covered ONLY by the Playwright e2e tests in
``test_molbuilder_e2e.py``, which exercise the store via the panel
DOM -- expensive (~5-10 s per test) and they can't isolate the
store's pure helpers from the DOM/network layers.

These tests use ``node`` with a minimal global stub (no DOM, no
real ``fetch``) so the pure rule-translation helpers and the
synchronous mutators run in <300 ms.  The async HTTP path
(``setSourceFile`` / ``applyFilter`` / ``writeLabel``) is
intentionally NOT covered here -- the server-roundtrip behaviour
needs a real Flask app and is owned by the Playwright tests
already in place.

Test layering per ``docs/protocols/playwright-tests.md`` § 1:
this file is the lowest layer (node JS unit) for the
``selection.store`` module.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT   = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/workspace/_selection-store-impl.js"


def _run_node(snippet: str) -> object:
    """Run a JS snippet under Node with the store module pre-loaded.

    As of Phase 9 (2026-06-13) the module no longer auto-mounts a
    public singleton; the workspace dispatcher owns the one
    process-wide instance.  Tests call
    ``window.molbuilder.selection._createStore()`` to spin up
    fresh isolated instances — that factory stays mounted under
    the existing namespace exactly so test harnesses (here + a
    future Playwright `_test` hook) can build their own.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    bootstrap = """
        global.window = global;
        // No DOM in this test environment.  The store doesn't actually
        // touch the DOM at module init -- only the HTTP wrappers
        // touch ``fetch``, which we stub below.
        global.fetch = () => Promise.resolve({
            ok: false,
            json: () => Promise.resolve({}),
        });
        global.AbortController = function () {
            this.signal = {};
            this.abort = () => {};
        };
        // Some store paths use ``queueMicrotask`` for the notify loop.
        // Node has it built in, but we ensure compatibility with
        // older runtimes.
        if (typeof global.queueMicrotask !== "function") {
            global.queueMicrotask = (cb) => Promise.resolve().then(cb);
        }
    """
    full = bootstrap + "\n" + MODULE.read_text() + "\n" + snippet
    proc = subprocess.run(
        [node, "--input-type=commonjs", "-e", full],
        capture_output=True, text=True, timeout=15,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"node exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\n"
            f"stdout:\n{proc.stdout}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


# Helper: build a "fresh store, set its atoms, then evaluate snippet" wrapper.
def _with_store(setup: str, body: str) -> str:
    """Return a snippet that:
      1. Creates a fresh store via the test-only ``_createStore`` export.
      2. Sets the store's atoms by calling ``adoptSession`` with a
         constructed session payload.  ``setup`` runs first to set
         up test-local data (e.g. atom counts).
      3. Runs ``body`` against the store.
    The snippet must end with a ``console.log(JSON.stringify(OUT))``.

    The store doesn't accept atoms directly via a public mutator
    (the canonical path is ``setSourceFile`` which fetches them); we
    poke ``state.atoms`` via the bypass-only ``adoptSession`` whose
    contract accepts a partial session restore.
    """
    return f"""
        const store = window.molbuilder.selection._createStore();
        {setup}
        {body}
    """


# --------------------------------------------------------------------
# Pure helpers (no state mutation needed)
# --------------------------------------------------------------------


class TestStoreAPISurface:
    """Pin the public API surface so a future refactor that drops a
    method (without bumping the contract version) fails CI."""

    def test_exposes_documented_methods(self):
        """Every method documented in atom-selection.md must be
        present on the store instance.  Catches accidental drops
        + renames before they surface in the Playwright tests."""
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "console.log(JSON.stringify({\n"
            "  reads:         typeof store.getState === 'function' && "
            "                  typeof store.subscribe === 'function',\n"
            "  source_file:   typeof store.setSourceFile === 'function' && "
            "                  typeof store.refreshAtoms === 'function' && "
            "                  typeof store.adoptSession === 'function' && "
            "                  typeof store.adoptAtoms === 'function' && "
            "                  typeof store.setLoader === 'function',\n"
            "  mode:          typeof store.setMode === 'function',\n"
            "  isolate:       typeof store.setIsolate === 'function',\n"
            "  kgrid:         typeof store.setKgrid === 'function',\n"
            "  click_editing: typeof store.toggleAtom === 'function' && "
            "                  typeof store.setSelection === 'function' && "
            "                  typeof store.addToSelection === 'function' && "
            "                  typeof store.removeFromSelection === 'function' && "
            "                  typeof store.selectAll === 'function' && "
            "                  typeof store.invertSelection === 'function' && "
            "                  typeof store.clearSelection === 'function',\n"
            "  filters:       typeof store.setFilters === 'function' && "
            "                  typeof store.addFilter === 'function' && "
            "                  typeof store.removeFilter === 'function' && "
            "                  typeof store.updateFilter === 'function' && "
            "                  typeof store.setCombinator === 'function' && "
            "                  typeof store.applyFilter === 'function',\n"
            "  sidecar:       typeof store.writeLabel === 'function',\n"
            "}));"
        )
        assert all(out.values()), f"missing methods: {out!r}"


class TestInitialState:
    """The initial state shape pins atom-selection.md § 2."""

    def test_initial_state_shape(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "console.log(JSON.stringify(store.getState()));"
        )
        assert out == {
            "sourceFile": None,
            "atoms":      [],
            "selection":  [],
            # ``pickOrder`` was added to the snapshot in task #304
            # (2026-06-09) — the click-order shadow for the
            # measurement-readout angle vertex.  Empty on fresh
            # store; updated in lock-step with selection by every
            # mutator.  See atom-selection.md § 2 + TestPickOrderSnapshot
            # below.
            "pickOrder":  [],
            "mode":       "click",
            # ``isolate`` ("show selected only") moved into the store
            # (Phase 5) so the panel checkbox + viewer adapter drive/read
            # it through the store instead of a cross-module handle.
            "isolate":    False,
            # k-grid display view-state (Phase 5 § 6.3): user-set enable + dims
            # (copies per lattice direction) + source (free/fixed); tiled at
            # render; lives in the store like isolate.
            "kgrid":      {"enabled": False, "dims": [1, 1, 1], "source": "free"},
            "filters":    [],
            "combinator": "or",
            "loading":    False,
            "error":      None,
        }

    def test_get_state_returns_snapshot_not_live(self):
        """``getState()`` must return a snapshot the caller can
        mutate without affecting the store -- otherwise React-style
        consumers that read-and-spread end up mutating the store's
        internal state via aliasing."""
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "const snap = store.getState();\n"
            "snap.selection.push(42);\n"
            "snap.mode = 'filter';\n"
            "snap.filters.push({kind: 'by_element', value: 'Au'});\n"
            "// Live store state must be unchanged.\n"
            "const fresh = store.getState();\n"
            "console.log(JSON.stringify({\n"
            "  selection: fresh.selection,\n"
            "  mode: fresh.mode,\n"
            "  filters: fresh.filters,\n"
            "}));"
        )
        assert out == {"selection": [], "mode": "click", "filters": []}


# --------------------------------------------------------------------
# Synchronous selection mutators
# --------------------------------------------------------------------


class TestAdoptAtoms:
    """``store.adoptAtoms(rawAtoms)`` — the in-memory sync path the
    BOMB-0 selection-staleness fix added (2026-06-07).  Each
    modifier-op response (Delete / Add / Orient / ...) now carries
    an ``atoms`` array; the front-end pushes it through this method
    instead of going through the disk-bound ``_fetchAtoms`` re-fetch
    that would return pre-op atoms (the disk hasn't changed yet)."""

    def test_replaces_atoms_with_normalised_rows(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.adoptAtoms([\n"
            "  {index: 0, element: 'C', regions: ['L-electrode'], is_frozen: false, atom_name: 'CA'},\n"
            "  {index: 1, element: 'H', regions: [],              is_frozen: true},\n"
            "]);\n"
            "console.log(JSON.stringify(store.getState().atoms));"
        )
        assert out == [
            {"index": 0, "element": "C",
             "labels": ["L-electrode"], "isFrozen": False,
             "atomName": "CA"},
            {"index": 1, "element": "H",
             "labels": [], "isFrozen": True},
        ]

    def test_filters_selection_to_in_range_indices(self):
        """A Delete op shrinks the atom list; any selection indices
        that no longer exist MUST be dropped.  Otherwise downstream
        ops resolve indices against the wrong list."""
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.setSelection([0, 2, 5]).then(() => {\n"
            "  // Pretend a Delete op shrank the structure to 2 atoms.\n"
            "  store.adoptAtoms([\n"
            "    {index: 0, element: 'C', regions: [], is_frozen: false},\n"
            "    {index: 1, element: 'C', regions: [], is_frozen: false},\n"
            "  ]);\n"
            "  console.log(JSON.stringify(store.getState().selection));\n"
            "});"
        )
        # 0 is still in range (kept); 2 and 5 are out of range (dropped).
        assert out == [0]

    def test_notifies_subscribers_once(self):
        """Single fanout for the bulk update.  Notification is
        microtask-deferred (the store coalesces synchronous
        mutations), so the test awaits microtasks before counting."""
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "let calls = 0;\n"
            "store.subscribe(() => { calls++; });\n"
            "// Subscribe fires immediately via the deferred notify\n"
            "// path; await it first so it doesn't inflate the count.\n"
            "Promise.resolve().then(() => {\n"
            "  calls = 0;\n"
            "  store.adoptAtoms([\n"
            "    {index: 0, element: 'C', regions: [], is_frozen: false},\n"
            "    {index: 1, element: 'H', regions: [], is_frozen: false},\n"
            "  ]);\n"
            "  return Promise.resolve();\n"
            "}).then(() => {\n"
            "  console.log(JSON.stringify(calls));\n"
            "});"
        )
        assert out == 1

    def test_rejects_non_array(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "try { store.adoptAtoms({not: 'an array'}); "
            "  console.log(JSON.stringify('resolved')); }\n"
            "catch (e) { console.log(JSON.stringify(e.name)); }"
        )
        assert out == "TypeError"


class TestAdoptSessionAtoms:
    """``adoptSession({sourceFile, selection, atoms})`` —
    BOMB-0 follow-up (2026-06-07) extension.  When ``atoms`` is
    supplied, the disk fetch is SKIPPED and atoms are installed
    in-memory.  This is the path /modify's restoreModifyState
    uses after a Delete op so the panel doesn't snap back to the
    pre-op atom list (the disk still has the original file)."""

    def test_pre_fetched_atoms_install_without_disk_fetch(self):
        """A session snapshot of the post-op atoms list (3 atoms
        shrunk to 2 by a Delete) must reinstall cleanly without
        going through /api/selection/atoms — there's no real
        server here; if the store tried to fetch, the test would
        fail."""
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.adoptSession({\n"
            "  sourceFile: 'projects/x/water.xyz',\n"
            "  selection:  [1],\n"
            "  atoms: [\n"
            "    {index: 0, element: 'H', regions: [], is_frozen: false},\n"
            "    {index: 1, element: 'H', regions: [], is_frozen: false},\n"
            "  ],\n"
            "}).then(() => {\n"
            "  const s = store.getState();\n"
            "  console.log(JSON.stringify({\n"
            "    src: s.sourceFile,\n"
            "    n:   s.atoms.length,\n"
            "    sel: s.selection,\n"
            "  }));\n"
            "});"
        )
        assert out == {
            "src": "projects/x/water.xyz",
            "n":   2,
            "sel": [1],
        }

    def test_pre_fetched_atoms_drop_out_of_range_selection(self):
        """If the snapshot's saved.selected has indices past the
        adopted atoms length (delete shrank the structure), drop
        them — same invariant as adoptAtoms."""
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.adoptSession({\n"
            "  sourceFile: 'projects/x/water.xyz',\n"
            "  selection:  [0, 2, 5],\n"
            "  atoms: [\n"
            "    {index: 0, element: 'H', regions: [], is_frozen: false},\n"
            "    {index: 1, element: 'H', regions: [], is_frozen: false},\n"
            "  ],\n"
            "}).then(() => {\n"
            "  console.log(JSON.stringify(store.getState().selection));\n"
            "});"
        )
        assert out == [0]

    def test_absent_atoms_falls_back_to_disk_fetch_path(self):
        """When the snapshot does NOT carry atoms (cross-tab
        handoff, pre-fix sessions), adoptSession still routes
        through the disk fetch — preserved for back-compat.
        We can't actually exercise the fetch under node, but we
        can verify atoms stays empty and sourceFile is set
        (no synchronous fill from a phantom atoms field)."""
        out = _run_node(
            "const store = window.molbuilder.selection._createStore({\n"
            "  fetch: () => Promise.reject(new Error('fetch-skip')),\n"
            "});\n"
            "store.adoptSession({\n"
            "  sourceFile: 'projects/x/water.xyz',\n"
            "  selection:  [],\n"
            "}).catch(() => {})\n"
            ".finally(() => {\n"
            "  const s = store.getState();\n"
            "  console.log(JSON.stringify({\n"
            "    src: s.sourceFile,\n"
            "    n:   s.atoms.length,\n"
            "  }));\n"
            "});"
        )
        assert out == {
            "src": "projects/x/water.xyz",
            "n":   0,
        }


class TestToggleAtom:

    def test_toggle_into_empty_selection(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.toggleAtom(3).then(() => {\n"
            "  console.log(JSON.stringify(store.getState().selection));\n"
            "});"
        )
        assert out == [3]

    def test_toggle_inserts_in_sorted_order(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "Promise.all([\n"
            "  store.toggleAtom(7),\n"
            "  store.toggleAtom(2),\n"
            "  store.toggleAtom(5),\n"
            "]).then(() => {\n"
            "  console.log(JSON.stringify(store.getState().selection));\n"
            "});"
        )
        # Sorted ascending regardless of insertion order.
        assert out == [2, 5, 7]

    def test_toggle_existing_removes(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "Promise.all([\n"
            "  store.toggleAtom(1),\n"
            "  store.toggleAtom(2),\n"
            "  store.toggleAtom(3),\n"
            "]).then(() => store.toggleAtom(2)).then(() => {\n"
            "  console.log(JSON.stringify(store.getState().selection));\n"
            "});"
        )
        assert out == [1, 3]

    def test_toggle_rejects_non_number(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.toggleAtom('not a number')\n"
            "  .then(() => { console.log(JSON.stringify('resolved')); })\n"
            "  .catch((e) => { console.log(JSON.stringify(e.name)); });"
        )
        assert out == "TypeError"


class TestSetSelection:

    def test_set_replaces_selection(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.toggleAtom(1)\n"
            "  .then(() => store.setSelection([5, 6, 7]))\n"
            "  .then(() => console.log(JSON.stringify(store.getState().selection)));"
        )
        assert out == [5, 6, 7]

    def test_set_sorts_and_dedupes(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.setSelection([7, 3, 7, 1, 3, 5]).then(() => {\n"
            "  console.log(JSON.stringify(store.getState().selection));\n"
            "});"
        )
        assert out == [1, 3, 5, 7]

    def test_set_filters_non_numbers(self):
        """The atom-selection spec says ``selection: number[]``.
        A misshapen array (a string or null sneaked in via the URL
        param) must be sanitised -- otherwise the next
        ``toggleAtom`` ``.indexOf`` comparison silently fails."""
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.setSelection([1, '2', null, 3.5, 4]).then(() => {\n"
            "  console.log(JSON.stringify(store.getState().selection));\n"
            "});"
        )
        # 3.5 IS a number but isn't an integer; setSelection's contract
        # filters to ``typeof === 'number'`` so 3.5 passes through.  The
        # acceptable shape is therefore [1, 3.5, 4] (after sort + dedup).
        assert out == [1, 3.5, 4]

    def test_set_rejects_non_array(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.setSelection('not array').then(() => {\n"
            "  console.log(JSON.stringify('resolved'));\n"
            "}).catch((e) => console.log(JSON.stringify(e.name)));"
        )
        assert out == "TypeError"


class TestAddRemoveSelection:

    def test_add_unions_into_selection(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.setSelection([1, 3])\n"
            "  .then(() => store.addToSelection([2, 3, 5]))\n"
            "  .then(() => console.log(JSON.stringify(store.getState().selection)));"
        )
        # 3 already present; union deduped + sorted.
        assert out == [1, 2, 3, 5]

    def test_remove_subtracts(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.setSelection([1, 2, 3, 4, 5])\n"
            "  .then(() => store.removeFromSelection([2, 4, 99]))\n"
            "  .then(() => console.log(JSON.stringify(store.getState().selection)));"
        )
        # 99 not in selection -- silently no-op for it; 2 + 4 removed.
        assert out == [1, 3, 5]


class TestSelectAllAndInvert:
    """``selectAll`` and ``invertSelection`` derive from
    ``state.atoms`` -- but the store doesn't accept atoms directly,
    so we use ``adoptSession`` to set them up."""

    def test_select_all_with_no_atoms_is_empty(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.selectAll().then(() => {\n"
            "  console.log(JSON.stringify(store.getState().selection));\n"
            "});"
        )
        assert out == []

    def test_invert_empty_with_no_atoms_is_empty(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.invertSelection().then(() => {\n"
            "  console.log(JSON.stringify(store.getState().selection));\n"
            "});"
        )
        assert out == []


class TestClearSelection:

    def test_clear_drops_all(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.setSelection([1, 2, 3])\n"
            "  .then(() => store.clearSelection())\n"
            "  .then(() => console.log(JSON.stringify(store.getState().selection)));"
        )
        assert out == []

    def test_clear_on_empty_is_noop(self):
        """Clearing an already-empty selection must NOT fire
        subscribers (otherwise spurious renders happen).  Verified
        via subscriber count."""
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "let fires = 0;\n"
            "// Skip the fire-once-on-subscribe to count only mutations.\n"
            "const unsub = store.subscribe(() => { fires += 1; });\n"
            "store.clearSelection().then(() => {\n"
            "  // 1 fire from subscribe, 0 from clear (already empty).\n"
            "  console.log(JSON.stringify({ fires: fires }));\n"
            "});"
        )
        assert out == {"fires": 1}


# --------------------------------------------------------------------
# Mode + combinator
# --------------------------------------------------------------------


class TestSetMode:

    def test_default_mode_is_click(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "console.log(JSON.stringify(store.getState().mode));"
        )
        assert out == "click"

    def test_switching_mode_preserves_selection(self):
        """Selection is shared across modes per atom-selection.md §3."""
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.setSelection([1, 2, 3])\n"
            "  .then(() => store.setMode('filter'))\n"
            "  .then(() => store.setMode('click'))\n"
            "  .then(() => console.log(JSON.stringify(store.getState().selection)));"
        )
        assert out == [1, 2, 3]

    def test_rejects_unknown_mode(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.setMode('xyzzy').then(() => {\n"
            "  console.log(JSON.stringify('resolved'));\n"
            "}).catch((e) => console.log(JSON.stringify(e.message)));"
        )
        assert "bad mode" in out


class TestCombinator:

    def test_default_is_or(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "console.log(JSON.stringify(store.getState().combinator));"
        )
        assert out == "or"

    def test_set_to_and(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.setCombinator('and').then(() => {\n"
            "  console.log(JSON.stringify(store.getState().combinator));\n"
            "});"
        )
        assert out == "and"

    def test_rejects_unknown(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.setCombinator('xor').then(() => {\n"
            "  console.log(JSON.stringify('resolved'));\n"
            "}).catch((e) => console.log(JSON.stringify(e.message)));"
        )
        assert "bad combinator" in out


# --------------------------------------------------------------------
# Filter drafts
# --------------------------------------------------------------------


class TestFilterDrafts:

    def test_add_filter(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.addFilter({kind: 'by_element', value: 'Au'}).then(() => {\n"
            "  console.log(JSON.stringify(store.getState().filters));\n"
            "});"
        )
        assert out == [{"kind": "by_element", "value": "Au"}]

    def test_set_filters_replaces(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.addFilter({kind: 'by_element', value: 'Au'})\n"
            "  .then(() => store.setFilters([\n"
            "    {kind: 'by_index', value: '0-3'},\n"
            "    {kind: 'by_label', value: 'L-electrode'},\n"
            "  ]))\n"
            "  .then(() => console.log(JSON.stringify(store.getState().filters)));"
        )
        assert out == [
            {"kind": "by_index", "value": "0-3"},
            {"kind": "by_label", "value": "L-electrode"},
        ]

    def test_remove_filter_at_index(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.setFilters([\n"
            "  {kind: 'by_element', value: 'Au'},\n"
            "  {kind: 'by_index',   value: '0-3'},\n"
            "  {kind: 'by_label',   value: 'X'},\n"
            "])\n"
            "  .then(() => store.removeFilter(1))\n"
            "  .then(() => console.log(JSON.stringify(\n"
            "    store.getState().filters.map(f => f.kind)\n"
            "  )));"
        )
        assert out == ["by_element", "by_label"]

    def test_update_filter_replaces_row(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.setFilters([\n"
            "  {kind: 'by_element', value: 'Au'},\n"
            "  {kind: 'by_index',   value: '0-3'},\n"
            "])\n"
            "  .then(() => store.updateFilter(0, "
            "    {kind: 'by_element', value: 'Fe'}))\n"
            "  .then(() => console.log(JSON.stringify(\n"
            "    store.getState().filters\n"
            "  )));"
        )
        assert out == [
            {"kind": "by_element", "value": "Fe"},
            {"kind": "by_index",   "value": "0-3"},
        ]

    def test_remove_filter_out_of_range_rejects(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.removeFilter(0).then(() => {\n"
            "  console.log(JSON.stringify('resolved'));\n"
            "}).catch((e) => console.log(JSON.stringify(e.message)));"
        )
        assert "out of range" in out

    def test_add_filter_rejects_non_object(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.addFilter('not an object').then(() => {\n"
            "  console.log(JSON.stringify('resolved'));\n"
            "}).catch((e) => console.log(JSON.stringify(e.name)));"
        )
        assert out == "TypeError"


# --------------------------------------------------------------------
# Subscribe contract
# --------------------------------------------------------------------


class TestPickOrderSnapshot:
    """Task #304 regression: the click-order shadow ``pickOrder`` MUST
    appear in ``getState()`` and in subscriber snapshots — pre-fix
    ``_snapshot()`` silently dropped it, so the only consumer
    (selection-panel.js's measurement readout) always saw
    ``undefined`` and fell back to the geometric vertex heuristic.
    The whole "vertex = user's 2nd click" semantic shipped dead
    end-to-end despite the store maintaining it correctly
    internally."""

    def test_pickOrder_present_in_getState(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "Promise.all([\n"
            "  store.toggleAtom(5),\n"
            "  store.toggleAtom(0),\n"
            "  store.toggleAtom(2),\n"
            "]).then(() => {\n"
            "  const s = store.getState();\n"
            "  console.log(JSON.stringify({\n"
            "    selection: s.selection,\n"
            "    pickOrder: s.pickOrder,\n"
            "  }));\n"
            "});"
        )
        # selection is sorted (set semantics); pickOrder preserves
        # click order so the angle-vertex consumer sees the user's
        # 2nd click as the middle atom.
        assert out == {
            "selection": [0, 2, 5],
            "pickOrder": [5, 0, 2],
        }

    def test_pickOrder_is_a_defensive_copy(self):
        """Mutating the returned ``pickOrder`` array must not poison
        the store — same contract as ``selection``."""
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "store.setSelection([1, 0, 2]).then(() => {\n"
            "  const snap = store.getState();\n"
            "  snap.pickOrder.push(999);\n"
            "  const fresh = store.getState();\n"
            "  console.log(JSON.stringify(fresh.pickOrder));\n"
            "});"
        )
        assert out == [1, 0, 2]

    def test_subscribe_snapshot_carries_pickOrder(self):
        """Subscribers (the panel renderer) see ``pickOrder`` on
        every notification, not just on a separate getState() call."""
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "let last = null;\n"
            "store.subscribe((s) => { last = s; });\n"
            "store.setSelection([3, 7, 1]).then(() => {\n"
            "  console.log(JSON.stringify({\n"
            "    selection: last.selection,\n"
            "    pickOrder: last.pickOrder,\n"
            "  }));\n"
            "});"
        )
        # selection sorted ascending; pickOrder preserves input
        # order from setSelection.
        assert out == {
            "selection": [1, 3, 7],
            "pickOrder": [3, 7, 1],
        }


class TestSubscribe:

    def test_subscribe_fires_immediately_with_state(self):
        """Per atom-selection.md § 4: subscribe MUST fire once
        synchronously (well, via microtask) with the current state
        so the subscriber can render its initial UI without a
        separate getState() call."""
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "let captured = null;\n"
            "store.subscribe((s) => { captured = s; });\n"
            "// Wait one microtask tick for the initial fire.\n"
            "Promise.resolve().then(() => {\n"
            "  console.log(JSON.stringify({\n"
            "    fired: captured !== null,\n"
            "    mode: captured ? captured.mode : null,\n"
            "    selection: captured ? captured.selection : null,\n"
            "  }));\n"
            "});"
        )
        assert out == {"fired": True, "mode": "click", "selection": []}

    def test_mutation_fires_subscriber(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "let fires = 0;\n"
            "store.subscribe(() => { fires += 1; });\n"
            "Promise.resolve()\n"
            "  .then(() => store.setSelection([1, 2]))\n"
            "  .then(() => store.toggleAtom(3))\n"
            "  .then(() => {\n"
            "    // 1 from subscribe + 1 from setSelection + 1 from toggleAtom = 3\n"
            "    console.log(JSON.stringify({ fires: fires }));\n"
            "  });"
        )
        assert out == {"fires": 3}

    def test_unsubscribe_stops_notifications(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "let fires = 0;\n"
            "const unsub = store.subscribe(() => { fires += 1; });\n"
            "Promise.resolve()\n"
            "  .then(() => { unsub(); })\n"
            "  .then(() => store.setSelection([1]))\n"
            "  .then(() => store.toggleAtom(2))\n"
            "  .then(() => {\n"
            "    // Only the initial fire; both mutations land AFTER unsubscribe.\n"
            "    console.log(JSON.stringify({ fires: fires }));\n"
            "  });"
        )
        assert out == {"fires": 1}

    def test_subscribe_rejects_non_function(self):
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "let err = null;\n"
            "try { store.subscribe('not a function'); }\n"
            "catch (e) { err = e.name; }\n"
            "console.log(JSON.stringify(err));"
        )
        assert out == "TypeError"

    def test_subscriber_exception_does_not_break_other_subscribers(self):
        """Per § 6 (error semantics): one buggy subscriber must NOT
        prevent other subscribers from receiving the notification."""
        out = _run_node(
            "const store = window.molbuilder.selection._createStore();\n"
            "let aFires = 0, bFires = 0;\n"
            "store.subscribe(() => { aFires += 1; throw new Error('A broken'); });\n"
            "store.subscribe(() => { bFires += 1; });\n"
            "Promise.resolve()\n"
            "  .then(() => store.setSelection([1]))\n"
            "  .then(() => {\n"
            "    // Both subscribers fired on initial-fire-on-register + on setSelection.\n"
            "    console.log(JSON.stringify({ aFires: aFires, bFires: bFires }));\n"
            "  });"
        )
        # Both must have fired twice: once on initial subscribe, once on mutation.
        # The thrown error from A must NOT prevent B from running.
        assert out["bFires"] >= 2, (
            f"subscriber B should have fired despite A's exception; "
            f"counts: {out!r}"
        )


class TestEphemeralStore:
    """Phase 5 (fused module): ``createEphemeralStore`` builds an ISOLATED
    selection carrying the ws.selection surface (renamed methods +
    ``indices``-shaped getState), so a readonly inspector owns its own
    selection without touching the workspace singleton."""

    def test_isolated_instances_with_panel_surface(self):
        out = _run_node("""
            const sel = window.molbuilder.selection;
            const a = sel.createEphemeralStore();
            const b = sel.createEphemeralStore();
            // Every method selection-panel + viewer-adapter call on the store.
            const need = ["add","addFilter","all","applyFilter","clear",
                          "getState","invert","invertSelection","removeFilter",
                          "setCombinator","setMode","subscribe","toggle",
                          "updateFilter","writeLabel"];
            a.toggle(3); a.toggle(1);
            console.log(JSON.stringify({
                surface: need.every(k => typeof a[k] === "function"),
                distinct: a !== b,
                aIndices: a.getState().indices,
                bIndices: b.getState().indices,          // isolation check
                indicesShape: "indices" in a.getState(), // NOT legacy "selection"
            }));
        """)
        assert out["surface"] is True
        assert out["distinct"] is True
        assert out["aIndices"] == [1, 3]
        assert out["bIndices"] == []          # b untouched by a's edits
        assert out["indicesShape"] is True


class TestKgridState:
    """k-grid view-state (Phase 5 § 6.3) lives in the store: setKgrid merges an
    ``{enabled?, dims?}`` patch, floors + clamps dims to >= 1, appears in the
    snapshot, and only notifies on a real change."""

    def test_setkgrid_merges_enable_and_normalizes_dims(self):
        out = _run_node("""
            const store = window.molbuilder.selection._createStore();
            store.setKgrid({ enabled: true });
            store.setKgrid({ dims: [2.9, 0, -1] });   // floor 2; 0 / -1 -> 1
            console.log(JSON.stringify(store.getState().kgrid));
        """)
        assert out == {"enabled": True, "dims": [2, 1, 1], "source": "free"}

    def test_setkgrid_dims_ignored_in_fixed_mode(self):
        # A .fdf presents its k-grid authoritatively (source+dims); a later BARE
        # user edit is ignored in fixed mode -- only enable toggles.
        out = _run_node("""
            const store = window.molbuilder.selection._createStore();
            store.setKgrid({ source: "fixed", dims: [2, 2, 1] });
            store.setKgrid({ dims: [4, 4, 4] });        // bare edit -> ignored
            store.setKgrid({ enabled: true });          // enable still works
            console.log(JSON.stringify(store.getState().kgrid));
        """)
        assert out == {"enabled": True, "dims": [2, 2, 1], "source": "fixed"}

    def test_setkgrid_free_dims_capped_by_atom_count(self):
        # 5000 atoms: k-grid is copies, so natoms * nx*ny*nz must stay <= 20000
        # (=> product <= 4).  A requested 3x3x3 (=27) is clamped down.
        out = _run_node("""
            (async () => {
                const store = window.molbuilder.selection._createStore();
                await store.adoptSession({
                    atoms: Array.from({length: 5000}, (_, i) => ({ index: i, element: "C" })),
                    selection: [],
                });
                store.setKgrid({ dims: [3, 3, 3] });
                const d = store.getState().kgrid.dims;
                console.log(JSON.stringify({ product: d[0] * d[1] * d[2] }));
            })();
        """)
        assert out["product"] <= 4     # 5000 * product <= 20000

    def test_setkgrid_noop_does_not_notify(self):
        out = _run_node("""
            const store = window.molbuilder.selection._createStore();
            let n = 0;
            store.subscribe(() => n++);            // fires once immediately
            const base = n;
            store.setKgrid({ enabled: false });    // already false
            store.setKgrid({ dims: [1, 1, 1] });   // already default
            console.log(JSON.stringify({ extra: n - base }));
        """)
        assert out["extra"] == 0
