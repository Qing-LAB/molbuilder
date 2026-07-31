"""Unit tests for the MolView-internal canvas-state store.

Pins the API of ``molbuilder/web/static/lib/molview/
_canvas-state-impl.js`` -- the in-memory geometry store (text + source
provenance + periodicity + dirty) that the MolView data model reads
behind ``molview.data`` (molview-module.md §19: the ``_``-prefixed store
files are MolView-INTERNAL).  This file scopes to that store's own
mutation contract; the public data surface + the persistence round-trip
are pinned in ``tests/test_workspace_dispatcher_js.py``.

Persistence is the workspace's concern, NOT this store's
(workspace-contract.md §4.1, sole key ``molbuilder.workspace.v1``); this
store only reads the persisted snapshot on init (via the shared snapshot
IO) and holds the live state.

Covered:
  * Initial state (empty canvas).
  * ``setStructure`` validates + replaces wholesale + resets ``dirty``.
  * ``replaceContent`` keeps/replaces periodicity (rides with geometry).
  * ``markDirty`` flips the dirty bit, but only once.
  * ``markSaved`` clears ``dirty`` and records ``last_save_to``.
  * ``clear`` wipes back to empty.
  * ``onChange`` subscribers fire on every mutation and unsubscribe.
  * Snapshot immutability (returned dicts are copies).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from _node_esm import run_node


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/molview/_canvas-state-impl.js"


def _run_node(snippet: str, *, prelude: str = "") -> object:
    """Run a JS snippet against ``_canvas-state-impl.js``.

    The module is now a native ES module (the MolView ESM migration): it mounts
    its API globally at ``window.molbuilder.molview._canvasState`` as a side
    effect (and ``export const canvasState``), so the shared ``run_node``
    dynamic-import harness loads it and we alias the mounted api to ``canvas``
    for the snippet -- exactly the object the old ``require()`` returned.

    ``prelude`` runs BEFORE the module import -- useful for priming
    sessionStorage with content the module should pick up on init.
    """
    globals_js = f"""
        const _store = new Map();
        global.sessionStorage = {{
            getItem:    k => _store.has(k) ? _store.get(k) : null,
            setItem:    (k, v) => _store.set(k, String(v)),
            removeItem: k => _store.delete(k),
            clear:      () => _store.clear(),
            // Expose the backing map for tests that want to assert on it.
            _peek:      () => Object.fromEntries(_store),
        }};
        {prelude}
    """
    wrapped = f"""
        const canvas = globalThis.molbuilder.molview._canvasState;
        await (async () => {{
            {snippet}
        }})().catch(err => {{
            console.log(JSON.stringify({{
                __test_unexpected_throw: true,
                message: err && err.message ? err.message : String(err),
                stack:   err && err.stack ? err.stack : null,
            }}));
        }});
    """
    out = run_node([MODULE], wrapped, globals_js=globals_js, timeout=15)
    if isinstance(out, dict):
        assert "__test_unexpected_throw" not in out, (
            "module threw: " + str(out)
        )
    return out


# ----- Surface presence ------------------------------------------ #


class TestSurfacePresence:

    def test_all_methods_callable(self):
        out = _run_node('''
            console.log(JSON.stringify({
                setStructure:      typeof canvas.setStructure,
                markDirty:         typeof canvas.markDirty,
                markSaved:         typeof canvas.markSaved,
                clear:             typeof canvas.clear,
                isEmpty:           typeof canvas.isEmpty,
                isDirty:           typeof canvas.isDirty,
                getSource:         typeof canvas.getSource,
                getStructure:      typeof canvas.getStructure,
                getLastSavedTo:    typeof canvas.getLastSavedTo,
                onChange:          typeof canvas.onChange,
                reloadFromStorage: typeof canvas.reloadFromStorage,
            }));
        ''')
        for fn in ("setStructure", "markDirty", "markSaved", "clear",
                   "isEmpty", "isDirty", "getSource", "getStructure",
                   "getLastSavedTo", "onChange", "reloadFromStorage"):
            assert out[fn] == "function", f"missing: {fn}"


# ----- Initial state --------------------------------------------- #


class TestInitialState:

    def test_starts_empty(self):
        out = _run_node('''
            console.log(JSON.stringify({
                isEmpty:        canvas.isEmpty(),
                isDirty:        canvas.isDirty(),
                getStructure:   canvas.getStructure(),
                getLastSavedTo: canvas.getLastSavedTo(),
                getSource:      canvas.getSource(),
            }));
        ''')
        assert out["isEmpty"] is True
        assert out["isDirty"] is False
        assert out["getStructure"] is None
        assert out["getLastSavedTo"] is None
        assert out["getSource"] == {
            "kind": "blank", "file": None, "generator_input": None,
        }


# ----- setStructure ---------------------------------------------- #


class TestSetStructure:

    def test_replaces_canvas_and_clears_dirty(self):
        """setStructure REPLACES wholesale and resets dirty -- a
        freshly-loaded structure is the canonical version; user
        modifications haven't happened yet."""
        out = _run_node('''
            canvas.setStructure(
                { source_format: "xyz", text: "1\\nC\\nC 0 0 0\\n" },
                { kind: "file", file: "/projects/p/a.xyz" }
            );
            // Even if we mark dirty first, a subsequent setStructure
            // wipes that dirty bit (it's a fresh canvas).
            canvas.markDirty();
            const wasDirty = canvas.isDirty();
            canvas.setStructure(
                { source_format: "pdb", text: "HETATM..." },
                { kind: "smiles", generator_input: { smiles: "CCO" } }
            );
            console.log(JSON.stringify({
                wasDirty:       wasDirty,
                isEmpty:        canvas.isEmpty(),
                isDirty:        canvas.isDirty(),
                getStructure:   canvas.getStructure(),
                getSource:      canvas.getSource(),
                getLastSavedTo: canvas.getLastSavedTo(),
            }));
        ''')
        assert out["wasDirty"] is True
        assert out["isEmpty"] is False
        assert out["isDirty"] is False
        assert out["getStructure"] == {
            "source_format": "pdb", "text": "HETATM...",
            "periodicity": None,   # no periodicity supplied to setStructure
            "annotations": None,   # F1: opaque carrier, absent here
        }
        assert out["getSource"] == {
            "kind": "smiles",
            "file": None,
            "generator_input": {"smiles": "CCO"},
        }
        assert out["getLastSavedTo"] is None

    def test_rejects_missing_text(self):
        out = _run_node('''
            let threw = false;
            try {
                canvas.setStructure(
                    { source_format: "xyz" },
                    { kind: "file" }
                );
            } catch (e) { threw = true; }
            console.log(JSON.stringify({
                threw:    threw,
                isEmpty:  canvas.isEmpty(),
            }));
        ''')
        assert out["threw"] is True
        assert out["isEmpty"] is True

    def test_rejects_unknown_source_format(self):
        out = _run_node('''
            let threw = false;
            try {
                canvas.setStructure(
                    { source_format: "cif", text: "..." },
                    { kind: "file" }
                );
            } catch (e) { threw = true; }
            console.log(JSON.stringify(threw));
        ''')
        assert out is True

    def test_rejects_missing_source_kind(self):
        out = _run_node('''
            let threw = false;
            try {
                canvas.setStructure(
                    { source_format: "xyz", text: "1\\nC\\nC 0 0 0\\n" },
                    {}
                );
            } catch (e) { threw = true; }
            console.log(JSON.stringify(threw));
        ''')
        assert out is True

    def test_periodicity_rides_with_geometry(self):
        """Periodicity travels with the geometry (workspace-contract.md §4.0):
        setStructure stores it; a text-only replaceContent KEEPS it; a modify op
        that recaptured a cell passes new periodicity and replaces it."""
        out = _run_node('''
            canvas.setStructure(
                { source_format: "xyz", text: "1\\nC\\nC 0 0 0\\n",
                  periodicity: { cell: [[5,0,0],[0,5,0],[0,0,10]],
                                 axis_kind: ["periodic","periodic","transport"],
                                 vacuum: [0,0,0] } },
                { kind: "file", file: "/p/a.xyz" }
            );
            const afterSet = canvas.getStructure().periodicity;
            canvas.replaceContent("2\\nC2\\nC 0 0 0\\nC 0 0 1\\n");   // text only
            const afterTextEdit = canvas.getStructure().periodicity;
            canvas.replaceContent("2\\nC2\\nC 0 0 0\\nC 0 0 1\\n",
                { cell: [[6,0,0],[0,6,0],[0,0,12]], axis_kind: null,
                  vacuum: [1,1,1] });                             // modify op
            const afterModify = canvas.getStructure().periodicity;
            console.log(JSON.stringify({afterSet, afterTextEdit, afterModify}));
        ''')
        assert out["afterSet"]["axis_kind"] == ["periodic", "periodic", "transport"]
        assert out["afterSet"]["vacuum"] == [0, 0, 0]
        assert out["afterTextEdit"]["vacuum"] == [0, 0, 0]     # text edit kept it
        assert out["afterModify"]["vacuum"] == [1, 1, 1]       # modify replaced it
        assert out["afterModify"]["cell"] == [[6, 0, 0], [0, 6, 0], [0, 0, 12]]

    def test_setperiodicity_applies_and_preserves_cell_origin(self):
        """§3c: setPeriodicity applies an explicit cell_origin AND preserves it
        across an UNRELATED edit (vacuum).  The editor is now spread-based, so it
        can no longer drop a field it doesn't hand-list -- the gap that let
        cell_origin vanish (structure-authority.md)."""
        out = _run_node('''
            canvas.setStructure(
                { source_format: "xyz", text: "1\\nC\\nC 0 0 0\\n",
                  periodicity: { cell: [[5,0,0],[0,5,0],[0,0,10]],
                                 axis_kind: ["periodic","periodic","transport"],
                                 vacuum: [0,0,0] } },
                { kind: "file", file: "/p/a.xyz" }
            );
            canvas.setPeriodicity({ cell_origin: [1.5, 2.5, 3.5] });
            const afterOrigin = canvas.getStructure().periodicity;
            canvas.setPeriodicity({ vacuum: [1, 1, 1] });   // unrelated edit
            const afterVacuum = canvas.getStructure().periodicity;
            console.log(JSON.stringify({ afterOrigin, afterVacuum }));
        ''')
        assert out["afterOrigin"]["cell_origin"] == [1.5, 2.5, 3.5]
        # The unrelated vacuum edit must NOT drop cell_origin (the field-preserve
        # guarantee of the spread-based editor).
        assert out["afterVacuum"]["cell_origin"] == [1.5, 2.5, 3.5]
        assert out["afterVacuum"]["vacuum"] == [1, 1, 1]
        assert out["afterVacuum"]["cell"] == [[5, 0, 0], [0, 5, 0], [0, 0, 10]]


# ----- markDirty ------------------------------------------------- #


class TestMarkDirty:

    def test_flips_dirty_bit(self):
        out = _run_node('''
            canvas.setStructure(
                { source_format: "xyz", text: "1\\nC\\nC 0 0 0\\n" },
                { kind: "file" }
            );
            const before = canvas.isDirty();
            canvas.markDirty();
            const after = canvas.isDirty();
            console.log(JSON.stringify({ before: before, after: after }));
        ''')
        assert out == {"before": False, "after": True}

    def test_noop_on_empty_canvas(self):
        """An empty canvas can't be dirty (no bytes to be lost).
        Pins that markDirty is a silent no-op on empty -- no notify,
        no state change."""
        out = _run_node('''
            const calls = [];
            canvas.onChange(s => calls.push(s.dirty));
            // Clear the initial-fire from subscribe (snapshot of empty).
            calls.length = 0;
            canvas.markDirty();
            console.log(JSON.stringify({
                isDirty: canvas.isDirty(),
                notifyCount: calls.length,
            }));
        ''')
        assert out["isDirty"] is False
        assert out["notifyCount"] == 0

    def test_idempotent_repeat_does_not_notify(self):
        """A second markDirty when already dirty is a no-op + no
        notify -- subscribers don't re-fire for the same state."""
        out = _run_node('''
            canvas.setStructure(
                { source_format: "xyz", text: "1\\nC\\nC 0 0 0\\n" },
                { kind: "file" }
            );
            canvas.markDirty();
            const calls = [];
            canvas.onChange(s => calls.push(s.dirty));
            // Drop the initial-fire from the subscribe.
            calls.length = 0;
            canvas.markDirty();
            canvas.markDirty();
            console.log(JSON.stringify({
                isDirty:     canvas.isDirty(),
                notifyCount: calls.length,
            }));
        ''')
        assert out["isDirty"] is True
        assert out["notifyCount"] == 0


# ----- markSaved ------------------------------------------------- #


class TestMarkSaved:

    def test_clears_dirty_and_records_path(self):
        out = _run_node('''
            canvas.setStructure(
                { source_format: "xyz", text: "1\\nC\\nC 0 0 0\\n" },
                { kind: "file" }
            );
            canvas.markDirty();
            canvas.markSaved("/projects/p/saved.xyz");
            console.log(JSON.stringify({
                isDirty:        canvas.isDirty(),
                getLastSavedTo: canvas.getLastSavedTo(),
            }));
        ''')
        assert out["isDirty"] is False
        assert out["getLastSavedTo"] == "/projects/p/saved.xyz"

    def test_rejects_empty_path(self):
        out = _run_node('''
            canvas.setStructure(
                { source_format: "xyz", text: "1\\nC\\nC 0 0 0\\n" },
                { kind: "file" }
            );
            let threw = false;
            try { canvas.markSaved(""); }
            catch (e) { threw = true; }
            console.log(JSON.stringify(threw));
        ''')
        assert out is True


# ----- clear ----------------------------------------------------- #


class TestClear:

    def test_wipes_to_empty(self):
        out = _run_node('''
            canvas.setStructure(
                { source_format: "xyz", text: "1\\nC\\nC 0 0 0\\n" },
                { kind: "smiles", generator_input: { smiles: "C" } }
            );
            canvas.markDirty();
            canvas.markSaved("/projects/p/x.xyz");
            canvas.clear();
            console.log(JSON.stringify({
                isEmpty:        canvas.isEmpty(),
                isDirty:        canvas.isDirty(),
                getStructure:   canvas.getStructure(),
                getLastSavedTo: canvas.getLastSavedTo(),
                getSource:      canvas.getSource(),
            }));
        ''')
        assert out["isEmpty"] is True
        assert out["isDirty"] is False
        assert out["getStructure"] is None
        assert out["getLastSavedTo"] is None
        assert out["getSource"]["kind"] == "blank"


# ----- onChange -------------------------------------------------- #


class TestOnChange:

    def test_fires_on_every_mutation(self):
        out = _run_node('''
            const calls = [];
            canvas.onChange(s => calls.push({
                empty: s.text == null,
                dirty: s.dirty,
                kind:  s.source && s.source.kind,
            }));
            canvas.setStructure(
                { source_format: "xyz", text: "1\\nC\\nC 0 0 0\\n" },
                { kind: "smiles" }
            );
            canvas.markDirty();
            canvas.markSaved("/p/x.xyz");
            canvas.clear();
            console.log(JSON.stringify(calls));
        ''')
        # No initial fire on subscribe (this primitive doesn't fire
        # the snapshot immediately on subscribe -- it only fires on
        # mutation).  Updates: 4 (setStructure, markDirty, markSaved,
        # clear).
        assert out == [
            {"empty": False, "dirty": False, "kind": "smiles"},
            {"empty": False, "dirty": True,  "kind": "smiles"},
            {"empty": False, "dirty": False, "kind": "smiles"},
            {"empty": True,  "dirty": False, "kind": "blank"},
        ]

    def test_unsubscribe_stops_callbacks(self):
        out = _run_node('''
            const calls = [];
            const unsub = canvas.onChange(s => calls.push(s.dirty));
            canvas.setStructure(
                { source_format: "xyz", text: "1\\nC\\nC 0 0 0\\n" },
                { kind: "file" }
            );
            unsub();
            canvas.markDirty();
            canvas.markSaved("/p/x.xyz");
            console.log(JSON.stringify(calls));
        ''')
        # One call (the setStructure), then unsub kills further fires.
        assert out == [False]

    def test_rejects_non_function(self):
        out = _run_node('''
            let threw = false;
            try { canvas.onChange("not a function"); }
            catch (e) { threw = true; }
            console.log(JSON.stringify(threw));
        ''')
        assert out is True

    def test_subscriber_throw_does_not_break_others(self):
        """Per Principle 6 (errors don't propagate): a throwing
        subscriber must not kill the publish loop for the others."""
        out = _run_node('''
            const calls = [];
            canvas.onChange(() => { throw new Error("boom"); });
            canvas.onChange(s => calls.push(s.dirty));
            canvas.setStructure(
                { source_format: "xyz", text: "1\\nC\\nC 0 0 0\\n" },
                { kind: "file" }
            );
            console.log(JSON.stringify(calls));
        ''')
        assert out == [False]


# ----- Snapshot semantics (no caller-visible mutation) ----------- #


class TestSnapshotImmutability:
    """``getSource`` / ``getStructure`` return CLONES -- a caller that
    mutates the returned object MUST NOT poison the canvas's private
    state.  Otherwise a generator that calls ``getSource()`` and
    mutates its ``generator_input`` would silently rewrite history."""

    def test_mutating_getSource_does_not_affect_canvas(self):
        out = _run_node('''
            canvas.setStructure(
                { source_format: "xyz", text: "1\\nC\\nC 0 0 0\\n" },
                { kind: "smiles", generator_input: { smiles: "CCO" } }
            );
            const src = canvas.getSource();
            src.kind = "POISONED";
            src.generator_input.smiles = "POISONED";
            console.log(JSON.stringify(canvas.getSource()));
        ''')
        assert out["kind"] == "smiles"
        assert out["generator_input"]["smiles"] == "CCO"


# Persistence is the workspace dispatcher's concern, not canvas-state's
# (workspace-contract.md §4.1 — sole persistence key
# ``molbuilder.workspace.v1``).  The dispatcher's round-trip tests
# (tests/test_workspace_dispatcher_js.py::TestPersistRoundtrip) cover
# what the legacy ``molbuilder.structure_canvas`` mirror tests used
# to cover; this file scopes to the in-memory state-mutation contract.
