"""Unit tests for the Structure-tab canvas-state primitive.

Pins the public API of ``molbuilder/web/static/lib/structure/
canvas-state.js`` -- the single source of truth for what's loaded
in the Structure tab canvas.

The primitive is the foundation Structure-tab generators, modifier
panels, save buttons, and dirty badges all subscribe to.  Breakage
here cascades into every panel; the tests cover:

  * Initial state (empty canvas).
  * ``setStructure`` validates input + replaces wholesale + resets
    ``dirty``.
  * ``markDirty`` flips the dirty bit, but only once (no spurious
    notifies on repeat calls); empty canvas is a no-op.
  * ``markSaved`` clears ``dirty`` and records ``last_save_to``.
  * ``clear`` wipes back to empty.
  * ``onChange`` subscribers fire after every state mutation and
    unsubscribe cleanly.
  * sessionStorage mirror survives a fresh module import (the
    refresh-survival contract).
  * sessionStorage parse errors fall back to empty state instead of
    throwing module init.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/structure/canvas-state.js"


def _run_node(snippet: str, *, prelude: str = "") -> object:
    """Run a JS snippet against canvas-state.js as a CommonJS module.

    The module's UMD branch sees ``module`` defined and exports its
    API on ``module.exports`` -- we ``require()`` it and assign to
    ``canvas`` for the snippet to use.

    ``prelude`` runs BEFORE the require call -- useful for priming
    sessionStorage with content the module should pick up on init.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    module_path = MODULE.resolve()
    bootstrap = f"""
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
        const canvas = require({json.dumps(str(module_path))});
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
                STORAGE_KEY:       typeof canvas.STORAGE_KEY,
            }));
        ''')
        for fn in ("setStructure", "markDirty", "markSaved", "clear",
                   "isEmpty", "isDirty", "getSource", "getStructure",
                   "getLastSavedTo", "onChange", "reloadFromStorage"):
            assert out[fn] == "function", f"missing: {fn}"
        assert out["STORAGE_KEY"] == "string"

    def test_storage_key_is_namespaced(self):
        """The key MUST live under the ``molbuilder.`` prefix so it
        doesn't collide with other libraries' sessionStorage usage."""
        out = _run_node('''
            console.log(JSON.stringify(canvas.STORAGE_KEY));
        ''')
        assert out.startswith("molbuilder.")
        assert "structure_canvas" in out


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


# ----- sessionStorage mirror ------------------------------------- #


class TestSessionStorageMirror:

    def test_state_persisted_to_sessionstorage(self):
        out = _run_node('''
            canvas.setStructure(
                { source_format: "xyz", text: "1\\nC\\nC 0 0 0\\n" },
                { kind: "file", file: "/p/a.xyz" }
            );
            const raw = sessionStorage.getItem(canvas.STORAGE_KEY);
            console.log(JSON.stringify(JSON.parse(raw)));
        ''')
        assert out["source_format"] == "xyz"
        assert out["text"].startswith("1\n")
        assert out["source"]["file"] == "/p/a.xyz"
        assert out["dirty"] is False

    def test_dirty_flag_persisted(self):
        out = _run_node('''
            canvas.setStructure(
                { source_format: "xyz", text: "1\\nC\\nC 0 0 0\\n" },
                { kind: "file" }
            );
            canvas.markDirty();
            const raw = sessionStorage.getItem(canvas.STORAGE_KEY);
            console.log(JSON.stringify(JSON.parse(raw).dirty));
        ''')
        assert out is True

    def test_clear_persists_empty_state(self):
        out = _run_node('''
            canvas.setStructure(
                { source_format: "xyz", text: "1\\nC\\nC 0 0 0\\n" },
                { kind: "file" }
            );
            canvas.clear();
            const raw = sessionStorage.getItem(canvas.STORAGE_KEY);
            console.log(JSON.stringify(JSON.parse(raw)));
        ''')
        assert out["text"] is None
        assert out["source"]["kind"] == "blank"

    def test_restores_from_sessionstorage_on_init(self):
        """The refresh-survival contract -- a primed sessionStorage
        is picked up on the FIRST public-API call after require()."""
        # Prime sessionStorage BEFORE the require call.  The module
        # restores lazily on the first method that needs state.
        primed = json.dumps({
            "source_format": "pdb",
            "text": "HETATM 1 ...",
            "source": {
                "kind": "smiles",
                "file": None,
                "generator_input": {"smiles": "CCO"},
            },
            "dirty": True,
            "last_save_to": "/p/saved.pdb",
        })
        out = _run_node(
            '''
            console.log(JSON.stringify({
                isEmpty:        canvas.isEmpty(),
                isDirty:        canvas.isDirty(),
                getStructure:   canvas.getStructure(),
                getLastSavedTo: canvas.getLastSavedTo(),
                getSource:      canvas.getSource(),
            }));
            ''',
            prelude=f'_store.set("molbuilder.structure_canvas", {json.dumps(primed)});',
        )
        assert out["isEmpty"] is False
        assert out["isDirty"] is True
        assert out["getStructure"] == {
            "source_format": "pdb", "text": "HETATM 1 ...",
        }
        assert out["getLastSavedTo"] == "/p/saved.pdb"
        assert out["getSource"]["kind"] == "smiles"

    def test_corrupt_sessionstorage_falls_back_to_empty(self):
        """A garbage payload (truncated JSON, manual tampering) MUST
        NOT crash the module -- the canvas just starts empty."""
        out = _run_node(
            '''
            console.log(JSON.stringify({
                isEmpty: canvas.isEmpty(),
                isDirty: canvas.isDirty(),
            }));
            ''',
            prelude='_store.set("molbuilder.structure_canvas", "not-valid-json{");',
        )
        assert out == {"isEmpty": True, "isDirty": False}

    def test_partial_sessionstorage_payload_picks_up_defaults(self):
        """An older serialisation that's missing newer fields
        (e.g. ``last_save_to``) MUST NOT crash -- missing fields
        get default values."""
        primed = json.dumps({
            "source_format": "xyz",
            "text": "1\nC\nC 0 0 0\n",
            # Missing source + dirty + last_save_to.
        })
        out = _run_node(
            '''
            console.log(JSON.stringify({
                isEmpty:        canvas.isEmpty(),
                isDirty:        canvas.isDirty(),
                getLastSavedTo: canvas.getLastSavedTo(),
                getSource:      canvas.getSource(),
            }));
            ''',
            prelude=f'_store.set("molbuilder.structure_canvas", {json.dumps(primed)});',
        )
        assert out["isEmpty"] is False
        assert out["isDirty"] is False
        assert out["getLastSavedTo"] is None
        assert out["getSource"]["kind"] == "blank"


# ----- reloadFromStorage ----------------------------------------- #


class TestReloadFromStorage:
    """Useful when external code (another tab, manual tampering)
    has updated sessionStorage and the in-memory state is stale."""

    def test_reload_picks_up_external_writes(self):
        out = _run_node('''
            // Establish initial state.
            canvas.setStructure(
                { source_format: "xyz", text: "1\\nC\\nC 0 0 0\\n" },
                { kind: "file" }
            );
            // Simulate an external writer (other tab, dev console)
            // overwriting the cell directly.
            sessionStorage.setItem(canvas.STORAGE_KEY, JSON.stringify({
                source_format: "pdb",
                text: "EXTERNAL",
                source: { kind: "smiles", file: null, generator_input: null },
                dirty: false,
                last_save_to: null,
            }));
            const beforeReload = canvas.getStructure();
            canvas.reloadFromStorage();
            const afterReload = canvas.getStructure();
            console.log(JSON.stringify({
                beforeReload: beforeReload,
                afterReload:  afterReload,
            }));
        ''')
        # Before reload, the in-memory state is still the XYZ.
        assert out["beforeReload"]["source_format"] == "xyz"
        # After reload, it picks up the PDB.
        assert out["afterReload"] == {
            "source_format": "pdb", "text": "EXTERNAL",
        }
