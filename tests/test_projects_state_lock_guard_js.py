"""Unit tests for the setShared lock-guard contract (sidebar gap
#177, defense-in-depth).

Per docs/web/projects.md, programmatic state
mutators (setShared, future navigateTo) MUST early-return
``{ok:false, error:"sidebar is locked"}`` while the lock is held.
The CSS ``pointer-events:none`` rule blocks user clicks but
tab-level navigators (the /results file-picker dropdown at
``lib/results/file-picker.js``) could otherwise sneak a directory
change past an active Save pipeline.

These tests stub sessionStorage + the module's lock entry points
under Node, exercise setShared in both locked and unlocked states,
and verify both the return value AND the side-effect (sessionStorage
+ subscriber fire) shape.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/projects/state.js"


def _run_node(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    module_url = MODULE.resolve().as_uri()
    # state.js touches sessionStorage + window.molbuilder + a bunch of
    # other modules; we stub the minimum it needs to import without
    # erroring.  api.js is imported but its functions aren't called
    # by the lock-guard tests so a stub module isn't required (the
    # static import works because Node fetches the file and it's
    # valid ESM).
    bootstrap = f"""
        // sessionStorage stub: in-memory key/value.
        const _store = new Map();
        global.sessionStorage = {{
            getItem:    k => _store.has(k) ? _store.get(k) : null,
            setItem:    (k, v) => _store.set(k, String(v)),
            removeItem: k => _store.delete(k),
            clear:      () => _store.clear(),
        }};
        global.window = global;
        global.document = {{
            addEventListener: () => {{}},
            getElementById:   () => null,
        }};
        // Locale fallback some sub-imports touch.
        global.navigator = {{ language: "en-US" }};
        global.fetch = async () => ({{
            ok: true, status: 200, json: async () => ({{ ok: true }})
        }});
        const statePromise = import("{module_url}");
        statePromise.then(async (state) => {{
            {snippet}
        }}).catch(err => {{
            console.log(JSON.stringify({{
                __test_unexpected_throw: true,
                message: err && err.message ? err.message : String(err),
                stack:   err && err.stack ? err.stack : null,
            }}));
        }});
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
    assert "__test_unexpected_throw" not in out, (
        "module threw unexpectedly: " + str(out)
    )
    return out


class TestSetSharedWhenUnlocked:

    def test_returns_ok_envelope(self):
        out = _run_node('''
            const r = state.setShared("/dir", "/dir/file.out");
            console.log(JSON.stringify(r));
        ''')
        assert out == {"ok": True}

    def test_updates_session_storage(self):
        out = _run_node('''
            state.setShared("/foo", "/foo/bar.out");
            console.log(JSON.stringify({
                dir:  sessionStorage.getItem("molbuilder.current_dir"),
                file: sessionStorage.getItem("molbuilder.current_file"),
            }));
        ''')
        assert out == {"dir": "/foo", "file": "/foo/bar.out"}

    def test_fires_subscribers(self):
        """Subscribers registered via projects.onChange receive the new
        {dir, file} payload after setShared.  onChange fires ONCE
        immediately on subscribe (per the docstring contract), so we
        expect 2 callbacks: the initial empty + the post-setShared."""
        out = _run_node('''
            const calls = [];
            state.projects.onChange(p => calls.push(p));
            state.setShared("/a", "/a/b.out");
            console.log(JSON.stringify(calls));
        ''')
        # Initial fire (empty) + post-setShared fire.
        assert len(out) == 2
        assert out[0] == {"dir": "", "file": ""}
        assert out[1] == {"dir": "/a", "file": "/a/b.out"}


class TestSetSharedWhenLocked:

    def test_returns_not_ok_envelope(self):
        out = _run_node('''
            state.projects.lock("Saving FDF…", []);
            const r = state.setShared("/dir", "/dir/file.out");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "sidebar is locked" in out["error"]
        # Includes the lock reason so the caller can show a useful
        # message to the user (or log it).
        assert "Saving FDF" in out["error"]

    def test_does_not_update_session_storage(self):
        out = _run_node('''
            state.projects.lock("Save in flight", []);
            state.setShared("/locked-target", "/locked-target/x.out");
            console.log(JSON.stringify({
                dir:  sessionStorage.getItem("molbuilder.current_dir"),
                file: sessionStorage.getItem("molbuilder.current_file"),
            }));
        ''')
        # Nothing was written.  Both keys absent (null) -- not even
        # an empty string.
        assert out == {"dir": None, "file": None}

    def test_does_not_fire_subscribers(self):
        """Locked setShared MUST NOT publish a fake selection change
        -- subscribers that re-render on change (e.g. /results
        viewer.js) would otherwise tear down the active inspector
        while the lock is still held."""
        out = _run_node('''
            const calls = [];
            state.projects.onChange(p => calls.push(p));
            state.projects.lock("Save in flight", []);
            state.setShared("/locked-target", "/locked-target/x.out");
            // Only the initial-fire from onChange subscribe; the
            // setShared attempt must not have published.
            console.log(JSON.stringify(calls));
        ''')
        # Just the initial fire (empty), no post-setShared event.
        assert out == [{"dir": "", "file": ""}]

    def test_unlock_restores_set_shared(self):
        """After unlock, setShared works normally again."""
        out = _run_node('''
            state.projects.lock("step 1", []);
            const blocked = state.setShared("/a", "/a/b");
            state.projects.unlock();
            const allowed = state.setShared("/c", "/c/d");
            console.log(JSON.stringify({
                blocked: blocked,
                allowed: allowed,
                final_dir:  sessionStorage.getItem("molbuilder.current_dir"),
                final_file: sessionStorage.getItem("molbuilder.current_file"),
            }));
        ''')
        assert out["blocked"]["ok"] is False
        assert out["allowed"] == {"ok": True}
        assert out["final_dir"] == "/c"
        assert out["final_file"] == "/c/d"


# ------------------------------------------------------------------ #
#  Per-tab selection memory (projects.md § 2, 2026-08-19)            #
# ------------------------------------------------------------------ #


def test_each_tab_keeps_its_own_place_and_a_fresh_tab_inherits():
    """The user's 2026-08-19 ask, executed: Results keeps its run folder
    while Modify keeps structure/, switching back returns each to its own
    place — and a tab never visited starts at the most recent place
    anywhere (the old shared behaviour, demoted to fallback)."""
    out = _run_node(
        """
        global.location = { pathname: "/results" };
        const m = state;
        m.projects.setShared("/p/proj/optimization/run-1", "");
        const resultsSees = m.projects.getCurrentDir();

        // The user switches to Modify and works somewhere else.
        global.location = { pathname: "/molbuilder" };
        m.projects.setShared("/p/proj/structure", "/p/proj/structure/a.xyz");
        const modifySees = m.projects.getCurrentDir();

        // Back to Results: its own place, not Modify's.
        global.location = { pathname: "/results" };
        const resultsAgain = m.projects.getCurrentDir();

        // A tab never visited inherits the most recent place anywhere.
        global.location = { pathname: "/spectrum-calculation" };
        const freshTab = m.projects.getCurrentDir();

        console.log(JSON.stringify({
            resultsSees, modifySees, resultsAgain, freshTab,
        }));
        """
    )
    assert out["resultsSees"] == "/p/proj/optimization/run-1"
    assert out["modifySees"] == "/p/proj/structure"
    assert out["resultsAgain"] == "/p/proj/optimization/run-1", (
        "switching back lost the tab's own place — the per-tab slot did "
        "not hold"
    )
    assert out["freshTab"] == "/p/proj/structure", (
        "a first visit must start at the most recent place anywhere"
    )


def test_a_handoff_lands_in_the_target_tabs_own_slot():
    """'Open in Molbuilder' writes the TARGET page's slots — raw shared-key
    writes only feed the fallback, which the target's own memory shadows
    (the silent break the door exists to prevent)."""
    out = _run_node(
        """
        // Modify has its own place already — the shadow case.
        global.location = { pathname: "/molbuilder" };
        const m = state;
        m.projects.setShared("/p/old/place", "");

        // Results hands a file across.
        global.location = { pathname: "/results" };
        const r = m.projects.handOffSelection(
            "/molbuilder", "/p/proj/optimization/run-1",
            "/p/proj/optimization/run-1/out.xyz");

        // Modify opens: the handoff won over its old place.
        global.location = { pathname: "/molbuilder" };
        console.log(JSON.stringify({
            ok: r.ok,
            dir: m.projects.getCurrentDir(),
            file: m.projects.getCurrentFile(),
        }));
        """
    )
    assert out["ok"] is True
    assert out["dir"] == "/p/proj/optimization/run-1"
    assert out["file"] == "/p/proj/optimization/run-1/out.xyz"
