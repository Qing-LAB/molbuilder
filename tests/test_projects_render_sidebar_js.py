"""Unit tests for ``renderSidebar(state)`` (sidebar gap M2, #166).

The renderSidebar function is the single point of selection-state-to-
DOM sync.  Wired as a ``projects.onChange`` subscriber so every
``setShared`` / ``navigateTo`` automatically syncs the entry-marker
+ status line -- no inline DOM-mutation calls from click handlers.

These tests stub a minimal DOM under Node (sufficient for
querySelector + classList + textContent) and verify:
  * setShared fires the subscriber synchronously.
  * Matching entries get ``.is-selected``.
  * Non-matching entries lose ``.is-selected``.
  * Blank file clears all selections + writes "No file selected."
  * Programmatic setShared from another module (the result-list
    dropdown's use case) updates the DOM via the subscriber.

DOM stubbing covers ONLY what list.js touches; this is not a
full jsdom.  The point is to pin the subscriber wiring + the
core DOM-sync logic without standing up a browser.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LIST_MODULE = ROOT / "molbuilder/web/static/lib/projects/list.js"


def _run_node(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    list_url = LIST_MODULE.resolve().as_uri()
    bootstrap = f"""
        // --- Minimal DOM stub --------------------------------- //
        // Each <li> has: classList (with add/remove/contains),
        // attributes (for [data-path="..."] selector), textContent,
        // style, appendChild, addEventListener, querySelector,
        // querySelectorAll, dataset.
        function _mkEl(tag) {{
            const el = {{
                tagName:        tag.toUpperCase(),
                _attrs:         new Map(),
                _children:      [],
                _listeners:     [],
                _classes:       new Set(),
                _textContent:   "",
                _innerHTML:     "",
                _title:         "",
                style:          {{ cssText: "" }},
                dataset:        {{}},
                classList: {{
                    add:      (...cs) => cs.forEach(c => el._classes.add(c)),
                    remove:   (...cs) => cs.forEach(c => el._classes.delete(c)),
                    contains: c => el._classes.has(c),
                    toggle:   c => el._classes.has(c)
                                    ? el._classes.delete(c)
                                    : el._classes.add(c),
                }},
                get className () {{
                    return [...el._classes].join(" ");
                }},
                set className (v) {{
                    el._classes = new Set(String(v).split(/\\s+/).filter(Boolean));
                }},
                get textContent () {{ return el._textContent; }},
                set textContent (v) {{
                    el._textContent = String(v);
                    el._innerHTML = "";
                    el._children = [];
                }},
                get innerHTML () {{ return el._innerHTML; }},
                set innerHTML (v) {{
                    el._innerHTML = String(v);
                    el._textContent = "";
                    el._children = [];
                }},
                get title () {{ return el._title; }},
                set title (v) {{ el._title = String(v); }},
                appendChild: c => {{ el._children.push(c); return c; }},
                addEventListener: (ev, fn) => {{
                    el._listeners.push({{ ev, fn }});
                }},
                setAttribute: (k, v) => el._attrs.set(k, String(v)),
                getAttribute: (k) => el._attrs.has(k) ? el._attrs.get(k) : null,
                querySelector: sel => _query(el, sel)[0] || null,
                querySelectorAll: sel => _query(el, sel),
            }};
            return el;
        }}
        function _query(root, sel) {{
            // Support a TINY subset:
            //   .class
            //   .class.is-selected
            //   .ps-entry[data-path="x"]
            const out = [];
            function walk(n) {{
                if (_matches(n, sel)) out.push(n);
                for (const c of (n._children || [])) walk(c);
            }}
            walk(root);
            return out;
        }}
        function _matches(n, sel) {{
            if (!n || !n._classes) return false;
            // ``.is-selected`` form
            const mClass = sel.match(/^\\.([a-zA-Z0-9_-]+)(?:\\.([a-zA-Z0-9_-]+))?$/);
            if (mClass) {{
                if (!n._classes.has(mClass[1])) return false;
                if (mClass[2] && !n._classes.has(mClass[2])) return false;
                return true;
            }}
            // ``.ps-entry[data-path="x"]`` form (used by renderSidebar)
            const mData = sel.match(/^\\.([a-zA-Z0-9_-]+)\\[data-path="(.*)"\\]$/);
            if (mData) {{
                if (!n._classes.has(mData[1])) return false;
                const dp = n._attrs && n._attrs.get && n._attrs.get("data-path");
                return dp === mData[2];
            }}
            // ``#id`` form (used by selection-status query)
            const mId = sel.match(/^#([a-zA-Z0-9_-]+)$/);
            if (mId) return n._attrs && n._attrs.get("id") === mId[1];
            // ``#parent .child`` two-level descendant for actions
            const mDesc = sel.match(/^#([a-zA-Z0-9_-]+)\\s+\\.([a-zA-Z0-9_-]+)$/);
            if (mDesc) {{
                // Find an ancestor with the id, then descendants by class.
                // For our stub: just check class match (we mounted a
                // ps-actions root with a .ps-selection child).
                return n._classes.has(mDesc[2]);
            }}
            return false;
        }}
        // Build the stub DOM: ps-list with two ps-entry children;
        // ps-actions with a ps-selection status span.
        const psList     = _mkEl("ul");
        psList.setAttribute("id", "ps-list");
        const entryA = _mkEl("li");
        entryA._classes.add("ps-entry");
        entryA.setAttribute("data-path", "/p/file-a.out");
        const entryB = _mkEl("li");
        entryB._classes.add("ps-entry");
        entryB.setAttribute("data-path", "/p/file-b.out");
        psList.appendChild(entryA);
        psList.appendChild(entryB);
        const psActions  = _mkEl("div");
        psActions.setAttribute("id", "ps-actions");
        const psSelection = _mkEl("span");
        psSelection._classes.add("ps-selection");
        psActions.appendChild(psSelection);
        const psCrumb = _mkEl("nav");
        psCrumb.setAttribute("id", "ps-breadcrumb");
        global.document = {{
            getElementById: id => {{
                if (id === "ps-list")       return psList;
                if (id === "ps-breadcrumb") return psCrumb;
                if (id === "ps-actions")    return psActions;
                return null;
            }},
            querySelector: sel => {{
                if (sel === "#ps-actions .ps-selection") return psSelection;
                return null;
            }},
            addEventListener: () => {{}},
            createElement: tag => _mkEl(tag),
        }};
        global.window = global;
        global.navigator = {{ language: "en-US" }};
        const _store = new Map();
        global.sessionStorage = {{
            getItem:    k => _store.has(k) ? _store.get(k) : null,
            setItem:    (k, v) => _store.set(k, String(v)),
            removeItem: k => _store.delete(k),
            clear:      () => _store.clear(),
        }};
        global.fetch = async () => ({{
            ok: true, status: 200, json: async () => ({{ ok: true }}),
        }});
        // No CSS available -- list.js's _cssEscape falls back to its
        // manual escape; we provide a no-op CSS object so the typeof
        // check returns "object" but CSS.escape is missing.
        global.CSS = undefined;
        // Import + run.
        const listPromise = import("{list_url}");
        listPromise.then(async (list) => {{
            // We need access to state.js too for setShared.
            // list.js re-exports nothing useful; instead we import
            // state directly and the two share the projects.* surface.
            const state = await import("{(ROOT / 'molbuilder/web/static/lib/projects/state.js').resolve().as_uri()}");
            list.initList();
            // Make helpers available to the snippet.
            global.list  = list;
            global.state = state;
            global.psList = psList;
            global.psSelection = psSelection;
            global.entryA = entryA;
            global.entryB = entryB;
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
        "module threw: " + str(out)
    )
    return out


class TestRenderSidebarSubscriber:

    def test_setShared_marks_matching_entry(self):
        out = _run_node('''
            state.setShared("/p", "/p/file-a.out");
            console.log(JSON.stringify({
                a_selected: entryA._classes.has("is-selected"),
                b_selected: entryB._classes.has("is-selected"),
            }));
        ''')
        assert out == {"a_selected": True, "b_selected": False}

    def test_setShared_swaps_selection(self):
        """When the user picks a different file, the new one becomes
        selected AND the previous one loses its mark.  Pin both
        sides to catch a regression that "adds without removing"."""
        out = _run_node('''
            state.setShared("/p", "/p/file-a.out");
            state.setShared("/p", "/p/file-b.out");
            console.log(JSON.stringify({
                a_selected: entryA._classes.has("is-selected"),
                b_selected: entryB._classes.has("is-selected"),
            }));
        ''')
        assert out == {"a_selected": False, "b_selected": True}

    def test_setShared_blank_file_clears_all(self):
        """Empty file ("") clears every entry's selection AND writes
        "No file selected." to the status line."""
        out = _run_node('''
            state.setShared("/p", "/p/file-a.out");
            state.setShared("/p", "");
            console.log(JSON.stringify({
                a_selected: entryA._classes.has("is-selected"),
                b_selected: entryB._classes.has("is-selected"),
                status:     psSelection._textContent,
            }));
        ''')
        assert out["a_selected"] is False
        assert out["b_selected"] is False
        # Cleared status reads "No file selected."
        assert out["status"] == "No file selected."

    def test_setShared_updates_selection_status_line(self):
        """The ``ps-selection`` span gets a "Selected: ..." innerHTML
        scaffold; the inner <strong> textContent + the title attribute
        are not testable under the Node DOM stub (which doesn't parse
        innerHTML strings), but the scaffold's presence proves
        renderSidebar ran the status-update branch."""
        out = _run_node('''
            state.setShared("/p", "/p/file-b.out");
            console.log(JSON.stringify({
                inner: psSelection._innerHTML,
            }));
        ''')
        assert "Selected:" in out["inner"]
        assert "<strong>" in out["inner"]

    def test_setShared_nonexistent_file_clears_marker(self):
        """If the file isn't in the currently-rendered list (e.g.
        renamed mid-flight, or the user navigated to a different
        dir), no entry gets marked AND the previous selection is
        cleared.  Honest UI: don't pretend a non-listed file is
        selected."""
        out = _run_node('''
            state.setShared("/p", "/p/file-a.out");
            // File doesn't exist in the stubbed list:
            state.setShared("/p", "/p/never-rendered.out");
            console.log(JSON.stringify({
                a_selected: entryA._classes.has("is-selected"),
                b_selected: entryB._classes.has("is-selected"),
            }));
        ''')
        # The non-listed file is still recorded in state but no
        # entry visually marks it.  No "magic" selection of a
        # non-rendered row.
        assert out["a_selected"] is False
        assert out["b_selected"] is False

    def test_locked_setShared_does_not_update_dom(self):
        """The lock guard (#177) rejects programmatic setShared while
        the lock is held; subscriber must NOT fire so the DOM stays
        synced to the pre-locked state."""
        out = _run_node('''
            state.setShared("/p", "/p/file-a.out");
            state.projects.lock("Saving FDF...", []);
            const r = state.setShared("/p", "/p/file-b.out");
            console.log(JSON.stringify({
                envelope:   r,
                a_selected: entryA._classes.has("is-selected"),
                b_selected: entryB._classes.has("is-selected"),
            }));
        ''')
        assert out["envelope"]["ok"] is False
        # DOM still shows file-a as selected; the locked setShared
        # was rejected before the subscriber would have run.
        assert out["a_selected"] is True
        assert out["b_selected"] is False

    def test_programmatic_setShared_updates_dom(self):
        """The /results result-list dropdown's use case: a
        non-sidebar module calls projects.setShared(dir, file) to
        move the cursor without re-listing the directory.  The
        subscriber syncs the sidebar's DOM even though the click
        came from outside."""
        out = _run_node('''
            state.projects.setShared("/p", "/p/file-b.out");
            console.log(JSON.stringify({
                a_selected: entryA._classes.has("is-selected"),
                b_selected: entryB._classes.has("is-selected"),
            }));
        ''')
        assert out == {"a_selected": False, "b_selected": True}
