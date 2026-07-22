"""Unit tests for the Structure-tab discard-unsaved warning modal.

Pins the public API of ``molbuilder/web/static/modify/structure/
warning-modal.js`` — the single-source-of-truth dialog for "you
have unsaved canvas modifications, continuing will discard them"
across the Structure tab.

The actual <dialog> rendering is exercised by Playwright e2e
once the Structure tab UI lands; these tests cover the contract
the tab UI depends on:

  * confirmDiscardUnsaved returns a Promise<boolean>.
  * Cancel resolves false; Discard resolves true; ESC resolves
    false (the native <dialog> "cancel" event).
  * Single-instance: two near-simultaneous calls share the SAME
    pending Promise — modals don't stack.
  * Default focus lands on Cancel (the safe action).
  * The DOM structure carries the documented title/body/buttons.
  * After resolution the dialog is detached from the DOM (no leak
    on repeated invocations).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/warning-modal.js"


# Minimum-viable DOM stub the modal helper needs:
#   createElement, an element shape supporting setAttribute /
#   className / id / textContent / type / appendChild /
#   querySelector / addEventListener / focus / parentNode /
#   removeChild / showModal / close / dispatchEvent.
#
# Kept inline in each test snippet via a shared prelude so the
# tests stay readable.
_DOM_PRELUDE = r"""
function _mkEl(tag) {
    const el = {
        tagName:       String(tag).toUpperCase(),
        className:     "",
        id:            "",
        textContent:   "",
        type:          "",
        _attrs:        new Map(),
        _children:     [],
        _listeners:    new Map(),
        _parent:       null,
        _open:         false,
        focused:       false,
        get parentNode() { return el._parent; },
        setAttribute: (k, v) => el._attrs.set(k, String(v)),
        getAttribute: (k) => el._attrs.has(k) ? el._attrs.get(k) : null,
        appendChild: (c) => {
            el._children.push(c);
            c._parent = el;
            return c;
        },
        removeChild: (c) => {
            const ix = el._children.indexOf(c);
            if (ix >= 0) el._children.splice(ix, 1);
            c._parent = null;
            return c;
        },
        addEventListener: (ev, fn) => {
            if (!el._listeners.has(ev)) el._listeners.set(ev, []);
            el._listeners.get(ev).push(fn);
        },
        dispatchEvent: (event) => {
            const lis = el._listeners.get(event.type) || [];
            for (const fn of lis.slice()) fn(event);
        },
        // <dialog>-like helpers.
        showModal: () => { el._open = true; },
        show:      () => { el._open = true; },
        close: () => {
            if (!el._open) return;
            el._open = false;
            // Native <dialog> fires a "close" event when close()
            // runs.  Mirror that so the helper's wired close
            // handler is exercised on the test path.
            el.dispatchEvent({ type: "close" });
        },
        focus: () => { el.focused = true; },
        // Walk children + self to find first by-attr match.
        querySelector: (sel) => {
            // Tiny subset: `[data-action="name"]` selectors.
            const m = sel.match(/^\[data-action="([^"]+)"\]$/);
            if (!m) return null;
            const want = m[1];
            function walk(n) {
                if (n.getAttribute && n.getAttribute("data-action") === want) {
                    return n;
                }
                for (const c of (n._children || [])) {
                    const r = walk(c);
                    if (r) return r;
                }
                return null;
            }
            return walk(el);
        },
    };
    return el;
}
const _body = _mkEl("body");
global.document = {
    createElement: (tag) => _mkEl(tag),
    body: _body,
};
global.window = global;
"""


def _run_node(snippet: str, *, prelude: str = "") -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    module_path = MODULE.resolve()
    bootstrap = f"""
        {_DOM_PRELUDE}
        {prelude}
        const modal = require({json.dumps(str(module_path))});
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

    def test_methods_callable(self):
        out = _run_node('''
            console.log(JSON.stringify({
                confirmDiscardUnsaved: typeof modal.confirmDiscardUnsaved,
                isOpen:                typeof modal.isOpen,
                _reset:                typeof modal._reset,
                TITLE:                 typeof modal.TITLE,
                BODY:                  typeof modal.BODY,
                CANCEL:                typeof modal.CANCEL,
                DISCARD:               typeof modal.DISCARD,
            }));
        ''')
        assert out["confirmDiscardUnsaved"] == "function"
        assert out["isOpen"]                == "function"
        assert out["_reset"]                == "function"
        for k in ("TITLE", "BODY", "CANCEL", "DISCARD"):
            assert out[k] == "string"

    def test_documented_copy_matches_architecture_doc(self):
        """The strings come from docs/tabs/architecture.md § 5.4.
        Pin them so a future translation / wording change has to
        update the test alongside the spec."""
        out = _run_node('''
            console.log(JSON.stringify({
                TITLE:   modal.TITLE,
                BODY:    modal.BODY,
                CANCEL:  modal.CANCEL,
                DISCARD: modal.DISCARD,
            }));
        ''')
        assert out["TITLE"] == "Unsaved modifications"
        assert out["BODY"] == (
            "You have unsaved changes to the current canvas. "
            "Continuing will discard them."
        )
        assert out["CANCEL"] == "Cancel"
        assert out["DISCARD"] == "Discard and continue"


# ----- Promise resolution ---------------------------------------- #


class TestPromiseResolution:

    def test_returns_a_thenable(self):
        out = _run_node('''
            const r = modal.confirmDiscardUnsaved();
            console.log(JSON.stringify({
                isThenable: r && typeof r.then === "function",
            }));
            modal._reset();
        ''')
        assert out["isThenable"] is True

    def test_cancel_resolves_false(self):
        out = _run_node('''
            const p = modal.confirmDiscardUnsaved();
            const dialog = document.body._children[document.body._children.length - 1];
            const cancelBtn = dialog.querySelector(\'[data-action="cancel"]\');
            cancelBtn._listeners.get("click")[0]();
            const v = await p;
            console.log(JSON.stringify(v));
        ''')
        assert out is False

    def test_discard_resolves_true(self):
        out = _run_node('''
            const p = modal.confirmDiscardUnsaved();
            const dialog = document.body._children[document.body._children.length - 1];
            const discardBtn = dialog.querySelector(\'[data-action="discard"]\');
            discardBtn._listeners.get("click")[0]();
            const v = await p;
            console.log(JSON.stringify(v));
        ''')
        assert out is True

    def test_esc_resolves_false(self):
        """Native <dialog> fires a 'cancel' event when ESC is pressed
        — the helper must treat that as a Cancel-equivalent (false)."""
        out = _run_node('''
            const p = modal.confirmDiscardUnsaved();
            const dialog = document.body._children[document.body._children.length - 1];
            dialog.dispatchEvent({ type: "cancel" });
            const v = await p;
            console.log(JSON.stringify(v));
        ''')
        assert out is False


# ----- DOM structure --------------------------------------------- #


class TestDOMStructure:

    def test_dialog_carries_documented_strings(self):
        out = _run_node('''
            const p = modal.confirmDiscardUnsaved();
            const dialog = document.body._children[document.body._children.length - 1];
            // Walk the children for the title (h2), body (p), and
            // both buttons.
            function findByTag(root, tag) {
                if (root.tagName === tag) return root;
                for (const c of (root._children || [])) {
                    const r = findByTag(c, tag);
                    if (r) return r;
                }
                return null;
            }
            const title = findByTag(dialog, "H2");
            const body  = findByTag(dialog, "P");
            const cancel  = dialog.querySelector(\'[data-action="cancel"]\');
            const discard = dialog.querySelector(\'[data-action="discard"]\');
            console.log(JSON.stringify({
                tag:           dialog.tagName,
                titleText:     title  ? title.textContent  : null,
                bodyText:      body   ? body.textContent   : null,
                cancelText:    cancel  ? cancel.textContent  : null,
                discardText:   discard ? discard.textContent : null,
                cancelType:    cancel  ? cancel.type  : null,
                discardType:   discard ? discard.type : null,
            }));
            modal._reset();
        ''')
        assert out["tag"] == "DIALOG"
        assert out["titleText"] == "Unsaved modifications"
        assert "unsaved changes" in out["bodyText"]
        assert out["cancelText"] == "Cancel"
        assert out["discardText"] == "Discard and continue"
        # Both buttons type="button" so the dialog form-default
        # submit semantics don't fire on click.
        assert out["cancelType"] == "button"
        assert out["discardType"] == "button"

    def test_aria_attributes_present(self):
        """Accessibility: dialog wires aria-labelledby + aria-describedby
        to the title + body so screen readers announce the modal."""
        out = _run_node('''
            modal.confirmDiscardUnsaved();
            const dialog = document.body._children[document.body._children.length - 1];
            console.log(JSON.stringify({
                labelledby:  dialog.getAttribute("aria-labelledby"),
                describedby: dialog.getAttribute("aria-describedby"),
            }));
            modal._reset();
        ''')
        assert out["labelledby"]
        assert out["describedby"]
        assert out["labelledby"] != out["describedby"]


# ----- Default focus --------------------------------------------- #


class TestDefaultFocus:

    def test_focus_lands_on_cancel(self):
        """Per § 5.4 Cancel is the default focus — the safe action.
        Pin the focus call so a future refactor that drops it gets
        caught."""
        out = _run_node('''
            modal.confirmDiscardUnsaved();
            const dialog = document.body._children[document.body._children.length - 1];
            const cancel  = dialog.querySelector(\'[data-action="cancel"]\');
            const discard = dialog.querySelector(\'[data-action="discard"]\');
            console.log(JSON.stringify({
                cancelFocused:  cancel.focused,
                discardFocused: discard.focused,
            }));
            modal._reset();
        ''')
        assert out["cancelFocused"] is True
        assert out["discardFocused"] is False


# ----- Single-instance ------------------------------------------- #


class TestSingleInstance:

    def test_isOpen_reflects_open_state(self):
        out = _run_node('''
            const before = modal.isOpen();
            modal.confirmDiscardUnsaved();
            const during = modal.isOpen();
            modal._reset();
            const after = modal.isOpen();
            console.log(JSON.stringify({
                before: before, during: during, after: after,
            }));
        ''')
        assert out == {"before": False, "during": True, "after": False}

    def test_concurrent_calls_share_promise(self):
        """A second call while a prior modal is open returns the
        same in-flight Promise — modals must NOT stack."""
        out = _run_node('''
            const p1 = modal.confirmDiscardUnsaved();
            const p2 = modal.confirmDiscardUnsaved();
            // Only one <dialog> in body.
            const dialogCount = document.body._children.filter(
                c => c.tagName === "DIALOG").length;
            const same = p1 === p2;
            modal._reset();
            const r1 = await p1;
            const r2 = await p2;
            console.log(JSON.stringify({
                same: same,
                dialogCount: dialogCount,
                r1: r1, r2: r2,
            }));
        ''')
        assert out["same"] is True
        assert out["dialogCount"] == 1
        # Both resolve to the same Cancel value because _reset()
        # cancels.
        assert out["r1"] is False
        assert out["r2"] is False

    def test_dialog_detached_from_dom_after_resolution(self):
        """After Cancel / Discard the <dialog> is removed from
        document.body so a future invocation lands a fresh element.
        Otherwise the body would accumulate one detached <dialog>
        per Save / Generate cycle."""
        out = _run_node('''
            // First cycle: Cancel.
            const p1 = modal.confirmDiscardUnsaved();
            const d1 = document.body._children[document.body._children.length - 1];
            d1.querySelector(\'[data-action="cancel"]\')
              ._listeners.get("click")[0]();
            await p1;
            const dialogsAfter1 = document.body._children.filter(
                c => c.tagName === "DIALOG").length;
            // Second cycle: Discard.
            const p2 = modal.confirmDiscardUnsaved();
            const d2 = document.body._children[document.body._children.length - 1];
            d2.querySelector(\'[data-action="discard"]\')
              ._listeners.get("click")[0]();
            await p2;
            const dialogsAfter2 = document.body._children.filter(
                c => c.tagName === "DIALOG").length;
            console.log(JSON.stringify({
                dialogsAfter1:  dialogsAfter1,
                dialogsAfter2:  dialogsAfter2,
                differentNodes: d1 !== d2,
            }));
        ''')
        # Both cycles leave the body with no detached <dialog>s.
        assert out["dialogsAfter1"] == 0
        assert out["dialogsAfter2"] == 0
        # And the second invocation built a NEW dialog node, not
        # reused the disposed first one.
        assert out["differentNodes"] is True

    def test_subsequent_calls_after_close_get_fresh_modal(self):
        """After a Cancel, a SUBSEQUENT call (not concurrent) gets
        a fresh modal that resolves on its own buttons."""
        out = _run_node('''
            const p1 = modal.confirmDiscardUnsaved();
            const d1 = document.body._children[document.body._children.length - 1];
            d1.querySelector(\'[data-action="cancel"]\')
              ._listeners.get("click")[0]();
            const r1 = await p1;
            const p2 = modal.confirmDiscardUnsaved();
            const d2 = document.body._children[document.body._children.length - 1];
            d2.querySelector(\'[data-action="discard"]\')
              ._listeners.get("click")[0]();
            const r2 = await p2;
            console.log(JSON.stringify({ r1: r1, r2: r2 }));
        ''')
        assert out == {"r1": False, "r2": True}


# ----- Dialog lifecycle ------------------------------------------ #


class TestDialogLifecycle:

    def test_showModal_called_on_open(self):
        """The helper opens with showModal() (NOT show()) — the
        difference is whether the page below is interactable while
        the modal is up.  An unsaved-changes confirmation MUST
        block interaction; show() leaves the page clickable."""
        out = _run_node('''
            // Wrap createElement so the dialog node tracks which
            // open method was called.
            const origCreate = document.createElement;
            document.createElement = function (tag) {
                const el = origCreate(tag);
                if (tag === "dialog") {
                    el._opensVia = null;
                    el.showModal = function () {
                        el._opensVia = "showModal";
                        el._open = true;
                    };
                    el.show = function () {
                        el._opensVia = "show";
                        el._open = true;
                    };
                }
                return el;
            };
            modal.confirmDiscardUnsaved();
            const dialog = document.body._children[document.body._children.length - 1];
            console.log(JSON.stringify({ opensVia: dialog._opensVia }));
            modal._reset();
        ''')
        assert out["opensVia"] == "showModal"

    def test_no_doc_returns_rejected_promise(self):
        """If somehow the helper runs without a document (Node
        without the DOM stub installed), it must NOT throw — it
        returns a rejected Promise the caller can surface as a
        UX-level error.

        Reproduces the no-document state by wiping global.document
        for this call only; the helper's fallback to ``root.document``
        is what surfaces the missing-DOM case in real life (a stale
        SSR import path, an embedded context without a window).
        """
        out = _run_node('''
            global.document = undefined;
            const p = modal.confirmDiscardUnsaved();
            let rejected = false;
            let msg = "";
            try { await p; }
            catch (e) { rejected = true; msg = e.message; }
            console.log(JSON.stringify({
                rejected: rejected,
                msg:      msg,
            }));
        ''')
        assert out["rejected"] is True
        assert "no document" in out["msg"]
