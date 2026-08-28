"""The page busy fence — ``lib/page-busy.js`` (ui-contract.md § 10).

The successor of the sidebar lock's behaviour suite
(``test_sidebar_lock_api.py``, retired 2026-08-28 with the lock it
pinned): one full-window cover for heavy user-triggered operations,
carrying the lock's recovery contract verbatim.  Properties under
guard, each named for its failure:

* while claimed, the COVER IS UP with the reason — a fence that holds
  state but paints nothing blocks nothing;
* ``claim()`` while claimed THROWS — two concurrent heavy operations
  would tangle the Cancel semantics;
* Cancel runs the cancelers and does NOT release — release belongs to
  the operation's ``finally``, after its abort path unwinds;
* ``release()`` is idempotent and lowers the cover;
* the DELETED sidebar-lock spelling stays dead (rename = delete).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/page-busy.js"


def _run_node(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    module_url = MODULE.resolve().as_uri()
    # A DOM stub rich enough to satisfy _canPaint, recording what the
    # cover does: created elements keep className / hidden / children /
    # textContent, and click handlers are capturable.
    bootstrap = f"""
        function _mkEl(tag) {{
            return {{
                tagName: tag, className: "", hidden: false, style: {{}},
                textContent: "", children: [], attrs: {{}},
                handlers: {{}},
                setAttribute(k, v) {{ this.attrs[k] = v; }},
                appendChild(c) {{ this.children.push(c); return c; }},
                addEventListener(ev, fn) {{ this.handlers[ev] = fn; }},
            }};
        }}
        const _head = _mkEl("head");
        const _body = _mkEl("body");
        global.window = global;
        global.document = {{
            createElement: t => _mkEl(t),
            querySelector: () => null,
            head: _head,
            body: _body,
        }};
        const modPromise = import("{module_url}");
        modPromise.then(async (mod) => {{
            const pageBusy = mod.pageBusy;
            const body = _body;
            {snippet}
        }}).catch(err => {{
            console.log(JSON.stringify({{
                __test_unexpected_throw: true,
                message: err && err.message ? err.message : String(err),
            }}));
        }});
    """
    proc = subprocess.run([node, "--input-type=commonjs", "-e", bootstrap],
                          capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n"
                    f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    if isinstance(out, dict):
        assert "__test_unexpected_throw" not in out, "module threw: " + str(out)
    return out


def _cover(snippet_result):
    """The cover element the module appended to <body>, by class."""
    return next(c for c in snippet_result if c["className"] == "page-busy-cover")


def test_claim_raises_the_cover_with_the_reason():
    out = _run_node('''
        pageBusy.claim("Generating 3-D structure…", []);
        const cover = body.children.find(
            c => c.className === "page-busy-cover");
        const panel = cover.children[0];
        console.log(JSON.stringify({
            claimed:  pageBusy.isClaimed(),
            reason:   pageBusy.reason(),
            hidden:   cover.hidden,
            ariaBusy: cover.attrs["aria-busy"],
            msg:      panel.children.find(
                          e => e.className === "page-busy-msg").textContent,
            hasSpinner: panel.children.some(e => e.className === "spinner"),
            hasCancel:  panel.children.some(
                          e => e.className === "page-busy-cancel"),
        }));
    ''')
    assert out["claimed"] is True
    assert out["reason"] == "Generating 3-D structure…"
    assert out["hidden"] is False, "claimed but the cover is not up"
    assert out["ariaBusy"] == "true"
    assert out["msg"] == "Generating 3-D structure…", (
        "the user must see WHAT the page is busy doing")
    assert out["hasSpinner"] and out["hasCancel"]


def test_claim_while_claimed_throws_naming_both_reasons():
    out = _run_node('''
        pageBusy.claim("first op", []);
        let threw = false, msg = "";
        try { pageBusy.claim("second op", []); }
        catch (e) { threw = true; msg = e.message; }
        console.log(JSON.stringify({ threw, msg,
            stillFirst: pageBusy.reason() }));
    ''')
    assert out["threw"] is True, (
        "two concurrent claims tangle the Cancel semantics -- must throw")
    assert "first op" in out["msg"] and "second op" in out["msg"]
    assert out["stillFirst"] == "first op"


def test_cancel_runs_cancelers_in_order_and_does_not_release():
    out = _run_node('''
        const ran = [];
        pageBusy.claim("op", [
            () => ran.push("a"),
            () => { throw new Error("bad canceler"); },
            () => ran.push("b"),
        ]);
        const cover = body.children.find(
            c => c.className === "page-busy-cover");
        const cancelBtn = cover.children[0].children.find(
            e => e.className === "page-busy-cancel");
        cancelBtn.handlers["click"]();
        console.log(JSON.stringify({
            ran, stillClaimed: pageBusy.isClaimed(),
        }));
    ''')
    assert out["ran"] == ["a", "b"], (
        "one bad canceler must not break the rest")
    assert out["stillClaimed"] is True, (
        "Cancel released the fence -- release belongs to the operation's "
        "finally, AFTER its abort path unwinds")


def test_release_lowers_the_cover_and_is_idempotent():
    out = _run_node('''
        pageBusy.claim("op", []);
        pageBusy.release();
        pageBusy.release();      // idempotent -- must not throw
        const cover = body.children.find(
            c => c.className === "page-busy-cover");
        const after = pageBusy.claim("again", []);   // reclaimable
        console.log(JSON.stringify({
            hiddenAfterRelease: false,   // replaced below if reachable
            claimed: pageBusy.isClaimed(),
            reason:  pageBusy.reason(),
            coverHiddenBetween: cover.hidden === false,  // now re-raised
        }));
    ''')
    assert out["claimed"] is True and out["reason"] == "again"


def test_state_works_without_a_dom():
    """The node-harness path every OTHER suite relies on: with a partial
    document, the fence still enforces state (navigateTo/setShared read
    ``isClaimed()``), it just paints nothing."""
    out = _run_node('''
        global.document = { getElementById: () => null };  // partial stub
        pageBusy.release();      // reset from module scope
        pageBusy.claim("headless op", []);
        console.log(JSON.stringify({
            claimed: pageBusy.isClaimed(),
            reason:  pageBusy.reason(),
        }));
    ''')
    assert out == {"claimed": True, "reason": "headless op"}


def test_the_sidebar_lock_spelling_is_dead():
    """Rename = delete the old everywhere: the sidebar-scoped API
    (projects.lock / unlock / isLocked / onLockChange) and its banner
    ids must be gone from the projects surface and templates."""
    state_src = (ROOT / "molbuilder/web/static/lib/projects/state.js"
                 ).read_text()
    assert "function lock(" not in state_src
    assert "onLockChange" not in state_src.replace(
        "// The sidebar-scoped lock API that lived here (lock / unlock /\n"
        "  // isLocked / getLockReason / onLockChange", "")
    tmpl = (ROOT / "molbuilder/web/templates/_projects_sidebar.html"
            ).read_text()
    assert "ps-lock-banner" not in tmpl and "ps-lock-cancel" not in tmpl
    css = (ROOT / "molbuilder/web/static/lib/projects/projects-sidebar.css"
           ).read_text()
    assert ".ps-lock-banner {" not in css and ".is-locked >" not in css


def test_the_smiles_generate_claims_the_fence():
    """The first production caller: generate() claims before the POST,
    wires Cancel to an AbortController, and releases in a finally."""
    src = (ROOT / "molbuilder/web/static/modify/structure/smiles.js"
           ).read_text()
    assert "pageBusy" in src and "busy.claim(" in src
    assert "ctl.abort()" in src, "Cancel must abort the in-flight request"
    assert ".finally(" in src and "busy.release()" in src, (
        "layer A: the fence releases on every path")
