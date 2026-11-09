"""The page busy fence — ``lib/page-busy.js`` (ui-contract.md § 10).

One full-window cover for heavy user-triggered operations.  Properties
under guard, each named for its failure:

* while claimed, the COVER IS UP with the reason — a fence that holds
  state but paints nothing blocks nothing;
* ``claim()`` while claimed THROWS — two concurrent heavy operations
  would tangle the Cancel semantics;
* Cancel runs the cancelers and does NOT release — release belongs to
  the operation's ``finally``, after its abort path unwinds;
* ``release()`` is idempotent and lowers the cover;
* the KEYBOARD is fenced too, and Cancel is shown only with cancelers --
  Enter on the button just clicked would otherwise send again, uncovered
  by any Cancel (§ 10);
* ``run`` -- the way an operation takes the fence -- releases when the
  operation throws, and inside an outer claim runs there, neither throwing
  nor lowering that claim: a stuck cover, or one lowered mid-operation, is
  what a hand-rolled claim got wrong;
* focus comes back WHERE THE PERSON LEFT IT -- to the control that held
  it, or, when the fence went up as Tab or a click moved focus, to where it
  went; a control the operation redrew found again, never guessed: a value
  typed and Tab otherwise left the person on the page's body.
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
    # textContent, and click handlers are capturable; the document keeps
    # its listeners, which ``fire`` calls as the browser would.
    bootstrap = f"""
        function _mkEl(tag) {{
            return {{
                tagName: tag, className: "", hidden: false, style: {{}},
                textContent: "", children: [], attrs: {{}},
                handlers: {{}},
                setAttribute(k, v) {{ this.attrs[k] = v; }},
                appendChild(c) {{ this.children.push(c); return c; }},
                addEventListener(ev, fn) {{ this.handlers[ev] = fn; }},
                contains(x) {{ return x === this || this.children.some(
                    (c) => c.contains(x)); }},
            }};
        }}
        const _head = _mkEl("head");
        const _body = _mkEl("body");
        global.window = global;
        const _on = {{}};
        global.document = {{
            createElement: t => _mkEl(t),
            querySelector: () => null,
            head: _head,
            body: _body,
            activeElement: _body,
            addEventListener(ev, fn) {{ (_on[ev] = _on[ev] || []).push(fn); }},
            removeEventListener(ev, fn) {{
                _on[ev] = (_on[ev] || []).filter((f) => f !== fn); }},
        }};
        const fire = (ev, e) => (_on[ev] || []).slice().forEach((f) => f(e));
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
    """`ui-contract.md` § 10.  THE SILENT FAILURE: every heavy operation leaving
    the page under its cover."""
    out = _run_node('''
        pageBusy.claim("op", []);
        const cover = body.children.find(
            c => c.className === "page-busy-cover");
        const up = cover.hidden;
        pageBusy.release();
        const lowered = cover.hidden;
        pageBusy.release();      // idempotent -- must not throw
        pageBusy.claim("again", []);   // reclaimable
        console.log(JSON.stringify({
            up, lowered,
            claimed: pageBusy.isClaimed(),
            reason:  pageBusy.reason(),
        }));
    ''')
    assert out["up"] is False
    assert out["lowered"] is True, "released, and the page is still covered"
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


def test_the_keyboard_is_fenced_and_cancel_shows_only_with_cancelers():
    """§ 10: a key pressed outside the cover does nothing while claimed, and
    focus rests on the cover -- on Cancel when there is one to run, on the
    panel when there is none -- until the release lets the page go."""
    out = _run_node('''
        let focused = null;
        const mk = document.createElement;
        document.createElement = t => Object.assign(mk(t), {
            contains(x) { return x === this || this.children.includes(x); },
            focus() { focused = this.className; },
        });
        const listeners = {};
        document.addEventListener = (ev, fn) => { listeners[ev] = fn; };
        document.removeEventListener = (ev, fn) => {
            if (listeners[ev] === fn) delete listeners[ev];
        };
        // the button just clicked holds focus as the fence goes up
        document.activeElement = { isConnected: true, focus() {} };
        pageBusy.claim("nothing to cancel", []);
        const cover = body.children.find(
            c => c.className === "page-busy-cover");
        const cancel = cover.children[0].children.find(
            e => e.className === "page-busy-cancel");
        const hiddenWithout = cancel.hidden;
        const focusWithout = focused;
        let stopped = false;
        listeners["keydown"]({ target: { closest: () => null },
                               preventDefault() { stopped = true; },
                               stopPropagation() {} });
        pageBusy.release();
        const fencedAfter = "keydown" in listeners;
        pageBusy.claim("cancellable", [() => {}]);
        const hiddenWith = cancel.hidden;
        const focusWith = focused;
        pageBusy.release();
        console.log(JSON.stringify({ hiddenWithout, focusWithout, stopped,
                                     fencedAfter, hiddenWith, focusWith }));
    ''')
    assert out["stopped"] is True, "a key outside the cover still acted"
    assert out["fencedAfter"] is False, "the release left the keyboard fenced"
    assert (out["hiddenWithout"], out["focusWithout"]) == (
        True, "page-busy-panel"), out
    assert (out["hiddenWith"], out["focusWith"]) == (
        False, "page-busy-cancel"), out


def test_run_releases_on_a_throw_and_composes_inside_a_claim():
    """§ 10: ``run`` claims, runs and releases in a ``finally``; held
    already, it runs the step inside the claim that holds it."""
    out = _run_node('''
        let during = null;
        const got = await pageBusy.run("Saving…", [], async () => {
            during = pageBusy.reason();
            return 7;
        });
        const afterOk = pageBusy.isClaimed();
        let thrown = null;
        try {
            await pageBusy.run("Failing…", [], async () => {
                throw new Error("the operation failed");
            });
        } catch (e) { thrown = e.message; }
        const afterThrow = pageBusy.isClaimed();
        pageBusy.claim("outer op", []);
        let inner = null;
        await pageBusy.run("a step", [], async () => {
            inner = pageBusy.reason();
        });
        const outerHeld = pageBusy.isClaimed() && pageBusy.reason();
        pageBusy.release();
        console.log(JSON.stringify({ during, got, afterOk, thrown,
                                     afterThrow, inner, outerHeld }));
    ''')
    assert (out["during"], out["got"], out["afterOk"]) == ("Saving…", 7, False)
    assert out["thrown"] == "the operation failed", out
    assert out["afterThrow"] is False, "a thrown operation left the cover up"
    assert out["inner"] == "outer op", "the step did not run inside the claim"
    assert out["outerHeld"] == "outer op", (
        "the step lowered the claim of the operation holding it")


#: A page of controls, enough to find one again: each node has its parent,
#: its id and attributes, and is in the page while its chain reaches it.
_PAGE = """
    function node(tag, attrs, parent) {
        const n = {
            tagName: tag.toUpperCase(), id: attrs.id || "",
            attrs: Object.assign({}, attrs), children: [],
            parentElement: parent || null, textContent: attrs.text || "",
            getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
            get isConnected() {
                let p = this; while (p.parentElement) p = p.parentElement;
                return p === page; },
            focus() {               // a disabled control takes no focus
                if (!("disabled" in this.attrs)) document.activeElement = this; },
            querySelectorAll(tag) {
                const out = [];
                const walk = (x) => x.children.forEach((c) => {
                    if (c.tagName === tag.toUpperCase()) out.push(c);
                    walk(c); });
                walk(this);
                return out; },
        };
        if (parent) parent.children.push(n);
        return n;
    }
    const page = node("main", {});
    // The cover's controls take focus as a browser's do.
    const make = document.createElement;
    document.createElement = (tag) => Object.assign(make(tag), {
        focus() { document.activeElement = this; } });
    document.getElementById = (id) => {
        let hit = null;
        const walk = (x) => x.children.forEach((c) => {
            if (!hit && c.id === id) hit = c; walk(c); });
        walk(page);
        return hit;
    };
    // A FOCUSED CONTROL TAKEN OUT says focus went to nothing, still in the
    // page as it says so -- as Chrome does.
    function detach(n) {
        if (document.activeElement === n) {
            fire("focusout", { target: n, relatedTarget: null });
            document.activeElement = body;
        }
        const up = n.parentElement;
        up.children.splice(up.children.indexOf(n), 1);
        n.parentElement = null;
    }
    // A run card: its panel carries an id, its fields their names.
    const card = node("div", { id: "ts-steppanel-2" }, page);
    const field = (name, attrs) => node("input",
        Object.assign({ "aria-label": name }, attrs || {}), card);
    // THE WRITE REDRAWS THE CARD: every field replaced by a new one.
    function redraw(names) {
        card.children.slice().forEach(detach);
        return names.map((nm) => field(nm));
    }
    async function typedAndLeft(left, to, endsAtOnce) {
        fire("focusin", { target: left });
        // the field's change runs with no control named focused
        document.activeElement = body;
        pageBusy.claim("Writing", []);
        // a write that ends inside the change, before the browser's move
        if (endsAtOnce) pageBusy.release();
        // THE BROWSER FINISHES ITS MOVE -- unless focus was taken meanwhile.
        const taken = document.activeElement !== body;
        fire("focusout", { target: left, relatedTarget:
                           taken ? document.activeElement : (to || null) });
        if (to && !taken) {
            fire("focusin", { target: to });
            document.activeElement = to;
        }
        await Promise.resolve();        // the operation waits on the server
    }
"""

#: (what happened, the steps, the field focus is on after the release)
_FOCUS = [
    ("Tab: focus moves on to the next field, found again by its name",
     """const a = field("value for wall_slack");
        const n = field("value for mpi_np");
        await typedAndLeft(a, n);
        redraw(["value for wall_slack", "value for mpi_np"]);""",
     "value for mpi_np"),
    ("a click on bare page: focus is left on no control",
     """const a = field("value for wall_slack");
        await typedAndLeft(a, null);
        redraw(["value for wall_slack"]);""",
     None),
    ("a button that held focus gets it back",
     """const go = node("button", { text: "Prep" }, card);
        fire("focusin", { target: go });
        document.activeElement = go;
        pageBusy.claim("Prepping", []);""",
     "Prep"),
    ("Shift+Tab to a group's tab: found again by its id",
     """const nav = node("nav", {}, page);
        const tab = node("button", { id: "ts-steptab-1",
                                     text: "group 2 \u00b7 device" }, nav);
        const a = field("value for wall_slack");
        await typedAndLeft(a, tab);
        detach(tab); detach(nav);
        redraw(["value for wall_slack"]);
        node("button", { id: "ts-steptab-1", text: "group 2, drawn again" },
             node("nav", {}, page));""",
     "group 2, drawn again"),
    ("a write ending inside Tab's move: focus moves on",
     """const a = field("value for time");
        const n = field("value for mem");
        await typedAndLeft(a, n, true);""",
     "value for mem"),
    ("a control the operation gave focus to keeps it",
     """const go = node("button", { text: "Prep" }, card);
        fire("focusin", { target: go });
        document.activeElement = go;
        pageBusy.claim("Prepping", []);
        const told = field("the refusal");
        fire("focusin", { target: told });
        told.focus();""",
     "the refusal"),
    ("two stages named alike, before Save refuses it: none is guessed",
     """const table = node("table", { id: "ts-stage-table" }, page);
        const cell = (nm) => node("input", { "aria-label": nm }, table);
        const a = cell("seed mesh_cutoff");
        const n = cell("seed mesh_cutoff");
        await typedAndLeft(a, n);
        detach(a); detach(n);
        cell("seed mesh_cutoff"); cell("seed mesh_cutoff");""",
     None),
    ("a second operation right after a button's: focus back on the button",
     """const go = node("button", { text: "Prep" }, card);
        fire("focusin", { target: go });
        document.activeElement = go;
        pageBusy.claim("Prepping", []);
        pageBusy.release();
        pageBusy.claim("Reading the folder again", []);
        await new Promise((r) => setTimeout(r, 0));   // it waits""",
     "Prep"),
    ("a button the operation disabled gets focus back once enabled again",
     """const go = node("button", { text: "Generate" }, card);
        fire("focusin", { target: go });
        document.activeElement = go;
        pageBusy.claim("Generating", []);
        go.attrs.disabled = "";
        await Promise.resolve();
        pageBusy.release();
        delete go.attrs.disabled;       // the caller, as its operation ends""",
     "Generate"),
    ("a control taken out with no operation running is not brought back",
     """const add = node("button", { text: "+ Add setting" }, card);
        fire("focusin", { target: add });
        document.activeElement = add;
        detach(add);                    // its card drawn again, unfenced
        node("button", { text: "+ Add setting" }, card);
        await Promise.resolve();
        // later, an operation that starts with nothing focused
        pageBusy.claim("Proposing the stages", []);""",
     None),
    ("the window left while a write runs: focus comes back where it went",
     """const a = field("value for wall_slack");
        const n = field("value for mpi_np");
        await typedAndLeft(a, n);
        document.hasFocus = () => false;
        fire("focusout", { target: n, relatedTarget: null });
        await Promise.resolve();
        redraw(["value for wall_slack", "value for mpi_np"]);""",
     "value for mpi_np"),
]


@pytest.mark.parametrize("what,steps,lands", _FOCUS,
                         ids=[r[0] for r in _FOCUS])
def test_focus_comes_back_where_the_person_left_it(what, steps, lands):
    """§ 10: a control that held focus gets it back at the release; a fence
    that went up while focus was moving -- a field's change runs as Tab or a
    click leaves it, the body named focused -- leaves the move to finish, and
    focus comes back to where it went; a control the operation redrew is
    found again by its id, or by its name in its container when one control
    there carries it.  Through 2.68.2 the release found nothing to give focus
    to, and a value typed and Tab left the person on the page's body.

    MUTATIONS THIS MUST FAIL AGAINST: the claim taking focus while it was
    moving; the release refocusing only a control still in the page; the
    record kept when focus went to nothing; a redraw taking the focused
    control out taken for a click on bare page; the record kept for a
    control already out of the page when an operation begins; the window
    leaving taken for a click on bare page; focus given back at once, before the caller
    ended its operation and inside a move the browser is making; a give-back
    skipped while another operation holds the fence; focus taken from a
    control that has it; a name two controls carry taken for the first."""
    out = _run_node(_PAGE + steps + """
        await Promise.resolve();        // the operation's last step returns
        pageBusy.release();
        await new Promise((r) => setTimeout(r, 0));
        const at = document.activeElement;
        console.log(JSON.stringify({ lands: (at && at !== body && at.attrs)
            ? (at.attrs["aria-label"] || at.textContent) : null }));
    """)
    assert out == {"lands": lands}, what
