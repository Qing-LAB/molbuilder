"""The one confirm -- ``molbuilder/web/static/lib/warning-modal.js``.

Every yes/no question the app asks goes through it (`ui-contract.md` § 11),
and it is built on the one dialog component (`lib/dialog.js`), so it answers
by that component's rules.  What these rows pin, each named for its failure:

  * HOW IT IS ANSWERED -- the costly button yes; Cancel, Esc and a click off
    the surface no; a click on its own padding nothing at all (a confirm that
    closed under a stray click would lose the person's answer);
  * THE FIXED COPY of the discard-unsaved question (`runtime.md` § 3), on a
    `<dialog>` the component dresses, its costly answer marked and last;
  * Cancel takes the focus, so the key that closes it keeps what you had;
  * ONE DIALOG AT A TIME -- a second question answers the first no;
  * an answered dialog leaves the page, and is opened modally.

Run under Node against a DOM stub small enough to read: the module is an ES
module and imports the component beside it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._node_esm import run_node

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/warning-modal.js"

#: What the module and the component touch, and nothing more: an element
#: holds its children, its listeners and its attributes; a `<dialog>` opens
#: and closes, firing `close` as the platform does; its box is 0..100.
_DOM = r"""
function _mkEl(tag) {
  const el = {
    tagName: String(tag).toUpperCase(), className: "", id: "",
    textContent: "", type: "", _attrs: new Map(), _children: [],
    _listeners: new Map(), _parent: null, _open: false, focused: false,
    _opensVia: null,
    get parentNode() { return el._parent; },
    setAttribute: (k, v) => el._attrs.set(k, String(v)),
    getAttribute: (k) => el._attrs.has(k) ? el._attrs.get(k) : null,
    appendChild: (c) => { el._children.push(c); c._parent = el; return c; },
    removeChild: (c) => {
      const ix = el._children.indexOf(c);
      if (ix >= 0) el._children.splice(ix, 1);
      c._parent = null; return c;
    },
    remove: () => { if (el._parent) el._parent.removeChild(el); },
    addEventListener: (ev, fn) => {
      if (!el._listeners.has(ev)) el._listeners.set(ev, []);
      el._listeners.get(ev).push(fn);
    },
    dispatchEvent: (event) => {
      for (const fn of (el._listeners.get(event.type) || []).slice()) fn(event);
    },
    showModal: () => { el._open = true; el._opensVia = "showModal"; },
    show: () => { el._open = true; el._opensVia = "show"; },
    close: () => {
      if (!el._open) return;
      el._open = false;
      el.dispatchEvent({ type: "close", target: el });
    },
    getBoundingClientRect: () => ({ left: 0, right: 100, top: 0, bottom: 100 }),
    focus: () => { el.focused = true; },
    querySelector: (sel) => {
      const m = sel.match(/^\[data-action="([^"]+)"\]$/);
      if (!m) return null;
      const walk = (n) => {
        if (n.getAttribute && n.getAttribute("data-action") === m[1]) return n;
        for (const c of (n._children || [])) { const r = walk(c); if (r) return r; }
        return null;
      };
      return walk(el);
    },
  };
  return el;
}
globalThis.document = { createElement: (t) => _mkEl(t), body: _mkEl("body") };
globalThis.__dialogs = () => document.body._children.filter(
  (c) => c.tagName === "DIALOG");
globalThis.__last = () => { const d = __dialogs(); return d[d.length - 1]; };
globalThis.__click = (d, action) =>
  d.querySelector('[data-action="' + action + '"]')._listeners.get("click")[0]();
globalThis.__settled = (p) => Promise.race([
  p.then((v) => ({ settled: true, value: v })),
  new Promise((r) => setTimeout(() => r({ settled: false }), 20)),
]);
"""


def _run(snippet: str):
    prelude = ("const { warningModal: modal } = await import("
               + json.dumps(MODULE.resolve().as_uri()) + ");\n")
    return run_node([], prelude + snippet, globals_js=_DOM)


#: How the question is answered, and what it answers.
_ANSWERS = [
    ("the costly button", "__click(d, 'discard');", True),
    ("Cancel", "__click(d, 'cancel');", False),
    ("Esc", "d.dispatchEvent({type: 'cancel', target: d,"
            " preventDefault() {}});", False),
    ("a click off the surface",
     "for (const t of ['pointerdown', 'click'])"
     " d.dispatchEvent({type: t, target: d, clientX: -5, clientY: 50});", False),
    ("a click on its own padding",
     "for (const t of ['pointerdown', 'click'])"
     " d.dispatchEvent({type: t, target: d, clientX: 50, clientY: 50});", None),
]


@pytest.mark.parametrize("how,act,answer", _ANSWERS,
                         ids=[a[0] for a in _ANSWERS])
def test_how_the_question_is_answered(how, act, answer):
    out = _run(f"""
        const p = modal.confirm({{ title: "Delete file", body: "Gone?" }});
        const d = __last();
        {act}
        console.log(JSON.stringify(await __settled(p)));
    """)
    if answer is None:
        assert out == {"settled": False}, f"{how} answered the question"
    else:
        assert out == {"settled": True, "value": answer}, out


def test_the_discard_question_carries_its_fixed_copy():
    out = _run("""
        modal.confirmDiscardUnsaved();
        const d = __last();
        const [title, body] = d._children;
        const cancel = d.querySelector('[data-action="cancel"]');
        const discard = d.querySelector('[data-action="discard"]');
        const row = cancel._parent._children;
        console.log(JSON.stringify({
          tag: d.tagName, cls: d.className,
          labelledby: d.getAttribute("aria-labelledby") === title.id,
          describedby: d.getAttribute("aria-describedby") === body.id,
          title: title.textContent, body: body.textContent,
          cancel: cancel.textContent, discard: discard.textContent,
          types: [cancel.type, discard.type], danger: discard.className,
          order: row.map((b) => b.getAttribute("data-action")),
          focused: [cancel.focused, discard.focused], opensVia: d._opensVia,
        }));
    """)
    assert out["tag"] == "DIALOG"
    assert out["cls"] == "mb-dialog molbuilder-warning-modal"
    assert out["labelledby"] and out["describedby"]
    assert (out["title"], out["body"], out["cancel"], out["discard"]) == (
        "Unsaved modifications",
        "You have unsaved changes to the current canvas. "
        "Continuing will discard them.",
        "Cancel", "Discard and continue")
    assert out["types"] == ["button", "button"]
    assert out["danger"] == "danger" and out["order"] == ["cancel", "discard"]
    assert out["focused"] == [True, False], "the focus is not on Cancel"
    assert out["opensVia"] == "showModal", "the page below stays clickable"


def test_one_dialog_at_a_time_and_an_answered_one_leaves():
    out = _run("""
        const p1 = modal.confirm({ title: "first" });
        const d1 = __last();
        const p2 = modal.confirm({ title: "second" });
        const first = await __settled(p1);
        const open = __dialogs().length;
        const d2 = __last();
        __click(d2, "discard");
        console.log(JSON.stringify({
          first, open, second: await p2, left: __dialogs().length,
          fresh: d1 !== d2,
        }));
    """)
    assert out["first"] == {"settled": True, "value": False}, (
        "a second question did not answer the first no")
    assert out["open"] == 1, "two confirms stood on the page at once"
    assert out["second"] is True and out["fresh"] is True
    assert out["left"] == 0, "an answered dialog stayed on the page"
