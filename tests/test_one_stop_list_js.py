"""The one list of choices -- ``lib/one-stop-list.js`` (ui-contract.md § 4.1).

The sidebar's rows, the spectrum's modes and the run history's states are
mounted by it, so its keys are every list's keys.  Each row of the table is a
behaviour, named for its failure:

* ONE TAB STOP -- the chosen item, else the first: a list of a hundred modes
  that costs a hundred presses to cross is no keyboard path;
* the ARROWS and Home / End move it, passing over an item put out of view
  of its own (a filter's ``display: none``) -- the stop moved onto it leaves
  the keys nowhere -- while a list inside a shut panel keeps its stop;
* a control inside an item is a stop only while its item holds the list's;
* a list redrawn under the keys keeps them on the item carrying its id;
* Enter does what a click on the item does -- the page's own click handler,
  so the mouse and the keys reach one action -- and, where choosing twice
  commits, Enter on the chosen item commits;
* a key pressed on a control inside an item is that control's -- a row's own
  menu button keeps its Enter.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

MODULE = (Path(__file__).resolve().parents[1]
          / "molbuilder/web/static/lib/one-stop-list.js")

#: A list of three items, the middle one chosen, one holding a control --
#: the elements answer the selectors the module asks (`.item`, `.item.on`),
#: each drawn unless ``undrawn`` (a filter's ``display: none``).
_PAGE = r"""
const fs = require("fs"), vm = require("vm");
let focused = null;
const clicks = [], commits = [];
const document = { activeElement: null,
                   getElementById: (id) => items.find((i) => i.id === id) || null,
                   // an element's own display: none when ``undrawn``
                   defaultView: { getComputedStyle: (el) =>
                       ({ display: el.undrawn ? "none" : "block" }) } };
function node(name, classes, parent) {
    return {
        name, id: name, classes: new Set(classes), parent, hidden: false,
        undrawn: false, tabIndex: -1, kids: [],
        matches(sel) {
            return sel.split(",").some((one) => one.trim().split(".")
                .filter(Boolean).every((c) => this.classes.has(c)));
        },
        closest(sel) { return this.matches(sel) ? this
                       : (this.parent && this.parent.closest(sel)); },
        querySelectorAll() { return this.kids; },
        focus() { focused = this.name; document.activeElement = this; },
        click() { clicks.push(this.name); },
    };
}
const handlers = {};
const list = {
    ownerDocument: document, matches: () => false, closest: () => null,
    addEventListener(ev, fn) { handlers[ev] = fn; },
    querySelectorAll: () => items,
    querySelector: (sel) => items.find((i) => i.matches(sel)) || null,
    contains: (el) => items.includes(el) || items.includes(el.parent),
};
let items = [node("a", ["item"], list), node("b", ["item", "on"], list),
             node("c", ["item"], list)];
const menu = node("b's menu", ["menu"], items[1]);
items[1].kids = [menu];
global.window = global;
vm.runInThisContext(fs.readFileSync(process.argv[2], "utf8"));
const key = (target, k) => handlers.keydown({ key: k, target,
                                              preventDefault() {} });
const stops = () => items.map((i) => i.tabIndex);
"""

#: (what, the steps after the page above, what they leave)
_ROWS = [
    ("mounted, the stop is on the chosen item",
     "list.mount({ items: '.item', current: '.item.on' });",
     {"stops": [-1, 0, -1]}),
    ("nothing chosen, the stop is on the first",
     "items[1].classes.delete('on');"
     " list.mount({ items: '.item', current: '.item.on' });",
     {"stops": [0, -1, -1]}),
    ("ArrowDown moves the stop and the focus",
     "list.mount({ items: '.item', current: '.item.on' });"
     " key(items[1], 'ArrowDown');",
     {"stops": [-1, -1, 0], "focused": "c"}),
    ("Home and End reach the ends",
     "list.mount({ items: '.item', current: '.item.on' });"
     " key(items[1], 'Home'); key(items[0], 'End');",
     {"stops": [-1, -1, 0], "focused": "c"}),
    ("a row a filter puts out of view is passed over, and holds no stop",
     "items[1].undrawn = true;"
     " list.mount({ items: '.item', current: '.item.on' });"
     " key(items[0], 'ArrowDown');",
     {"focused": "c", "stops": [-1, -1, 0]}),
    ("focus on a control inside an item moves the stop to that item",
     "list.mount({ items: '.item', current: '.item.on' });"
     " key(items[1], 'ArrowDown'); handlers.focusin({ target: menu });",
     {"stops": [-1, 0, -1]}),
    ("a control inside an item is a stop only while the item holds it",
     "list.mount({ items: '.item', current: '.item.on' });"
     " const held = menu.tabIndex; key(items[1], 'ArrowDown');"
     " out.menu = [held, menu.tabIndex];",
     {"menu": [0, -1]}),
    ("redrawn under the keys, they stay on the item carrying its id",
     "const api = list.mount({ items: '.item', current: '.item.on' });"
     " items[2].focus();"
     " api.redraw(() => { items = [node('a', ['item'], list),"
     "   node('b', ['item', 'on'], list), node('c', ['item'], list)]; });"
     " out.again = document.activeElement === items[2];",
     {"again": True, "stops": [-1, -1, 0]}),
    ("redrawn under the keys, their item gone, they go to the stop",
     "const api = list.mount({ items: '.item', current: '.item.on' });"
     " items[2].focus();"
     " api.redraw(() => { items = [node('a', ['item'], list),"
     "   node('b', ['item', 'on'], list)]; });"
     " out.again = document.activeElement === items[1];",
     {"again": True}),
    ("redrawn with the keys elsewhere, focus is left where it is",
     "const api = list.mount({ items: '.item', current: '.item.on' });"
     " api.redraw(() => { items = items.slice(); });",
     {"focused": None}),
    ("a list in a disposed module listens through its owner's scope",
     "const said = [];"
     " list.mount({ items: '.item', current: '.item.on',"
     "  listeners: { on(target, ev, fn) { said.push(ev);"
     "                target.addEventListener(ev, fn); } } });"
     " out.said = said.sort();",
     {"said": ["focusin", "keydown"]}),
    ("Enter does what a click on the item does",
     "list.mount({ items: '.item', current: '.item.on' });"
     " key(items[0], 'Enter'); key(items[2], ' ');",
     {"clicks": ["a", "c"], "commits": []}),
    ("Enter on the chosen item commits, where choosing twice commits",
     "list.mount({ items: '.item', current: '.item.on',"
     "  commit: (i) => commits.push(i.name) });"
     " key(items[1], 'Enter'); key(items[0], 'Enter');",
     {"clicks": ["a"], "commits": ["b"]}),
    ("a key on a control inside an item is that control's",
     "list.mount({ items: '.item', current: '.item.on' });"
     " key(menu, 'Enter'); key(menu, 'ArrowDown');",
     {"clicks": [], "focused": None, "stops": [-1, 0, -1]}),
    ("redrawn with the keys elsewhere, settle puts the stop on the chosen item",
     "const api = list.mount({ items: '.item', current: '.item.on' });"
     " key(items[1], 'ArrowDown'); document.activeElement = null;"
     " items.forEach((i) => { i.tabIndex = -1; }); api.settle();",
     {"stops": [-1, 0, -1]}),
    ("settled with the keys on an item, the stop stays with them",
     "const api = list.mount({ items: '.item', current: '.item.on' });"
     " key(items[1], 'ArrowDown'); api.settle();",
     {"stops": [-1, -1, 0], "focused": "c"}),
    ("focus reaching an item moves the stop to it",
     "list.mount({ items: '.item', current: '.item.on' });"
     " handlers.focusin({ target: items[2] });",
     {"stops": [-1, -1, 0]}),
]


@pytest.mark.parametrize("what,steps,leaves", _ROWS,
                         ids=[r[0] for r in _ROWS])
def test_one_tab_stop_the_arrows_and_enter_as_a_click(tmp_path, what, steps,
                                                      leaves):
    """MUTATIONS THIS MUST FAIL AGAINST: the stop on every item; the arrows
    landing on an item not drawn; a control inside an item a stop whatever
    its item; a redraw leaving the keys where they fell; Enter running
    anything but the item's click; Enter on the chosen item clicking where a
    commit is given; a key on a control inside an item taken as the
    item's."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    driver = tmp_path / "drive.js"
    driver.write_text(
        _PAGE
        + "const real = window.molbuilder.oneStopList;"
        + " list.mount = (o) => real.mount(list, o); const out = {};\n"
        + steps
        + "\nprocess.stdout.write(JSON.stringify(Object.assign(out,"
          " { stops: stops(), focused, clicks, commits })));")
    out = json.loads(subprocess.run(
        [node, str(driver), str(MODULE)], capture_output=True, text=True,
        timeout=20, check=True).stdout)
    assert {k: out[k] for k in leaves} == leaves, what
