"""The one tab strip -- ``lib/tab-strip.js`` (ui-contract.md § 4.1).

Every strip of tabs in the app is mounted by it, so its keyboard is every
strip's keyboard.  Under guard, each named for its failure:

* ONE TAB STOP -- the selected tab is the only one Tab reaches, or a strip of
  seven costs seven presses to cross;
* the ARROW KEYS move the selection, wrapping, and pass over a tab that is
  hidden or disabled -- a key that lands on a tab nobody can see selects
  nothing a person can read;
* each tab and its panel paired by id, the panel shown by ``hidden`` alone.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

MODULE = (Path(__file__).resolve().parents[1]
          / "molbuilder/web/static/lib/tab-strip.js")

_DRIVER = r"""
const fs = require("fs"), vm = require("vm");
function el(attrs) {
    const a = Object.assign({}, attrs);
    return {
        attrs: a, hidden: false, disabled: false, tabIndex: 0, id: a.id || "",
        dataset: { tab: a["data-tab"] }, handlers: {},
        getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
        setAttribute(k, v) { this.attrs[k] = String(v); },
        addEventListener(ev, fn) { this.handlers[ev] = fn; },
        focus() { focused = this.dataset.tab; },
    };
}
let focused = null;
const panels = {
    "p-a": el({ id: "p-a" }), "p-b": el({ id: "p-b" }), "p-c": el({ id: "p-c" }),
};
const tabs = [
    el({ role: "tab", "data-tab": "a", "aria-controls": "p-a" }),
    el({ role: "tab", "data-tab": "b", "aria-controls": "p-b",
         "aria-selected": "true" }),
    el({ role: "tab", "data-tab": "c", "aria-controls": "p-c" }),
];
tabs[2].disabled = true;
global.window = global;
global.document = { getElementById: (id) => panels[id] || null };
const strip = { ownerDocument: global.document,
                querySelectorAll: () => tabs };
vm.runInThisContext(fs.readFileSync(process.argv[2], "utf8"));
const changes = [];
window.molbuilder.tabStrip.mount(strip, {
    onChange: (name, meta) => changes.push([name, !!meta.byUser]) });
const snap = () => ({
    selected: tabs.map((t) => t.getAttribute("aria-selected")),
    stops: tabs.map((t) => t.tabIndex),
    shown: ["p-a", "p-b", "p-c"].filter((p) => !panels[p].hidden),
});
const out = { mounted: snap(),
              labelled: panels["p-a"].getAttribute("aria-labelledby")
                        === tabs[0].id && !!tabs[0].id };
const key = (t, k) => t.handlers.keydown({ key: k, preventDefault() {} });
key(tabs[1], "ArrowRight");          // c is disabled: wraps to a
out.right = snap(); out.focusRight = focused;
key(tabs[0], "End");
out.end = snap();
out.changes = changes;
process.stdout.write(JSON.stringify(out));
"""


def test_one_tab_stop_and_the_arrow_keys_over_what_can_be_chosen(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    driver = tmp_path / "drive.js"
    driver.write_text(_DRIVER)
    out = json.loads(subprocess.run(
        [node, str(driver), str(MODULE)], capture_output=True, text=True,
        timeout=20, check=True).stdout)
    # MOUNTED as the markup says: b selected, the one tab stop, its panel.
    assert out["mounted"] == {"selected": ["false", "true", "false"],
                              "stops": [-1, 0, -1], "shown": ["p-b"]}
    assert out["labelled"], "the panel is not labelled by its tab"
    # ArrowRight from b passes the disabled c and wraps to a.
    assert out["right"] == {"selected": ["true", "false", "false"],
                            "stops": [0, -1, -1], "shown": ["p-a"]}
    assert out["focusRight"] == "a"
    # End is the last tab that can be chosen: b.
    assert out["end"]["selected"] == ["false", "true", "false"]
    assert out["changes"] == [["a", True], ["b", True]]
