"""What a page keeps in this browser, through its one door (`web/runtime.md`
§ 3, `lib/page-prefs.js`) -- run in node on its own Web Storage, the two
storages a browser has (`process/testing.md` § 4)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._node_esm import TO_THE_BRIM, run_node

_PREFS = (Path(__file__).resolve().parents[1] / "molbuilder" / "web" / "static"
          / "lib" / "page-prefs.js")

#: What a storage holds, read through its own API.
_KEYS = ("const keysOf = (s) => "
         "Array.from({ length: s.length }, (_, i) => s.key(i));\n")

#: A STORAGE THE BROWSER REFUSES, three ways (`runtime.md` § 3: storage the
#: browser refuses reads as nothing kept).  A full quota is made for real
#: (`_node_esm.TO_THE_BRIM`, Node enforcing its own); every call refused, or
#: storage blocked outright -- the storage itself refused when it is asked
#: for -- are not states Node can be brought to (`process/testing.md` § 3b: a
#: failure that cannot be caused for real), so the real `Storage`, or the
#: window's door to it, throws the browser's own error.
_REFUSALS = {
    "a full quota": TO_THE_BRIM + """
toTheBrim(sessionStorage);
""",
    "every call refused": """
for (const m of ["getItem", "setItem", "removeItem"]) {
    Storage.prototype[m] = function () {
        throw new DOMException("The operation is insecure.", "SecurityError");
    };
}
""",
    "storage blocked": """
Object.defineProperty(globalThis, "sessionStorage", { get() {
    throw new DOMException("The operation is insecure.", "SecurityError");
} });
""",
}


def _run(snippet: str, *, refused: str = ""):
    return run_node(
        [_PREFS],
        _KEYS + "const prefs = window.molbuilder.pagePrefs;\n" + snippet,
        globals_js=_REFUSALS.get(refused, ""), storage=True)


@pytest.mark.parametrize("lasting, kept_in", [
    (False, "sessionStorage"), (True, "localStorage")],
    ids=["for the session", "after the browser closes"])
def test_a_preference_reads_back_under_the_one_prefix(lasting, kept_in):
    """`runtime.md` § 3: every key `molbuilder.` and its name, in the
    session's storage unless `{lasting: true}` and in that one only; null
    drops it.  THE SILENT FAILURES: a preference kept in the wrong storage
    -- a width forgotten when the browser closes, a filter kept for good --
    or under a key of a page's own (to 2.69.26, eight accessors in six files
    under five prefixes), or a cleared one coming back."""
    opts = json.dumps({"lasting": lasting})
    other = ("sessionStorage" if kept_in == "localStorage"
             else "localStorage")
    out = _run(f"""
        prefs.write("panel_open", "1", {opts});
        const back = prefs.read("panel_open", {opts});
        const keys = keysOf({kept_in});
        const elsewhere = keysOf({other});
        // A NEIGHBOUR, so dropping one value is seen to drop that one alone.
        prefs.write("filter", "xyz", {opts});
        prefs.write("panel_open", null, {opts});
        console.log(JSON.stringify({{ back, keys, elsewhere,
            dropped: prefs.read("panel_open", {opts}),
            neighbour: prefs.read("filter", {opts}) }}));
    """)
    assert out == {"back": "1", "keys": ["molbuilder.panel_open"],
                   "elsewhere": [], "dropped": None,
                   "neighbour": "xyz"}, out


def test_an_empty_value_is_kept():
    """`runtime.md` § 3: `null` drops a value, "" is one.  THE SILENT
    FAILURE: an empty value read as nothing kept -- the sidebar's cleared
    file falling back to the pick another page made, which
    `test_projects_state_busy_guard_js.py` asks of the selection itself."""
    out = _run("""
        prefs.write("current_file.results", "");
        const kept = prefs.read("current_file.results");
        prefs.write("current_file.results", null);
        console.log(JSON.stringify({ kept,
            dropped: prefs.read("current_file.results") }));
    """)
    assert out == {"kept": "", "dropped": None}, out


@pytest.mark.parametrize("refused", list(_REFUSALS))
def test_storage_the_browser_refuses_reads_as_nothing_kept(refused):
    """`runtime.md` § 3: a private window or a full quota refuses the
    storage; the page goes on -- nothing read, nothing thrown -- and `write`
    answers false, so a caller can tell the value was not kept."""
    out = _run("""
        const wrote = prefs.write("panel_open", "1");
        console.log(JSON.stringify({ wrote, back: prefs.read("panel_open") }));
    """, refused=refused)
    assert out == {"wrote": False, "back": None}, out
