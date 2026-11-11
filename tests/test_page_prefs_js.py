"""A page's own layout, kept in this browser (`web/runtime.md` § 3,
`lib/page-prefs.js`) -- run in node with the two storages a browser has."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._node_esm import run_node

_PREFS = (Path(__file__).resolve().parents[1] / "molbuilder" / "web" / "static"
          / "lib" / "page-prefs.js")

#: The browser's two storages, as a page has them -- each a Map a test can
#: read, or one that refuses as a private window or a full quota does.
_STORAGES = """
function storage(refuses) {
    const kept = new Map();
    const no = () => { throw new Error("refused"); };
    return { kept,
        getItem: refuses ? no : (k) => (kept.has(k) ? kept.get(k) : null),
        setItem: refuses ? no : (k, v) => { kept.set(k, String(v)); },
        removeItem: refuses ? no : (k) => { kept.delete(k); } };
}
globalThis.sessionStorage = storage(globalThis.__refuses);
globalThis.localStorage = storage(globalThis.__refuses);
"""


def _run(snippet: str, *, refuses: bool = False):
    return run_node(
        [_PREFS],
        "const prefs = window.molbuilder.pagePrefs;\n" + snippet,
        globals_js=f"globalThis.__refuses = {json.dumps(refuses)};\n"
                   + _STORAGES)


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
        const keys = [...{kept_in}.kept.keys()];
        const elsewhere = [...{other}.kept.keys()];
        prefs.write("panel_open", null, {opts});
        console.log(JSON.stringify({{ back, keys, elsewhere,
            dropped: prefs.read("panel_open", {opts}) }}));
    """)
    assert out == {"back": "1", "keys": ["molbuilder.panel_open"],
                   "elsewhere": [], "dropped": None}, out


def test_storage_the_browser_refuses_reads_as_nothing_kept():
    """`runtime.md` § 3: a private window or a full quota refuses the
    storage; a preference is a convenience, so the page goes on -- nothing
    read, nothing written, nothing thrown."""
    out = _run("""
        prefs.write("panel_open", "1");
        console.log(JSON.stringify({ back: prefs.read("panel_open") }));
    """, refuses=True)
    assert out == {"back": None}, out
