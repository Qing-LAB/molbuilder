"""Inspector-registry contract under Node -- what does not depend on the
order the page registers presenters in: each presenter's ``isResult`` flag,
and ``register``'s refusal of an incomplete presenter and its replacing one
of the same name.

Which viewer a file opens in, lists under and is headed by DOES depend on
that order, which only the browser runs; those questions are
``tests/data/presenters.toml``'s rows, asked on the live page by
``test_inspector_registry_e2e.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.module

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "molbuilder/web/static"


# --------------------------------------------------------------------- #
#  Node bootstrap                                                        #
# --------------------------------------------------------------------- #


from _node_esm import run_node

# THE PRESENTERS THE PAGE LOADS, READ FROM THE PAGE: the element builder and
# the registry first, then every inspector module results.html names -- never
# a second copy of that list here.
# structure.js is an ES module (it imports { mount }); the others are classic
# IIFEs.  run_node dynamic-imports each -- a classic IIFE runs + publishes its
# global, an ES module runs + exports; structure.js's `import` transitively
# loads the whole molview graph, which the DOM/storage stub below lets load.
_PAGE = (ROOT / "molbuilder/web/templates/results.html").read_text(encoding="utf-8")
_INSPECTOR_MODULES = [STATIC / "lib/dom.js", STATIC / "lib/inspectors/registry.js"] + [
    STATIC / m
    for m in re.findall(r"filename='(lib/inspectors/[^']+\.js)'", _PAGE)
    if not m.endswith(("registry.js", "lifecycle.js"))]

_STUB = """
globalThis.document = {
    createElement: () => ({ appendChild(){}, setAttribute(){}, style:{},
                            classList:{ add(){}, remove(){}, toggle(){}, contains:()=>false },
                            querySelector:()=>null, querySelectorAll:()=>[], addEventListener(){} }),
    createTextNode: () => ({}),
    createDocumentFragment: () => ({ appendChild(){} }),
    getElementById: () => null, querySelector: () => null, querySelectorAll: () => [],
    addEventListener(){},
};
globalThis.localStorage   = { getItem:()=>null, setItem(){}, removeItem(){} };
globalThis.sessionStorage = { getItem:()=>null, setItem(){}, removeItem(){} };
globalThis.fetch = () => Promise.resolve({ ok:true, json:()=>Promise.resolve({}), text:()=>Promise.resolve("") });
globalThis.molbuilder = globalThis.molbuilder || {};
globalThis.molbuilder.runtime = { register:()=>undefined, whenReady:()=>Promise.resolve() };
"""


def _run_node(snippet: str) -> object:
    """ES-module harness: dynamic-import the registry + inspector modules (in results.html order),
    then run the snippet.  The stub lets the molview graph structure.js pulls in load in Node;
    static_root resolves structure.js's browser-absolute ``/static/…`` import of molview/index.js."""
    return run_node(_INSPECTOR_MODULES, snippet, globals_js=_STUB, static_root=STATIC)


# --------------------------------------------------------------------- #
#  isResult flag — inspector opt-in vs catch-all                        #
# --------------------------------------------------------------------- #


_IS_RESULT_FLAGS = [
    ("trajectory", True, "trajectory_opts_in"),
    ("spectra", True, "spectra_opts_in"),
    ("structure", True, "structure_opts_in"),
    # source is a catch-all viewer; MUST stay out of the picker —
    # otherwise the dropdown floods with input files + READMEs.
    ("source", False, "source_opts_out"),
]


@pytest.fixture(scope="module")
def is_result_flags() -> dict[str, object]:
    return _run_node(r"""
        const out = {};
        for (const i of window.molbuilder.inspectors.list()) {
            out[i.name] = i.isResult;
        }
        console.log(JSON.stringify(out));
    """)


@pytest.mark.parametrize(
    "inspector_name,expected",
    [pytest.param(n, exp, id=tid) for n, exp, tid in _IS_RESULT_FLAGS],
)
def test_inspector_isResult_flag(
    is_result_flags, inspector_name, expected,
):
    """The ``isResult`` flag distinguishes inspectors that belong in
    the /results picker dropdown from catch-all viewers."""
    assert is_result_flags.get(inspector_name) is expected


# --------------------------------------------------------------------- #
#  Distinct-shape tests — kept as one-offs                              #
# --------------------------------------------------------------------- #


def test_register_rejects_missing_required_fields():
    """The registry validates the Inspector interface at register
    time — a missing field is a programming error, not a runtime
    surprise."""
    errs = _run_node(r"""
        const out = [];
        const cases = [
            {},
            {name: "x"},
            {name: "x", displayName: "X"},
            {name: "x", displayName: "X", match: () => true},
        ];
        for (const c of cases) {
            try { window.molbuilder.inspectors.register(c); out.push(null); }
            catch (e) { out.push(e.message || String(e)); }
        }
        console.log(JSON.stringify(out));
    """)
    assert all(msg is not None for msg in errs), (
        f"registry accepted an incomplete inspector definition: {errs!r}"
    )


def test_register_is_idempotent_on_name():
    """Re-registering the same name replaces the previous entry —
    the contract that lets a placeholder be swapped for a real
    implementation without code changes elsewhere."""
    result = _run_node(r"""
        const reg = window.molbuilder.inspectors;
        const before = reg.list().length;
        const fake = {
            name: "source",
            displayName: "Source TEST REPLACEMENT",
            match: () => false,
            mount: () => ({dispose: () => {}}),
        };
        reg.register(fake);
        const after = reg.list().length;
        const replaced = reg.list().find(i => i.name === "source");
        console.log(JSON.stringify(
            {before, after, displayName: replaced.displayName}));
    """)
    assert result["after"] == result["before"], (
        "re-registering same name should NOT grow the list")
    assert result["displayName"] == "Source TEST REPLACEMENT"
