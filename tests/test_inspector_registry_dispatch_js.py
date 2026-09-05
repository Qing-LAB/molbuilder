"""Inspector-registry contract — L2 module tests.

Drives the registry's pick / pickResult / resultCategory / isResult
+ list / register surface under Node with minimal DOM stubs.  Same
contracts as the predecessor Playwright tests with no Chromium
bring-up.

History — this file first landed 2026-06-13 as a sweep that demoted
32 tests out of ``test_inspector_registry_e2e.py`` +
``test_results_folder_dispatch_e2e.py``.  The first pass shipped 35
near-identical per-case test functions, each spinning up its own
Node subprocess to call one pure JS function.  A retrospective audit
the same day flagged the over-decomposition: the original e2e shape
got mechanically copied into L2 without parametrizing.

This revision rebuilds the file with module-scoped fixtures that
batch each contract's cases through a single Node invocation, then
parametrize over the cached result dict.  Net source: ~35 function
defs → 5 parametrized + 3 one-offs.  Wall-clock: ~35 * 200ms = ~7s
→ ~5 * 200ms = ~1s.  Per-case failure messages stay specific via
pytest.param ids.

Per docs/process/testing.md + § 7 (L5 → L2 demotion
when the test's only payload is "call this pure JS function and
check its return") and § 8 (parametrize when N tests assert the
same contract over different inputs).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.module

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "molbuilder/web/static"


# --------------------------------------------------------------------- #
#  Node bootstrap — shared by all batched fixtures                      #
# --------------------------------------------------------------------- #


from _node_esm import run_node

# Mirror the results.html load order EXACTLY (first-match-wins routing).  structure.js is now an ES
# module (it imports { mount }); the others are classic IIFEs.  run_node dynamic-imports each -- a
# classic IIFE runs + publishes its global, an ES module runs + exports; structure.js's `import`
# transitively loads the whole molview graph, which the DOM/storage stub below lets load in Node.
_INSPECTOR_MODULES = [
    STATIC / "lib/inspectors/registry.js",
    # Listed first among the matchers on /results, because an exact
    # basename is the most specific predicate there is -- and source.js
    # claims every .json, so anything less would lose job-set.json to it.
    STATIC / "lib/inspectors/bench-summary.js",
    STATIC / "lib/inspectors/_partial_inspector_factory.js",
    STATIC / "lib/inspectors/trajectory.js",
    STATIC / "lib/inspectors/spectra.js",
    STATIC / "lib/inspectors/structure.js",
    STATIC / "lib/inspectors/source.js",
]

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


def _batch_eval(filenames: list[str], js_expr: str) -> dict[str, object]:
    """Apply ``js_expr`` (a JS expression whose value may depend on
    the bound variable ``p``) to each filename in one Node call;
    return a dict keyed by filename.  Powers the batched fixtures
    below."""
    return _run_node(
        f"const paths = {json.dumps(filenames)};\n"
        f"const out = {{}};\n"
        f"for (const p of paths) {{ out[p] = {js_expr}; }}\n"
        f"console.log(JSON.stringify(out));"
    )


# --------------------------------------------------------------------- #
#  Dispatch contract — registry.pick(filename) → inspector             #
# --------------------------------------------------------------------- #


# (filename, expected_inspector_name_or_None, pytest-id).  Each case
# is a distinct contract — see inline comments for the regression /
# invariant each one guards.  The id is what pytest reports on
# failure, so make it speaking.
_DISPATCH_CASES = [
    # 2026-06-12 regression: geomeTRIC's multi-frame *_geom_optim.xyz
    # MUST hit trajectory, not structure (structure was claiming this
    # file and rendering frame 0 as a single static structure).
    ("/tmp/BDT/optimization/BDT_geom_optim.xyz", "trajectory",
     "geom_optim_xyz→trajectory"),
    # Older PySCF wrappers used *_optim.xyz without the _geom prefix.
    ("/tmp/foo_optim.xyz", "trajectory", "plain_optim_xyz→trajectory"),
    # Plain user-named .xyz (single structure) MUST go to structure —
    # the trajectory match is narrowed to *_optim.xyz precisely so
    # plain .xyz doesn't get hijacked.
    ("/tmp/water_demo/water.xyz", "structure", "plain_xyz→structure"),
    # molwatch unified trajectory log.
    ("/tmp/run/foo.molwatch.log", "trajectory", "molwatch_log→trajectory"),
    # SIESTA stdout redirect.
    ("/tmp/run/foo.out", "trajectory", "siesta_out→trajectory"),
    # Plain .log (PySCF stdout etc.) falls through trajectory's narrow
    # .molwatch.log match → source.
    ("/tmp/run/foo.log", "source", "plain_log→source"),
    # .pdb is the second structure-inspector trigger.
    ("/tmp/protein.pdb", "structure", "pdb→structure"),
    # source claims a handful of plain text extensions (.txt is one).
    ("/tmp/notes.txt", "source", "txt→source"),
    # A sweep's plan, claimed by exact basename.  source.js matches every
    # .json and is listed after it precisely so this one gets there first.
    ("/p/calc/01_coarse/bench/job-set.json", "bench-summary",
     "job_set_json→bench_summary"),
    # ...and the exactness cuts both ways: a DIFFERENT .json is still the
    # text viewer's, so claiming job-set.json cost nothing else.
    ("/p/calc/01_coarse/bench/bench-result.json", "source",
     "other_json→source"),
    # .fdf is a SIESTA INPUT file — source viewer, not a result.
    ("/projects/foo/spectrum/run.fdf", "source", "fdf→source"),
    # Compound extension precedence: .spectra.json MUST win over plain
    # .json (which source would also claim).  Load order in
    # results.html makes spectra register before source.
    ("/projects/foo/spectrum/water.spectra.json", "spectra",
     "compound_spectra_json→spectra"),
    # Plain .json → source (first-match-wins after spectra missed).
    ("/projects/foo/user/config.json", "source", "plain_json→source"),
    # Truly unknown extension → null (not all unknowns fall through).
    ("/projects/foo/spectrum/run.unknown_ext", None, "unknown_ext→null"),
    # Empty path → null.
    ("", None, "empty_path→null"),
]


@pytest.fixture(scope="module")
def pick_results() -> dict[str, object]:
    """Drive ``registry.pick`` over every dispatch case in one Node
    call.  Module-scoped so all parametrized cases share the cost."""
    return _batch_eval(
        [c[0] for c in _DISPATCH_CASES],
        "(window.molbuilder.inspectors.pick(p) || {name: null}).name",
    )


@pytest.mark.parametrize(
    "filename,expected",
    [pytest.param(f, exp, id=tid) for f, exp, tid in _DISPATCH_CASES],
)
def test_registry_pick_dispatches_filename(
    pick_results, filename, expected,
):
    """``registry.pick(filename)`` routes to the documented inspector
    (or null when no inspector claims it).  Each parametrized case is
    a distinct contract — see inline comments in ``_DISPATCH_CASES``."""
    assert pick_results[filename] == expected


# --------------------------------------------------------------------- #
#  pickResult — the picker-only subset (isResult-gated dispatch)        #
# --------------------------------------------------------------------- #


# pickResult returns the matching inspector iff its isResult flag is
# True; otherwise null.  Used by /results to gate which files show up
# in the dropdown.
_PICK_RESULT_CASES = [
    ("/projects/foo/bar.out", "trajectory", "out→trajectory"),
    ("/projects/foo/run.molwatch.log", "trajectory",
     "molwatch_log→trajectory"),
    ("/projects/foo/raman.spectra.json", "spectra",
     "spectra_json→spectra"),
    ("/projects/foo/optimized.xyz", "structure", "xyz→structure"),
    ("/projects/foo/protein.pdb", "structure", "pdb→structure"),
    # .fdf is an INPUT file (source matches but isResult:false) — must
    # NOT appear in the picker.
    ("/projects/foo/inputs/job.fdf", None, "fdf_input→null"),
    # Plain .log: source matches but isResult:false.
    ("/projects/foo/build.log", None, "plain_log→null"),
    ("/projects/foo/data.xyzzy", None, "unknown→null"),
    ("", None, "empty_path→null"),
]


@pytest.fixture(scope="module")
def pick_result_results() -> dict[str, object]:
    return _batch_eval(
        [c[0] for c in _PICK_RESULT_CASES],
        "(window.molbuilder.inspectors.pickResult(p) || {name: null}).name",
    )


@pytest.mark.parametrize(
    "filename,expected",
    [pytest.param(f, exp, id=tid) for f, exp, tid in _PICK_RESULT_CASES],
)
def test_registry_pickResult_filters_by_isResult(
    pick_result_results, filename, expected,
):
    """``pickResult`` returns the matching inspector iff
    ``isResult:true``; null otherwise.  Gate for the /results picker
    dropdown."""
    assert pick_result_results[filename] == expected


# --------------------------------------------------------------------- #
#  resultCategory — engine-flavoured optgroup headers                   #
# --------------------------------------------------------------------- #


# Pin exact spellings — a typo silently changes the user-facing UI
# (group headers in the picker dropdown).
_RESULT_CATEGORY_CASES = [
    ("/projects/foo/bar.out", "SIESTA optimization", "out→siesta_opt"),
    # ENGINE-NEUTRAL, and this case asserted "PySCF optimization" until
    # 2026-09-04.  SIESTA writes a `.molwatch.log` for every run
    # (`config/siesta.py`: `write_molwatch_log = True`); measured, 34 of
    # the first 40 real logs in `projects/` declare `# engine: siesta`.
    # So the picker filed most SIESTA runs under PySCF while the plot
    # title -- which gets the engine from the server -- said SIESTA.
    # The browser cannot know: the engine is a fact about the run
    # DIRECTORY and the picker has only a filename.
    #
    # This does NOT split the group the geomeTRIC cases below exist to
    # keep together: `absorbs()` folds `*_geom_optim.xyz` into the
    # `.molwatch.log` master whenever one is present (`results.md`
    # § 2.3), so the two never appear in the dropdown at the same time.
    ("/projects/foo/run.molwatch.log", "Optimization",
     "molwatch_log→engine_neutral"),
    ("/projects/foo/r.spectra.json", "PySCF spectrum",
     "spectra_json→pyscf_spectrum"),
    ("/projects/foo/q.xyz", "Structure", "xyz→structure"),
    ("/projects/foo/p.pdb", "Structure", "pdb→structure"),
    # geomeTRIC's optimisation traces belong in the SAME dropdown group
    # as the run's other artifacts (`.molwatch.log`), or the user hunts
    # for them under a generic header.  Both spellings, because the
    # `_geom_` infix appeared only in later PySCF wrappers and older
    # runs on disk still carry the short form.
    #
    # Added 2026-09-03, replacing the source-grep that stood in for it
    # (`"_optim.xyz" in body and "PySCF" in body`, over the regex-sliced
    # resultCategory function body).  That pin could not tell WHICH
    # label came back — the two strings merely had to co-occur somewhere
    # in the function — so it passed on any bucket at all.
    ("/projects/foo/BDT_geom_optim.xyz", "PySCF optimization",
     "geom_optim_xyz→pyscf_opt"),
    ("/projects/foo/BDT_optim.xyz", "PySCF optimization",
     "plain_optim_xyz→pyscf_opt"),
]


@pytest.fixture(scope="module")
def result_category_results() -> dict[str, object]:
    return _batch_eval(
        [c[0] for c in _RESULT_CATEGORY_CASES],
        "(function () {"
        "  const r = window.molbuilder.inspectors.pickResult(p);"
        "  return r ? r.resultCategory(p) : null;"
        "})()",
    )


@pytest.mark.parametrize(
    "filename,expected",
    [pytest.param(f, exp, id=tid) for f, exp, tid in _RESULT_CATEGORY_CASES],
)
def test_registry_resultCategory_labels(
    result_category_results, filename, expected,
):
    """``resultCategory`` returns the engine-flavoured group header
    for the picker dropdown.  The exact spelling IS the contract; a
    typo here changes the user-visible UI."""
    assert result_category_results[filename] == expected


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


def test_four_inspectors_self_registered_on_load():
    """Loading the four inspector modules registers each.  The
    registry's ``list()`` must report all four by name."""
    names = _run_node(
        "console.log(JSON.stringify("
        "  window.molbuilder.inspectors.list().map(i => i.name)));"
    )
    assert set(names) >= {"source", "structure", "trajectory", "spectra"}


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
