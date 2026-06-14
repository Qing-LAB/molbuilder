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

Per docs/protocols/test-strategy.md § 2 + § 7 (L5 → L2 demotion
when the test's only payload is "call this pure JS function and
check its return") and § 8 (parametrize when N tests assert the
same contract over different inputs).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.module

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "molbuilder/web/static"


# --------------------------------------------------------------------- #
#  Node bootstrap — shared by all batched fixtures                      #
# --------------------------------------------------------------------- #


def _run_node(snippet: str) -> object:
    """Load the registry + 4 inspector modules under Node + run the
    snippet.  Returns the JSON-parsed last line of stdout.

    The script-tag order in ``templates/results.html`` is load-bearing
    — first-match-wins means a different order mis-routes compound
    extensions (``.spectra.json`` → source instead of spectra).
    Mirror that order exactly.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    bootstrap = """
        // Minimal DOM stubs — the registry's match() functions are
        // pure, but the inspector module IIFEs reach for `document` /
        // `window` at load time when they wire helpers.  No-op stubs
        // let the modules load; no test in this file reaches mount.
        global.window = global;
        global.globalThis = global;
        global.document = {
            createElement: () => ({
                appendChild: () => undefined,
                setAttribute: () => undefined,
                style: {},
            }),
            createTextNode: () => ({}),
            createDocumentFragment: () => ({ appendChild: () => undefined }),
            getElementById: () => null,
            querySelector: () => null,
            querySelectorAll: () => [],
        };
        global.window.molbuilder = global.window.molbuilder || {};
        global.window.molbuilder.runtime = {
            register: () => undefined,
            whenReady: () => Promise.resolve(),
        };
    """
    modules = [
        "lib/inspectors/registry.js",
        "lib/inspectors/_partial_inspector_factory.js",
        "lib/inspectors/trajectory.js",
        "lib/inspectors/spectra.js",
        "lib/inspectors/structure.js",
        "lib/inspectors/source.js",
    ]
    full = bootstrap + "\n"
    for rel in modules:
        full += (STATIC / rel).read_text() + "\n"
    full += snippet
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".js", delete=False, encoding="utf-8") as tmp:
        tmp.write(full)
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            [node, tmp_path],
            capture_output=True, text=True, timeout=15,
        )
    finally:
        try: Path(tmp_path).unlink()
        except OSError: pass
    if proc.returncode != 0:
        pytest.fail(
            f"node exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\n"
            f"stdout:\n{proc.stdout}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


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
    ("/projects/foo/run.molwatch.log", "PySCF optimization",
     "molwatch_log→pyscf_opt"),
    ("/projects/foo/r.spectra.json", "PySCF spectrum",
     "spectra_json→pyscf_spectrum"),
    ("/projects/foo/q.xyz", "Structure", "xyz→structure"),
    ("/projects/foo/p.pdb", "Structure", "pdb→structure"),
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
