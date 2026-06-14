"""Inspector-registry filename dispatch contract (L2 module test).

Demotes the dispatch checks from
``tests/test_results_folder_dispatch_e2e.py::TestRegistryDispatchPerFile``
out of the Playwright e2e tier and into the cheap module tier.
Each of the 5 dispatch tests in the e2e file did the same thing:
spin up Chromium (~1.7-2 s each), call
``window.molbuilder.inspectors.pick(path)``, and assert the
returned inspector's ``name``.  None of them actually verified DOM
mount — the assertion was a pure JS function call.

This file loads the registry + the 4 inspector modules under Node
with a minimal ``document`` stub, then drives ``pick(filename)``
for each fixture filename directly.  ~50 ms total instead of ~10
seconds; same contract enforced.

Per docs/protocols/test-strategy.md § 2 + § 7, this is the
canonical L5 → L2 demotion shape: when an e2e test's only payload
is "call this pure JS function and check its return", the cost
of Chromium is wasted.

The original e2e file keeps one representative DOM-mount test
(the "geom_optim.xyz mounts trajectory" case) so the full chain
from filename → registry pick → embed setStructure stays e2e-
covered.  The other 4 e2e tests get retired in the companion
edit.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.module   # L2 — single-module behavioural test

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "molbuilder/web/static"


def _run_node(snippet: str) -> object:
    """Load the registry + inspector modules under Node + run the
    snippet.  Returns the JSON-parsed last line of stdout.  Minimal
    DOM stubs: the registry + ``match`` functions don't touch the
    DOM, but inspector mount-side code does — we never reach mount
    in these tests.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    bootstrap = """
        // Minimal DOM stubs — match() doesn't touch the DOM, but
        // the inspector modules' IIFEs do reach for ``document`` /
        // ``window`` at load time when they wire helpers.  No-op
        // stubs let the module load succeed.
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
    # The order matters: registry + factory before inspectors that
    # consume them.  Match the live script-tag order in
    # templates/results.html exactly — first-match-wins means a
    # different order would mis-route compound extensions
    # (.spectra.json → source instead of spectra).  Spectra inspector
    # added 2026-06-13 alongside the 28-test demotion from
    # test_inspector_registry_e2e.py.
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


def _pick(filename: str) -> str | None:
    """Drive ``registry.pick(filename)`` under Node, return the
    matched inspector's name or null."""
    return _run_node(
        f"const r = window.molbuilder.inspectors.pick({json.dumps(filename)});"
        f"console.log(JSON.stringify(r ? r.name : null));"
    )


# --------------------------------------------------------------------- #
#  The five canonical dispatch cases                                    #
# --------------------------------------------------------------------- #


class TestRegistryDispatch:
    """The 5 filename → inspector mappings the registry must honour.

    Mirrors the assertions from the legacy
    ``TestRegistryDispatchPerFile`` e2e suite, minus the Playwright
    bring-up.  Each test takes ~50 ms instead of ~2 s.
    """

    def test_geom_optim_xyz_picks_trajectory(self):
        """``*_geom_optim.xyz`` (geomeTRIC's multi-frame trajectory)
        MUST hit the trajectory inspector, not the structure
        inspector.  The 2026-06-12 regression that drove the whole
        e2e file: structure was claiming this file and rendering
        frame 0 as a single static structure."""
        assert _pick("/tmp/BDT/optimization/BDT_geom_optim.xyz") == "trajectory"

    def test_plain_optim_xyz_picks_trajectory(self):
        """Older PySCF wrappers used ``*_optim.xyz`` (no ``_geom``);
        trajectory must claim both forms."""
        assert _pick("/tmp/foo_optim.xyz") == "trajectory"

    def test_plain_xyz_picks_structure(self):
        """Plain user-named ``.xyz`` (single structure, not the
        conventional ``_optim`` trajectory naming) MUST go to the
        structure inspector — the trajectory expansion is narrowed
        to ``*_optim.xyz`` precisely so plain ``.xyz`` doesn't get
        hijacked."""
        assert _pick("/tmp/water_demo/water.xyz") == "structure"

    def test_molwatch_log_picks_trajectory(self):
        """molwatch's unified-format log is the trajectory inspector's
        primary file type."""
        assert _pick("/tmp/run/foo.molwatch.log") == "trajectory"

    def test_siesta_out_picks_trajectory(self):
        """SIESTA's stdout-redirect is the trajectory inspector's
        SIESTA-flavoured payload."""
        assert _pick("/tmp/run/foo.out") == "trajectory"

    def test_pyscf_log_picks_source(self):
        """``.log`` (PySCF's stdout) falls through trajectory's
        match (which only takes ``.molwatch.log``) and lands on the
        source inspector — the catch-all viewer for arbitrary
        text."""
        assert _pick("/tmp/run/foo.log") == "source"

    def test_pdb_picks_structure(self):
        """``.pdb`` is the second structure-inspector trigger."""
        assert _pick("/tmp/protein.pdb") == "structure"

    def test_unknown_extension_picks_source(self):
        """Catch-all: filename the trajectory/structure matchers
        don't claim falls through to source."""
        assert _pick("/tmp/notes.txt") == "source"


def _run_query(expr: str) -> object:
    """Run a single expression against the registry and return its
    JSON-encoded value."""
    return _run_node(
        f"console.log(JSON.stringify({expr}));"
    )


# --------------------------------------------------------------------- #
#  Registry contract — was tests/test_inspector_registry_e2e.py         #
#  ::TestRegistryContract (12 tests demoted 2026-06-13)                 #
# --------------------------------------------------------------------- #


class TestRegistryContract:
    """The registry's register/pick/list contract.  All assertions
    here previously required a Playwright bring-up just to call a
    JS function and check its return; now they run under Node in
    ~30 ms each."""

    def test_four_inspectors_self_registered_on_load(self):
        """Loading the four inspector modules registers each.  The
        registry's list() must report all four by name."""
        names = _run_query(
            "window.molbuilder.inspectors.list().map(i => i.name)"
        )
        assert set(names) >= {"source", "structure", "trajectory", "spectra"}

    def test_pick_returns_null_for_unknown_extension(self):
        assert _pick("/projects/foo/spectrum/run.unknown_ext") is None

    def test_pick_returns_null_for_empty_path(self):
        assert _pick("") is None

    def test_pick_dispatches_xyz_to_structure(self):
        assert _pick("/projects/foo/struct/water.xyz") == "structure"

    def test_pick_dispatches_pdb_to_structure(self):
        assert _pick("/projects/foo/struct/peptide.pdb") == "structure"

    def test_pick_dispatches_fdf_to_source(self):
        assert _pick("/projects/foo/spectrum/run.fdf") == "source"

    def test_pick_dispatches_compound_log_to_trajectory(self):
        """Compound extension ``.molwatch.log`` MUST win over plain
        ``.log`` (which source would also claim)."""
        assert _pick("/projects/foo/spectrum/run.molwatch.log") == "trajectory"

    def test_pick_dispatches_compound_json_to_spectra(self):
        """Same first-match-wins invariant for ``.spectra.json`` vs
        plain ``.json``."""
        assert _pick("/projects/foo/spectrum/water.spectra.json") == "spectra"

    def test_pick_dispatches_plain_json_to_source(self):
        assert _pick("/projects/foo/user/config.json") == "source"

    def test_register_rejects_missing_required_fields(self):
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

    def test_register_is_idempotent_on_name(self):
        """Re-registering the same name replaces the previous entry
        (the contract that lets a placeholder be swapped for a real
        implementation without code changes elsewhere)."""
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


# --------------------------------------------------------------------- #
#  Picker contract — was tests/test_inspector_registry_e2e.py           #
#  ::TestPickerContract (16 tests demoted 2026-06-13)                   #
# --------------------------------------------------------------------- #


def _pick_result(filename: str) -> "str | None":
    """Return the name of the inspector pickResult selects, or None."""
    return _run_node(
        f"const r = window.molbuilder.inspectors.pickResult({json.dumps(filename)});"
        f"console.log(JSON.stringify(r ? r.name : null));"
    )


def _result_category(filename: str) -> "str | None":
    """Drive ``pickResult(path).resultCategory(path)`` and return the
    label string (or None when pickResult returned null)."""
    return _run_node(
        f"const r = window.molbuilder.inspectors.pickResult({json.dumps(filename)});"
        f"console.log(JSON.stringify("
        f"  r ? r.resultCategory({json.dumps(filename)}) : null));"
    )


class TestPickerIsResultFlag:
    """The ``isResult`` flag distinguishes inspectors whose matches
    belong in the /results picker dropdown vs catch-all viewers."""

    def test_result_inspectors_opt_in(self):
        """trajectory + spectra + structure declare isResult:true."""
        flags = _run_node(r"""
            const list = window.molbuilder.inspectors.list();
            const out = {};
            for (const i of list) out[i.name] = i.isResult;
            console.log(JSON.stringify(out));
        """)
        assert flags["trajectory"] is True
        assert flags["spectra"] is True
        assert flags["structure"] is True

    def test_source_inspector_opts_out(self):
        """source is a catch-all viewer and MUST stay out of the
        picker — otherwise the dropdown floods with input files +
        READMEs."""
        is_result = _run_node(
            "console.log(JSON.stringify("
            "  window.molbuilder.inspectors.list()"
            "    .find(i => i.name === 'source').isResult));"
        )
        assert is_result is False


class TestPickResult:
    """``pickResult(file)`` returns the inspector iff its
    ``isResult`` flag is true; otherwise null.  Used by the
    /results file-picker to gate which files appear in the
    dropdown."""

    def test_pickResult_returns_trajectory_for_out(self):
        assert _pick_result("/projects/foo/bar.out") == "trajectory"

    def test_pickResult_returns_trajectory_for_molwatch_log(self):
        assert _pick_result("/projects/foo/run.molwatch.log") == "trajectory"

    def test_pickResult_returns_spectra_for_spectra_json(self):
        assert _pick_result("/projects/foo/raman.spectra.json") == "spectra"

    def test_pickResult_returns_structure_for_xyz(self):
        assert _pick_result("/projects/foo/optimized.xyz") == "structure"

    def test_pickResult_returns_structure_for_pdb(self):
        assert _pick_result("/projects/foo/protein.pdb") == "structure"

    def test_pickResult_returns_null_for_source_only_extension(self):
        """source matches .fdf but is isResult:false → pickResult
        returns null.  The .fdf is an input file, not a result."""
        assert _pick_result("/projects/foo/inputs/job.fdf") is None

    def test_pickResult_returns_null_for_plain_log(self):
        """source matches .log (plain extension); not a result."""
        assert _pick_result("/projects/foo/build.log") is None

    def test_pickResult_returns_null_for_unknown_extension(self):
        assert _pick_result("/projects/foo/data.xyzzy") is None

    def test_pickResult_returns_null_for_empty_path(self):
        assert _pick_result("") is None


class TestResultCategory:
    """Engine-flavoured ``<optgroup>`` headers in the picker
    dropdown.  Pin the exact spellings — a typo silently changes
    the user-facing UI."""

    def test_resultCategory_out_is_siesta_optimization(self):
        assert _result_category("/projects/foo/bar.out") == "SIESTA optimization"

    def test_resultCategory_molwatch_log_is_pyscf_optimization(self):
        assert _result_category(
            "/projects/foo/run.molwatch.log") == "PySCF optimization"

    def test_resultCategory_spectra_json_is_pyscf_spectrum(self):
        assert _result_category(
            "/projects/foo/r.spectra.json") == "PySCF spectrum"

    def test_resultCategory_xyz_is_structure(self):
        assert _result_category("/projects/foo/q.xyz") == "Structure"

    def test_resultCategory_pdb_is_structure(self):
        assert _result_category("/projects/foo/p.pdb") == "Structure"
