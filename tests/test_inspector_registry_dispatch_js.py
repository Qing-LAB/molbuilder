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
    # consume them.
    modules = [
        "lib/inspectors/registry.js",
        "lib/inspectors/_partial_inspector_factory.js",
        "lib/inspectors/trajectory.js",
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
