"""End-to-end contract: opening a real engine result folder mounts
the right inspector for each file.

The audit gap that motivated this file:
==========================================

The five-lens fresh-eyes audit (2026-06-12) covered code-vs-doc
drift, security, scientific correctness, concurrency, and test-
coverage gaps.  None of those lenses asked the most basic question
a user actually asks: **"I opened my PySCF result folder.  Did the
results tab pick up the right file and mount the right inspector?"**

That blind spot let a real regression survive across five days
(commit 41415e0 on 2026-06-07 → user report on 2026-06-12).  PySCF
and geomeTRIC write their multi-frame trajectory at
``<job>_geom_optim.xyz``; the trajectory inspector only matched
``.molwatch.log`` + ``.out``, so the file fell to the structure
inspector and rendered as a single static frame.  The user had no
way to access the trajectory animation or per-frame plots that
already existed in the file on disk.

This file ships the missing contract test: a small set of
scenarios that exercise the actual user flow (sidebar selection →
file picker enumeration → registry dispatch → inspector mount) on
realistic in-memory fixtures.  Each scenario asserts the inspector
shape (frame strip vs structure card vs source card), so a future
refactor that re-narrows a match — or breaks the registry's first-
match-wins ordering — fails this test rather than silently sending
the user back into the wrong viewer.

Inspector identification (DOM signatures used below):

  * trajectory inspector
      ``.mol-viewer-frame-strip`` (auto-mounted by the embed when
      ``animation.kind === "trajectory"``)
      AND ``#viewer`` host div
  * structure inspector
      ``.structure-card``  (per-card scaffold built by
      ``lib/inspectors/structure.js``)
  * source inspector
      ``.source-card``
  * spectra inspector
      ``.spectra-results-summary`` or similar (not exercised here)
"""
from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


# --------------------------------------------------------------------- #
#  Server fixture                                                       #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _register_tmp_as_picker_root(tmp_path, monkeypatch):
    from molbuilder import diagnostics
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None,
        conda_envs=frozenset(),
    )
    cls = type(caps)
    monkeypatch.setattr(
        cls, "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),),
    )
    diagnostics.set_capabilities(caps)


# --------------------------------------------------------------------- #
#  Fixture builders                                                     #
# --------------------------------------------------------------------- #


def _xyz_single_frame_water(stem: str = "water") -> str:
    """Generic single-structure XYZ — what the structure inspector
    is supposed to claim."""
    return (
        "3\n"
        f"{stem} initial geometry\n"
        "O    0.000000    0.000000    0.000000\n"
        "H    0.957000    0.000000    0.000000\n"
        "H   -0.239000    0.927000    0.000000\n"
    )


def _xyz_multi_frame_pyscf_optim(n_frames: int = 4) -> str:
    """A geomeTRIC-shaped multi-frame XYZ — the trajectory that
    PySCF's geom-opt wrapper writes as ``<job>_geom_optim.xyz``.
    Comment line uses the ``Iteration K Energy E`` pattern the
    PySCFParser content-sniffs for energy extraction; the
    underlying ``can_parse`` accepts the file even without that
    pattern (any well-formed multi-frame XYZ is OK)."""
    out = []
    energy0 = -76.4123456
    for i in range(n_frames):
        # Tiny perturbation per frame so the diff between frames is
        # visible if anything probes it.
        dz = i * 0.005
        e = energy0 + i * 0.0012
        out.append(
            "3\n"
            f"Iteration {i} Energy {e:.10f}\n"
            f"O    0.000000    0.000000    {dz:.6f}\n"
            "H    0.957000    0.000000    0.000000\n"
            "H   -0.239000    0.927000    0.000000\n"
        )
    return "".join(out)


def _project_dir(tmp_path: Path, *parts: str) -> Path:
    """Build ``tmp_path/<parts>`` and return it (mkdir parents)."""
    p = tmp_path.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------- #
#  Direct registry-mount tests (no file picker, fast)                   #
# --------------------------------------------------------------------- #


def _open_results(page, base_url):
    """Land on /results with the inspector registry mounted but no
    file selected.  Tests below drive the registry mount() directly
    so they exercise the dispatch path without any picker noise."""
    page.goto(f"{base_url}/results")
    page.wait_for_function(
        "() => window.molbuilder "
        "      && window.molbuilder.inspectors "
        "      && window.molbuilder.inspectors.list "
        "      && window.molbuilder.inspectors.list().length >= 4"
    )
    page.wait_for_selector("#inspector-host", timeout=5000)


class TestRegistryDispatchPerFile:
    """Smoke test for the FULL e2e chain (page loads → inspectors
    register → registry.pick works).

    Pre-2026-06-13 this class held 5 separate Playwright tests, one
    per filename pattern.  Each took ~1.8 s of Chromium startup to
    call ``window.molbuilder.inspectors.pick(path)`` — a pure JS
    function call that doesn't need a browser at all.  Per
    docs/process/testing.md the dispatch logic was
    demoted to the L2 module tier in
    ``tests/test_inspector_registry_dispatch_js.py`` (Node-level,
    ~50 ms total instead of ~10 s).

    What stays here: ONE regression smoke that verifies the
    INTEGRATION still works end-to-end — page loads, inspectors
    actually register on /results, registry's ``pick()`` is
    reachable via ``window.molbuilder.inspectors``.  Picks the
    geom_optim.xyz case because it's the original regression-pin
    (PySCF/geomeTRIC trajectory was silently rendered as static
    structure for 5 days; 2026-06-12 user report).
    """

    def test_geom_optim_xyz_mounts_trajectory_not_structure(
            self, page, flask_server, tmp_path, monkeypatch):
        """The regression that drove this whole file:
        ``<job>_geom_optim.xyz`` (geomeTRIC's multi-frame trajectory)
        MUST hit the trajectory inspector, not the structure inspector.
        Before 2026-06-12 the structure inspector claimed it and
        rendered frame 0 as a single static structure.

        Kept at L5 (Playwright) because its job is to verify the
        full chain works: page mounts → inspectors module loads →
        registry is populated → pick() is reachable from JS.  The
        4 sibling cases (.molwatch.log, .out, .pyscf.log, plain
        .xyz) moved to L2 — the function-call assertion doesn't
        need a browser; see ``test_inspector_registry_dispatch_js.py``.
        """
        _register_tmp_as_picker_root(tmp_path, monkeypatch)
        proj = _project_dir(tmp_path, "BDT", "optimization")
        traj = proj / "BDT_geom_optim.xyz"
        traj.write_text(_xyz_multi_frame_pyscf_optim(n_frames=4))
        _open_results(page, flask_server)
        # Drive the registry's pick() directly + record which inspector
        # was chosen.  This is the lowest-noise way to verify the
        # contract: the test fails if pick() returns the structure
        # inspector OR null OR anything other than ``trajectory``.
        picked = page.evaluate(
            "(path) => {"
            "  const reg = window.molbuilder.inspectors;"
            "  const i = reg.pick(path);"
            "  return i ? i.name : null;"
            "}",
            str(traj),
        )
        assert picked == "trajectory", (
            f"BDT_geom_optim.xyz should mount the trajectory "
            f"inspector but the registry picked: {picked!r}.  "
            f"This is the 2026-06-12 regression: the trajectory "
            f"inspector's match() must claim *_optim.xyz / "
            f"*_geom_optim.xyz.")


