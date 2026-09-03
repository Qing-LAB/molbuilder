"""End-to-end: the "Auto-detect defaults" chip flow on /molbuilder
surfaces the right (charge, spin, treatment) verdict for a noble-
metal cluster.

Why an e2e here?
================

The chemistry analyzer (``molbuilder.chemistry.analyze_structure``)
is heavily unit-tested.  But the regression that drove this file
(2026-06-13, task #363) was a UI surface gap: noble-metal Au /
Pd clusters were correctly classified as closed-shell-singlet by
the analyzer, then a stale code path on the chip side surfaced
"open-shell" anyway.  L2 tests on the analyzer never caught it
because the analyzer was right; L2 tests on the chip builder
never caught it because the bug was in how the response flowed
through ``viewer.js``'s click handler.

This test pins the FULL chain end-to-end:

  1. Load an Au-only XYZ into the canvas via the same path the
     sidebar Load button uses (``molbuilderTab.commitFile``).
  2. Click the ``#auto-detect-btn`` (the user-visible affordance).
  3. Wait for ``.workflow-detection-chip`` text to land.
  4. Assert that text contains ``closed-shell singlet`` AND does
     NOT contain ``open-shell``.

If a future regression revives the closed-shell→open-shell
inversion ANYWHERE in the chain (analyzer rule change, adapter
translation drift, chip builder text drift, click handler wiring
break), this test fails.

Au cluster choice
-----------------

Smallest fixture that exercises the noble-metal cluster rule:
4 Au atoms in a tight triangular packing.  Open-d count is 0
because the half-filled-d outer shell of free Au is countered
by the cluster-context closure rule (see chemistry analyzer
``check_noble_metal_cluster``).  Same shape would catch a Pd
regression too if it were swapped here; one of either suffices
for this gate.
"""
from __future__ import annotations

from textwrap import dedent

import pytest


pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


_AU4_XYZ = dedent("""\
    4
    Au4 trigonal pyramid (test fixture for noble-metal cluster rule)
    Au    0.000    0.000    0.000
    Au    2.886    0.000    0.000
    Au    1.443    2.500    0.000
    Au    1.443    0.833    2.357
    """)


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _register_tmp_as_picker_root(tmp_path, monkeypatch):
    from molbuilder import diagnostics
    _orig = diagnostics.get_capabilities()
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset(),
    )
    cls = type(caps)
    monkeypatch.setattr(
        cls, "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),),
    )
    diagnostics.set_capabilities(caps)
    monkeypatch.setattr(diagnostics, "_snapshot", _orig)


def test_auto_detect_chip_au_cluster_is_closed_shell(
        page, flask_server, tmp_path, monkeypatch):
    """Au4 cluster → Auto-detect button → chip MUST say closed-shell
    singlet (NOT open-shell).  Regression-pin for the 2026-06-13
    noble-metal inversion class.
    """
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    xyz = tmp_path / "au4.xyz"
    xyz.write_text(_AU4_XYZ)

    # The Auto-detect button lives on /structure-optimization (index.html),
    # not /molbuilder (which is the build/edit tab without an analyze
    # card).  index.html's viewer.js mounts its own canvas + listens
    # for sidebar commits.
    page.goto(f"{flask_server}/structure-optimization")
    page.wait_for_function(
        "() => !!window.molbuilder "
        "      && !!window.molbuilder.projects "
        "      && !!document.getElementById('auto-detect-btn')",
        timeout=10000,
    )

    # Drive the canonical commit path: publishCommit fires the
    # onCommit subscribers which include viewer.js's _commitStructure
    # — that fetches /api/files/read, loads the XYZ into the embed,
    # and enables the auto-detect button.
    page.evaluate(
        "(args) => window.molbuilder.projects.publishCommit("
        "  args.dir, args.file)",
        {"dir": str(tmp_path.resolve()),
         "file": str(xyz.resolve())},
    )

    # The Auto-detect button is disabled until the structure load
    # resolves.  Wait for both the load to finish (3Dmol shows the
    # atom count) and the button to enable.
    page.wait_for_function(
        "() => {"
        "  const btn = document.getElementById('auto-detect-btn');"
        "  return !!btn && !btn.disabled;"
        "}",
        timeout=10000,
    )
    page.evaluate(
        "() => document.getElementById('auto-detect-btn').click()"
    )

    # The chip appears AFTER the response lands.  Selector covers
    # both Profile + Budget chip slots (the analyzer's verdict is
    # surfaced on both per the workflow-group framework).
    page.wait_for_function(
        "() => {"
        "  const chips = document.querySelectorAll("
        "    '.workflow-detection-chip');"
        "  for (const c of chips) {"
        "    if (c.textContent && c.textContent.trim().length > 0) "
        "      return true;"
        "  }"
        "  return false;"
        "}",
        timeout=8000,
    )

    chip_texts = page.evaluate(
        "() => Array.from(document.querySelectorAll("
        "  '.workflow-detection-chip'"
        ")).map(c => c.textContent || '')"
    )
    # At least one chip must have rendered the Profile-line text
    # (the one we care about for shell-classification).  Find the
    # first non-empty chip with metal cluster prose.
    profile_chips = [t for t in chip_texts if "cluster" in t]
    assert profile_chips, (
        f"No detection chip surfaced a ``cluster`` line for the "
        f"Au4 fixture.  Chip texts captured: {chip_texts!r}.  "
        f"Expected at least one chip to read ``4 atoms · Au "
        f"cluster · closed-shell singlet``."
    )
    for text in profile_chips:
        assert "closed-shell" in text, (
            f"Profile chip for Au4 cluster reads {text!r}; expected "
            f"``closed-shell singlet``.  This is the 2026-06-13 "
            f"noble-metal inversion regression class -- the analyzer "
            f"correctly classifies Au clusters as closed-shell, but "
            f"some path between analyzer.suggested_treatment and the "
            f"chip text is mis-reading the verdict."
        )
        assert "open-shell" not in text, (
            f"Profile chip for Au4 cluster reads {text!r}; contains "
            f"``open-shell``.  Pure Au clusters in noble-metal "
            f"context MUST surface as closed-shell singlet per "
            f"chemistry analyzer ``check_noble_metal_cluster``."
        )
