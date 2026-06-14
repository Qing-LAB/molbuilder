"""L2 source-text invariant: Generate POSTs across optimization /
spectrum / transport tabs carry in-body ``frozen_atoms`` + ``regions``.

2026-06-14 viewer-is-truth contract: every tab that lets the user
Load a structure and then click Generate MUST hold the file's
labels (frozen_atoms + regions) in tab-local memory at Load time
and ship them in the Generate POST body.  The server's
``apply_labels_to_struct`` (molbuilder/web/blueprints/_shared.py)
applies them verbatim and SKIPS the disk-sidecar re-read when
these keys are present.

Pre-fix every tab relied on the server re-reading a
``.molstruct.json`` next to whatever ``structure_path`` was sent --
which produced silent no-op behaviour when the path drifted (the
BDT-stage-2 incident, see project_round3_audit and the
viewer-is-truth memory).  The in-body labels contract makes that
class of bug impossible: the bytes the user sees in the viewer
travel with the labels the user sees in the panel.

These tests pin the source text at the POST sites so a refactor
can't quietly remove the keys without surfacing here at CI time.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "molbuilder/web/static"

VIEWER_JS = STATIC / "viewer.js"
SPECTRA_CORE = STATIC / "lib/spectra/core.js"
TRANSPORT_CORE = STATIC / "lib/transport/core.js"
SIDECAR_LABELS = STATIC / "lib/structure/sidecar-labels.js"


def _post_body_around(src: str, url_fragment: str, window: int = 6000) -> str:
    """Return the chars surrounding the fetch(...) for url_fragment.
    The POST body shape lives INSIDE the fetch call (``JSON.
    stringify({...})``), so we have to look AFTER the URL token,
    plus a small lead-in to catch any local-variable defaults.
    """
    ix = src.find(url_fragment)
    assert ix > 0, f"no fetch(...) for {url_fragment!r} in source"
    start = max(0, ix - 500)
    end = min(len(src), ix + window)
    return src[start:end]


@pytest.fixture(scope="module")
def viewer_src() -> str:
    return VIEWER_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def spectra_src() -> str:
    return SPECTRA_CORE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def transport_src() -> str:
    return TRANSPORT_CORE.read_text(encoding="utf-8")


# --------------------------------------------------------------------- #
#  Sidecar helper mount                                                  #
# --------------------------------------------------------------------- #


def test_sidecar_labels_helper_exists():
    """The single-source-of-truth client helper for sidecar-label
    fetching MUST live at ``lib/structure/sidecar-labels.js`` so
    every tab uses the same path-computation + 404-tolerance
    rules."""
    assert SIDECAR_LABELS.exists(), (
        f"sidecar-labels helper missing at {SIDECAR_LABELS.relative_to(ROOT)}"
    )
    text = SIDECAR_LABELS.read_text(encoding="utf-8")
    assert "window.molbuilder.sidecarLabels" in text or \
           "root.molbuilder.sidecarLabels" in text, (
        "helper must mount on ``molbuilder.sidecarLabels``"
    )
    assert "fetch" in text and "sidecarPathFor" in text, (
        "helper must expose ``fetch`` and ``sidecarPathFor``"
    )


# --------------------------------------------------------------------- #
#  Optimization tab — both SIESTA and PySCF Generate                     #
# --------------------------------------------------------------------- #


class TestOptimizationTabContract:
    """viewer.js drives /api/build/fdf and /api/build/pyscf."""

    def test_state_holds_frozen_atoms_and_regions(self, viewer_src):
        """The tab-local ``state`` object must include
        ``frozen_atoms`` and ``regions`` so they survive between
        Load and Generate."""
        assert re.search(r"frozen_atoms\s*:\s*\[\s*\]", viewer_src), (
            "state must initialise ``frozen_atoms: []``"
        )
        assert re.search(r"regions\s*:\s*\{\s*\}", viewer_src), (
            "state must initialise ``regions: {}``"
        )

    def test_siesta_post_body_carries_labels(self, viewer_src):
        body = _post_body_around(viewer_src, 'fetch("/api/build/fdf"')
        assert "frozen_atoms:" in body and "state.frozen_atoms" in body, (
            "SIESTA Generate POST body must carry "
            "``frozen_atoms: state.frozen_atoms`` (or || []).  See the "
            "viewer-is-truth contract."
        )
        assert "regions:" in body and "state.regions" in body, (
            "SIESTA Generate POST body must carry ``regions: state.regions``"
        )

    def test_pyscf_post_body_carries_labels(self, viewer_src):
        body = _post_body_around(viewer_src, 'fetch("/api/build/pyscf"')
        assert "frozen_atoms:" in body and "state.frozen_atoms" in body
        assert "regions:" in body and "state.regions" in body

    def test_load_handler_populates_state_from_sidecar(self, viewer_src):
        """The commit-load path must call sidecarLabels.fetch and
        assign the result to state.frozen_atoms / state.regions."""
        assert "sidecarLabels" in viewer_src, (
            "viewer.js must reach sidecarLabels helper at Load time"
        )
        assert "state.frozen_atoms" in viewer_src
        assert "state.regions" in viewer_src


# --------------------------------------------------------------------- #
#  Spectrum tab                                                          #
# --------------------------------------------------------------------- #


class TestSpectrumTabContract:
    """lib/spectra/core.js drives /api/spectra/render."""

    def test_committed_label_state_declared(self, spectra_src):
        assert "_committedFrozenAtoms" in spectra_src, (
            "spectrum tab must declare a committed-load frozen-atoms "
            "variable populated at Load time (not the live sidebar pick)"
        )
        assert "_committedRegions" in spectra_src

    def test_post_body_carries_labels(self, spectra_src):
        """The body literal for /api/spectra/render is declared far
        above the fetch call (many conditional pushes between),
        so a windowed-chunk check is too tight.  Pin file-scope
        instead: the exact key-binding must exist alongside the
        url, both within the same module."""
        assert "/api/spectra/render" in spectra_src
        assert re.search(
            r"frozen_atoms\s*:\s*_committedFrozenAtoms",
            spectra_src,
        ), ("spectrum tab Generate POST must carry "
            "``frozen_atoms: _committedFrozenAtoms`` per the "
            "viewer-is-truth contract.")
        assert re.search(
            r"regions\s*:\s*_committedRegions",
            spectra_src,
        ), ("spectrum tab Generate POST must carry "
            "``regions: _committedRegions``")

    def test_load_handler_populates_label_state(self, spectra_src):
        """``_commitStructureForSpectra`` must call sidecarLabels
        and assign the result to ``_committedFrozenAtoms`` /
        ``_committedRegions``."""
        assert "sidecarLabels" in spectra_src
        # Both vars should appear inside a function definition that
        # is also the commit handler; cheapest is to verify they
        # appear at all and that the assignment shape is right.
        assert re.search(
            r"_committedFrozenAtoms\s*=\s*labels\.frozen_atoms",
            spectra_src,
        ), "spectrum tab must populate _committedFrozenAtoms from sidecar"


# --------------------------------------------------------------------- #
#  Transport tab                                                         #
# --------------------------------------------------------------------- #


class TestTransportTabContract:
    """lib/transport/core.js drives /api/transport/render."""

    def test_committed_label_state_declared(self, transport_src):
        assert "_currentFrozenAtoms" in transport_src
        assert "_currentRegions" in transport_src

    def test_post_body_carries_labels(self, transport_src):
        # Same shape: the body is a JSON.stringify({...}) right
        # before fetch.
        body = _post_body_around(
            transport_src, 'fetch("/api/transport/render"'
        )
        if "/api/transport/render" not in body:
            body = transport_src
        assert ("frozen_atoms:" in body
                and "_currentFrozenAtoms" in body), (
            "transport tab Generate POST must carry "
            "``frozen_atoms: _currentFrozenAtoms``"
        )
        assert ("regions:" in body
                and "_currentRegions" in body)

    def test_load_handler_populates_label_state(self, transport_src):
        assert "sidecarLabels" in transport_src
        assert re.search(
            r"_currentFrozenAtoms\s*=\s*labels\.frozen_atoms",
            transport_src,
        )


# --------------------------------------------------------------------- #
#  Template mounts: all three tabs load the helper                       #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("template_path", [
    "molbuilder/web/templates/index.html",
    "molbuilder/web/templates/spectra.html",
    "molbuilder/web/templates/transport_calculation.html",
])
def test_template_mounts_sidecar_labels_helper(template_path):
    """Each of the three Generate-bearing templates must load
    ``lib/structure/sidecar-labels.js``."""
    path = ROOT / template_path
    assert path.exists(), f"template not found: {template_path}"
    text = path.read_text(encoding="utf-8")
    assert "lib/structure/sidecar-labels.js" in text, (
        f"{template_path} must include "
        "<script src=\"…/lib/structure/sidecar-labels.js\"> before "
        "the tab's own core.js"
    )
