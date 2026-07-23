"""L2 source-text invariant: Generate POSTs across optimization /
spectrum / transport tabs carry in-body ``frozen_atoms`` + ``regions``.

Viewer-is-truth contract: every tab that lets the user Load a structure
and then click Generate MUST ship the file's labels (frozen_atoms +
regions) in the Generate POST body.  The server's ``apply_labels_to_struct``
(molbuilder/web/blueprints/_shared.py) applies them verbatim and SKIPS the
disk-sidecar re-read when these keys are present.

**Source of the labels (post MolView migration):** the labels are read
straight off the concealed MolView data model — ``molview.data.getFrozen()``
/ ``getRegions()`` — at Generate time.  The earlier design fetched them via a
client ``modify/structure/sidecar-labels.js`` helper at Load time and cached
them in tab-local ``_committed*`` / ``_current*`` variables; that helper +
those caches were removed when the tabs migrated to reading the model
directly (the sidecar is now applied server-side by ``/api/build/load``).  The
single read at Generate time is the ONE source of truth — no second fetch, no
path-pointer indirection, so the bytes the user sees in the viewer travel with
the labels the user sees in the panel.

These tests pin the source text at the POST sites so a refactor can't quietly
drop the keys (or reintroduce a second read path) without surfacing at CI time.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "molbuilder/web/static"

VIEWER_JS = STATIC / "structure-optimization" / "viewer.js"
SPECTRA_CORE = STATIC / "lib/spectra/core.js"
TRANSPORT_CORE = STATIC / "lib/transport/core.js"


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
#  Optimization tab — both SIESTA and PySCF Generate                     #
# --------------------------------------------------------------------- #


class TestOptimizationTabContract:
    """viewer.js drives /api/build/fdf and /api/build/pyscf.

    Symmetric with the Spectrum + Transport contracts below: labels are read
    FRESH off the model (``getFrozen`` / ``getRegions``) at Generate time -- there
    is NO ``state.*`` mirror to desync (unified-API access; the 2026-07 audit
    removed the load-time mirror this tab alone still kept)."""

    def test_siesta_post_body_carries_labels_from_model(self, viewer_src):
        body = _post_body_around(viewer_src, 'fetch("/api/build/fdf"')
        assert "frozen_atoms:" in body and "getFrozen" in body, (
            "SIESTA Generate POST body must carry ``frozen_atoms`` sourced from "
            "``getFrozen()`` (read FRESH off the model, not a state.* mirror)."
        )
        assert "regions:" in body and "getRegions" in body, (
            "SIESTA Generate POST body must carry ``regions`` from ``getRegions()``"
        )

    def test_pyscf_post_body_carries_labels_from_model(self, viewer_src):
        body = _post_body_around(viewer_src, 'fetch("/api/build/pyscf"')
        assert "frozen_atoms:" in body and "getFrozen" in body
        assert "regions:" in body and "getRegions" in body

    def test_no_stale_label_mirror(self, viewer_src):
        """The old load-time ``state.frozen_atoms`` / ``state.regions`` mirror was
        removed with the 2026-07 unified-API audit -- the model read at Generate
        time is the ONE source; guard against its reintroduction (matches the
        Spectrum + Transport anti-cache guards)."""
        assert "state.frozen_atoms" not in viewer_src, (
            "no state.* label mirror: read getFrozen() fresh at emit"
        )
        assert "state.regions" not in viewer_src, (
            "no state.* label mirror: read getRegions() fresh at emit"
        )


# --------------------------------------------------------------------- #
#  Spectrum tab                                                          #
# --------------------------------------------------------------------- #


class TestSpectrumTabContract:
    """lib/spectra/core.js drives /api/spectra/render, reading labels
    off molview.data at Generate time."""

    def test_post_body_carries_labels_from_model(self, spectra_src):
        """The Generate POST must carry ``frozen_atoms`` / ``regions``
        read from the model (``getFrozen`` / ``getRegions``)."""
        assert "/api/spectra/render" in spectra_src
        # The body literal is declared far above the fetch (a multi-line
        # ternary reads the model), so pin file-scope: the key + the model
        # read must both exist in the module.
        assert "frozen_atoms:" in spectra_src and "getFrozen()" in spectra_src, (
            "spectrum tab Generate POST must carry ``frozen_atoms`` sourced "
            "from ``getFrozen()`` per the viewer-is-truth contract.")
        assert "regions:" in spectra_src and "getRegions()" in spectra_src, (
            "spectrum tab Generate POST must carry ``regions`` sourced from "
            "``getRegions()``")

    def test_no_stale_committed_label_cache(self, spectra_src):
        """The old load-time ``_committed*`` cache was removed with the
        MolView migration — the model read at Generate time is the ONE
        source; guard against its reintroduction."""
        assert "_committedFrozenAtoms" not in spectra_src
        assert "_committedRegions" not in spectra_src


# --------------------------------------------------------------------- #
#  Transport tab                                                         #
# --------------------------------------------------------------------- #


class TestTransportTabContract:
    """lib/transport/core.js drives /api/transport/render, reading labels
    off molview.data at Generate time."""

    def test_post_body_carries_labels_from_model(self, transport_src):
        assert "getFrozen" in transport_src and "getRegions" in transport_src
        assert re.search(
            r"frozen_atoms\s*=\s*[^\n]*getFrozen\s*\(", transport_src
        ), ("transport tab Generate POST must set "
            "``frozen_atoms = …getFrozen()``")
        assert re.search(
            r"regions\s*=\s*[^\n]*getRegions\s*\(", transport_src
        ), ("transport tab Generate POST must set ``regions = …getRegions()``")

    def test_no_stale_current_label_cache(self, transport_src):
        """The old ``_current*`` load-time cache was removed with the
        MolView migration."""
        assert "_currentFrozenAtoms" not in transport_src
        assert "_currentRegions" not in transport_src
