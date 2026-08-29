"""L2 source-text invariant: a Generate POST carries the structure ONCE, from
ONE read of the viewer.

**The rule (molview.md § 9.3):** "after an edit, a request built from the viewer
carries that edit in EVERY part of what it sends -- no piece can be older than
another, because it all came from one read of the structure."

`exportFile()` is that one read: it returns the atoms, their positions at the
displayed frame, the labels and the cell, assembled by the viewer.  A tab sends
that envelope and nothing else about the structure.

**What these tests used to pin, and why it changed (2026-08-03).**  They
required the body to carry `frozen_atoms:` and `regions:` keys, read from
`getFrozen()` / `getRegions()` at emit.  That was the right INSTINCT -- the
labels must be live, never a load-time mirror -- expressed as the wrong
invariant.  Each tab ended up reading the model FOUR times for one request (the
envelope, then the frozen list, then the regions, then the cell), and the server
overwrote the envelope's copy with the later ones, so the envelope was dead
weight and "read together" was false exactly where it is load-bearing.  Every
read was fresh; four fresh reads are still four.

So the invariant is now the stronger one: **one read, and no second copy.**  The
anti-mirror guards below are kept unchanged -- a cached label set is still the
failure they were written for.

These are source-text tests so a refactor cannot quietly reintroduce a second
read path without surfacing.
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


def _assert_one_read(body: str, what: str) -> None:
    """The body carries the viewer's own envelope, and nothing that repeats it.

    `exportFile()` already holds the labels and the cell.  A body that ALSO
    sends `frozen_atoms` / `regions` / `periodicity` read them again, at another
    moment, from another call -- which is the failure § 9.3 names: a request
    whose pieces are not all the same age.
    """
    assert "structure:" in body, (
        f"{what} POST does not carry a `structure` at all: it must send the "
        f"viewer's own envelope"
    )
    assert "exportFile" in body or "_structureForRequest" in body or "_out.structure" in body, (
        f"{what} POST builds its structure some other way than asking the "
        f"viewer for it -- a hand-built envelope is how the cell ended up under "
        f"a `metadata.periodicity` key the receiver refuses"
    )
    for repeated in ("frozen_atoms:", "regions:", "periodicity:"):
        assert repeated not in body, (
            f"{what} POST sends `{repeated}` beside the envelope, which already "
            f"carries it. That is a SECOND read of the same fact at a second "
            f"moment -- and the server prefers the later copy, so the envelope "
            f"stops being what is judged (molview.md § 9.3)"
        )


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
    """The tab's POST body is ONE read of the viewer.

    Symmetric with the Spectrum + Transport contracts below: labels are read
    FRESH off the model (``getFrozen`` / ``getRegions``) at request time --
    there is NO ``state.*`` mirror to desync (unified-API access; the 2026-07
    audit removed the load-time mirror this tab alone still kept).

    **The two Generate POSTs are gone** (2026-08-15): the tab collects
    parameters and hands them on rather than producing a deck
    (``web/task-setup-plan.md`` 2 -- *the browser describes and observes, the
    terminal acts*).  ``/api/build/preflight`` is the surviving POST and it
    carries the same property, so the guard moved onto it rather than being
    parked.  That matters: the property is about how the body is ASSEMBLED,
    not about which endpoint receives it, and the next surface this tab posts
    to will need it just as much.
    """

    def test_preflight_body_is_one_read_of_the_viewer(self, viewer_src):
        """The live panel must judge the structure on screen, not a cell
        fetched a moment after the atoms."""
        _assert_one_read(
            _post_body_around(viewer_src, 'fetch("/api/build/preflight"'),
            "preflight")

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
    """lib/spectra/core.js hands over through lib/task-handover.js,
    reading the structure off molview.data at Send time (P3: the
    render POST retired with its route)."""

    def test_send_is_one_read_of_the_viewer(self, spectra_src):
        """The send carries the envelope and no second copy of it."""
        ix = spectra_src.find("async function sendToTaskSetup")
        assert ix > 0, "the send handler moved; this test cannot see it"
        body = spectra_src[ix:ix + 1600]
        assert "taskHandover.send" in body
        _assert_one_read(body, "spectra send")

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
    """lib/transport/core.js describes the COMPOSITE (P7b): the tab
    posts NO structure at all -- its structure IS the junction citation,
    and the labels travel with the cited calculation's own files."""

    def test_the_tab_posts_no_structure(self, transport_src):
        """The composite's structure arrives at prep from the citation
        (transport-design.md 4.1).  A structure in the tab's send would
        be a SECOND source for the same facts -- exactly the class
        molview.md 9.3a exists to prevent, one level up."""
        assert "calculation: \"transport\"" in transport_src
        assert "junction: _junction" in transport_src
        code = re.sub(r"/\*.*?\*/", "", transport_src, flags=re.S)
        code = re.sub(r"^\s*//.*$", "", code, flags=re.M)
        assert "api/transport/render" not in code, (
            "the Generate lane retired with the bundle road (P7)")
        for retired in ("frozen_atoms", "getFrozen", "getRegions"):
            assert retired not in code, (
                f"``{retired}`` is back in the transport tab's code -- "
                f"the composite reads nothing off the viewer.")

    def test_no_stale_current_label_cache(self, transport_src):
        """The old ``_current*`` load-time cache was removed with the
        MolView migration."""
        assert "_currentFrozenAtoms" not in transport_src
        assert "_currentRegions" not in transport_src
