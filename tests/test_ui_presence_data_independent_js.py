"""L2 source-text invariant: UI presence is data-independent.

2026-06-14 architectural rule (from the hide-frozen-row precedent
+ post-ship audit): controls / panels / sections that the user has
muscle memory of MUST NOT appear and disappear based on data
shape.  Their PRESENCE is a stable affordance; only their EFFECT
depends on data.

Pins from this session:
  * trajectory-inspector hide-frozen toggle -- always visible.  (The
    ``#hide-frozen-row`` wrapper + its ``refreshHideFrozenAvailability``
    show/hide logic were retired in the MolView migration, task #34;
    the toggle now lives in the always-rendered flat force-controls
    block, so there is no JS visibility transition left to pin.)
  * spectra-inspector ``.modes-table .es-col`` headers -- always
    visible (no ``th.hidden = !anyES`` write).
  * mol-viewer-embed Animation Export section -- always visible
    (no ``sect.hidden = !hasAnim`` write).

These tests scan the relevant JS source files for the assignment
patterns that previously hid each element, and FAIL if the
pattern reappears.  Brittle to refactors that rename the elements
or change the hide-trigger, but that's the point: if either side
moves, the test must be revisited.
"""
from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "molbuilder/web/static"


@pytest.fixture(scope="module")
def spectra_core_src() -> str:
    return (STATIC / "lib/spectra/core.js").read_text(encoding="utf-8")


# --------------------------------------------------------------------- #
#  Spectra ES-column headers                                             #
# --------------------------------------------------------------------- #


def test_spectra_es_columns_not_hidden_on_no_es_data(spectra_core_src):
    """The ``.es-col`` table-header elements MUST remain visible
    regardless of whether any mode has electronic_structure data.

    Pre-2026-06-14 ``lib/spectra/core.js:1183`` did
    ``th.hidden = !anyES``, vanishing the entire ES column block
    whenever the loaded results lacked ES data.  Reappearing on
    the NEXT load broke column-position muscle memory.
    """
    # The smoking-gun assignment ``th.hidden = !anyES`` is what
    # vanished the headers pre-fix.  Pin its absence + a few
    # plausible rewrites (``!data``, .hidden = !anyES on a
    # different selector, the ES-column-specific variant).
    forbidden_patterns = [
        "th.hidden = !anyES",
        "th.hidden = !data",
        "es_col.hidden = !",
        ".hidden = !anyES",
    ]
    for bad in forbidden_patterns:
        assert bad not in spectra_core_src, (
            f"forbidden hide-on-no-data pattern in "
            f"lib/spectra/core.js: ``{bad}``.  ES column headers "
            f"are unconditionally visible per the 2026-06-14 "
            f"``UI presence is data-independent`` contract."
        )


# RETIRED 2026-08-01: the two Animation-Export tests.
#
# They read `lib/viewer/mol-viewer-embed.js` and asserted on its source. NO PAGE
# LOADS THAT FILE -- no template links it, no module imports it -- so both tests
# passed while pinning nothing that runs. MolView draws its own Export menu and
# its own frame bar now.
#
# What replaced them, derived from molview.md rather than from the source:
#   * the frame bar appears only once there is more than one frame --
#     tests/test_molview_mount.py::test_the_frame_bar_appears_only_once_there_is_more_than_one_frame
#   * the Export menu is built and reachable --
#     tests/test_molview_mount.py::test_an_open_menu_is_placed_against_its_own_trigger
