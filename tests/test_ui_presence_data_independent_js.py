"""L2 source-text invariant: UI presence is data-independent.

2026-06-14 architectural rule (from the hide-frozen-row precedent
+ post-ship audit): controls / panels / sections that the user has
muscle memory of MUST NOT appear and disappear based on data
shape.  Their PRESENCE is a stable affordance; only their EFFECT
depends on data.

Pins:
  * spectra-inspector ``.modes-table .es-col`` headers -- visible
    whatever the data (no ``th.hidden = !anyES`` write); only a route
    with no probe at all, SIESTA's, shows none (`web/spectra.md`
    § 9b.3) -- a property of the route, not of the data.

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
