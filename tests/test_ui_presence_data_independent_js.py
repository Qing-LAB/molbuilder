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


@pytest.fixture(scope="module")
def viewer_embed_src() -> str:
    return (STATIC / "lib/mol-viewer-embed.js").read_text(
        encoding="utf-8"
    )


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


# --------------------------------------------------------------------- #
#  mol-viewer-embed Animation Export section                             #
# --------------------------------------------------------------------- #


def test_animation_export_section_never_hidden(viewer_embed_src):
    """The Animation Export section MUST remain visible regardless
    of whether an animation is currently mounted.

    Pre-2026-06-14 ``lib/mol-viewer-embed.js:1768`` did
    ``sect.hidden = !hasAnim``, vanishing the Save/Download
    buttons (and their label) on every static-structure load.
    A user who learned the section was on the Export tab didn't
    find it on the next load.
    """
    bad_patterns = [
        "sect.hidden = !hasAnim",
        "section.hidden = !hasAnim",
        'sect.hidden = !state.current.animation',
        # The aria-true form is a common variant.
        'aria-hidden", "true"',  # only flag if specifically on the section
    ]
    # Only the substring search; aria-hidden is OK elsewhere in the
    # file (knob buttons, etc.).  We narrow by requiring ``data-
    # section="animation"`` to appear in the same window.
    assert "sect.hidden = !hasAnim" not in viewer_embed_src, (
        "lib/mol-viewer-embed.js must not write "
        "``sect.hidden = !hasAnim`` -- the Animation Export "
        "section is unconditionally visible per the 2026-06-14 "
        "UI-presence contract.  Disable the buttons inside when "
        "there's no animation instead."
    )


def test_animation_buttons_disabled_when_no_anim(viewer_embed_src):
    """The contract's other half: when there's no animation, the
    buttons inside the Animation Export section MUST be disabled
    so the user sees the no-op state visibly.  Pin the disable
    pattern so a future refactor that drops it (and reverts to
    hide-the-section) doesn't slip past."""
    # The fix uses ``btn.setAttribute("disabled", "")`` on each
    # button inside the section.
    assert "setAttribute(\"disabled\", \"\")" in viewer_embed_src \
        or 'setAttribute("disabled", "")' in viewer_embed_src, (
            "Animation Export section MUST disable its buttons "
            "when no animation is mounted (the data-independent-"
            "presence contract requires this)."
        )
    assert 'data-section="animation"' in viewer_embed_src, (
        "regression: the ``data-section=\"animation\"`` selector "
        "was removed; the disable-pattern guard above relies on "
        "the section being identifiable.  Update both."
    )
