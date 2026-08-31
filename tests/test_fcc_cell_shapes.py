"""`FCC_ORTHOGONAL_CHOICES` must agree with the ASE it describes.

`science/junction-cell.md` § 2b states which cell shapes each fcc surface can
be built with, and `modify.FCC_ORTHOGONAL_CHOICES` is that table in code.  A
hardcoded copy of a library's behaviour drifts the moment the library changes,
so this builds EVERY combination and asserts ASE agrees -- both directions:
what the table allows must build, and what it withholds must not.

The rule matters because a UI cannot offer a choice that does not exist.  The
Slab panel's "Orthogonal cell" box started unchecked on every surface, and
unchecked is the one setting a (100) slab cannot be built with, so a default
(100) request came back a 400.
"""
from __future__ import annotations

import pytest

from molbuilder.modify import (
    FCC_ORTHOGONAL_CHOICES,
    SUPPORTED_FCC_PLANES,
    _build_ase_slab,
)

#: An (m, n) ASE accepts for every shape it can build at all, so a failure is
#: the plane/orthogonal rule and never the size rule.  (111)-orthogonal needs
#: an even `n`, which is why n = 2.
_MN = (2, 2)


def _builds(plane: str, orthogonal: bool) -> bool:
    try:
        _build_ase_slab("Au", plane, (_MN[0], _MN[1], 3), orthogonal, 4.078)
        return True
    except ValueError:
        return False


@pytest.mark.parametrize("plane", SUPPORTED_FCC_PLANES)
def test_every_plane_has_an_entry(plane):
    assert plane in FCC_ORTHOGONAL_CHOICES, (
        f"fcc({plane}) is offered but the table does not say which cell "
        f"shapes it has, so the panel cannot know what to offer")


@pytest.mark.parametrize("plane", SUPPORTED_FCC_PLANES)
def test_what_the_table_allows_ase_builds(plane):
    for orthogonal in FCC_ORTHOGONAL_CHOICES[plane]:
        assert _builds(plane, orthogonal), (
            f"the table offers orthogonal={orthogonal} on fcc({plane}) but "
            f"ASE cannot build it -- the panel would offer a 400")


@pytest.mark.parametrize("plane", SUPPORTED_FCC_PLANES)
def test_what_the_table_withholds_ase_refuses(plane):
    for orthogonal in (False, True):
        if orthogonal in FCC_ORTHOGONAL_CHOICES[plane]:
            continue
        assert not _builds(plane, orthogonal), (
            f"ASE builds orthogonal={orthogonal} on fcc({plane}) but the "
            f"table withholds it -- the panel hides a shape that works")


def test_the_choice_is_real_on_exactly_one_surface():
    """Not a restatement of the table -- the reason § 2b exists.

    If a future ASE gained non-orthogonal (100), the note the panel shows
    ("there is no choice to make on this surface") would become a lie while
    every test above still passed.
    """
    free = [p for p, v in FCC_ORTHOGONAL_CHOICES.items() if len(v) > 1]
    assert free == ["111"], (
        f"§ 2b says the cell shape is a choice on fcc(111) alone; the table "
        f"now says {free}.  Update the document, then this test")


def test_the_meta_route_serves_it():
    """The panel must read the rule, not carry a copy."""
    from molbuilder.web.app import create_app
    j = create_app(config={}).test_client().get("/api/modify/meta").get_json()
    assert j["orthogonal_choices"] == {
        p: list(v) for p, v in FCC_ORTHOGONAL_CHOICES.items()}
