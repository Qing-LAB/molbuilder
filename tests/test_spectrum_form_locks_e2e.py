"""The Spectrum tab's PySCF form locks what the orbital check's selection would
ignore (`web/spectra.md` § 9a.1; user, 2026-09-28): the explicit list is live
under `explicit` alone, and the frequency window under `all` alone -- `skip`
selects nothing and `explicit` names its modes, so the window enters neither.

Driven the way a person drives it: the page on a live server, the selection
changed through its own control -- and first, the page as it loads, before any
change: the default `skip` locks all three.

MUTATIONS THIS MUST FAIL AGAINST: the window left out of the lock map (it stays
editable under `skip` and `explicit`, where it enters nothing); the lock
applied on a change only, never at load.
"""
from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

pytestmark = [pytest.mark.e2e]

#: the explicit list, then the window's two bounds -- the catalogue's items
#: under the PySCF form's own prefix (`blueprints/_shared._item_to_field`)
_FIELDS = ("py-es-explicit-indices", "py-freq-min-cm1", "py-freq-max-cm1")


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def test_the_selection_locks_what_it_would_ignore(page, flask_server):
    page.goto(f"{flask_server}/spectrum-calculation")
    page.wait_for_selector("#py-es-mode-selection", timeout=20000)
    at_load = page.evaluate(
        "(ids) => [document.getElementById('py-es-mode-selection').value]"
        ".concat(ids.map(i => !document.getElementById(i).disabled))",
        list(_FIELDS))
    assert at_load == ["skip", False, False, False], at_load
    live = {}
    for choice in ("skip", "all", "explicit"):
        page.select_option("#py-es-mode-selection", value=choice)
        live[choice] = page.evaluate(
            "(ids) => ids.map(i => !document.getElementById(i).disabled)",
            list(_FIELDS))
    assert live == {"skip":     [False, False, False],
                    "all":      [False, True, True],
                    "explicit": [True, False, False]}, live
