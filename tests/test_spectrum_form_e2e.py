"""The Spectrum tab's form, in the browser, driven the way a person drives it:
the page on a live server, a structure picked in the sidebar and loaded, the
form's own controls changed.
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
    """The Spectrum tab's PySCF form locks what the orbital check's selection
    would ignore (`web/spectra.md` § 9a.1; user, 2026-09-28): the explicit
    list is live under `explicit` alone, and the frequency window under
    `all` alone -- `skip` selects nothing and `explicit` names its modes, so
    the window enters neither.

    First the page as it loads, before any change: nothing chosen -- a new
    calculation's form holds what the person gives it, and a blank is not
    chosen (`web/form-schema.md`, plan K7) -- and the kind's `skip`, which
    then applies, locks all three.

    MUTATIONS THIS MUST FAIL AGAINST: the window left out of the lock map
    (it stays editable under `skip` and `explicit`, where it enters
    nothing); the lock applied on a change only, never at load."""
    page.goto(f"{flask_server}/spectrum-calculation")
    page.wait_for_selector("#py-es-mode-selection", timeout=20000)
    at_load = page.evaluate(
        "(ids) => [document.getElementById('py-es-mode-selection').value]"
        ".concat(ids.map(i => !document.getElementById(i).disabled))",
        list(_FIELDS))
    assert at_load == ["", False, False, False], at_load
    live = {}
    for choice in ("skip", "all", "explicit"):
        page.select_option("#py-es-mode-selection", value=choice)
        live[choice] = page.evaluate(
            "(ids) => ids.map(i => !document.getElementById(i).disabled)",
            list(_FIELDS))
    assert live == {"skip":     [False, False, False],
                    "all":      [False, True, True],
                    "explicit": [True, False, False]}, live


def test_every_finding_the_live_check_returns_is_on_the_page(
        page, flask_server, tmp_path, monkeypatch):
    """Every finding the tab's live check returns for the structure loaded is
    drawn on the page, a row each, by the one renderer
    (`science/validation.md` § 4.1 R2, R3; `web/ui-contract.md` § 5.1).

    THE SILENT FAILURE: the tab's script finds no renderer on the page -- it
    reads it off a global that only a separate entry point publishes -- and
    draws nothing, so a structure the check has something to say about
    reads as a clean one.

    Water picked in the sidebar and loaded, as a person loads it; what the
    check answered is read off the page's own request.

    MUTATION THIS MUST FAIL AGAINST: the page without the renderer's
    global entry point."""
    from test_chemistry_card_e2e import (_ready_for_a_pick,
                                         _register_tmp_as_picker_root)
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    water = tmp_path / "w" / "water.xyz"
    water.parent.mkdir()
    water.write_text("3\nwater\nO 0 0 0\nH 0.957 0 0\nH -0.239 0.927 0\n")
    answers = []
    page.on("response", lambda r: answers.append(r)
            if "/api/build/preflight" in r.url else None)
    page.goto(f"{flask_server}/spectrum-calculation",
              wait_until="domcontentloaded")
    _ready_for_a_pick(page)
    page.evaluate(
        "(p) => window.molbuilder.projects.setShared("
        "  p.substring(0, p.lastIndexOf('/')), p)", str(water))
    page.wait_for_function(
        "() => !document.getElementById('load-from-sidebar-btn').disabled",
        timeout=5000)
    with page.expect_response(lambda r: "/api/build/preflight" in r.url,
                              timeout=15000):
        page.locator("#load-from-sidebar-btn").click()
    page.wait_for_load_state("networkidle")
    said = answers[-1].json()["issues"]
    assert said, "the live check said nothing about water"
    drawn = page.evaluate(
        "() => [...document.querySelectorAll('li.issue-item')].map(li => ({"
        "  severity: li.dataset.severity,"
        "  message: li.querySelector('.issue-msg').textContent,"
        "  where: (li.querySelector('.issue-where') || {}).textContent"
        "         || null }))")
    missing = [f for f in ({"severity": i["severity"],
                            "message": i["message"],
                            "where": i.get("where") or None}
                           for i in said)
               if f not in drawn]
    assert not missing, missing
