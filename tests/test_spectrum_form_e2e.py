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


#: Raman for, then its range's two ends -- the shared part's fields, drawn
#: once above the engine strip (`form-schema.md` § 1.3) -- and SIESTA's
#: Light along, in SIESTA's own part, waiting on Compute Raman in the
#: shared one
_RAMAN = ("common-raman-modes", "common-raman-freq-min-cm1",
          "common-raman-freq-max-cm1", "p-optical-direction")


def test_raman_for_is_live_only_while_raman_is_asked(page, flask_server):
    """`web/spectra.md` § 5 step 3: *Raman for* is greyed out until *Compute
    Raman* is ticked, its range's two ends until *Modes in a frequency
    range* is chosen -- each field's `active_when`, followed by the one
    renderer, in the part both engines share; and SIESTA's optical settings,
    in its own part, follow the same tick across the parts.

    At load nothing is chosen: Compute Raman's blank is its default, off,
    so all three are greyed out.

    MUTATIONS THIS MUST FAIL AGAINST: the renderer reading `active_when`
    once at load (ticking leaves them greyed); a blank read as no value
    rather than the default (the range live under Raman for's blank, which
    is all modes); a part looking only in itself for the field it waits on
    (SIESTA's optical settings greyed for good)."""
    page.goto(f"{flask_server}/spectrum-calculation")
    page.wait_for_selector("#common-compute-raman", timeout=20000)
    # THE CHOICES READ AS THEIR WORDS (the catalogue's `choice_labels`),
    # their values the choices themselves.
    assert page.evaluate(
        "() => [...document.getElementById('common-raman-modes').options]"
        ".filter(o => o.value).map(o => [o.value, o.textContent])") == [
        ["all", "All modes"], ["range", "Modes in a frequency range"]]
    # A FIELD IS LIVE WHEN ITS CONTROLS TAKE AN EDIT -- every box of a
    # three-number field (Light along), whose id names its row of boxes.
    live = lambda: page.evaluate(  # noqa: E731
        "(ids) => ids.map(i => [...document.getElementById(i)"
        ".closest('.schema-field').querySelectorAll('input, select')]"
        ".every(c => !c.disabled))",
        list(_RAMAN))
    states = {"at load": live()}
    page.check("#common-compute-raman")
    states["Raman on"] = live()
    page.select_option("#common-raman-modes", value="range")
    states["a range"] = live()
    page.uncheck("#common-compute-raman")
    states["Raman off again"] = live()
    assert states == {"at load":         [False, False, False, False],
                      "Raman on":        [True, False, False, True],
                      "a range":         [True, True, True, True],
                      "Raman off again": [False, False, False, False]}, states


def test_every_finding_the_live_check_returns_is_on_the_page(
        page, flask_server, tmp_path, monkeypatch):
    """Every finding the tab's live check returns for the structure loaded is
    drawn on the page, a row each, by the one renderer
    (`science/validation.md` § 4.1 R2, R3; `web/ui-contract.md` § 5.1).

    THE SILENT FAILURE: the tab's script finds no renderer on the page -- it
    reads it off a global that only a separate entry point publishes -- and
    draws nothing, so a structure the check has something to say about
    reads as a clean one.

    Water set as the sidebar's selection and loaded with the tab's Load
    button; what the check answered is read off the page's own request --
    a finding about the structure among it -- and the rows drawn in the
    tab's findings panel and its form's parts are those findings.

    MUTATIONS THIS MUST FAIL AGAINST: the page without the renderer's
    global entry point; the tab drawing only the findings about a field; a
    finding drawn twice."""
    from support.findings import assert_every_finding_drawn
    from test_chemistry_card_e2e import (_ready_for_a_pick,
                                         _register_tmp_as_picker_root)
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    water = tmp_path / "w" / "water.xyz"
    water.parent.mkdir()
    water.write_text("3\nwater\nO 0 0 0\nH 0.957 0 0\nH -0.239 0.927 0\n")
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
                              timeout=15000) as answered:
        page.locator("#load-from-sidebar-btn").click()
    said = answered.value.json()["issues"]
    assert any(not (i.get("where") or "").startswith("config.")
               for i in said), ("the live check said nothing about water "
                                "itself", said)
    assert_every_finding_drawn(
        page, ["#spectra-issues", "#spectra-form-state", "#spectra-form-common",
               "#spectra-form-pyscf", "#spectra-form-siesta"],
        said)


def test_an_edit_in_the_shared_part_is_checked_live_beside_its_field(
        page, flask_server, tmp_path, monkeypatch):
    """`web/spectra.md` § 5 step 3: the settings both engines share are
    checked as they are edited, like the engine's own -- the live check's
    request follows an edit above the strip, and its finding lands beside
    the field there.

    THE SILENT FAILURE: the check listening to the engine's part alone --
    a Raman range stated nowhere would go unflagged until Send.

    MUTATION THIS MUST FAIL AGAINST: the live check's listener on the
    engine forms' wrapper only (no request after the shared part's edit)."""
    from test_chemistry_card_e2e import (_ready_for_a_pick,
                                         _register_tmp_as_picker_root)
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    water = tmp_path / "w" / "water.xyz"
    water.parent.mkdir()
    water.write_text("3\nwater\nO 0 0 0\nH 0.957 0 0\nH -0.239 0.927 0\n")
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
    page.check("#common-compute-raman")
    with page.expect_response(lambda r: "/api/build/preflight" in r.url,
                              timeout=15000) as answered:
        page.select_option("#common-raman-modes", value="range")
    said = [i for i in answered.value.json()["issues"]
            if i.get("where") == "config.raman_modes"]
    assert said and said[0]["severity"] == "error", said
    page.wait_for_function(
        "() => (document.getElementById('common-raman-modes')"
        ".closest('.schema-field').querySelector('.field-issues') || {})"
        ".textContent", timeout=10000)


def test_a_charge_stated_in_the_chemistry_card_is_what_the_form_carries(
        page, flask_server, tmp_path, monkeypatch):
    """`web/spectra.md` § 5 step 2, `form-schema.md` § 1.3: the charge and
    spin are drawn in the chemistry card, once for both engines, and read
    with the form's other parts -- the live check sends what the Send
    sends (`collectParams`), so a charge typed there is in its request.

    THE SILENT FAILURE: the fields drawn in step 2 but not read with the
    form -- a stated charge never reaching the description, the run
    worked out at another.

    MUTATIONS THIS MUST FAIL AGAINST: the state part left out of the
    form's parts (`_formParts`); the live check not listening in the
    chemistry card."""
    from test_chemistry_card_e2e import (_ready_for_a_pick,
                                         _register_tmp_as_picker_root)
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    water = tmp_path / "w" / "water.xyz"
    water.parent.mkdir()
    water.write_text("3\nwater\nO 0 0 0\nH 0.957 0 0\nH -0.239 0.927 0\n")
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
    page.wait_for_selector("#chemistry-card #common-net-charge",
                           timeout=15000)
    with page.expect_request(lambda r: "/api/build/preflight" in r.url,
                             timeout=15000) as asked:
        page.fill("#common-net-charge", "1")
        page.press("#common-net-charge", "Tab")
    assert asked.value.post_data_json["params"]["net_charge"] == 1
