"""The Transport tab, driven the way a person drives it: the page on a live
server, the junction cited through the tree picker, a rung's value typed into
its tab, the Send pressed.

PINS: ``docs/engines/siesta.md`` § 6.1 (a component the kind fixes is drawn
locked, its reason beside it); ``docs/web/form-schema.md`` § 1.1's ``fixed``
row; ``docs/workflow.md`` § 9 (the description's own check, gate ③, at every
describe); ``docs/web/handover-procedure.md`` § 2.1 and
``docs/science/validation.md`` § 4.1 R2, R2a (the findings go to the tab's
findings panel through the one renderer; the status line says how many came
back and the road from here).

PREVENTS, each read in the code before 2026-09-30:

* the transport axis's k count an editable box on the shared panel, whose
  value no rung reads;
* the Transport tab's Send running only the codec: a rung's value past its
  hard limit refused first at prep, and a bias point outside its item's
  recommended range said nowhere;
* the hand-over listing each notice as a bullet in the status line -- a
  second renderer -- and a refusal's findings dropped.

Nothing here launches an engine: the Send writes the description and stops.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

from test_transport_prep import _CITE, _junction_struct, _write_junction

pytestmark = [pytest.mark.e2e]

_MS = 20000


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


@pytest.fixture
def tab(page, flask_server, isolated_projects_root):
    """The Transport tab open on an empty folder for the calculation, a
    concluded junction relaxation in the tree beside it -- the sidebar's
    selection is the folder, as the person leaves it after making one."""
    _write_junction(isolated_projects_root, _junction_struct())
    dest = isolated_projects_root / "J" / "transport" / "T"
    dest.mkdir(parents=True)
    slot = json.dumps(str(dest))
    page.add_init_script(
        "try {"
        f" sessionStorage.setItem("
        f"'molbuilder.current_dir.transport-calculation', {slot});"
        f" sessionStorage.setItem('molbuilder.current_dir', {slot});"
        "} catch (_) {}")
    errors: list = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{flask_server}/transport-calculation")
    page.wait_for_selector("#transport-rung-tabs .tab-btn", timeout=_MS)
    page.wait_for_selector("#transport-shared-container #t-kgrid-z",
                           timeout=_MS)
    # the picker opens on the projects root, which the sidebar resolves
    # once at load: a click before that opens nothing
    page.wait_for_function(
        "() => !!(window.molbuilder && window.molbuilder.projects"
        " && window.molbuilder.projects.getProjectsRoot())", timeout=_MS)
    yield page, dest
    assert not errors, errors


def _cite(page):
    """Cite the relaxation through the picker: expand the tree down to the
    attempt, choose it, confirm; the shared panel then answers the cited
    deck's values."""
    page.locator("#transport-junction-btn").click()
    *folders, attempt = _CITE.split("/")
    for name in folders:
        twisty = page.locator(
            f"dialog li.tp-node[data-path$='/{name}'] > .tp-row > .tp-twisty")
        twisty.wait_for(state="visible", timeout=_MS)
        twisty.click()
    row = page.locator(
        f"dialog li.tp-node[data-path$='/{folders[-1]}/{attempt}'] > .tp-row")
    row.wait_for(state="visible", timeout=_MS)
    row.click()
    page.locator("dialog [data-action='confirm']").click()
    page.wait_for_function(
        "() => !document.getElementById('transport-send-btn').disabled",
        timeout=_MS)
    # the cited deck's kgrid is 4 4 2: its x count on the panel is the sign
    # the panel was drawn again for the citation
    page.wait_for_function(
        "() => { const x = document.querySelector("
        "'#transport-shared-container #t-kgrid-x');"
        " return !!x && x.value === '4'; }", timeout=_MS)


def _send(page):
    """Press Send; the describe door's answer."""
    with page.expect_response("**/api/transport/describe") as answer:
        page.locator("#transport-send-btn").click()
    return answer.value


def _rows(page):
    return [(r.get_attribute("data-severity"),
             r.locator(".issue-msg").inner_text())
            for r in page.locator(
                "#transport-send-findings li.issue-item").all()]


def test_the_transport_axis_is_drawn_locked_with_its_reason(tab):
    """The shared panel's third k count is the transport axis, which no rung
    reads: it is drawn locked at 1 with the reason beside it, and the x and
    y counts stay the person's.

    MUTATIONS THIS MUST FAIL AGAINST: the schema not sending `fixed` (the
    box is editable); the triple ignoring it (the same); the reason not
    written beside the control."""
    from molbuilder.kmesh import WHY_OPEN
    page, _dest = tab
    _cite(page)
    z = page.locator("#transport-shared-container #t-kgrid-z")
    assert z.is_disabled() and z.input_value() == "1"
    assert page.locator("#transport-shared-container #t-kgrid-x").is_enabled()
    reason = page.locator(
        "#transport-shared-container label:has(#t-kgrid-z) .lock-reason")
    assert reason.inner_text() == f"↳ z is fixed at 1: {WHY_OPEN}"


def test_the_send_says_what_the_description_check_found(tab):
    """A lead sampled once along transport is refused at the Send, where
    changing it is still free: the refusal's finding is a row in the tab's
    findings panel and nothing is written.  Put right, with a bias point
    outside its recommended range, the Send writes the description and the
    warning is a row of its own -- the refusal's row gone, since every send
    redraws the panel -- while the status line says how many came back and
    the road from here, and repeats none of them.

    MUTATIONS THIS MUST FAIL AGAINST: the describe door not running the
    description's own check (the refused value describes; the far point says
    nothing); the tab handing the hand-over no findings panel, or the page
    not loading the renderer (no rows); a refusal's findings not handed on;
    the status line listing the notices again."""
    page, dest = tab
    _cite(page)

    page.locator("#transport-rung-tabs .tab-btn[data-tab='electrode_L']"
                 ).click()
    kz = page.locator("#transport-rung-panel-electrode_L #t-electrode-kz")
    kz.wait_for(state="visible", timeout=_MS)
    kz.fill("1")
    assert _send(page).status == 400
    page.wait_for_selector("#transport-send-findings li.issue-item",
                           timeout=_MS)
    refused = [m for s, m in _rows(page) if s == "error"]
    assert len(refused) == 1, _rows(page)
    assert ("stage 'electrode_L' sets electrode_kz = 1: it must be greater "
            "than 1") in refused[0], refused
    assert not (dest / "task.json").exists()

    kz.fill("40")
    page.locator("#transport-bias").fill("0.0, 6.0")
    assert _send(page).status == 200
    status = page.locator("#transport-send-status")
    status.locator("text=came back").wait_for(timeout=_MS)
    rows = _rows(page)
    assert not [r for r in rows if r[0] == "error"], rows
    assert [r for r in rows if r[0] == "warn" and
            "the bias list holds 6 V, outside the recommended range" in r[1]
            ], rows
    line = status.inner_text()
    count = "one finding" if len(rows) == 1 else f"{len(rows)} findings"
    assert f"{count} came back — read them" in line, line
    assert "Next: prep run seed" in line, line
    assert not [m for _s, m in rows if m in line], line
    assert json.loads((dest / "task.json").read_text())["bias"] == {
        "voltages_v": [0.0, 6.0]}
