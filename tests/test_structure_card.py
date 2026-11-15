"""The inspect card the Structure-optimization and Spectrum tabs share
(`lib/structure-card.js`, `web/tabs.md` § 3, `web/spectra.md` § 5): a
structure loaded from the sidebar comes back when the tab is shown again, as
the tab's own draft, and so do what the forms hold, kept in this browser tab
-- on both tabs, said in one sentence.

THE SILENT FAILURES: each tab carried its own copy of the card.  The Spectrum
tab's lost the structure and the form on a reload (the user, 2026-11-14: keep
them, as the Structure-optimization tab does); the Structure-optimization
tab's counted a file loaded before it had, so a load that failed could not be
asked again by a double-click; and the two said a load in two sentences.

And what goes on the canvas goes on one at a time, the newest asked winning.
THE SILENT FAILURE: a second file double-clicked while the first was still
loading landed and was named -- then the first's late answer replaced it on
the canvas, so the tab described, and would have sent, the first structure
while its readout named the second.

Driven as a person drives the page: the road's water saved as a pair in the
projects tree, picked in the sidebar, loaded with the Load button, a charge
typed, the page reloaded; for the race, a slow server is the first load's
answer held back until the second file is asked.
"""
from __future__ import annotations

import re

import pytest


from support.sidebar import WAIT_MS, committed, picked


#: A second structure beside the road's water: the road's H2, as a pair.
SECOND = "P/structure/h2.xyz"

TABS = ["structure-optimization", "spectrum-calculation"]


@pytest.fixture(scope="module")
def tree(isolated_projects_root_module):
    """The projects tree, the road's water saved in it as a pair, and its H2
    beside it (:data:`SECOND`), each saved as the road saves one."""
    from support.road import H2, WATER, structure_saved
    structure_saved(isolated_projects_root_module, WATER)
    structure_saved(isolated_projects_root_module, H2, at=SECOND)
    return isolated_projects_root_module


@pytest.mark.parametrize("tab", TABS)
def test_a_describing_tab_keeps_its_structure_and_forms_across_a_reload(
        page, app_at, tree, tab):
    from playwright.sync_api import expect
    from support.road import STRUCTURE
    picked(page, app_at, tab, tree / STRUCTURE)
    page.click("#load-from-sidebar-btn")
    status = page.locator("#load-status")
    expect(status).to_have_text("Loaded structure.xyz — 3 atoms.",
                                timeout=WAIT_MS)
    # THE CHARGE OF THE FORM ON SHOW -- the engine form the strip shows on
    # Structure optimization, the shared charge and spin on Spectrum.
    charge = page.get_by_label("Net charge").filter(visible=True).first
    committed(page, charge, "-1")
    page.reload()
    expect(status).to_have_text("Restored structure.xyz — 3 atoms.",
                                timeout=WAIT_MS)
    expect(page.locator("#info-atoms")).to_have_text("3")
    expect(charge).to_have_value("-1", timeout=WAIT_MS)


@pytest.mark.parametrize("tab", TABS)
def test_a_file_picked_while_another_loads_is_the_one_the_tab_holds(
        page, app_at, tree, tab):
    """The water's load answered late; the H2 double-clicked meanwhile is
    what the tab holds once both have answered -- read off the electrons the
    chemistry card counts on the structure the tab would send, asked again
    by a charge typed afterwards (H2 at -1: 3; the water: 11)."""
    from playwright.sync_api import expect
    from support.road import STRUCTURE
    from support.sidebar import walk_to
    picked(page, app_at, tab, tree / STRUCTURE)
    held = []
    page.route("**/api/build/load",
               lambda r: r.continue_() if held else held.append(r))
    page.click("#load-from-sidebar-btn")
    for _ in range(WAIT_MS // 50):
        if held:
            break
        page.wait_for_timeout(50)
    assert held, "the water's load never asked the server"
    walk_to(page, tree / SECOND, use=True)
    with page.expect_response(
            lambda r: r.request.post_data == held[0].request.post_data):
        held[0].continue_()
    expect(page.locator("#load-status")).to_have_text(
        "Loaded h2.xyz — 2 atoms.", timeout=WAIT_MS)
    committed(page, page.get_by_label("Net charge").filter(visible=True).first,
              "-1")
    counted = page.locator("dt:text-is('Electrons') + dd .chemistry-value")
    expect(counted.filter(has_text=re.compile(r"^3$"))).not_to_have_count(
        0, timeout=WAIT_MS)
