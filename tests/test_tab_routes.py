"""Tab routing contract — canonical paths + nav bar.

Pins the routing contract from `docs/web/tabs.md` § 3:

  * `/molbuilder`, `/structure-optimization`,
    `/spectrum-calculation`, `/transport-calculation`, `/results`
    each render their tab's template and return 200.

  * `/` is also bound to the Molbuilder page so the bare host
    lands on the interactive workspace.

  * No other route aliases — pre-1.0 cleanup dropped every
    legacy-path 301; renamed paths break by design (commit + push,
    bookmark updates, move on).

  * The nav bar on every canonical page shows all five tabs in the
    documented order with the correct active-tab marker.
"""
from __future__ import annotations

import re

import pytest


@pytest.fixture
def web():
    """Flask test client with default config (no auth, no projects
    root); enough to render templates and follow redirects."""
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client()


# --------------------------------------------------------------------- #
#  Canonical routes — each renders its expected template                #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("path,expected_in_body", [
    ("/molbuilder",              "molview-host"),
    ("/structure-optimization",  "load-from-sidebar-btn"),
    ("/spectrum-calculation",    "send-to-task-setup"),
    ("/transport-calculation",   "Transport calculation"),
    ("/results",                 "results-current-file"),
])
def test_canonical_path_renders(web, path, expected_in_body):
    """Each canonical path must return 200 and render the expected
    template body."""
    r = web.get(path)
    assert r.status_code == 200, (
        f"{path!r} returned {r.status_code}; expected 200"
    )
    body = r.get_data(as_text=True)
    assert expected_in_body in body, (
        f"{path!r} body missing expected marker {expected_in_body!r}"
    )


def test_root_redirects_to_first_tab(web):
    """The bare ``/`` 302-redirects to whichever path is FIRST in
    ``molbuilder.web.tabs.TABS`` — no hard-coded path here so a
    tab reorder picks up the new first tab automatically."""
    from molbuilder.web.tabs import TABS
    expected = TABS[0]["path"]
    r = web.get("/")
    assert r.status_code == 302, (
        f"/ returned {r.status_code}; expected 302 to {expected}"
    )
    assert r.headers["Location"].endswith(expected), (
        f"/ redirected to {r.headers.get('Location')!r}; "
        f"expected {expected!r}"
    )


def test_root_landing_path_follows_TABS(web):
    """If someone reorders TABS so a different tab is first, ``/``
    must redirect to that new first tab without an app.py edit.
    Pins the single-source-of-truth principle."""
    from molbuilder.web.tabs import TABS, landing_path
    assert landing_path() == TABS[0]["path"]


# --------------------------------------------------------------------- #
#  No legacy aliases                                                    #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("legacy_path", ["/modify", "/structure", "/spectra"])
def test_pre_rename_paths_return_404(web, legacy_path):
    """Pre-1.0 cleanup: renamed paths return 404, NOT a 301
    redirect.  Bookmarks to the old URLs break by design; the
    canonical name is the single source of truth in code, tests,
    and docs."""
    r = web.get(legacy_path)
    assert r.status_code == 404, (
        f"{legacy_path!r} returned {r.status_code}; expected 404 "
        f"(no legacy aliases per architecture.md § 3.2)"
    )


# --------------------------------------------------------------------- #
#  Nav bar — every canonical page shows the 5-tab nav in order          #
# --------------------------------------------------------------------- #


# The expected (path, label) pairs come from the same TABS list the
# header iterates over — reordering or renaming a tab in tabs.py
# updates BOTH the rendered nav and this test in one place.
def _expected_nav_links():
    from molbuilder.web.tabs import TABS
    return [(t["path"], t["label"]) for t in TABS]


_EXPECTED_NAV_LINKS = _expected_nav_links()


@pytest.mark.parametrize("page_path", [p for p, _ in _EXPECTED_NAV_LINKS])
def test_nav_bar_lists_all_five_tabs_in_order(web, page_path):
    """Every tab page renders the same 5-tab bar in the same order
    with the same labels.  A regression that drops a tab or reorders
    them surfaces here.

    NOTE: the regex relies on each ``<a class="app-tab">`` containing
    only plain-text label content (no nested ``<svg>`` icons etc.).
    If a future change adds nested HTML inside the tab anchors, switch
    to ``lxml.html`` parsing rather than relaxing this assertion."""
    body = web.get(page_path).get_data(as_text=True)
    # Find every <a class="app-tab ..."> link in body order.
    found = re.findall(
        r'<a[^>]*href="([^"]+)"[^>]*class="app-tab[^"]*"[^>]*>([^<]+)</a>',
        body,
    )
    assert found == _EXPECTED_NAV_LINKS, (
        f"{page_path!r} nav links do not match the contract.\n"
        f"  expected: {_EXPECTED_NAV_LINKS}\n"
        f"  got:      {found}"
    )


@pytest.mark.parametrize("page_path,expected_active_href", [
    ("/molbuilder",              "/molbuilder"),
    ("/structure-optimization",  "/structure-optimization"),
    ("/spectrum-calculation",    "/spectrum-calculation"),
    ("/transport-calculation",   "/transport-calculation"),
    ("/results",                 "/results"),
])
def test_canonical_page_marks_its_own_tab_active(
        web, page_path, expected_active_href):
    """Each canonical page must mark exactly one tab as is-active,
    and it must be its own tab."""
    body = web.get(page_path).get_data(as_text=True)
    # Every link with is-active class.
    active = re.findall(
        r'<a[^>]*href="([^"]+)"[^>]*class="[^"]*is-active[^"]*"',
        body,
    )
    assert active == [expected_active_href], (
        f"{page_path!r} expected only {expected_active_href!r} active; "
        f"got {active!r}"
    )
