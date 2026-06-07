"""Tab routing contract — canonical paths + 301 redirects + nav bar.

Pins the routing contract from `docs/tabs/architecture.md` § 3:

  * `/structure`, `/structure-optimization`,
    `/spectrum-calculation`, `/transport-calculation`, `/results`
    each render their tab's template and return 200.

  * Legacy URLs 301-redirect to their new homes:
      `/` → `/structure`
      `/modify` → `/structure`
      `/spectra` → `/spectrum-calculation`

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
    ("/structure",               "selection-host"),
    ("/structure-optimization",  "build-btn"),
    ("/spectrum-calculation",    "generate-btn"),
    ("/transport-calculation",   "Transport calculation"),
    ("/results",                 "results-current-file"),
])
def test_canonical_path_renders(web, path, expected_in_body):
    """Each new canonical path must return 200 and render the
    expected template body."""
    r = web.get(path)
    assert r.status_code == 200, (
        f"{path!r} returned {r.status_code}; expected 200"
    )
    body = r.get_data(as_text=True)
    assert expected_in_body in body, (
        f"{path!r} body missing expected marker {expected_in_body!r}"
    )


# --------------------------------------------------------------------- #
#  Legacy URLs — 301 redirects                                          #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("legacy,canonical", [
    ("/",        "/structure"),
    ("/modify",  "/structure"),
    ("/spectra", "/spectrum-calculation"),
])
def test_legacy_path_301_redirects(web, legacy, canonical):
    """Every old URL the previous tab set used must 301-redirect to
    its new home so existing bookmarks survive.  The status MUST be
    301 (permanent), not 302, so search engines and clients update
    their references."""
    r = web.get(legacy)
    assert r.status_code == 301, (
        f"{legacy!r} returned {r.status_code}; expected 301 permanent"
    )
    assert r.headers["Location"].endswith(canonical), (
        f"{legacy!r} redirected to {r.headers.get('Location')!r}; "
        f"expected {canonical!r}"
    )


# --------------------------------------------------------------------- #
#  Nav bar — every canonical page shows the 5-tab nav in order          #
# --------------------------------------------------------------------- #


_EXPECTED_NAV_LINKS = [
    ("/structure",               "Structure"),
    ("/structure-optimization",  "Structure optimization"),
    ("/spectrum-calculation",    "Spectrum calculation"),
    ("/transport-calculation",   "Transport calculation"),
    ("/results",                 "Results"),
]


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
    ("/structure",               "/structure"),
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
