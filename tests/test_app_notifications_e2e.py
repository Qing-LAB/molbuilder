"""App-wide system-notification bar (docs/web/notifications.md).

Pins the contract: a STACK the user clears INDIVIDUALLY, dedup by id, and the
``molbuilder:persist-error`` -> error-row wiring.  The workspace side of that
rule is `workspace.md` § 5's ``onPersistError`` -- *"be told when a background
save fails -- never silent"* -- and § 6's *"tells you when a save to the server
failed"*; this file pins the surface it arrives on.

THE FAILURE IS A REAL ONE: the workspace's state folder
(`workspace_storage._state_dir`) made unwritable, a structure loaded on the
Molbuilder tab, and every save the tab sends after it answered 500 by the real
route; the folder writable again and the next save lands.  The second message
is the tab's own: the Transform tab turning the ruler on says so
(`modify/viewer.js`, `_rulerForTransform`).
"""
from __future__ import annotations

import re

import pytest


pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

from support.sidebar import WAIT_MS, walk_to  # noqa: E402

#: The workspace's two routes that write (`lib/workspace/dispatcher.js`).
_WRITES = re.compile(r"/api/workspace-storage/(write|prune)$")

#: The Molbuilder tab's saves an act ENDS with, by the workspace id they are
#: written under (`dispatcher.js` ``workspaceId``: ``ws-`` and the tag; the
#: tab's viewer is ``modify``, `selection-bootstrap.js`): the view's own lane,
#: written 400 ms after the structure, the selection or the ruler changed
#: (`lib/molview/ui-context.js`), and the draft, written after a Save state's
#: point (`lib/molview/history.js`).  The dispatcher sends its writes one
#: after another, in the order asked, so the act's last one answered is
#: every earlier one answered.
_VIEW_LANE, _DRAFT = "ws-modify-ui", "ws-modify-draft"


@pytest.fixture(scope="module")
def flask_server(isolated_projects_root_module):
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


@pytest.fixture
def unwritable(isolated_projects_root):
    """A structure in the projects tree, and the workspace's state folder in
    made unwritable -- every save refused by the disk -- with the way to
    make it writable again: ``(the structure, give_back)``.  The folder is
    the store's own answer (`workspace_storage._state_dir`), where the suite
    keeps each test's store (`conftest._isolated_workspace_store`)."""
    from molbuilder.web.blueprints.workspace_storage import _state_dir
    xyz = isolated_projects_root.resolve() / "P" / "structure" / "water.xyz"
    xyz.parent.mkdir(parents=True)
    xyz.write_text("3\nwater\nO 0 0 0\nH 0.957 0 0\nH -0.239 0.927 0\n")
    states = _state_dir()
    states.mkdir(parents=True)
    states.chmod(0o500)
    try:
        yield xyz, lambda: states.chmod(0o700)
    finally:
        states.chmod(0o700)


def _ending_with(page, lane, act) -> None:
    """``act`` done, and the save it ends with -- the one to ``lane`` --
    answered."""
    with page.expect_response(
            lambda r: (_WRITES.search(r.url) is not None
                       and (r.request.post_data_json or {})
                       .get("workspace_id") == lane), timeout=WAIT_MS):
        act()


def _load_failing(page, base, xyz):
    """The Molbuilder tab, ``xyz`` loaded from the sidebar while every save
    fails, and the load's saves all in: the workspace's writes the page sent
    and the answers it received, kept as they come --
    ``{"sent": [...], "answered": [status, ...]}``."""
    writes = {"sent": [], "answered": []}
    page.on("request", lambda r: writes["sent"].append(r.url)
            if _WRITES.search(r.url) else None)
    page.on("response", lambda r: writes["answered"].append(r.status)
            if _WRITES.search(r.url) else None)
    page.goto(f"{base}/molbuilder", wait_until="domcontentloaded")
    page.wait_for_selector("#molview-host .molviewer-card canvas",
                           timeout=WAIT_MS)
    walk_to(page, xyz)
    _ending_with(page, _VIEW_LANE,
                 page.locator("#load-candidate-btn").click)
    page.wait_for_selector(".app-notification.app-notification--error",
                           timeout=WAIT_MS)
    return writes


def _failed(answered) -> list:
    """The answers the page takes for a failed save (`dispatcher.js`: a
    response not ``ok``)."""
    return [status for status in answered if not 200 <= status < 300]


def _counted(page, writes, dismissed: int = 0) -> int:
    """The error row's count once it has caught up with the page's saves:
    every save the page sent answered, and the row counting each one that
    failed since it was raised -- all but the ``dismissed`` first, which a
    row dismissed before it counted -- the page's count and the answers read
    at one moment, nothing arriving between the two reads, asked every
    100 ms within the one bound (`support.sidebar.WAIT_MS`): the answers are
    heard by the test and the count is the page's, so no one wait inside the
    page can ask both.  The act's last save is waited on first
    (:func:`_ending_with`), so none is still to be sent."""
    seen = None
    for _ in range(WAIT_MS // 100):
        sent, answered = len(writes["sent"]), list(writes["answered"])
        counted = page.evaluate(
            "() => window.molbuilder.notify.list()"
            "  .filter((r) => r.id === 'persist-error')"
            "  .reduce((n, r) => n + r.count, 0)")
        seen = (sent, answered, counted)
        if ((len(writes["sent"]), writes["answered"]) == (sent, answered)
                and sent == len(answered)
                and counted == len(_failed(answered)) - dismissed):
            return counted
        page.wait_for_timeout(100)
    raise AssertionError(
        f"the saves (sent, answered) and the row's count never agreed: {seen}")


def test_failed_saves_are_one_error_row_counting_each(
        page, flask_server, unwritable):
    """A failed save is said, never silent: one error row with the save
    warning's words, and a burst of failures is that one row with a ``×N``
    counter, not a stack (`notifications.md` § 1, *Dedup by id*; § 3;
    `workspace.md` § 5, ``onPersistError``).

    THE SILENT FAILURES: the bar's listener for the save failure gone -- a
    save that failed shows nothing, and the person keeps working on edits
    the server never got; every failed save of a burst stacking its own row
    -- fifty rows for one failure (§ 1: *one row, not fifty*)."""
    xyz, _ = unwritable
    writes = _load_failing(page, flask_server, xyz)
    n = _counted(page, writes)
    failed = _failed(writes["answered"])
    assert set(failed) == {500}, writes
    rows = page.locator(".app-notification")
    assert rows.count() == 1
    assert "Couldn't save state to disk" in rows.first.inner_text()
    lst = page.evaluate("() => window.molbuilder.notify.list()")
    assert [(r["id"], r["level"], r["count"]) for r in lst] == [
        ("persist-error", "error", n)], (lst, failed)
    assert len(failed) >= 2, (
        f"a load sends more than one save; {len(failed)} failed")
    assert f"×{n}" in rows.first.inner_text()


def test_stack_of_two_and_individual_clearance(page, flask_server, unwritable):
    """Two messages are a stack the person clears one at a time: a row's
    × removes that row alone, *Clear all* takes every row, and the bar
    hides once it is empty (`notifications.md` § 1-§ 2).

    THE SILENT FAILURE: a × that clears the whole bar, or a new message
    replacing the one before it."""
    xyz, _ = unwritable
    writes = _load_failing(page, flask_server, xyz)
    _counted(page, writes)
    # The ruler turned on is the view's to save, and that save fails too.
    _ending_with(page, _VIEW_LANE,
                 page.locator("#optab-btn-transform").click)
    page.wait_for_function("() => window.molbuilder.notify.list().length === 2")
    _counted(page, writes)
    assert page.locator(".app-notification").count() == 2

    # Dismiss ONLY the persist error via its × button; the info row remains.
    err = page.locator(".app-notification--error")
    err.locator(".app-notification__dismiss").click()
    page.wait_for_function("() => window.molbuilder.notify.list().length === 1")
    dismissed = len(_failed(writes["answered"]))
    remaining = page.evaluate("() => window.molbuilder.notify.list()")
    assert remaining[0]["id"] == "ruler-on-transform-tab"
    assert page.locator(".app-notification--error").count() == 0

    # Another save, still refused, raises the warning again beside the
    # tab's message -- and the bar's *Clear all (2)* takes both: the bar
    # hides again.
    _ending_with(page, _DRAFT, page.locator("#save-state").click)
    page.wait_for_function("() => window.molbuilder.notify.list().length === 2")
    _counted(page, writes, dismissed)
    page.locator(".app-notification__clear-all").click()
    page.wait_for_function("() => window.molbuilder.notify.list().length === 0")
    assert page.locator("#app-notifications").is_hidden()


def test_a_save_that_lands_lowers_the_warning_and_only_it(
        page, flask_server, unwritable):
    """Saving recovered -- the warning must come down, and only it
    (`notifications.md` § 3: *a save that succeeds lowers it*, *only that
    row, the others stay*).

    Until 2026-09-05 nothing lowered it: `dispatcher.js` emitted
    ``molbuilder:persist-error`` on failure and had NO success signal at all,
    so the row raised by the first failed write outlived the failure. After a
    server outage that morning the route was answering 200 again while the
    banner still said saving was broken, its ``(×N)`` frozen at the last
    failure, and the only way out was to dismiss it by hand.  `clear(id)` is
    per-row: a recovery that called `clearAll()` would take the other row
    with it, which is why one stands beside the warning here.
    """
    xyz, give_back = unwritable
    writes = _load_failing(page, flask_server, xyz)
    _counted(page, writes)
    _ending_with(page, _VIEW_LANE,
                 page.locator("#optab-btn-transform").click)
    page.wait_for_function("() => window.molbuilder.notify.list().length === 2")

    give_back()
    with page.expect_response(
            lambda r: r.url.endswith("/api/workspace-storage/write") and r.ok,
            timeout=WAIT_MS):
        page.locator("#save-state").click()

    page.wait_for_function(
        "() => !window.molbuilder.notify.list()"
        "  .some(r => r.id === 'persist-error')", timeout=WAIT_MS)
    assert page.locator(".app-notification--error").count() == 0, (
        "the persist warning survived a successful write")
    lst = page.evaluate("() => window.molbuilder.notify.list()")
    assert [r["id"] for r in lst] == ["ruler-on-transform-tab"], (
        f"recovery took an unrelated notification with it: {lst}")
